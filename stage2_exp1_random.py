from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stage1_exp2 import SharedFrameEncoder, get_device
from stage2_exp1 import (
    DepthProbe,
    build_loader,
    compute_metrics,
    ensure_dir,
    move_batch,
    pixel_to_feature_index,
    plot_history,
    save_json,
)


class FrozenRandomBackboneDepthProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = SharedFrameEncoder()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.probe = DepthProbe(feature_dim=64)

    def forward(self, img_t: torch.Tensor, ball_center_2d_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat_t = self.encoder(img_t)

        u = pixel_to_feature_index(ball_center_2d_t[:, 0])
        v = pixel_to_feature_index(ball_center_2d_t[:, 1])
        batch_indices = torch.arange(feat_t.shape[0], device=feat_t.device)
        local_feat = feat_t[batch_indices, :, v, u]
        return self.probe(local_feat)


def run_epoch(
    loader: torch.utils.data.DataLoader,
    model: FrozenRandomBackboneDepthProbe,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    model.encoder.eval()
    totals = {"loss": 0.0, "mae": 0.0, "rmse": 0.0}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            pred = model(batch["img_t"], batch["ball_center_2d_t"])
            target = batch["depth"]
            loss = torch.nn.functional.l1_loss(pred, target)
            metrics = compute_metrics(pred, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            for key, value in metrics.items():
                totals[key] += float(value.detach().cpu())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train(
    data_root: str | Path,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    device = get_device(device)
    torch.manual_seed(seed)
    train_loader = build_loader(data_root, "train", batch_size, shuffle=True)
    val_loader = build_loader(data_root, "val", batch_size, shuffle=False)

    model = FrozenRandomBackboneDepthProbe().to(device)
    optimizer = torch.optim.Adam(model.probe.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_rmse": [],
        "val_rmse": [],
    }
    best_val = float("inf")
    best_path = Path(output_dir) / "best.pt"

    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(train_loader, model, optimizer, device)
        val_stats = run_epoch(val_loader, model, None, device)
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_mae"].append(train_stats["mae"])
        history["val_mae"].append(val_stats["mae"])
        history["train_rmse"].append(train_stats["rmse"])
        history["val_rmse"].append(val_stats["rmse"])

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "history": history,
            "config": {
                "data_root": str(data_root),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "device": device,
                "seed": seed,
                "backbone_type": "random_frozen",
            },
        }
        torch.save(payload, Path(output_dir) / "last.pt")
        if val_stats["mae"] < best_val:
            best_val = val_stats["mae"]
            torch.save(payload, best_path)

    plot_history(history, Path(output_dir) / "curves.png")
    save_json(history, Path(output_dir) / "history.json")
    summary = {
        "best_checkpoint": str(best_path),
        "best_val_mae_m": best_val,
        "device": device,
        "seed": seed,
        "backbone_type": "random_frozen",
    }
    save_json(summary, Path(output_dir) / "summary.json")
    return summary


def evaluate(
    data_root: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    from stage2_exp1 import BallCenterDepthProbeDataset
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = ensure_dir(output_dir)
    device = get_device(device)
    dataset = BallCenterDepthProbeDataset(data_root, split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    payload = torch.load(checkpoint, map_location=device)
    model = FrozenRandomBackboneDepthProbe().to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["img_t"].to(device), batch["ball_center_2d_t"].to(device)).cpu().numpy()
            target = batch["depth"].cpu().numpy()
            preds.append(pred)
            targets.append(target)

    pred_arr = np.concatenate(preds, axis=0)
    target_arr = np.concatenate(targets, axis=0)
    diff = pred_arr - target_arr
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    metrics = {
        "split": split,
        "num_samples": int(len(dataset)),
        "mae_m": mae,
        "rmse_m": rmse,
    }
    save_json(metrics, Path(output_dir) / "metrics.json")

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.scatter(target_arr[:, 0], pred_arr[:, 0], s=8, alpha=0.5)
    lo = min(target_arr[:, 0].min(), pred_arr[:, 0].min())
    hi = max(target_arr[:, 0].max(), pred_arr[:, 0].max())
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1)
    ax.set_title("depth")
    ax.set_xlabel("target (m)")
    ax.set_ylabel("pred (m)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "prediction_scatter.png", dpi=160)
    plt.close(fig)
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2 Experiment 1 random control: frozen random encoder plus the same depth probe."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the random frozen depth probe.")
    train_parser.add_argument("--data-root", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--device", type=str, default="auto")
    train_parser.add_argument("--seed", type=int, default=7)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved random-control checkpoint.")
    eval_parser.add_argument("--data-root", type=str, required=True)
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--output-dir", type=str, required=True)
    eval_parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.command == "train":
        train(
            data_root=args.data_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )
        return

    if args.command == "eval":
        evaluate(
            data_root=args.data_root,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
