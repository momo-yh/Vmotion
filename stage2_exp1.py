from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from stage1_exp2 import SharedFrameEncoder, ensure_dir, get_device, save_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SIZE = 128
FEATURE_SIZE = 8
STRIDE = IMAGE_SIZE // FEATURE_SIZE


def pixel_to_feature_index(coord: torch.Tensor) -> torch.Tensor:
    scaled = torch.floor(coord / STRIDE).to(torch.long)
    return torch.clamp(scaled, 0, FEATURE_SIZE - 1)


class BallCenterDepthProbeDataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.sample_dirs = sorted((self.root / split).glob("sample_*"))
        if not self.sample_dirs:
            raise FileNotFoundError(f"No sample directories found in {(self.root / split)!s}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_dir = self.sample_dirs[index]
        img_t = np.asarray(Image.open(sample_dir / "img_t.png").convert("RGB"), dtype=np.float32) / 255.0
        img_t1 = np.asarray(Image.open(sample_dir / "img_t1.png").convert("RGB"), dtype=np.float32) / 255.0
        with (sample_dir / "meta.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        ball_center_2d_t = torch.tensor(meta["ball_center_2d_t"], dtype=torch.float32)
        ball_center_3d_t = torch.tensor(meta["ball_center_3d_t"], dtype=torch.float32)
        return {
            "img_t": torch.from_numpy(img_t).permute(2, 0, 1),
            "img_t1": torch.from_numpy(img_t1).permute(2, 0, 1),
            "ball_center_2d_t": ball_center_2d_t,
            "depth": ball_center_3d_t[2:3],
        }


class DepthProbe(nn.Module):
    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class FrozenMotionBackboneDepthProbe(nn.Module):
    def __init__(self, backbone_checkpoint: str | Path, device: str) -> None:
        super().__init__()
        payload = torch.load(backbone_checkpoint, map_location=device)
        self.encoder = SharedFrameEncoder()

        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in payload["model"].items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError("No encoder weights found in the Stage 1 Experiment 2 checkpoint.")
        self.encoder.load_state_dict(encoder_state)
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


def build_loader(data_root: str | Path, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = BallCenterDepthProbeDataset(data_root, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    diff = pred - target
    mae = diff.abs().mean()
    rmse = torch.sqrt((diff**2).mean())
    return {"mae": mae, "rmse": rmse}


def run_epoch(
    loader: DataLoader,
    model: FrozenMotionBackboneDepthProbe,
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
            loss = F.l1_loss(pred, target)
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


def plot_history(history: dict[str, list[float]], path: str | Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_mae"], label="train")
    axes[1].plot(epochs, history["val_mae"], label="val")
    axes[1].set_title("depth mae")
    axes[1].set_ylabel("meters")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train(
    data_root: str | Path,
    backbone_checkpoint: str | Path,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    device = get_device(device)
    train_loader = build_loader(data_root, "train", batch_size, shuffle=True)
    val_loader = build_loader(data_root, "val", batch_size, shuffle=False)

    model = FrozenMotionBackboneDepthProbe(backbone_checkpoint, device).to(device)
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
                "backbone_checkpoint": str(backbone_checkpoint),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "device": device,
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
    output_dir = ensure_dir(output_dir)
    device = get_device(device)
    dataset = BallCenterDepthProbeDataset(data_root, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    payload = torch.load(checkpoint, map_location=device)
    backbone_checkpoint = payload["config"]["backbone_checkpoint"]
    model = FrozenMotionBackboneDepthProbe(backbone_checkpoint, device).to(device)
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
        description="Stage 2 Experiment 1: frozen depth probe on the Stage 1 Experiment 2 backbone."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the frozen depth probe.")
    train_parser.add_argument("--data-root", type=str, required=True)
    train_parser.add_argument("--backbone-checkpoint", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--device", type=str, default="auto")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved probe checkpoint.")
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
            backbone_checkpoint=args.backbone_checkpoint,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
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
