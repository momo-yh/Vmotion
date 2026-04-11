from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from two_ball_motion_common import (
    DepthProbe,
    LocalCorrelation,
    RegionEncoder,
    TwoBallMotionDataset,
    collate_keep_strings,
    compute_regression_metrics,
    ensure_dir,
    feature_coords_from_image,
    plot_history,
    save_json,
    set_seed,
)


def build_loader(data_root: str, split: str, batch_size: int) -> DataLoader:
    dataset = TwoBallMotionDataset(data_root, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_keep_strings)


def make_feature_cache(
    data_root: str,
    split: str,
    cache_path: Path,
    encoder: RegionEncoder,
    correlation: LocalCorrelation,
    target_ball: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return payload["x"], payload["y"]

    loader = build_loader(data_root, split, batch_size=32)
    encoder.eval()
    correlation.eval()
    xs = []
    ys = []
    with torch.no_grad():
        for batch in loader:
            z_t = encoder(batch["img_t"])
            z_t1 = encoder(batch["img_t1"])
            corr = correlation(z_t, z_t1)
            centers = batch[f"{target_ball}_center_2d"]
            yy, xx = feature_coords_from_image(centers, corr.shape[2], corr.shape[3])
            feat = corr[torch.arange(corr.shape[0]), :, yy, xx]
            depth = batch[f"{target_ball}_depth"]
            xs.append(feat.cpu())
            ys.append(depth.cpu())
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    torch.save({"x": x, "y": y}, cache_path)
    return x, y


def run_probe_epoch(model: DepthProbe, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> tuple[float, dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses = []
    preds = []
    targets = []
    for x, y in loader:
        pred = model(x)
        loss = F.l1_loss(pred, y)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    metrics = compute_regression_metrics(pred_np[:, None], target_np[:, None])
    metrics["loss"] = float(np.mean(losses))
    return float(np.mean(losses)), metrics


def mean_depth_baseline(data_root: str, target_ball: str) -> dict[str, float]:
    train_ds = TwoBallMotionDataset(data_root, "train")
    test_ds = TwoBallMotionDataset(data_root, "test")
    train_depth = np.array([float(train_ds[i][f"{target_ball}_depth"]) for i in range(len(train_ds))], dtype=np.float32)
    test_depth = np.array([float(test_ds[i][f"{target_ball}_depth"]) for i in range(len(test_ds))], dtype=np.float32)
    pred = np.full_like(test_depth, fill_value=float(train_depth.mean()))
    mae = float(np.abs(pred - test_depth).mean())
    rmse = float(np.sqrt(((pred - test_depth) ** 2).mean()))
    return {"mae": mae, "rmse": rmse, "mean_depth": float(train_depth.mean())}


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(output_dir / "cache")
    radius = args.radius
    encoder = RegionEncoder(out_channels=128)
    if not args.random_encoder:
        checkpoint = torch.load(args.backbone_checkpoint, map_location="cpu")
        encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in checkpoint["model_state"].items() if k.startswith("encoder.")})
        radius = checkpoint["radius"]
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    correlation = LocalCorrelation(radius=radius)

    x_train, y_train = make_feature_cache(args.data_root, "train", cache_dir / f"{args.target_ball}_train.pt", encoder, correlation, args.target_ball)
    x_val, y_val = make_feature_cache(args.data_root, "val", cache_dir / f"{args.target_ball}_val.pt", encoder, correlation, args.target_ball)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = DepthProbe(input_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_probe_epoch(model, train_loader, optimizer)
        _, val_metrics = run_probe_epoch(model, val_loader, None)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae_m": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae_m": val_metrics["mae"],
            "val_rmse_m": val_metrics["rmse"],
        }
        history.append(row)
        save_json(history, output_dir / "history.json")
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            best_epoch = epoch
            torch.save(
                {
                    "probe_state": model.state_dict(),
                    "target_ball": args.target_ball,
                    "input_dim": int(x_train.shape[1]),
                    "random_encoder": bool(args.random_encoder),
                    "radius": radius,
                },
                output_dir / "best.pt",
            )
        print(f"epoch {epoch:03d} | train_mae={train_metrics['mae']:.6f} | val_mae={val_metrics['mae']:.6f}")

    plot_history(history, output_dir / "training_curve.png", f"Depth Probe ({args.target_ball})", ["train_mae_m", "val_mae_m"])
    save_json(
        {
            "best_val_mae_m": best_val,
            "best_epoch": best_epoch,
            "target_ball": args.target_ball,
            "random_encoder": bool(args.random_encoder),
            "mean_baseline_test": mean_depth_baseline(args.data_root, args.target_ball),
        },
        output_dir / "summary.json",
    )


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    loader = DataLoader(TensorDataset(*load_eval_features(args.features_path)), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = DepthProbe(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["probe_state"])
    _, metrics = run_probe_epoch(model, loader, None)
    save_json(metrics, output_dir / "metrics.json")
    print(metrics)


def load_eval_features(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return payload["x"], payload["y"]


def cache_eval(args: argparse.Namespace) -> None:
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    radius = args.radius
    encoder = RegionEncoder(out_channels=128)
    if not args.random_encoder:
        checkpoint = torch.load(args.backbone_checkpoint, map_location="cpu")
        encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in checkpoint["model_state"].items() if k.startswith("encoder.")})
        radius = checkpoint["radius"]
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    correlation = LocalCorrelation(radius=radius)
    make_feature_cache(args.data_root, args.split, output_path, encoder, correlation, args.target_ball)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate two-ball depth probes on local correlation features.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data-root", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, required=True)
    train_parser.add_argument("--target-ball", type=str, required=True, choices=["small", "large"])
    train_parser.add_argument("--backbone-checkpoint", type=str)
    train_parser.add_argument("--random-encoder", action="store_true")
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--radius", type=int, default=6)
    train_parser.add_argument("--seed", type=int, default=7)

    cache_parser = sub.add_parser("cache")
    cache_parser.add_argument("--data-root", type=str, required=True)
    cache_parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    cache_parser.add_argument("--output-path", type=str, required=True)
    cache_parser.add_argument("--target-ball", type=str, required=True, choices=["small", "large"])
    cache_parser.add_argument("--backbone-checkpoint", type=str)
    cache_parser.add_argument("--random-encoder", action="store_true")
    cache_parser.add_argument("--radius", type=int, default=6)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--features-path", type=str, required=True)
    eval_parser.add_argument("--output-dir", type=str, required=True)
    eval_parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "cache":
        cache_eval(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
