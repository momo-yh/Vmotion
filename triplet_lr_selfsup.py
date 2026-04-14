from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from triplet_lr_common import (
    TripletLRSelfSupModel,
    TripletLRTwoBallDataset,
    collate_keep_strings,
    ensure_dir,
    plot_history,
    save_json,
    set_seed,
)


def build_loader(data_root: str, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TripletLRTwoBallDataset(data_root, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_keep_strings)


def run_epoch(model: TripletLRSelfSupModel, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: str) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses = []
    perpoint_losses = []
    sharpness_losses = []
    abs_x = []
    valid_frac = []
    mask_frac = []
    active_points = []
    peak_conf = []
    for batch in loader:
        out = model(
            batch["img_t"].to(device),
            batch["img_t1"].to(device),
            batch["img_t2"].to(device),
            batch["tau1_x"].to(device),
            batch["tau2_x"].to(device),
            batch["K"].to(device),
        )
        loss = out["loss"]
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        perpoint_losses.append(float(out["perpoint_loss"].item()))
        sharpness_losses.append(float(out["sharpness_loss"].item()))
        valid = out["valid"]
        abs_err_x = torch.abs(out["q_pred_x"] - out["q_actual_x"])
        abs_x.append(float((abs_err_x * valid.float()).sum().item() / valid.float().sum().clamp_min(1.0).item()))
        valid_frac.append(float(valid.float().mean().item()))
        mask_frac.append(float(out["mask"].mean().item()))
        active_points.append(float(out["active_points"].mean().item()))
        peak_conf.append(float(out["peak_a1"].mean().item()))
    return {
        "loss": float(np.mean(losses)),
        "perpoint_loss": float(np.mean(perpoint_losses)),
        "sharpness_loss": float(np.mean(sharpness_losses)),
        "abs_x_feat": float(np.mean(abs_x)),
        "valid_fraction": float(np.mean(valid_frac)),
        "mask_mean": float(np.mean(mask_frac)),
        "active_points": float(np.mean(active_points)),
        "peak_a1_mean": float(np.mean(peak_conf)),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    train_loader = build_loader(args.data_root, "train", args.batch_size, shuffle=True)
    val_loader = build_loader(args.data_root, "val", args.batch_size, shuffle=False)
    model = TripletLRSelfSupModel(
        radius=args.radius,
        corr_temperature=args.corr_temperature,
        lambda_sharp=args.lambda_sharp,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, args.device)
        val_metrics = run_epoch(model, val_loader, None, args.device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_perpoint_loss": train_metrics["perpoint_loss"],
            "train_sharpness_loss": train_metrics["sharpness_loss"],
            "train_abs_x_feat": train_metrics["abs_x_feat"],
            "train_valid_fraction": train_metrics["valid_fraction"],
            "train_mask_mean": train_metrics["mask_mean"],
            "train_active_points": train_metrics["active_points"],
            "train_peak_a1_mean": train_metrics["peak_a1_mean"],
            "val_loss": val_metrics["loss"],
            "val_perpoint_loss": val_metrics["perpoint_loss"],
            "val_sharpness_loss": val_metrics["sharpness_loss"],
            "val_abs_x_feat": val_metrics["abs_x_feat"],
            "val_valid_fraction": val_metrics["valid_fraction"],
            "val_mask_mean": val_metrics["mask_mean"],
            "val_active_points": val_metrics["active_points"],
            "val_peak_a1_mean": val_metrics["peak_a1_mean"],
        }
        history.append(row)
        save_json(history, output_dir / "history.json")
        if val_metrics["loss"] < best_val - args.min_delta:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "radius": args.radius,
                    "best_epoch": best_epoch,
                    "downsample": model.downsample,
                    "depth_min": model.depth_min,
                    "depth_max": model.depth_max,
                    "eps": model.eps,
                    "corr_temperature": model.corr_temperature,
                    "lambda_sharp": model.lambda_sharp,
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "model_state": model.state_dict(),
                "radius": args.radius,
                "epoch": epoch,
                "downsample": model.downsample,
                "depth_min": model.depth_min,
                "depth_max": model.depth_max,
                "eps": model.eps,
                "corr_temperature": model.corr_temperature,
                "lambda_sharp": model.lambda_sharp,
            },
            output_dir / "last.pt",
        )
        print(
            f"epoch {epoch:03d} | train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | val_sharp={val_metrics['sharpness_loss']:.4f} | val_peak={val_metrics['peak_a1_mean']:.3f} | "
            f"val_mask={val_metrics['mask_mean']:.3f} | val_valid={val_metrics['valid_fraction']:.3f}"
        )
        if epochs_without_improvement >= args.patience:
            print(f"early stop at epoch {epoch:03d} | no val-loss improvement for {args.patience} epochs")
            break

    plot_history(history, output_dir / "training_curve.png", "Three-Frame LR Self-Supervised Training", ["train_loss", "val_loss"])
    plot_history(history, output_dir / "sharpness_curve.png", "Sharpness Loss", ["train_sharpness_loss", "val_sharpness_loss"])
    plot_history(history, output_dir / "peak_curve.png", "Peak Confidence", ["train_peak_a1_mean", "val_peak_a1_mean"])
    plot_history(history, output_dir / "mask_curve.png", "Mask Activity", ["train_mask_mean", "val_mask_mean", "train_active_points", "val_active_points"])
    save_json(
        {
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "stopped_early": epochs_without_improvement >= args.patience,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "corr_temperature": model.corr_temperature,
            "lambda_sharp": model.lambda_sharp,
        },
        output_dir / "summary.json",
    )


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = TripletLRSelfSupModel(
        radius=ckpt["radius"],
        downsample=ckpt["downsample"],
        depth_min=ckpt["depth_min"],
        depth_max=ckpt["depth_max"],
        eps=ckpt["eps"],
        corr_temperature=ckpt.get("corr_temperature", 0.07),
        lambda_sharp=ckpt.get("lambda_sharp", 0.02),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    loader = build_loader(args.data_root, args.split, args.batch_size, shuffle=False)
    metrics = run_epoch(model, loader, None, args.device)
    save_json(metrics, output_dir / "metrics.json")
    print(metrics)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the three-frame left-right self-supervised model.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data-root", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--radius", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--device", type=str, default="cpu")
    train_parser.add_argument("--patience", type=int, default=4)
    train_parser.add_argument("--min-delta", type=float, default=1e-4)
    train_parser.add_argument("--corr-temperature", type=float, default=0.07)
    train_parser.add_argument("--lambda-sharp", type=float, default=0.02)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--data-root", type=str, required=True)
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--output-dir", type=str, required=True)
    eval_parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.cmd == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
