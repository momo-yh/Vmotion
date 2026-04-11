from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from two_ball_motion_common import (
    TranslationModel,
    TwoBallMotionDataset,
    collate_keep_strings,
    compute_regression_metrics,
    ensure_dir,
    plot_history,
    save_json,
    set_seed,
    translation_scales,
)


def build_loader(data_root: str, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TwoBallMotionDataset(data_root, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_keep_strings)


def run_epoch(model: TranslationModel, loader: DataLoader, optimizer: torch.optim.Optimizer | None, sx: float, sy: float) -> tuple[float, dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    preds = []
    targets = []
    losses = []
    for batch in loader:
        img_t = batch["img_t"]
        img_t1 = batch["img_t1"]
        target = batch["translation_xy"]
        pred, _ = model(img_t, img_t1)
        pred_norm = torch.stack([pred[:, 0] / sx, pred[:, 1] / sy], dim=1)
        target_norm = torch.stack([target[:, 0] / sx, target[:, 1] / sy], dim=1)
        loss = 0.5 * F.mse_loss(pred_norm, target_norm)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    metrics = compute_regression_metrics(pred_np, target_np)
    metrics["loss"] = float(np.mean(losses))
    return float(np.mean(losses)), metrics


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    sx, sy = translation_scales(args.data_root)
    train_loader = build_loader(args.data_root, "train", args.batch_size, shuffle=True)
    val_loader = build_loader(args.data_root, "val", args.batch_size, shuffle=False)
    model = TranslationModel(radius=args.radius)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, sx, sy)
        _, val_metrics = run_epoch(model, val_loader, None, sx, sy)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae_m": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae_m": val_metrics["mae"],
            "val_mae_tx_m": val_metrics["mae_x"],
            "val_mae_ty_m": val_metrics["mae_y"],
        }
        history.append(row)
        save_json(history, output_dir / "history.json")
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "radius": args.radius,
                    "sx": sx,
                    "sy": sy,
                    "best_epoch": best_epoch,
                },
                output_dir / "best.pt",
            )
        torch.save(
            {
                "model_state": model.state_dict(),
                "radius": args.radius,
                "sx": sx,
                "sy": sy,
                "epoch": epoch,
            },
            output_dir / "last.pt",
        )
        print(
            f"epoch {epoch:03d} | train_mae={train_metrics['mae']:.6f} | "
            f"val_mae={val_metrics['mae']:.6f} | val_tx={val_metrics['mae_x']:.6f} | val_ty={val_metrics['mae_y']:.6f}"
        )

    plot_history(history, output_dir / "training_curve.png", "Two-Ball Stage 1 Translation", ["train_mae_m", "val_mae_m"])
    save_json(
        {
            "best_val_mae_m": best_val,
            "best_epoch": best_epoch,
            "radius": args.radius,
            "sx": sx,
            "sy": sy,
        },
        output_dir / "summary.json",
    )


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = TranslationModel(radius=checkpoint["radius"])
    model.load_state_dict(checkpoint["model_state"])
    loader = build_loader(args.data_root, args.split, args.batch_size, shuffle=False)
    _, metrics = run_epoch(model, loader, None, checkpoint["sx"], checkpoint["sy"])
    save_json(metrics, output_dir / "metrics.json")
    print(metrics)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the two-ball local-correlation translation model.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data-root", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--radius", type=int, default=6)
    train_parser.add_argument("--seed", type=int, default=7)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--data-root", type=str, required=True)
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--output-dir", type=str, required=True)
    eval_parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    eval_parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.cmd == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
