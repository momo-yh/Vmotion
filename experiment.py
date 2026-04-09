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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class MotionPairDataset(Dataset):
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

        transform = torch.tensor(meta["T_t_to_t1"], dtype=torch.float32)
        return {
            "img_t": torch.from_numpy(img_t).permute(2, 0, 1),
            "img_t1": torch.from_numpy(img_t1).permute(2, 0, 1),
            "translation": transform[:3, 3],
        }


class FramePairTranslationRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, img_t: torch.Tensor, img_t1: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img_t, img_t1], dim=1)
        return self.head(self.features(x))


def build_loader(data_root: str | Path, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = MotionPairDataset(data_root, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    diff = pred - target
    mae = diff.abs().mean()
    rmse = torch.sqrt((diff**2).mean())
    axis_mae = diff.abs().mean(dim=0)
    return {
        "mae": mae,
        "rmse": rmse,
        "mae_x": axis_mae[0],
        "mae_y": axis_mae[1],
        "mae_z": axis_mae[2],
    }


def run_epoch(
    loader: DataLoader,
    model: FramePairTranslationRegressor,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "mae_x": 0.0, "mae_y": 0.0, "mae_z": 0.0}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            pred = model(batch["img_t"], batch["img_t1"])
            target = batch["translation"]
            loss = F.smooth_l1_loss(pred, target)
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
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_mae"], label="train")
    axes[1].plot(epochs, history["val_mae"], label="val")
    axes[1].set_title("translation mae")
    axes[1].set_ylabel("meters")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, history["val_mae_x"], label="x")
    axes[2].plot(epochs, history["val_mae_y"], label="y")
    axes[2].plot(epochs, history["val_mae_z"], label="z")
    axes[2].set_title("val axis mae")
    axes[2].set_ylabel("meters")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train(
    data_root: str | Path,
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

    model = FramePairTranslationRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_rmse": [],
        "val_rmse": [],
        "val_mae_x": [],
        "val_mae_y": [],
        "val_mae_z": [],
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
        history["val_mae_x"].append(val_stats["mae_x"])
        history["val_mae_y"].append(val_stats["mae_y"])
        history["val_mae_z"].append(val_stats["mae_z"])

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
    dataset = MotionPairDataset(data_root, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    payload = torch.load(checkpoint, map_location=device)
    model = FramePairTranslationRegressor().to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            img_t = batch["img_t"].to(device)
            img_t1 = batch["img_t1"].to(device)
            pred = model(img_t, img_t1).cpu().numpy()
            target = batch["translation"].cpu().numpy()
            preds.append(pred)
            targets.append(target)

    pred_arr = np.concatenate(preds, axis=0)
    target_arr = np.concatenate(targets, axis=0)
    diff = pred_arr - target_arr
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    axis_mae = np.mean(np.abs(diff), axis=0)

    metrics = {
        "split": split,
        "num_samples": int(len(dataset)),
        "mae_m": mae,
        "rmse_m": rmse,
        "mae_x_m": float(axis_mae[0]),
        "mae_y_m": float(axis_mae[1]),
        "mae_z_m": float(axis_mae[2]),
    }
    save_json(metrics, Path(output_dir) / "metrics.json")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ["tx", "ty", "tz"]
    for i, ax in enumerate(axes):
        ax.scatter(target_arr[:, i], pred_arr[:, i], s=8, alpha=0.5)
        lo = min(target_arr[:, i].min(), pred_arr[:, i].min())
        hi = max(target_arr[:, i].max(), pred_arr[:, i].max())
        ax.plot([lo, hi], [lo, hi], color="red", linewidth=1)
        ax.set_title(labels[i])
        ax.set_xlabel("target (m)")
        ax.set_ylabel("pred (m)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "prediction_scatter.png", dpi=160)
    plt.close(fig)
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-file stage1 baseline for camera translation regression.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the baseline model.")
    train_parser.add_argument("--data-root", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--device", type=str, default="auto")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved checkpoint.")
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
