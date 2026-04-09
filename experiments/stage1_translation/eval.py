from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_gen.dataset import MotionPairDataset

from .model import FramePairTranslationRegressor

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
    parser = argparse.ArgumentParser(description="Evaluate the simple camera translation regressor.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    evaluate(
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
