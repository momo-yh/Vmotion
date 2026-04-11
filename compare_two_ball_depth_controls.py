from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def compute_mean_baseline(data_root: str | Path, target_ball: str) -> tuple[float, float]:
    root = Path(data_root)
    train_vals = []
    test_vals = []
    for p in sorted((root / "train").glob("sample_*")):
        meta = load_json(p / "meta.json")
        train_vals.append(float(meta[f"{target_ball}_ball_center_3d_t"][2]))
    for p in sorted((root / "test").glob("sample_*")):
        meta = load_json(p / "meta.json")
        test_vals.append(float(meta[f"{target_ball}_ball_center_3d_t"][2]))

    train_arr = np.array(train_vals, dtype=np.float32)
    test_arr = np.array(test_vals, dtype=np.float32)
    mean_pred = float(train_arr.mean())
    mae = float(np.abs(test_arr - mean_pred).mean())
    rmse = float(np.sqrt(((test_arr - mean_pred) ** 2).mean()))
    return mae, rmse


def make_plot(
    data_root: str | Path,
    trained_small_metrics: str | Path,
    trained_large_metrics: str | Path,
    random_small_metrics: str | Path,
    random_large_metrics: str | Path,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trained_small = load_json(trained_small_metrics)
    trained_large = load_json(trained_large_metrics)
    random_small = load_json(random_small_metrics)
    random_large = load_json(random_large_metrics)

    mean_small_mae, mean_small_rmse = compute_mean_baseline(data_root, "small")
    mean_large_mae, mean_large_rmse = compute_mean_baseline(data_root, "large")

    labels = ["Small ball", "Large ball"]
    mae_values = {
        "trained": [trained_small["mae_m"], trained_large["mae_m"]],
        "random": [random_small["mae_m"], random_large["mae_m"]],
        "mean": [mean_small_mae, mean_large_mae],
    }
    rmse_values = {
        "trained": [trained_small["rmse_m"], trained_large["rmse_m"]],
        "random": [random_small["rmse_m"], random_large["rmse_m"]],
        "mean": [mean_small_rmse, mean_large_rmse],
    }

    x = np.arange(len(labels))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for ax, title, values in [
        (axes[0], "Depth MAE", mae_values),
        (axes[1], "Depth RMSE", rmse_values),
    ]:
        ax.bar(x - width, values["trained"], width, label="Motion-trained")
        ax.bar(x, values["random"], width, label="Random frozen")
        ax.bar(x + width, values["mean"], width, label="Mean baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("meters")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    fig.suptitle("Two-ball depth probe comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "two_ball_depth_probe_comparison.png", dpi=160)
    plt.close(fig)

    summary = {
        "trained_small": trained_small,
        "trained_large": trained_large,
        "random_small": random_small,
        "random_large": random_large,
        "mean_baseline_small": {"mae_m": mean_small_mae, "rmse_m": mean_small_rmse},
        "mean_baseline_large": {"mae_m": mean_large_mae, "rmse_m": mean_large_rmse},
        "output_image": str(output_dir / "two_ball_depth_probe_comparison.png"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two-ball depth probe controls.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--trained-small-metrics", type=str, required=True)
    parser.add_argument("--trained-large-metrics", type=str, required=True)
    parser.add_argument("--random-small-metrics", type=str, required=True)
    parser.add_argument("--random-large-metrics", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    make_plot(
        data_root=args.data_root,
        trained_small_metrics=args.trained_small_metrics,
        trained_large_metrics=args.trained_large_metrics,
        random_small_metrics=args.random_small_metrics,
        random_large_metrics=args.random_large_metrics,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
