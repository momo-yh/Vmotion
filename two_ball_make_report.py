from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from two_ball_motion_common import ensure_dir, load_json, save_json

matplotlib.use("Agg")


def make_bar_plot(stage1_metrics: dict, small_trained: dict, small_random: dict, large_trained: dict, large_random: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(["tx", "ty"], [stage1_metrics["mae_x"], stage1_metrics["mae_y"]], color=["#4c78a8", "#f58518"])
    axes[0].set_title("Stage 1 test MAE")
    axes[0].set_ylabel("meters")

    labels = ["small\ntrained", "small\nrandom", "large\ntrained", "large\nrandom"]
    maes = [small_trained["mae"], small_random["mae"], large_trained["mae"], large_random["mae"]]
    axes[1].bar(labels, maes, color=["#54a24b", "#9d9da1", "#e45756", "#bab0ab"])
    axes[1].set_title("Stage 2 depth MAE")
    axes[1].set_ylabel("meters")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(output_path: Path, metrics: dict) -> None:
    lines = [
        "# Two-Ball XY Translation Report",
        "",
        "## Setup",
        "",
        "- Dataset: `outputs_multiple/datasets/two_ball_xy_translation_dataset`",
        "- Motion scope: left-right and up-down translation only",
        "- Stage 1 model: local-correlation translation decoder",
        "- Stage 2 probe: frozen local-correlation feature to depth",
        "",
        "## Stage 1",
        "",
        f"- test MAE: `{metrics['stage1']['mae']:.6f} m`",
        f"- `tx` MAE: `{metrics['stage1']['mae_x']:.6f} m`",
        f"- `ty` MAE: `{metrics['stage1']['mae_y']:.6f} m`",
        "",
        "## Stage 2",
        "",
        f"- small ball depth MAE, trained encoder: `{metrics['small_trained']['mae']:.6f} m`",
        f"- small ball depth MAE, random encoder: `{metrics['small_random']['mae']:.6f} m`",
        f"- large ball depth MAE, trained encoder: `{metrics['large_trained']['mae']:.6f} m`",
        f"- large ball depth MAE, random encoder: `{metrics['large_random']['mae']:.6f} m`",
        "",
        "## Interpretation",
        "",
        f"- The trained encoder improves small-ball depth MAE by `{(1.0 - metrics['small_trained']['mae'] / metrics['small_random']['mae']) * 100.0:.1f}%` over the random encoder.",
        f"- The trained encoder improves large-ball depth MAE by `{(1.0 - metrics['large_trained']['mae'] / metrics['large_random']['mae']) * 100.0:.1f}%` over the random encoder.",
        "- If the trained encoder clearly beats the random control, the depth signal is coming from motion-trained structure rather than probe capacity alone.",
        "",
        "## Output PNGs",
        "",
        "- `stage1_translation/training_curve.png`",
        "- `stage2_depth_small_trained/training_curve.png`",
        "- `stage2_depth_large_trained/training_curve.png`",
        "- `summary_barplot.png`",
        "- `correlation_visualization/correlation_visualization.png`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact report for the two-ball experiment bundle.")
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    report_dir = ensure_dir(root / "report")
    stage1_metrics = load_json(root / "stage1_translation_eval_test" / "metrics.json")
    small_trained = load_json(root / "stage2_depth_small_trained_eval_test" / "metrics.json")
    small_random = load_json(root / "stage2_depth_small_random_eval_test" / "metrics.json")
    large_trained = load_json(root / "stage2_depth_large_trained_eval_test" / "metrics.json")
    large_random = load_json(root / "stage2_depth_large_random_eval_test" / "metrics.json")

    metrics = {
        "stage1": stage1_metrics,
        "small_trained": small_trained,
        "small_random": small_random,
        "large_trained": large_trained,
        "large_random": large_random,
    }
    make_bar_plot(stage1_metrics, small_trained, small_random, large_trained, large_random, report_dir / "summary_barplot.png")
    save_json(metrics, report_dir / "summary_metrics.json")
    write_report(report_dir / "report.md", metrics)


if __name__ == "__main__":
    main()
