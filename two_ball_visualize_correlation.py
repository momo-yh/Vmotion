from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image

from two_ball_motion_common import (
    LocalCorrelation,
    RegionEncoder,
    TwoBallMotionDataset,
    feature_coords_from_image,
    load_json,
)

matplotlib.use("Agg")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize local correlation windows for the two-ball dataset.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TwoBallMotionDataset(args.data_root, args.split)
    sample = dataset[args.index]
    sample_dir = Path(args.data_root) / args.split / sample["sample_id"]
    img_t = Image.open(sample_dir / "img_t.png").convert("RGB")
    img_t1 = Image.open(sample_dir / "img_t1.png").convert("RGB")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    encoder = RegionEncoder(out_channels=128)
    encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in checkpoint["model_state"].items() if k.startswith("encoder.")})
    encoder.eval()
    correlation = LocalCorrelation(radius=checkpoint["radius"])

    batch_t = sample["img_t"].unsqueeze(0)
    batch_t1 = sample["img_t1"].unsqueeze(0)
    z_t = encoder(batch_t)
    z_t1 = encoder(batch_t1)
    corr = correlation(z_t, z_t1)[0]
    k = 2 * checkpoint["radius"] + 1

    small_y, small_x = feature_coords_from_image(sample["small_center_2d"].unsqueeze(0), corr.shape[1], corr.shape[2])
    large_y, large_x = feature_coords_from_image(sample["large_center_2d"].unsqueeze(0), corr.shape[1], corr.shape[2])
    small_map = corr[:, small_y.item(), small_x.item()].reshape(k, k).cpu().numpy()
    large_map = corr[:, large_y.item(), large_x.item()].reshape(k, k).cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(img_t)
    axes[0, 0].scatter([float(sample["small_center_2d"][0]), float(sample["large_center_2d"][0])], [float(sample["small_center_2d"][1]), float(sample["large_center_2d"][1])], c=["cyan", "orange"], s=35)
    axes[0, 0].set_title("img_t")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_t1)
    meta = load_json(sample_dir / "meta.json")
    axes[0, 1].scatter(
        [meta["small_ball_center_2d_t1"][0], meta["large_ball_center_2d_t1"][0]],
        [meta["small_ball_center_2d_t1"][1], meta["large_ball_center_2d_t1"][1]],
        c=["cyan", "orange"],
        s=35,
    )
    axes[0, 1].set_title("img_t+1")
    axes[0, 1].axis("off")

    im0 = axes[1, 0].imshow(small_map, cmap="viridis")
    axes[1, 0].set_title("small-ball local correlation")
    plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im1 = axes[1, 1].imshow(large_map, cmap="viridis")
    axes[1, 1].set_title("large-ball local correlation")
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_dir / "correlation_visualization.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
