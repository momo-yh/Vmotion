from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_generation import ensure_dir, save_json, save_preview_grid, transform_world_to_camera
from data_generation_multiple import (
    LARGE_BALL_COLOR,
    MultiBallDatasetRenderConfig,
    MultiBallSceneRenderer,
    SMALL_BALL_COLOR,
    centers_separated_in_image,
    make_intrinsics,
    make_transform,
    rotation_matrix_to_euler_xyz_deg,
    sample_ball_pair,
    sample_camera_pose,
    scene_quality_ok,
    table_visibility_ok,
    validate_projection,
)


@dataclass(frozen=True)
class TripletLRConfig(MultiBallDatasetRenderConfig):
    motion_translation_min: float = 0.045
    motion_translation_max: float = 0.08
    min_measurable_displacement_px: float = 8.0


def sample_lr_motion(rng: np.random.Generator, cfg: TripletLRConfig) -> np.ndarray:
    magnitude = rng.uniform(cfg.motion_translation_min, cfg.motion_translation_max)
    sign = rng.choice([-1.0, 1.0])
    translation = np.array([sign * magnitude, 0.0, 0.0], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    return make_transform(rotation, translation)


def project_balls(world_to_camera: np.ndarray, K: np.ndarray, small_ball_world: np.ndarray, large_ball_world: np.ndarray, cfg: TripletLRConfig):
    small_3d = transform_world_to_camera(world_to_camera, small_ball_world)
    large_3d = transform_world_to_camera(world_to_camera, large_ball_world)
    valid_small, small_2d, small_r = validate_projection(small_3d, K, cfg.small_ball_radius, cfg)
    valid_large, large_2d, large_r = validate_projection(large_3d, K, cfg.large_ball_radius, cfg)
    return {
        "small_3d": small_3d,
        "large_3d": large_3d,
        "valid": bool(valid_small and valid_large),
        "small_2d": small_2d,
        "large_2d": large_2d,
        "small_r": small_r,
        "large_r": large_r,
    }


def sample_episode(renderer: MultiBallSceneRenderer, rng: np.random.Generator) -> dict[str, np.ndarray]:
    cfg = renderer.cfg
    K = renderer.K
    for _ in range(2048):
        small_ball_world, large_ball_world = sample_ball_pair(rng, cfg)
        world_to_camera_t = sample_camera_pose(rng, small_ball_world, large_ball_world, cfg)
        T_t_to_t1 = sample_lr_motion(rng, cfg)
        T_t1_to_t2 = sample_lr_motion(rng, cfg)
        world_to_camera_t1 = T_t_to_t1 @ world_to_camera_t
        world_to_camera_t2 = T_t1_to_t2 @ world_to_camera_t1

        proj_t = project_balls(world_to_camera_t, K, small_ball_world, large_ball_world, cfg)
        proj_t1 = project_balls(world_to_camera_t1, K, small_ball_world, large_ball_world, cfg)
        proj_t2 = project_balls(world_to_camera_t2, K, small_ball_world, large_ball_world, cfg)
        if not (proj_t["valid"] and proj_t1["valid"] and proj_t2["valid"]):
            continue
        if not centers_separated_in_image(proj_t["small_2d"], proj_t["small_r"], proj_t["large_2d"], proj_t["large_r"]):
            continue
        if not centers_separated_in_image(proj_t1["small_2d"], proj_t1["small_r"], proj_t1["large_2d"], proj_t1["large_r"]):
            continue
        if not centers_separated_in_image(proj_t2["small_2d"], proj_t2["small_r"], proj_t2["large_2d"], proj_t2["large_r"]):
            continue
        disp_ok = all(
            [
                abs(float(proj_t1["small_2d"][0] - proj_t["small_2d"][0])) >= cfg.min_measurable_displacement_px,
                abs(float(proj_t2["small_2d"][0] - proj_t1["small_2d"][0])) >= cfg.min_measurable_displacement_px,
                abs(float(proj_t1["large_2d"][0] - proj_t["large_2d"][0])) >= cfg.min_measurable_displacement_px,
                abs(float(proj_t2["large_2d"][0] - proj_t1["large_2d"][0])) >= cfg.min_measurable_displacement_px,
            ]
        )
        if not disp_ok:
            continue
        table_ok = all(
            [
                table_visibility_ok(world_to_camera_t, K, small_ball_world, cfg),
                table_visibility_ok(world_to_camera_t, K, large_ball_world, cfg),
                table_visibility_ok(world_to_camera_t1, K, small_ball_world, cfg),
                table_visibility_ok(world_to_camera_t1, K, large_ball_world, cfg),
                table_visibility_ok(world_to_camera_t2, K, small_ball_world, cfg),
                table_visibility_ok(world_to_camera_t2, K, large_ball_world, cfg),
            ]
        )
        if not table_ok:
            continue

        img_t = renderer.render(world_to_camera_t, small_ball_world, large_ball_world)
        img_t1 = renderer.render(world_to_camera_t1, small_ball_world, large_ball_world)
        img_t2 = renderer.render(world_to_camera_t2, small_ball_world, large_ball_world)
        if not (scene_quality_ok(img_t, img_t1) and scene_quality_ok(img_t1, img_t2)):
            continue

        return {
            "img_t": img_t.astype(np.uint8),
            "img_t1": img_t1.astype(np.uint8),
            "img_t2": img_t2.astype(np.uint8),
            "K": K.astype(np.float32),
            "T_t_to_t1": T_t_to_t1.astype(np.float32),
            "T_t1_to_t2": T_t1_to_t2.astype(np.float32),
            "small_ball_center_world": small_ball_world.astype(np.float32),
            "large_ball_center_world": large_ball_world.astype(np.float32),
            "world_to_camera_t": world_to_camera_t.astype(np.float32),
            "world_to_camera_t1": world_to_camera_t1.astype(np.float32),
            "world_to_camera_t2": world_to_camera_t2.astype(np.float32),
            "small_ball_center_3d_t": proj_t["small_3d"].astype(np.float32),
            "small_ball_center_3d_t1": proj_t1["small_3d"].astype(np.float32),
            "small_ball_center_3d_t2": proj_t2["small_3d"].astype(np.float32),
            "large_ball_center_3d_t": proj_t["large_3d"].astype(np.float32),
            "large_ball_center_3d_t1": proj_t1["large_3d"].astype(np.float32),
            "large_ball_center_3d_t2": proj_t2["large_3d"].astype(np.float32),
            "small_ball_center_2d_t": proj_t["small_2d"].astype(np.float32),
            "small_ball_center_2d_t1": proj_t1["small_2d"].astype(np.float32),
            "small_ball_center_2d_t2": proj_t2["small_2d"].astype(np.float32),
            "large_ball_center_2d_t": proj_t["large_2d"].astype(np.float32),
            "large_ball_center_2d_t1": proj_t1["large_2d"].astype(np.float32),
            "large_ball_center_2d_t2": proj_t2["large_2d"].astype(np.float32),
        }
    raise RuntimeError("Failed to sample a valid three-frame episode.")


def save_sample(sample: dict[str, np.ndarray], sample_dir: str | Path) -> None:
    sample_dir = ensure_dir(sample_dir)
    Image.fromarray(sample["img_t"]).save(sample_dir / "img_t.png")
    Image.fromarray(sample["img_t1"]).save(sample_dir / "img_t1.png")
    Image.fromarray(sample["img_t2"]).save(sample_dir / "img_t2.png")
    meta = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in sample.items() if not k.startswith("img_")}
    meta["small_ball_radius"] = float(sample["small_ball_center_world"][1])
    meta["large_ball_radius"] = float(sample["large_ball_center_world"][1])
    save_json(meta, sample_dir / "meta.json")


def make_preview_triplet(sample: dict[str, np.ndarray]) -> Image.Image:
    images = [Image.fromarray(sample["img_t"]), Image.fromarray(sample["img_t1"]), Image.fromarray(sample["img_t2"])]
    gap = 12
    footer_h = 68
    width = sum(img.width for img in images) + 2 * gap
    height = images[0].height + footer_h
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x_off = 0
    centers = [
        ("small_ball_center_2d_t", "large_ball_center_2d_t"),
        ("small_ball_center_2d_t1", "large_ball_center_2d_t1"),
        ("small_ball_center_2d_t2", "large_ball_center_2d_t2"),
    ]
    for idx, img in enumerate(images):
        canvas.paste(img, (x_off, 0))
        s_key, l_key = centers[idx]
        sx, sy = map(float, sample[s_key])
        lx, ly = map(float, sample[l_key])
        draw.ellipse((x_off + sx - 3, sy - 3, x_off + sx + 3, sy + 3), fill=tuple(int(c) for c in SMALL_BALL_COLOR))
        draw.ellipse((x_off + lx - 3, ly - 3, x_off + lx + 3, ly + 3), fill=tuple(int(c) for c in LARGE_BALL_COLOR))
        x_off += img.width + gap
    t1 = sample["T_t_to_t1"][:3, 3]
    t2 = sample["T_t1_to_t2"][:3, 3]
    draw.text((6, images[0].height + 8), f"t->t1 tx={t1[0]:+.3f}m, t1->t2 tx={t2[0]:+.3f}m", fill=(40, 40, 40))
    draw.text((6, images[0].height + 28), f"S u: {sample['small_ball_center_2d_t'][0]:.1f}->{sample['small_ball_center_2d_t1'][0]:.1f}->{sample['small_ball_center_2d_t2'][0]:.1f}", fill=(40, 40, 40))
    draw.text((6, images[0].height + 48), f"L u: {sample['large_ball_center_2d_t'][0]:.1f}->{sample['large_ball_center_2d_t1'][0]:.1f}->{sample['large_ball_center_2d_t2'][0]:.1f}", fill=(40, 40, 40))
    return canvas


def generate_split(root: str | Path, split: str, count: int, seed: int, cfg: TripletLRConfig | None = None) -> None:
    root = ensure_dir(root)
    split_dir = ensure_dir(root / split)
    renderer = MultiBallSceneRenderer(cfg)
    rng = np.random.default_rng(seed)
    previews = []
    for index in range(count):
        sample = sample_episode(renderer, rng)
        save_sample(sample, split_dir / f"sample_{index:06d}")
        if len(previews) < cfg.preview_examples_per_split:
            previews.append(make_preview_triplet(sample))
    save_json({"split": split, "count": count, "seed": seed}, split_dir / "manifest.json")
    save_preview_grid(previews, root / f"{split}_preview_grid.png")


def generate_dataset(output_root: str | Path, train_count: int, val_count: int, test_count: int, seed: int = 7, cfg: TripletLRConfig | None = None) -> None:
    cfg = cfg or TripletLRConfig(motion_mode="xyz")
    output_root = ensure_dir(output_root)
    generate_split(output_root, "train", train_count, seed, cfg)
    generate_split(output_root, "val", val_count, seed + 1, cfg)
    generate_split(output_root, "test", test_count, seed + 2, cfg)
    save_json(
        {
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "seed": seed,
            "config": asdict(cfg),
            "scope": "three-frame left-right only",
        },
        output_root / "dataset_manifest.json",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the three-frame left-right-only two-ball dataset.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TripletLRConfig(motion_mode="xyz")
    generate_dataset(args.output_root, args.train_count, args.val_count, args.test_count, args.seed, cfg)


if __name__ == "__main__":
    main()
