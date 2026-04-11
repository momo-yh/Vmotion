from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

from data_generation import (
    BACKGROUND_BOTTOM,
    BACKGROUND_TOP,
    PREVIEW_DOT_COLOR,
    PREVIEW_TEXT_COLOR,
    SHADOW_COLOR,
    TABLE_COLOR,
    TABLE_EDGE_COLOR,
    draw_line,
    draw_polygon,
    ensure_dir,
    look_at_world_to_camera_cv,
    make_background,
    make_intrinsics,
    make_transform,
    point_inside_convex_polygon,
    polygon_mask,
    project_point,
    project_points,
    rotation_matrix_to_euler_xyz_deg,
    save_json,
    save_preview_grid,
    scene_quality_ok,
    table_corners_world,
    transform_world_to_camera,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MultiBallDatasetRenderConfig:
    image_width: int = 128
    image_height: int = 128
    fx: float = 100.0
    fy: float = 100.0
    cx: float = 64.0
    cy: float = 64.0
    small_ball_radius: float = 0.028
    large_ball_radius: float = 0.055
    table_half_x: float = 0.95
    table_z_min: float = 0.05
    table_z_max: float = 1.60
    ball_x_min: float = -0.32
    ball_x_max: float = 0.32
    ball_z_min: float = 0.24
    ball_z_max: float = 0.86
    min_ball_separation: float = 0.16
    motion_translation_max: float = 0.08
    camera_height_min: float = 0.24
    camera_height_max: float = 0.42
    camera_distance_min: float = 0.52
    camera_distance_max: float = 0.82
    camera_azimuth_deg_min: float = -18.0
    camera_azimuth_deg_max: float = 18.0
    min_ball_radius_px: float = 4.0
    max_ball_radius_px: float = 24.0
    min_table_coverage_ratio: float = 0.16
    preview_examples_per_split: int = 8
    motion_mode: str = "xyz"


SMALL_BALL_COLOR = np.array([50, 114, 205], dtype=np.uint8)
SMALL_BALL_HIGHLIGHT = np.array([135, 186, 248], dtype=np.uint8)
LARGE_BALL_COLOR = np.array([205, 75, 52], dtype=np.uint8)
LARGE_BALL_HIGHLIGHT = np.array([245, 165, 128], dtype=np.uint8)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), eps)


def sample_relative_motion(rng: np.random.Generator, cfg: MultiBallDatasetRenderConfig) -> np.ndarray:
    if cfg.motion_mode == "xy":
        translation = np.array(
            [
                rng.uniform(-cfg.motion_translation_max, cfg.motion_translation_max),
                rng.uniform(-cfg.motion_translation_max, cfg.motion_translation_max),
                0.0,
            ],
            dtype=np.float64,
        )
    elif cfg.motion_mode == "xyz":
        translation = rng.uniform(-cfg.motion_translation_max, cfg.motion_translation_max, size=3)
    else:
        raise ValueError(f"Unsupported motion_mode: {cfg.motion_mode}")
    rotation = np.eye(3, dtype=np.float64)
    return make_transform(rotation, translation)


def sample_ball_pair(rng: np.random.Generator, cfg: MultiBallDatasetRenderConfig) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(512):
        small = np.array(
            [
                rng.uniform(cfg.ball_x_min, cfg.ball_x_max),
                cfg.small_ball_radius,
                rng.uniform(cfg.ball_z_min, cfg.ball_z_max),
            ],
            dtype=np.float64,
        )
        large = np.array(
            [
                rng.uniform(cfg.ball_x_min, cfg.ball_x_max),
                cfg.large_ball_radius,
                rng.uniform(cfg.ball_z_min, cfg.ball_z_max),
            ],
            dtype=np.float64,
        )
        diff_xz = small[[0, 2]] - large[[0, 2]]
        if float(np.linalg.norm(diff_xz)) >= cfg.min_ball_separation:
            return small, large
    raise RuntimeError("Failed to sample a valid two-ball configuration.")


def sample_camera_pose(rng: np.random.Generator, small_ball: np.ndarray, large_ball: np.ndarray, cfg: MultiBallDatasetRenderConfig) -> np.ndarray:
    midpoint = 0.5 * (small_ball + large_ball)
    distance = rng.uniform(cfg.camera_distance_min, cfg.camera_distance_max)
    azimuth = np.deg2rad(rng.uniform(cfg.camera_azimuth_deg_min, cfg.camera_azimuth_deg_max))
    eye = np.array(
        [
            midpoint[0] + distance * np.sin(azimuth),
            rng.uniform(cfg.camera_height_min, cfg.camera_height_max),
            midpoint[2] - distance * np.cos(azimuth),
        ],
        dtype=np.float64,
    )
    table_anchor = np.array([0.0, 0.03, 0.72], dtype=np.float64)
    target = 0.6 * midpoint + 0.4 * table_anchor
    target[1] = max(target[1], 0.05)
    return look_at_world_to_camera_cv(eye, target)


def draw_table(img: np.ndarray, world_to_camera: np.ndarray, K: np.ndarray, cfg: MultiBallDatasetRenderConfig) -> bool:
    corners = table_corners_world(cfg)
    pixels, camera_points = project_points(corners, world_to_camera, K)
    if pixels is None or camera_points is None:
        return False
    draw_polygon(img, pixels, TABLE_COLOR, alpha=1.0)
    for i in range(4):
        width = 2 if i in (0, 1) else 1
        draw_line(img, pixels[i], pixels[(i + 1) % 4], TABLE_EDGE_COLOR, width=width)

    mask = polygon_mask((img.shape[0], img.shape[1]), pixels)
    if mask.any():
        yy, xx = np.where(mask)
        tone = 0.88 + 0.10 * ((yy.astype(np.float32) - yy.min()) / max(float(yy.max() - yy.min()), 1.0))
        textured = img[yy, xx].astype(np.float32) * tone[:, None]
        img[yy, xx] = np.clip(textured, 0.0, 255.0).astype(np.uint8)
    return True


def draw_shadow(img: np.ndarray, world_to_camera: np.ndarray, K: np.ndarray, ball_center_world: np.ndarray, radius_world: float) -> None:
    contact = np.array([ball_center_world[0], 0.0, ball_center_world[2]], dtype=np.float64)
    center = project_point(transform_world_to_camera(world_to_camera, contact), K)
    if center is None:
        return
    depth = transform_world_to_camera(world_to_camera, contact)[2]
    if depth <= 1e-6:
        return
    radius_x = K[0, 0] * radius_world / depth * 1.15
    radius_y = radius_x * 0.35
    image = Image.fromarray(img)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bbox = (
        float(center[0] - radius_x),
        float(center[1] - radius_y),
        float(center[0] + radius_x),
        float(center[1] + radius_y),
    )
    draw.ellipse(bbox, fill=(int(SHADOW_COLOR[0]), int(SHADOW_COLOR[1]), int(SHADOW_COLOR[2]), 60))
    img[:] = np.asarray(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))


def draw_ball(
    img: np.ndarray,
    world_to_camera: np.ndarray,
    K: np.ndarray,
    ball_center_world: np.ndarray,
    radius_world: float,
    color: np.ndarray,
    highlight: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    center_camera = transform_world_to_camera(world_to_camera, ball_center_world)
    center = project_point(center_camera, K)
    if center is None:
        return None, 0.0
    radius = K[0, 0] * radius_world / float(center_camera[2])
    if radius <= 1.0:
        return center, radius

    image = Image.fromarray(img)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bbox = (
        float(center[0] - radius),
        float(center[1] - radius),
        float(center[0] + radius),
        float(center[1] + radius),
    )
    draw.ellipse(bbox, fill=tuple(int(c) for c in np.append(color, 255)))

    highlight_center = (float(center[0] - 0.28 * radius), float(center[1] - 0.32 * radius))
    highlight_radius = 0.42 * radius
    draw.ellipse(
        (
            highlight_center[0] - highlight_radius,
            highlight_center[1] - highlight_radius,
            highlight_center[0] + highlight_radius,
            highlight_center[1] + highlight_radius,
        ),
        fill=tuple(int(c) for c in np.append(highlight, 110)),
    )
    img[:] = np.asarray(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))
    return center, radius


class MultiBallSceneRenderer:
    def __init__(self, cfg: MultiBallDatasetRenderConfig | None = None) -> None:
        self.cfg = cfg or MultiBallDatasetRenderConfig()
        self.K = make_intrinsics(self.cfg)

    def render(self, world_to_camera: np.ndarray, small_ball_world: np.ndarray, large_ball_world: np.ndarray) -> np.ndarray:
        img = make_background(self.cfg)
        draw_table(img, world_to_camera, self.K, self.cfg)

        draw_shadow(img, world_to_camera, self.K, small_ball_world, self.cfg.small_ball_radius)
        draw_shadow(img, world_to_camera, self.K, large_ball_world, self.cfg.large_ball_radius)

        balls = [
            (small_ball_world, self.cfg.small_ball_radius, SMALL_BALL_COLOR, SMALL_BALL_HIGHLIGHT),
            (large_ball_world, self.cfg.large_ball_radius, LARGE_BALL_COLOR, LARGE_BALL_HIGHLIGHT),
        ]
        balls.sort(key=lambda item: float(transform_world_to_camera(world_to_camera, item[0])[2]), reverse=True)
        for center_world, radius_world, color, highlight in balls:
            draw_ball(img, world_to_camera, self.K, center_world, radius_world, color, highlight)
        return img


def validate_projection(point_camera: np.ndarray, K: np.ndarray, radius_world: float, cfg: MultiBallDatasetRenderConfig) -> tuple[bool, np.ndarray | None, float]:
    proj = project_point(point_camera, K)
    if proj is None:
        return False, None, 0.0
    radius_px = K[0, 0] * radius_world / float(point_camera[2])
    valid = (
        proj[0] >= radius_px
        and proj[0] <= (cfg.image_width - 1) - radius_px
        and proj[1] >= radius_px
        and proj[1] <= (cfg.image_height - 1) - radius_px
        and cfg.min_ball_radius_px <= radius_px <= cfg.max_ball_radius_px
    )
    return bool(valid), proj, float(radius_px)


def table_visibility_ok(
    world_to_camera: np.ndarray,
    K: np.ndarray,
    ball_center_world: np.ndarray,
    cfg: MultiBallDatasetRenderConfig,
) -> bool:
    corners = table_corners_world(cfg)
    table_pixels, _ = project_points(corners, world_to_camera, K)
    if table_pixels is None:
        return False
    mask = polygon_mask((cfg.image_height, cfg.image_width), table_pixels)
    coverage = float(mask.mean())
    if coverage < cfg.min_table_coverage_ratio:
        return False

    contact_world = np.array([ball_center_world[0], 0.0, ball_center_world[2]], dtype=np.float64)
    contact_camera = transform_world_to_camera(world_to_camera, contact_world)
    contact_px = project_point(contact_camera, K)
    if contact_px is None:
        return False
    if not point_inside_convex_polygon(contact_px.astype(np.float32), table_pixels.astype(np.float32)):
        return False
    if not (0.0 <= contact_px[0] <= cfg.image_width - 1 and 0.0 <= contact_px[1] <= cfg.image_height - 1):
        return False
    return True


def centers_separated_in_image(center_a: np.ndarray, radius_a: float, center_b: np.ndarray, radius_b: float) -> bool:
    return float(np.linalg.norm(center_a - center_b)) >= 0.6 * (radius_a + radius_b)


def sample_episode(renderer: MultiBallSceneRenderer, rng: np.random.Generator) -> dict[str, np.ndarray]:
    cfg = renderer.cfg
    K = renderer.K
    for _ in range(1024):
        small_ball_world, large_ball_world = sample_ball_pair(rng, cfg)
        world_to_camera_t = sample_camera_pose(rng, small_ball_world, large_ball_world, cfg)
        T_t_to_t1 = sample_relative_motion(rng, cfg)
        world_to_camera_t1 = T_t_to_t1 @ world_to_camera_t

        small_3d_t = transform_world_to_camera(world_to_camera_t, small_ball_world)
        small_3d_t1 = transform_world_to_camera(world_to_camera_t1, small_ball_world)
        large_3d_t = transform_world_to_camera(world_to_camera_t, large_ball_world)
        large_3d_t1 = transform_world_to_camera(world_to_camera_t1, large_ball_world)

        valid_small_t, small_2d_t, small_r_t = validate_projection(small_3d_t, K, cfg.small_ball_radius, cfg)
        valid_small_t1, small_2d_t1, small_r_t1 = validate_projection(small_3d_t1, K, cfg.small_ball_radius, cfg)
        valid_large_t, large_2d_t, large_r_t = validate_projection(large_3d_t, K, cfg.large_ball_radius, cfg)
        valid_large_t1, large_2d_t1, large_r_t1 = validate_projection(large_3d_t1, K, cfg.large_ball_radius, cfg)
        if not all([valid_small_t, valid_small_t1, valid_large_t, valid_large_t1]):
            continue
        if not centers_separated_in_image(small_2d_t, small_r_t, large_2d_t, large_r_t):
            continue
        if not centers_separated_in_image(small_2d_t1, small_r_t1, large_2d_t1, large_r_t1):
            continue
        if not table_visibility_ok(world_to_camera_t, K, small_ball_world, cfg):
            continue
        if not table_visibility_ok(world_to_camera_t, K, large_ball_world, cfg):
            continue
        if not table_visibility_ok(world_to_camera_t1, K, small_ball_world, cfg):
            continue
        if not table_visibility_ok(world_to_camera_t1, K, large_ball_world, cfg):
            continue

        img_t = renderer.render(world_to_camera_t, small_ball_world, large_ball_world)
        img_t1 = renderer.render(world_to_camera_t1, small_ball_world, large_ball_world)
        if not scene_quality_ok(img_t, img_t1):
            continue

        return {
            "img_t": img_t.astype(np.uint8),
            "img_t1": img_t1.astype(np.uint8),
            "K": K.astype(np.float32),
            "T_t_to_t1": T_t_to_t1.astype(np.float32),
            "small_ball_center_world": small_ball_world.astype(np.float32),
            "large_ball_center_world": large_ball_world.astype(np.float32),
            "small_ball_center_3d_t": small_3d_t.astype(np.float32),
            "small_ball_center_3d_t1": small_3d_t1.astype(np.float32),
            "large_ball_center_3d_t": large_3d_t.astype(np.float32),
            "large_ball_center_3d_t1": large_3d_t1.astype(np.float32),
            "small_ball_center_2d_t": small_2d_t.astype(np.float32),
            "small_ball_center_2d_t1": small_2d_t1.astype(np.float32),
            "large_ball_center_2d_t": large_2d_t.astype(np.float32),
            "large_ball_center_2d_t1": large_2d_t1.astype(np.float32),
            "world_to_camera_t": world_to_camera_t.astype(np.float32),
            "world_to_camera_t1": world_to_camera_t1.astype(np.float32),
        }
    raise RuntimeError("Failed to sample a valid two-ball render episode.")


def save_sample(sample: dict[str, np.ndarray], sample_dir: str | Path) -> None:
    sample_dir = ensure_dir(sample_dir)
    Image.fromarray(sample["img_t"]).save(sample_dir / "img_t.png")
    Image.fromarray(sample["img_t1"]).save(sample_dir / "img_t1.png")

    meta = {
        "K": sample["K"].tolist(),
        "T_t_to_t1": sample["T_t_to_t1"].tolist(),
        "small_ball_radius": float(sample["small_ball_center_world"][1]),
        "large_ball_radius": float(sample["large_ball_center_world"][1]),
        "small_ball_center_world": sample["small_ball_center_world"].tolist(),
        "large_ball_center_world": sample["large_ball_center_world"].tolist(),
        "small_ball_center_3d_t": sample["small_ball_center_3d_t"].tolist(),
        "small_ball_center_3d_t1": sample["small_ball_center_3d_t1"].tolist(),
        "large_ball_center_3d_t": sample["large_ball_center_3d_t"].tolist(),
        "large_ball_center_3d_t1": sample["large_ball_center_3d_t1"].tolist(),
        "small_ball_center_2d_t": sample["small_ball_center_2d_t"].tolist(),
        "small_ball_center_2d_t1": sample["small_ball_center_2d_t1"].tolist(),
        "large_ball_center_2d_t": sample["large_ball_center_2d_t"].tolist(),
        "large_ball_center_2d_t1": sample["large_ball_center_2d_t1"].tolist(),
        "world_to_camera_t": sample["world_to_camera_t"].tolist(),
        "world_to_camera_t1": sample["world_to_camera_t1"].tolist(),
    }
    save_json(meta, sample_dir / "meta.json")


def draw_preview_marker(draw: ImageDraw.ImageDraw, center: tuple[float, float], fill: tuple[int, int, int], label: str) -> None:
    x, y = center
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=fill)
    draw.text((x + 5, y - 6), label, fill=fill)


def make_preview_pair(sample: dict[str, np.ndarray]) -> Image.Image:
    image_t = Image.fromarray(sample["img_t"])
    image_t1 = Image.fromarray(sample["img_t1"])
    gap = 20
    footer_h = 62
    canvas = Image.new("RGB", (image_t.width + image_t1.width + gap, image_t.height + footer_h), color=(255, 255, 255))
    canvas.paste(image_t, (0, 0))
    canvas.paste(image_t1, (image_t.width + gap, 0))
    draw = ImageDraw.Draw(canvas)

    sx0, sy0 = map(float, sample["small_ball_center_2d_t"])
    lx0, ly0 = map(float, sample["large_ball_center_2d_t"])
    sx1, sy1 = map(float, sample["small_ball_center_2d_t1"])
    lx1, ly1 = map(float, sample["large_ball_center_2d_t1"])
    sx1 += image_t.width + gap
    lx1 += image_t.width + gap

    draw_preview_marker(draw, (sx0, sy0), (40, 100, 220), "S")
    draw_preview_marker(draw, (lx0, ly0), (220, 80, 40), "L")
    draw_preview_marker(draw, (sx1, sy1), (40, 100, 220), "S")
    draw_preview_marker(draw, (lx1, ly1), (220, 80, 40), "L")

    t = sample["T_t_to_t1"][:3, 3]
    r = rotation_matrix_to_euler_xyz_deg(sample["T_t_to_t1"][:3, :3])
    summary_1 = (
        f"S du={sample['small_ball_center_2d_t1'][0]-sample['small_ball_center_2d_t'][0]:+.1f}px, "
        f"dv={sample['small_ball_center_2d_t1'][1]-sample['small_ball_center_2d_t'][1]:+.1f}px"
    )
    summary_2 = (
        f"L du={sample['large_ball_center_2d_t1'][0]-sample['large_ball_center_2d_t'][0]:+.1f}px, "
        f"dv={sample['large_ball_center_2d_t1'][1]-sample['large_ball_center_2d_t'][1]:+.1f}px"
    )
    summary_3 = f"cam t=({t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f}) m, r=({r[0]:+.1f},{r[1]:+.1f},{r[2]:+.1f}) deg"
    draw.text((6, image_t.height + 6), summary_1, fill=tuple(int(c) for c in PREVIEW_TEXT_COLOR))
    draw.text((6, image_t.height + 24), summary_2, fill=tuple(int(c) for c in PREVIEW_TEXT_COLOR))
    draw.text((6, image_t.height + 42), summary_3, fill=tuple(int(c) for c in PREVIEW_TEXT_COLOR))
    return canvas


def generate_split(root: str | Path, split: str, count: int, seed: int, cfg: MultiBallDatasetRenderConfig | None = None) -> None:
    root = ensure_dir(root)
    split_dir = ensure_dir(root / split)
    renderer = MultiBallSceneRenderer(cfg)
    rng = np.random.default_rng(seed)
    preview_images: list[Image.Image] = []

    for index in range(count):
        sample = sample_episode(renderer, rng)
        sample_dir = split_dir / f"sample_{index:06d}"
        save_sample(sample, sample_dir)
        if len(preview_images) < renderer.cfg.preview_examples_per_split:
            preview_images.append(make_preview_pair(sample))

    save_json({"split": split, "count": count, "seed": seed}, split_dir / "manifest.json")
    save_preview_grid(preview_images, root / f"{split}_preview_grid.png")


def generate_dataset(
    output_root: str | Path,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int = 7,
    cfg: MultiBallDatasetRenderConfig | None = None,
) -> None:
    cfg = cfg or MultiBallDatasetRenderConfig()
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
        },
        output_root / "dataset_manifest.json",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the two-ball motion-pair dataset.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--motion-mode", type=str, default="xyz", choices=["xy", "xyz"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = MultiBallDatasetRenderConfig(motion_mode=args.motion_mode)
    generate_dataset(
        output_root=args.output_root,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
