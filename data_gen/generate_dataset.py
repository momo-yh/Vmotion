from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DatasetRenderConfig:
    image_width: int = 128
    image_height: int = 128
    fx: float = 100.0
    fy: float = 100.0
    cx: float = 64.0
    cy: float = 64.0
    ball_radius: float = 0.04
    table_half_x: float = 0.95
    table_z_min: float = 0.05
    table_z_max: float = 1.60
    ball_x_min: float = -0.25
    ball_x_max: float = 0.25
    ball_z_min: float = 0.25
    ball_z_max: float = 0.75
    motion_translation_max: float = 0.08
    camera_height_min: float = 0.24
    camera_height_max: float = 0.42
    camera_distance_min: float = 0.52
    camera_distance_max: float = 0.78
    camera_azimuth_deg_min: float = -18.0
    camera_azimuth_deg_max: float = 18.0
    min_ball_radius_px: float = 5.0
    max_ball_radius_px: float = 20.0
    min_table_coverage_ratio: float = 0.16
    preview_examples_per_split: int = 8


BACKGROUND_TOP = np.array([240, 244, 250], dtype=np.uint8)
BACKGROUND_BOTTOM = np.array([250, 251, 253], dtype=np.uint8)
TABLE_COLOR = np.array([174, 166, 154], dtype=np.uint8)
TABLE_EDGE_COLOR = np.array([116, 106, 94], dtype=np.uint8)
BALL_COLOR = np.array([205, 75, 52], dtype=np.uint8)
BALL_HIGHLIGHT = np.array([245, 165, 128], dtype=np.uint8)
SHADOW_COLOR = np.array([70, 64, 60], dtype=np.uint8)
PREVIEW_DOT_COLOR = np.array([20, 235, 45], dtype=np.uint8)
PREVIEW_TEXT_COLOR = np.array([35, 35, 35], dtype=np.uint8)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), eps)


def make_intrinsics(cfg: DatasetRenderConfig) -> np.ndarray:
    return np.array(
        [
            [cfg.fx, 0.0, cfg.cx],
            [0.0, cfg.fy, cfg.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def look_at_world_to_camera_cv(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = normalize((target - eye).astype(np.float64))
    right = normalize(np.cross(forward, world_up))
    down = normalize(np.cross(forward, right))
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ eye
    return make_transform(rotation, translation)


def transform_world_to_camera(world_to_camera: np.ndarray, point_world: np.ndarray) -> np.ndarray:
    return world_to_camera[:3, :3] @ point_world + world_to_camera[:3, 3]


def project_point(point_camera: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    z = float(point_camera[2])
    if z <= 1e-6:
        return None
    u = K[0, 0] * float(point_camera[0]) / z + K[0, 2]
    v = K[1, 1] * float(point_camera[1]) / z + K[1, 2]
    return np.array([u, v], dtype=np.float32)


def project_points(points_world: np.ndarray, world_to_camera: np.ndarray, K: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    points_camera = (world_to_camera[:3, :3] @ points_world.T).T + world_to_camera[:3, 3]
    if np.any(points_camera[:, 2] <= 1e-6):
        return None, None
    pixels = np.stack(
        [
            K[0, 0] * points_camera[:, 0] / points_camera[:, 2] + K[0, 2],
            K[1, 1] * points_camera[:, 1] / points_camera[:, 2] + K[1, 2],
        ],
        axis=1,
    )
    return pixels.astype(np.float32), points_camera.astype(np.float32)


def table_corners_world(cfg: DatasetRenderConfig) -> np.ndarray:
    return np.array(
        [
            [-cfg.table_half_x, 0.0, cfg.table_z_min],
            [cfg.table_half_x, 0.0, cfg.table_z_min],
            [cfg.table_half_x, 0.0, cfg.table_z_max],
            [-cfg.table_half_x, 0.0, cfg.table_z_max],
        ],
        dtype=np.float64,
    )


def sample_ball_center(rng: np.random.Generator, cfg: DatasetRenderConfig) -> np.ndarray:
    return np.array(
        [
            rng.uniform(cfg.ball_x_min, cfg.ball_x_max),
            cfg.ball_radius,
            rng.uniform(cfg.ball_z_min, cfg.ball_z_max),
        ],
        dtype=np.float64,
    )


def sample_camera_pose(rng: np.random.Generator, ball_center: np.ndarray, cfg: DatasetRenderConfig) -> np.ndarray:
    distance = rng.uniform(cfg.camera_distance_min, cfg.camera_distance_max)
    azimuth = np.deg2rad(rng.uniform(cfg.camera_azimuth_deg_min, cfg.camera_azimuth_deg_max))
    eye = np.array(
        [
            ball_center[0] + distance * np.sin(azimuth),
            rng.uniform(cfg.camera_height_min, cfg.camera_height_max),
            ball_center[2] - distance * np.cos(azimuth),
        ],
        dtype=np.float64,
    )
    table_anchor = np.array([0.0, 0.03, 0.72], dtype=np.float64)
    target = 0.65 * ball_center + 0.35 * table_anchor
    target[1] = max(target[1], 0.05)
    return look_at_world_to_camera_cv(eye, target)


def sample_relative_motion(rng: np.random.Generator, cfg: DatasetRenderConfig) -> np.ndarray:
    translation = rng.uniform(-cfg.motion_translation_max, cfg.motion_translation_max, size=3)
    rotation = np.eye(3, dtype=np.float64)
    return make_transform(rotation, translation)


def make_background(cfg: DatasetRenderConfig) -> np.ndarray:
    h, w = cfg.image_height, cfg.image_width
    blend = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = BACKGROUND_TOP.astype(np.float32)[None, None, :]
    bottom = BACKGROUND_BOTTOM.astype(np.float32)[None, None, :]
    bg = top * (1.0 - blend) + bottom * blend
    return np.broadcast_to(bg, (h, w, 3)).astype(np.uint8).copy()


def point_inside_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    sign = None
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        current = 0 if abs(cross) < 1e-6 else (1 if cross > 0 else -1)
        if current == 0:
            continue
        if sign is None:
            sign = current
        elif sign != current:
            return False
    return True


def polygon_mask(image_shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    h, w = image_shape
    mask_image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon([tuple(map(float, p)) for p in polygon], fill=255)
    return np.asarray(mask_image) > 0


def draw_polygon(img: np.ndarray, polygon: np.ndarray, color: np.ndarray, alpha: float = 1.0) -> None:
    mask = polygon_mask((img.shape[0], img.shape[1]), polygon)
    if not mask.any():
        return
    base = img[mask].astype(np.float32)
    target = color.astype(np.float32)
    img[mask] = np.clip(base * (1.0 - alpha) + target * alpha, 0.0, 255.0).astype(np.uint8)


def draw_line(img: np.ndarray, p0: np.ndarray, p1: np.ndarray, color: np.ndarray, width: int = 1) -> None:
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    draw.line((float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])), fill=tuple(int(c) for c in color), width=width)
    img[:] = np.asarray(image)


def draw_table(img: np.ndarray, world_to_camera: np.ndarray, K: np.ndarray, cfg: DatasetRenderConfig) -> bool:
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


def draw_shadow(img: np.ndarray, world_to_camera: np.ndarray, K: np.ndarray, ball_center_world: np.ndarray, cfg: DatasetRenderConfig) -> None:
    contact = np.array([ball_center_world[0], 0.0, ball_center_world[2]], dtype=np.float64)
    center = project_point(transform_world_to_camera(world_to_camera, contact), K)
    if center is None:
        return
    depth = transform_world_to_camera(world_to_camera, contact)[2]
    if depth <= 1e-6:
        return
    radius_x = cfg.fx * cfg.ball_radius / depth * 1.15
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


def draw_ball(img: np.ndarray, world_to_camera: np.ndarray, K: np.ndarray, ball_center_world: np.ndarray, cfg: DatasetRenderConfig) -> tuple[np.ndarray | None, float]:
    center_camera = transform_world_to_camera(world_to_camera, ball_center_world)
    center = project_point(center_camera, K)
    if center is None:
        return None, 0.0
    radius = cfg.fx * cfg.ball_radius / float(center_camera[2])
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
    draw.ellipse(bbox, fill=tuple(int(c) for c in np.append(BALL_COLOR, 255)))

    highlight_center = (float(center[0] - 0.28 * radius), float(center[1] - 0.32 * radius))
    highlight_radius = 0.42 * radius
    draw.ellipse(
        (
            highlight_center[0] - highlight_radius,
            highlight_center[1] - highlight_radius,
            highlight_center[0] + highlight_radius,
            highlight_center[1] + highlight_radius,
        ),
        fill=tuple(int(c) for c in np.append(BALL_HIGHLIGHT, 110)),
    )
    img[:] = np.asarray(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))
    return center, radius


class SimpleSceneRenderer:
    def __init__(self, cfg: DatasetRenderConfig | None = None) -> None:
        self.cfg = cfg or DatasetRenderConfig()
        self.K = make_intrinsics(self.cfg)

    def render(self, world_to_camera: np.ndarray, ball_center_world: np.ndarray) -> np.ndarray:
        img = make_background(self.cfg)
        draw_table(img, world_to_camera, self.K, self.cfg)
        draw_shadow(img, world_to_camera, self.K, ball_center_world, self.cfg)
        draw_ball(img, world_to_camera, self.K, ball_center_world, self.cfg)
        return img


def validate_projection(point_camera: np.ndarray, K: np.ndarray, cfg: DatasetRenderConfig) -> tuple[bool, np.ndarray | None, float]:
    proj = project_point(point_camera, K)
    if proj is None:
        return False, None, 0.0
    radius_px = cfg.fx * cfg.ball_radius / float(point_camera[2])
    valid = (
        proj[0] >= radius_px
        and proj[0] <= (cfg.image_width - 1) - radius_px
        and proj[1] >= radius_px
        and proj[1] <= (cfg.image_height - 1) - radius_px
        and cfg.min_ball_radius_px <= radius_px <= cfg.max_ball_radius_px
    )
    return bool(valid), proj, float(radius_px)


def scene_quality_ok(img_t: np.ndarray, img_t1: np.ndarray) -> bool:
    if float(img_t.mean()) < 20.0 or float(img_t1.mean()) < 20.0:
        return False
    if float(img_t.std()) < 8.0 or float(img_t1.std()) < 8.0:
        return False
    return True


def rotation_matrix_to_euler_xyz_deg(rotation: np.ndarray) -> np.ndarray:
    sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation[2, 1], rotation[2, 2])
        y = np.arctan2(-rotation[2, 0], sy)
        z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        x = np.arctan2(-rotation[1, 2], rotation[1, 1])
        y = np.arctan2(-rotation[2, 0], sy)
        z = 0.0
    return np.rad2deg(np.array([x, y, z], dtype=np.float64))


def draw_preview_dot(draw: ImageDraw.ImageDraw, center: tuple[float, float], radius: int = 2) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=tuple(int(c) for c in PREVIEW_DOT_COLOR))


def make_preview_pair(
    img_t: np.ndarray,
    img_t1: np.ndarray,
    ball_2d_t: np.ndarray,
    ball_2d_t1: np.ndarray,
    T_t_to_t1: np.ndarray,
) -> Image.Image:
    image_t = Image.fromarray(img_t)
    image_t1 = Image.fromarray(img_t1)
    gap = 20
    footer_h = 62
    canvas = Image.new("RGB", (image_t.width + image_t1.width + gap, image_t.height + footer_h), color=(255, 255, 255))
    canvas.paste(image_t, (0, 0))
    canvas.paste(image_t1, (image_t.width + gap, 0))
    draw = ImageDraw.Draw(canvas)
    x0, y0 = float(ball_2d_t[0]), float(ball_2d_t[1])
    x1, y1 = float(ball_2d_t1[0] + image_t.width + gap), float(ball_2d_t1[1])
    draw_preview_dot(draw, (x0, y0), radius=2)
    draw_preview_dot(draw, (x1, y1), radius=2)

    du = float(ball_2d_t1[0] - ball_2d_t[0])
    dv = float(ball_2d_t1[1] - ball_2d_t[1])
    t = T_t_to_t1[:3, 3]
    r = rotation_matrix_to_euler_xyz_deg(T_t_to_t1[:3, :3])
    summary_1 = f"pixel shift: du={du:+.1f}px, dv={dv:+.1f}px"
    summary_2 = f"cam t=({t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f}) m"
    summary_3 = f"cam r=({r[0]:+.1f},{r[1]:+.1f},{r[2]:+.1f}) deg"
    draw.text((6, image_t.height + 6), summary_1, fill=tuple(int(c) for c in PREVIEW_TEXT_COLOR))
    draw.text((6, image_t.height + 24), summary_2, fill=tuple(int(c) for c in PREVIEW_TEXT_COLOR))
    draw.text((6, image_t.height + 42), summary_3, fill=tuple(int(c) for c in PREVIEW_TEXT_COLOR))
    return canvas


def preview_from_sample(sample: dict[str, np.ndarray]) -> Image.Image:
    return make_preview_pair(
        sample["img_t"],
        sample["img_t1"],
        sample["ball_center_2d_t"],
        sample["ball_center_2d_t1"],
        sample["T_t_to_t1"],
    )


def table_visibility_ok(
    world_to_camera: np.ndarray,
    K: np.ndarray,
    ball_center_world: np.ndarray,
    cfg: DatasetRenderConfig,
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


def sample_episode(renderer: SimpleSceneRenderer, rng: np.random.Generator) -> dict[str, np.ndarray]:
    cfg = renderer.cfg
    K = renderer.K
    for _ in range(512):
        ball_center_world = sample_ball_center(rng, cfg)
        world_to_camera_t = sample_camera_pose(rng, ball_center_world, cfg)
        T_t_to_t1 = sample_relative_motion(rng, cfg)
        world_to_camera_t1 = T_t_to_t1 @ world_to_camera_t

        ball_center_3d_t = transform_world_to_camera(world_to_camera_t, ball_center_world)
        ball_center_3d_t1 = transform_world_to_camera(world_to_camera_t1, ball_center_world)
        valid_t, ball_2d_t, _ = validate_projection(ball_center_3d_t, K, cfg)
        valid_t1, ball_2d_t1, _ = validate_projection(ball_center_3d_t1, K, cfg)
        if not (valid_t and valid_t1):
            continue
        if not table_visibility_ok(world_to_camera_t, K, ball_center_world, cfg):
            continue
        if not table_visibility_ok(world_to_camera_t1, K, ball_center_world, cfg):
            continue

        img_t = renderer.render(world_to_camera_t, ball_center_world)
        img_t1 = renderer.render(world_to_camera_t1, ball_center_world)
        if not scene_quality_ok(img_t, img_t1):
            continue

        return {
            "img_t": img_t.astype(np.uint8),
            "img_t1": img_t1.astype(np.uint8),
            "K": K.astype(np.float32),
            "T_t_to_t1": T_t_to_t1.astype(np.float32),
            "ball_center_3d_t": ball_center_3d_t.astype(np.float32),
            "ball_center_2d_t": ball_2d_t.astype(np.float32),
            "ball_center_2d_t1": ball_2d_t1.astype(np.float32),
            "ball_center_world": ball_center_world.astype(np.float32),
            "world_to_camera_t": world_to_camera_t.astype(np.float32),
            "world_to_camera_t1": world_to_camera_t1.astype(np.float32),
        }
    raise RuntimeError("Failed to sample a valid render episode.")


def save_sample(sample: dict[str, np.ndarray], sample_dir: str | Path) -> None:
    sample_dir = ensure_dir(sample_dir)
    Image.fromarray(sample["img_t"]).save(sample_dir / "img_t.png")
    Image.fromarray(sample["img_t1"]).save(sample_dir / "img_t1.png")

    meta = {
        "K": sample["K"].tolist(),
        "T_t_to_t1": sample["T_t_to_t1"].tolist(),
        "ball_center_3d_t": sample["ball_center_3d_t"].tolist(),
        "ball_center_2d_t": sample["ball_center_2d_t"].tolist(),
        "ball_center_2d_t1": sample["ball_center_2d_t1"].tolist(),
        "ball_center_world": sample["ball_center_world"].tolist(),
        "world_to_camera_t": sample["world_to_camera_t"].tolist(),
        "world_to_camera_t1": sample["world_to_camera_t1"].tolist(),
    }
    save_json(meta, sample_dir / "meta.json")


def save_preview_grid(preview_images: list[Image.Image], out_path: str | Path) -> None:
    if not preview_images:
        return
    cols = 2
    rows = int(np.ceil(len(preview_images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for idx, (ax, image) in enumerate(zip(axes.ravel(), preview_images)):
        ax.imshow(np.asarray(image))
        ax.set_title(f"sample_{idx:06d}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def generate_split(root: str | Path, split: str, count: int, seed: int, cfg: DatasetRenderConfig | None = None) -> None:
    root = ensure_dir(root)
    split_dir = ensure_dir(root / split)
    renderer = SimpleSceneRenderer(cfg)
    rng = np.random.default_rng(seed)
    preview_images: list[Image.Image] = []

    for index in range(count):
        sample = sample_episode(renderer, rng)
        sample_dir = split_dir / f"sample_{index:06d}"
        save_sample(sample, sample_dir)
        if len(preview_images) < renderer.cfg.preview_examples_per_split:
            preview_images.append(preview_from_sample(sample))

    save_json({"split": split, "count": count, "seed": seed}, split_dir / "manifest.json")
    save_preview_grid(preview_images, root / f"{split}_preview_grid.png")


def generate_dataset(
    output_root: str | Path,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int = 7,
    cfg: DatasetRenderConfig | None = None,
) -> None:
    cfg = cfg or DatasetRenderConfig()
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
    parser = argparse.ArgumentParser(description="Generate the rendered motion-pair dataset.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    generate_dataset(
        output_root=args.output_root,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
