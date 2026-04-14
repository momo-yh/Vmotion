from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_preview_grid(images: list[Image.Image], path: str | Path, cols: int = 2) -> None:
    if not images:
        return
    tile_w = max(img.width for img in images)
    tile_h = max(img.height for img in images)
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(255, 255, 255))
    for idx, img in enumerate(images):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        canvas.paste(img, (x, y))
    canvas.save(path)


def make_transform(tx: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = tx
    return T


@dataclass(frozen=True)
class ComplexTripletConfig:
    image_width: int = 128
    image_height: int = 128
    fx: float = 100.0
    fy: float = 100.0
    cx: float = 64.0
    cy: float = 64.0
    motion_translation_min: float = 0.045
    motion_translation_max: float = 0.08
    min_measurable_displacement_px: float = 8.0
    scene_center_world_z: float = 0.95
    min_objects: int = 3
    max_objects: int = 4
    sphere_radius_small_min: float = 0.035
    sphere_radius_small_max: float = 0.060
    sphere_radius_large_min: float = 0.085
    sphere_radius_large_max: float = 0.130
    cube_half_small_min: float = 0.032
    cube_half_small_max: float = 0.055
    cube_half_large_min: float = 0.070
    cube_half_large_max: float = 0.105
    world_x_min: float = -0.55
    world_x_max: float = 0.55
    world_y_min: float = -0.06
    world_y_max: float = 0.24
    world_z_min: float = 0.55
    world_z_max: float = 1.55
    min_center_gap: float = 0.12
    preview_examples_per_split: int = 8
    background_z: float = 4.0


PALETTE = [
    np.array([220, 78, 78], dtype=np.float32),
    np.array([72, 125, 220], dtype=np.float32),
    np.array([234, 234, 234], dtype=np.float32),
    np.array([82, 176, 110], dtype=np.float32),
    np.array([238, 156, 72], dtype=np.float32),
    np.array([148, 118, 212], dtype=np.float32),
]


def intrinsics(cfg: ComplexTripletConfig) -> np.ndarray:
    return np.array(
        [[cfg.fx, 0.0, cfg.cx], [0.0, cfg.fy, cfg.cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def ray_grid(cfg: ComplexTripletConfig) -> tuple[np.ndarray, np.ndarray]:
    uu, vv = np.meshgrid(np.arange(cfg.image_width, dtype=np.float32), np.arange(cfg.image_height, dtype=np.float32))
    rx = (uu - cfg.cx) / cfg.fx
    ry = (vv - cfg.cy) / cfg.fy
    return rx, ry


def make_blank_background(cfg: ComplexTripletConfig) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0 : cfg.image_height, 0 : cfg.image_width].astype(np.float32)
    xx = (xx - cfg.cx) / cfg.image_width
    yy = (yy - cfg.cy) / cfg.image_height
    radial = np.sqrt(xx * xx + yy * yy)
    shade = 250.0 - 12.0 * radial
    shade += 5.0 * (yy + 0.1)
    shade = np.clip(shade, 242.0, 254.0)
    color = np.repeat(shade[..., None], 3, axis=2).astype(np.uint8)
    depth = np.zeros((cfg.image_height, cfg.image_width), dtype=np.float32)
    return color, depth


def sample_object(rng: np.random.Generator, cfg: ComplexTripletConfig, idx: int) -> dict:
    kind = "sphere" if idx % 2 == 0 else "cube"
    size_bucket = "large" if idx < 2 else "small"
    if kind == "sphere":
        if size_bucket == "large":
            size = float(rng.uniform(cfg.sphere_radius_large_min, cfg.sphere_radius_large_max))
        else:
            size = float(rng.uniform(cfg.sphere_radius_small_min, cfg.sphere_radius_small_max))
    else:
        if size_bucket == "large":
            size = float(rng.uniform(cfg.cube_half_large_min, cfg.cube_half_large_max))
        else:
            size = float(rng.uniform(cfg.cube_half_small_min, cfg.cube_half_small_max))
    center = np.array(
        [
            rng.uniform(cfg.world_x_min, cfg.world_x_max),
            rng.uniform(cfg.world_y_min, cfg.world_y_max),
            rng.uniform(cfg.world_z_min, cfg.world_z_max),
        ],
        dtype=np.float32,
    )
    base = PALETTE[idx % len(PALETTE)].copy()
    jitter = rng.uniform(-10.0, 10.0, size=3).astype(np.float32)
    color = np.clip(base + jitter, 60.0, 248.0)
    return {
        "type": kind,
        "center_world": center,
        "size": size,
        "base_color": color,
        "size_bucket": size_bucket,
    }


def non_overlapping(objects: list[dict], cfg: ComplexTripletConfig) -> bool:
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a = objects[i]["center_world"]
            b = objects[j]["center_world"]
            size_gap = objects[i]["size"] + objects[j]["size"] + cfg.min_center_gap
            if np.linalg.norm(a - b) < size_gap:
                return False
    return True


def projected_separation_ok(records: list[dict], cfg: ComplexTripletConfig) -> bool:
    for i in range(len(records)):
        ci = np.asarray(records[i]["center_2d"], dtype=np.float32)
        ri = float(records[i]["radius_px"])
        inside_i = 0.0 <= ci[0] < cfg.image_width and 0.0 <= ci[1] < cfg.image_height
        if not inside_i:
            return False
        for j in range(i + 1, len(records)):
            cj = np.asarray(records[j]["center_2d"], dtype=np.float32)
            rj = float(records[j]["radius_px"])
            inside_j = 0.0 <= cj[0] < cfg.image_width and 0.0 <= cj[1] < cfg.image_height
            if not inside_j:
                return False
            min_gap = 1.00 * (ri + rj) + 2.0
            if np.linalg.norm(ci - cj) < min_gap:
                return False
    return True


def sample_scene(rng: np.random.Generator, cfg: ComplexTripletConfig) -> list[dict]:
    for _ in range(4096):
        count = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))
        objects = [sample_object(rng, cfg, idx) for idx in range(count)]
        if non_overlapping(objects, cfg):
            return objects
    raise RuntimeError("Failed to sample a non-overlapping floating scene.")


def project_point(point_cam: np.ndarray, cfg: ComplexTripletConfig) -> np.ndarray:
    return np.array(
        [cfg.fx * point_cam[0] / point_cam[2] + cfg.cx, cfg.fy * point_cam[1] / point_cam[2] + cfg.cy],
        dtype=np.float32,
    )


def lambert(base_color: np.ndarray, diffuse: np.ndarray, ambient: float = 0.32) -> np.ndarray:
    light = ambient + (1.0 - ambient) * np.clip(diffuse, 0.0, 1.0)
    out = base_color[None, :] * light[..., None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def sphere_shading(local_normal: np.ndarray, base_color: np.ndarray) -> np.ndarray:
    light_dir = np.array([-0.18, -0.12, 1.0], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    diffuse = local_normal @ light_dir
    spec = np.clip(local_normal @ light_dir, 0.0, 1.0) ** 14
    color = lambert(base_color, diffuse, ambient=0.64)
    rim = np.clip(1.0 - local_normal[:, 2], 0.0, 1.0) ** 1.5
    color = np.clip(color.astype(np.float32) + 52.0 * spec[:, None] - 7.0 * rim[:, None], 0.0, 255.0)
    return color.astype(np.uint8)


def cube_face_normal(hit_points: np.ndarray, center: np.ndarray, half: float) -> np.ndarray:
    local = (hit_points - center) / max(half, 1e-6)
    idx = np.argmax(np.abs(local), axis=1)
    normal = np.zeros_like(local)
    normal[np.arange(local.shape[0]), idx] = np.sign(local[np.arange(local.shape[0]), idx])
    return normal


def cube_shading(hit_points: np.ndarray, center: np.ndarray, half: float, base_color: np.ndarray) -> np.ndarray:
    normal = cube_face_normal(hit_points, center, half)
    light_dir = np.array([-0.18, -0.12, 1.0], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    diffuse = normal @ light_dir
    color = lambert(base_color, diffuse, ambient=0.66).astype(np.float32)
    local = (hit_points - center) / max(half, 1e-6)
    edge = np.maximum.reduce(np.abs(local), axis=1)
    rim = np.clip((edge - 0.7) / 0.3, 0.0, 1.0)
    color *= (0.99 - 0.05 * rim[:, None])
    return np.clip(color, 0.0, 255.0).astype(np.uint8)


def render_objects(
    cfg: ComplexTripletConfig,
    rx: np.ndarray,
    ry: np.ndarray,
    cam_x: float,
    color: np.ndarray,
    depth: np.ndarray,
    objects: list[dict],
) -> list[dict]:
    dirs = np.stack([rx, ry, np.ones_like(rx)], axis=-1)
    records: list[dict] = []
    visible_depth = np.full_like(rx, np.inf, dtype=np.float32)

    for obj in objects:
        center_cam = obj["center_world"].copy()
        center_cam[0] -= cam_x
        base = obj["base_color"]

        if obj["type"] == "sphere":
            radius = obj["size"]
            dot = dirs[..., 0] * center_cam[0] + dirs[..., 1] * center_cam[1] + center_cam[2]
            a = (dirs * dirs).sum(axis=-1)
            c0 = float(center_cam.dot(center_cam) - radius * radius)
            disc = dot * dot - a * c0
            hit = disc > 0.0
            if np.any(hit):
                root = np.sqrt(np.maximum(disc, 0.0))
                lam = (dot - root) / a
                hit &= lam > 0.0
                replace = hit & (lam < visible_depth)
                if np.any(replace):
                    pts = dirs[replace] * lam[replace][..., None]
                    local = (pts - center_cam) / radius
                    color[replace] = sphere_shading(local, base)
                    visible_depth[replace] = lam[replace].astype(np.float32)
        else:
            half = obj["size"]
            bmin = center_cam - half
            bmax = center_cam + half
            inv_dirs = np.full_like(dirs, np.inf)
            np.divide(1.0, dirs, out=inv_dirs, where=np.abs(dirs) > 1e-8)
            t0 = bmin * inv_dirs
            t1 = bmax * inv_dirs
            tmin = np.minimum(t0, t1)
            tmax = np.maximum(t0, t1)
            t_enter = np.max(tmin, axis=-1)
            t_exit = np.min(tmax, axis=-1)
            hit = (t_enter > 0.0) & (t_enter <= t_exit)
            replace = hit & (t_enter < visible_depth)
            if np.any(replace):
                pts = dirs[replace] * t_enter[replace][..., None]
                color[replace] = cube_shading(pts, center_cam, half, base)
                visible_depth[replace] = t_enter[replace].astype(np.float32)

        depth = np.where(np.isfinite(visible_depth), visible_depth.astype(np.float32), depth)
        records.append(
            {
                "type": obj["type"],
                "center_3d": center_cam.astype(np.float32),
                "center_2d": project_point(center_cam, cfg),
                "size": float(obj["size"]),
                "radius_px": float(cfg.fx * obj["size"] / center_cam[2]),
            }
        )

    depth[np.isinf(visible_depth)] = 0.0
    depth[np.isfinite(visible_depth)] = visible_depth[np.isfinite(visible_depth)]
    return records


def render_frame(cfg: ComplexTripletConfig, cam_x: float, objects: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rx, ry = ray_grid(cfg)
    color, depth = make_blank_background(cfg)
    records = render_objects(cfg, rx, ry, cam_x, color, depth, objects)
    return color, depth.astype(np.float32), records


def sample_lr_motion(rng: np.random.Generator, cfg: ComplexTripletConfig) -> float:
    magnitude = float(rng.uniform(cfg.motion_translation_min, cfg.motion_translation_max))
    sign = float(rng.choice([-1.0, 1.0]))
    return sign * magnitude


def visible_and_displaced(records0: list[dict], records1: list[dict], cfg: ComplexTripletConfig) -> bool:
    visible_count = 0
    for r0, r1 in zip(records0, records1):
        c0 = r0["center_2d"]
        c1 = r1["center_2d"]
        inside = (
            0.0 <= float(c0[0]) < cfg.image_width
            and 0.0 <= float(c0[1]) < cfg.image_height
            and 0.0 <= float(c1[0]) < cfg.image_width
            and 0.0 <= float(c1[1]) < cfg.image_height
        )
        moved = abs(float(c1[0] - c0[0])) >= cfg.min_measurable_displacement_px
        if inside and moved:
            visible_count += 1
    return visible_count >= max(2, len(records0) - 1)


def sample_episode(rng: np.random.Generator, cfg: ComplexTripletConfig) -> dict[str, np.ndarray]:
    K = intrinsics(cfg)
    for _ in range(4096):
        objects = sample_scene(rng, cfg)
        tx1 = sample_lr_motion(rng, cfg)
        tx2 = sample_lr_motion(rng, cfg)
        img_t, depth_t, rec_t = render_frame(cfg, 0.0, objects)
        img_t1, depth_t1, rec_t1 = render_frame(cfg, tx1, objects)
        img_t2, depth_t2, rec_t2 = render_frame(cfg, tx1 + tx2, objects)
        if not projected_separation_ok(rec_t, cfg):
            continue
        if not projected_separation_ok(rec_t1, cfg):
            continue
        if not projected_separation_ok(rec_t2, cfg):
            continue
        if not visible_and_displaced(rec_t, rec_t1, cfg):
            continue
        if not visible_and_displaced(rec_t1, rec_t2, cfg):
            continue
        ref_world = np.array([0.0, 0.0, cfg.scene_center_world_z], dtype=np.float32)
        ref_t = project_point(ref_world, cfg)
        ref_t1 = project_point(np.array([ref_world[0] - tx1, ref_world[1], ref_world[2]], dtype=np.float32), cfg)
        ref_t2 = project_point(np.array([ref_world[0] - tx1 - tx2, ref_world[1], ref_world[2]], dtype=np.float32), cfg)
        if abs(float(ref_t1[0] - ref_t[0])) < cfg.min_measurable_displacement_px:
            continue
        if abs(float(ref_t2[0] - ref_t1[0])) < cfg.min_measurable_displacement_px:
            continue
        return {
            "img_t": img_t,
            "img_t1": img_t1,
            "img_t2": img_t2,
            "depth_t": depth_t,
            "depth_t1": depth_t1,
            "depth_t2": depth_t2,
            "K": K,
            "T_t_to_t1": make_transform(tx1),
            "T_t1_to_t2": make_transform(tx2),
            "world_to_camera_t": make_transform(0.0),
            "world_to_camera_t1": make_transform(tx1),
            "world_to_camera_t2": make_transform(tx1 + tx2),
            "objects_t": rec_t,
            "objects_t1": rec_t1,
            "objects_t2": rec_t2,
            "reference_center_world": ref_world,
            "reference_center_2d_t": ref_t,
            "reference_center_2d_t1": ref_t1,
            "reference_center_2d_t2": ref_t2,
        }
    raise RuntimeError("Failed to sample a valid floating-object triplet.")


def make_preview_triplet(sample: dict[str, np.ndarray]) -> Image.Image:
    images = [Image.fromarray(sample["img_t"]), Image.fromarray(sample["img_t1"]), Image.fromarray(sample["img_t2"])]
    gap = 12
    pad_top = 24
    footer_h = 52
    width = sum(img.width for img in images) + 2 * gap
    height = images[0].height + pad_top + footer_h
    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    x_off = 0
    titles = ["Frame 1", "Frame 2", "Frame 3"]
    for title, img in zip(titles, images):
        draw.text((x_off + img.width / 2 - 20, 4), title, fill=(70, 70, 70))
        canvas.paste(img, (x_off, pad_top))
        x_off += img.width + gap
    tx1 = float(sample["T_t_to_t1"][0, 3])
    tx2 = float(sample["T_t1_to_t2"][0, 3])
    obj_count = len(sample["objects_t"])
    sizes = ", ".join(f"{row['type'][0].upper()}:{row['size']:.2f}" for row in sample["objects_t"][:4])
    draw.text((8, pad_top + images[0].height + 8), f"objects={obj_count}, t->t1 tx={tx1:+.3f}m, t1->t2 tx={tx2:+.3f}m", fill=(45, 45, 45))
    draw.text((8, pad_top + images[0].height + 28), f"sample sizes: {sizes}", fill=(75, 75, 75))
    return canvas


def save_sample(sample: dict[str, np.ndarray], sample_dir: str | Path) -> None:
    sample_dir = ensure_dir(sample_dir)
    Image.fromarray(sample["img_t"]).save(sample_dir / "img_t.png")
    Image.fromarray(sample["img_t1"]).save(sample_dir / "img_t1.png")
    Image.fromarray(sample["img_t2"]).save(sample_dir / "img_t2.png")
    np.save(sample_dir / "depth_t.npy", sample["depth_t"])
    np.save(sample_dir / "depth_t1.npy", sample["depth_t1"])
    np.save(sample_dir / "depth_t2.npy", sample["depth_t2"])
    meta: dict[str, object] = {}
    for key, value in sample.items():
        if key.startswith("img_") or key.startswith("depth_"):
            continue
        if isinstance(value, np.ndarray):
            meta[key] = value.tolist()
        elif isinstance(value, list):
            meta[key] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in row.items()} for row in value]
        else:
            meta[key] = value
    meta["scene_type"] = "floating_multi_object_blank_background"
    meta["object_count"] = len(sample["objects_t"])
    save_json(meta, sample_dir / "meta.json")


def generate_split(root: str | Path, split: str, count: int, seed: int, cfg: ComplexTripletConfig) -> None:
    root = ensure_dir(root)
    split_dir = ensure_dir(root / split)
    rng = np.random.default_rng(seed)
    previews: list[Image.Image] = []
    for index in range(count):
        sample = sample_episode(rng, cfg)
        save_sample(sample, split_dir / f"sample_{index:06d}")
        if len(previews) < cfg.preview_examples_per_split:
            previews.append(make_preview_triplet(sample))
    save_json({"split": split, "count": count, "seed": seed}, split_dir / "manifest.json")
    save_preview_grid(previews, root / f"{split}_preview_grid.png")


def generate_dataset(output_root: str | Path, train_count: int, val_count: int, test_count: int, seed: int, cfg: ComplexTripletConfig) -> None:
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
            "scope": "three-frame left-right only, floating spheres and cubes on blank background",
        },
        output_root / "dataset_manifest.json",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a floating multi-object three-frame dataset with blank background.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--train-count", type=int, default=1200)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--test-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = ComplexTripletConfig()
    generate_dataset(args.output_root, args.train_count, args.val_count, args.test_count, args.seed, cfg)


if __name__ == "__main__":
    main()
