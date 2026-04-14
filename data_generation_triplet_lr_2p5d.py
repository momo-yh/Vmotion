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
class SpriteTripletConfig:
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
    depth_min: float = 0.45
    depth_max: float = 1.40
    min_objects: int = 4
    max_objects: int = 6
    radius_px_small_min: float = 7.0
    radius_px_small_max: float = 12.0
    radius_px_large_min: float = 14.0
    radius_px_large_max: float = 24.0
    center_x_margin: float = 18.0
    center_y_margin: float = 18.0
    preview_examples_per_split: int = 8


PALETTE = [
    np.array([220, 78, 78], dtype=np.uint8),
    np.array([72, 125, 220], dtype=np.uint8),
    np.array([234, 234, 234], dtype=np.uint8),
    np.array([82, 176, 110], dtype=np.uint8),
    np.array([238, 156, 72], dtype=np.uint8),
    np.array([148, 118, 212], dtype=np.uint8),
]


def make_background(cfg: SpriteTripletConfig) -> np.ndarray:
    yy, xx = np.mgrid[0 : cfg.image_height, 0 : cfg.image_width].astype(np.float32)
    xx = (xx - cfg.cx) / cfg.image_width
    yy = (yy - cfg.cy) / cfg.image_height
    radial = np.sqrt(xx * xx + yy * yy)
    shade = 248.0 - 10.0 * radial + 4.0 * (yy + 0.1)
    shade = np.clip(shade, 240.0, 254.0)
    return np.repeat(shade[..., None], 3, axis=2).astype(np.uint8)


def sample_tx(rng: np.random.Generator, cfg: SpriteTripletConfig) -> float:
    magnitude = float(rng.uniform(cfg.motion_translation_min, cfg.motion_translation_max))
    sign = float(rng.choice([-1.0, 1.0]))
    return sign * magnitude


def object_size(rng: np.random.Generator, cfg: SpriteTripletConfig, idx: int) -> float:
    if idx < 2:
        return float(rng.uniform(cfg.radius_px_large_min, cfg.radius_px_large_max))
    return float(rng.uniform(cfg.radius_px_small_min, cfg.radius_px_small_max))


def sample_scene(rng: np.random.Generator, cfg: SpriteTripletConfig) -> list[dict]:
    count = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))
    for _ in range(4096):
        objects: list[dict] = []
        ok = True
        for idx in range(count):
            radius_px = object_size(rng, cfg, idx)
            shape = "circle" if idx % 2 == 0 else "square"
            depth = float(rng.uniform(cfg.depth_min, cfg.depth_max))
            cx = float(rng.uniform(cfg.center_x_margin, cfg.image_width - cfg.center_x_margin))
            cy = float(rng.uniform(cfg.center_y_margin, cfg.image_height - cfg.center_y_margin))
            color = PALETTE[idx % len(PALETTE)].copy()
            obj = {
                "shape": shape,
                "center_xy": np.array([cx, cy], dtype=np.float32),
                "depth": depth,
                "radius_px": radius_px,
                "color": color,
            }
            for prev in objects:
                dist = np.linalg.norm(obj["center_xy"] - prev["center_xy"])
                if dist < 0.9 * (obj["radius_px"] + prev["radius_px"]):
                    ok = False
                    break
            if not ok:
                break
            objects.append(obj)
        if ok:
            return objects
    raise RuntimeError("Failed to sample a 2.5D sprite scene.")


def displacement_px(depth: float, tx: float, cfg: SpriteTripletConfig) -> float:
    return float(cfg.fx * tx / depth)


def object_positions(objects: list[dict], tx_cumulative: float, cfg: SpriteTripletConfig) -> list[dict]:
    posed = []
    for obj in objects:
        dx = displacement_px(float(obj["depth"]), tx_cumulative, cfg)
        center = obj["center_xy"] + np.array([dx, 0.0], dtype=np.float32)
        posed.append(
            {
                "shape": obj["shape"],
                "center_xy": center.astype(np.float32),
                "depth": float(obj["depth"]),
                "radius_px": float(obj["radius_px"]),
                "color": obj["color"],
            }
        )
    return posed


def inside_image(obj: dict, cfg: SpriteTripletConfig) -> bool:
    cx, cy = map(float, obj["center_xy"])
    r = float(obj["radius_px"])
    return r <= cx < cfg.image_width - r and r <= cy < cfg.image_height - r


def painter_render(objects: list[dict], cfg: SpriteTripletConfig) -> tuple[np.ndarray, np.ndarray]:
    img = make_background(cfg).astype(np.float32)
    depth = np.zeros((cfg.image_height, cfg.image_width), dtype=np.float32)
    yy, xx = np.mgrid[0 : cfg.image_height, 0 : cfg.image_width].astype(np.float32)
    order = sorted(objects, key=lambda row: row["depth"], reverse=True)
    for obj in order:
        cx, cy = obj["center_xy"]
        r = float(obj["radius_px"])
        if obj["shape"] == "circle":
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            nx = (xx[mask] - cx) / max(r, 1e-6)
            ny = (yy[mask] - cy) / max(r, 1e-6)
            nz = np.sqrt(np.maximum(1.0 - nx * nx - ny * ny, 0.0))
            diffuse = 0.68 + 0.32 * np.clip(-0.18 * nx - 0.12 * ny + 0.95 * nz, 0.0, 1.0)
            color = obj["color"][None, :].astype(np.float32) * diffuse[:, None]
            img[mask] = np.clip(color, 0.0, 255.0)
            depth[mask] = float(obj["depth"])
        else:
            half = r
            mask = (np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= half)
            local_x = (xx[mask] - cx) / max(half, 1e-6)
            local_y = (yy[mask] - cy) / max(half, 1e-6)
            face = 0.74 + 0.18 * np.clip(1.0 - 0.4 * local_x - 0.25 * local_y, 0.0, 1.0)
            edge = np.maximum(np.abs(local_x), np.abs(local_y))
            face *= 0.97 - 0.08 * np.clip((edge - 0.72) / 0.28, 0.0, 1.0)
            color = obj["color"][None, :].astype(np.float32) * face[:, None]
            img[mask] = np.clip(color, 0.0, 255.0)
            depth[mask] = float(obj["depth"])
    return img.astype(np.uint8), depth


def sample_episode(rng: np.random.Generator, cfg: SpriteTripletConfig) -> dict[str, np.ndarray]:
    K = np.array([[cfg.fx, 0.0, cfg.cx], [0.0, cfg.fy, cfg.cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    for _ in range(4096):
        objects = sample_scene(rng, cfg)
        tx1 = sample_tx(rng, cfg)
        tx2 = sample_tx(rng, cfg)
        posed_t = object_positions(objects, 0.0, cfg)
        posed_t1 = object_positions(objects, tx1, cfg)
        posed_t2 = object_positions(objects, tx1 + tx2, cfg)
        if not all(inside_image(obj, cfg) for obj in posed_t + posed_t1 + posed_t2):
            continue
        moved_1 = [
            abs(float(obj1["center_xy"][0] - obj0["center_xy"][0])) >= cfg.min_measurable_displacement_px
            for obj0, obj1 in zip(posed_t, posed_t1)
        ]
        moved_2 = [
            abs(float(obj2["center_xy"][0] - obj1["center_xy"][0])) >= cfg.min_measurable_displacement_px
            for obj1, obj2 in zip(posed_t1, posed_t2)
        ]
        min_required = max(2, len(objects) // 2)
        if sum(moved_1) < min_required:
            continue
        if sum(moved_2) < min_required:
            continue
        img_t, depth_t = painter_render(posed_t, cfg)
        img_t1, depth_t1 = painter_render(posed_t1, cfg)
        img_t2, depth_t2 = painter_render(posed_t2, cfg)
        ref_depth = cfg.scene_center_world_z
        ref_disp1 = abs(displacement_px(ref_depth, tx1, cfg))
        ref_disp2 = abs(displacement_px(ref_depth, tx2, cfg))
        if ref_disp1 < cfg.min_measurable_displacement_px or ref_disp2 < cfg.min_measurable_displacement_px:
            continue
        return {
            "img_t": img_t,
            "img_t1": img_t1,
            "img_t2": img_t2,
            "depth_t": depth_t.astype(np.float32),
            "depth_t1": depth_t1.astype(np.float32),
            "depth_t2": depth_t2.astype(np.float32),
            "K": K,
            "T_t_to_t1": make_transform(tx1),
            "T_t1_to_t2": make_transform(tx2),
            "world_to_camera_t": make_transform(0.0),
            "world_to_camera_t1": make_transform(tx1),
            "world_to_camera_t2": make_transform(tx1 + tx2),
            "objects_t": posed_t,
            "objects_t1": posed_t1,
            "objects_t2": posed_t2,
        }
    raise RuntimeError("Failed to sample a valid 2.5D triplet.")


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
    for title, img in zip(["Frame 1", "Frame 2", "Frame 3"], images):
        draw.text((x_off + img.width / 2 - 20, 4), title, fill=(70, 70, 70))
        canvas.paste(img, (x_off, pad_top))
        x_off += img.width + gap
    tx1 = float(sample["T_t_to_t1"][0, 3])
    tx2 = float(sample["T_t1_to_t2"][0, 3])
    depths = ", ".join(f"{obj['shape'][0].upper()}:{obj['depth']:.2f}" for obj in sample["objects_t"][:4])
    draw.text((8, pad_top + images[0].height + 8), f"objects={len(sample['objects_t'])}, t->t1 tx={tx1:+.3f}m, t1->t2 tx={tx2:+.3f}m", fill=(45, 45, 45))
    draw.text((8, pad_top + images[0].height + 28), f"sample depths: {depths}", fill=(75, 75, 75))
    return canvas


def save_sample(sample: dict[str, np.ndarray], sample_dir: str | Path) -> None:
    sample_dir = ensure_dir(sample_dir)
    Image.fromarray(sample["img_t"]).save(sample_dir / "img_t.png")
    Image.fromarray(sample["img_t1"]).save(sample_dir / "img_t1.png")
    Image.fromarray(sample["img_t2"]).save(sample_dir / "img_t2.png")
    np.save(sample_dir / "depth_t.npy", sample["depth_t"])
    np.save(sample_dir / "depth_t1.npy", sample["depth_t1"])
    np.save(sample_dir / "depth_t2.npy", sample["depth_t2"])
    meta = {}
    for key, value in sample.items():
        if key.startswith("img_") or key.startswith("depth_"):
            continue
        if isinstance(value, np.ndarray):
            meta[key] = value.tolist()
        elif isinstance(value, list):
            meta[key] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in row.items()} for row in value]
        else:
            meta[key] = value
    meta["scene_type"] = "sprite_2p5d_multi_object"
    meta["object_count"] = len(sample["objects_t"])
    save_json(meta, sample_dir / "meta.json")


def generate_split(root: str | Path, split: str, count: int, seed: int, cfg: SpriteTripletConfig) -> None:
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


def generate_dataset(output_root: str | Path, train_count: int, val_count: int, test_count: int, seed: int, cfg: SpriteTripletConfig) -> None:
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
            "scope": "three-frame left-right only, fast 2.5D sprite dataset",
        },
        output_root / "dataset_manifest.json",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a fast 2.5D multi-object three-frame dataset.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--train-count", type=int, default=1200)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--test-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = SpriteTripletConfig()
    generate_dataset(args.output_root, args.train_count, args.val_count, args.test_count, args.seed, cfg)


if __name__ == "__main__":
    main()
