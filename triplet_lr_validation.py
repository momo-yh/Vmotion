from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from triplet_lr_common import (
    LocalCorrelation,
    RegionEncoder,
    TripletLRSelfSupModel,
    TripletLRTwoBallDataset,
    collate_keep_strings,
    correlation_offsets,
    ensure_dir,
    load_json,
    plot_history,
    save_json,
    set_seed,
)

matplotlib.use("Agg")

TABLE_X_MIN = -0.95
TABLE_X_MAX = 0.95
TABLE_Z_MIN = 0.05
TABLE_Z_MAX = 1.60


def load_backbone(checkpoint_path: str | Path, device: str) -> TripletLRSelfSupModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = TripletLRSelfSupModel(
        radius=ckpt["radius"],
        downsample=ckpt["downsample"],
        depth_min=ckpt["depth_min"],
        depth_max=ckpt["depth_max"],
        eps=ckpt["eps"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def table_depth_map(cx: np.ndarray, cy: np.ndarray, world_to_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    camera_origin_world = -rotation.T @ translation
    dirs_camera = np.stack([cx, cy, np.ones_like(cx)], axis=-1)
    dirs_world = dirs_camera @ rotation
    denom = dirs_world[..., 1]
    valid = np.abs(denom) > 1e-6
    lam = np.full_like(cx, np.inf, dtype=np.float32)
    lam[valid] = (-camera_origin_world[1] / denom[valid]).astype(np.float32)
    valid &= lam > 0.0
    points_world = camera_origin_world[None, None, :] + lam[..., None] * dirs_world
    valid &= points_world[..., 0] >= TABLE_X_MIN
    valid &= points_world[..., 0] <= TABLE_X_MAX
    valid &= points_world[..., 2] >= TABLE_Z_MIN
    valid &= points_world[..., 2] <= TABLE_Z_MAX
    z = np.full_like(cx, np.inf, dtype=np.float32)
    z[valid] = lam[valid]
    return z, valid


def sphere_depth_map(cx: np.ndarray, cy: np.ndarray, center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    a = cx * cx + cy * cy + 1.0
    dot = cx * center[0] + cy * center[1] + center[2]
    c0 = center[0] ** 2 + center[1] ** 2 + center[2] ** 2 - radius * radius
    disc = dot * dot - a * c0
    valid = disc > 0.0
    z = np.full_like(cx, np.inf, dtype=np.float32)
    if np.any(valid):
        root = np.sqrt(np.maximum(disc[valid], 0.0))
        lam = (dot[valid] - root) / a[valid]
        good = lam > 0.0
        z_valid = np.full_like(root, np.inf, dtype=np.float32)
        z_valid[good] = lam[good].astype(np.float32)
        z[valid] = z_valid
    return z, np.isfinite(z)


def compute_depth_map(meta: dict, key_suffix: str = "t", height: int = 128, width: int = 128) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(meta["K"], dtype=np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx0 = float(K[0, 2])
    cy0 = float(K[1, 2])
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    cx = (uu - cx0) / fx
    cy = (vv - cy0) / fy

    depth = np.full((height, width), np.inf, dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    table_z, table_valid = table_depth_map(cx, cy, np.asarray(meta[f"world_to_camera_{key_suffix}"], dtype=np.float32))
    depth[table_valid] = table_z[table_valid]
    valid |= table_valid

    for center_key, radius_key in [
        (f"small_ball_center_3d_{key_suffix}", "small_ball_radius"),
        (f"large_ball_center_3d_{key_suffix}", "large_ball_radius"),
    ]:
        z_sphere, valid_sphere = sphere_depth_map(cx, cy, np.asarray(meta[center_key], dtype=np.float32), float(meta[radius_key]))
        replace = valid_sphere & (z_sphere < depth)
        depth[replace] = z_sphere[replace]
        valid |= replace
    depth[~valid] = 0.0
    return depth, valid.astype(np.float32)


def downsample_depth(depth: np.ndarray, mask: np.ndarray, out_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    depth_t = torch.from_numpy(depth)[None, None]
    mask_t = torch.from_numpy(mask)[None, None]
    factor = depth.shape[0] // out_size
    depth_sum = F.avg_pool2d(depth_t * mask_t, kernel_size=factor, stride=factor) * (factor * factor)
    mask_sum = F.avg_pool2d(mask_t, kernel_size=factor, stride=factor) * (factor * factor)
    depth_lr = depth_sum / mask_sum.clamp_min(1e-6)
    valid_lr = (mask_sum > 0.0).float()
    return depth_lr[0, 0], valid_lr[0, 0]


class TripletValidationDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str) -> None:
        self.base = TripletLRTwoBallDataset(data_root, split)
        self.data_root = Path(data_root)
        self.split = split

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        meta = load_json(self.data_root / self.split / item["sample_id"] / "meta.json")
        depth, mask = compute_depth_map(meta, key_suffix="t1")
        depth_lr, mask_lr = downsample_depth(depth, mask, out_size=32)
        item["depth_lr"] = depth_lr
        item["mask_lr"] = mask_lr
        item["depth_full"] = torch.from_numpy(depth)
        item["mask_full"] = torch.from_numpy(mask)
        return item


class PointwiseDepthDecoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def select_extreme_samples(data_root: str | Path, split: str) -> dict[str, str]:
    split_dir = Path(data_root) / split
    best: dict[str, tuple[float, str]] = {}
    for sample_dir in split_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        meta = load_json(sample_dir / "meta.json")
        tx = float(meta["T_t_to_t1"][0][3])
        key = "right" if tx > 0 else "left"
        score = abs(tx)
        if key not in best or score > best[key][0]:
            best[key] = (score, sample_dir.name)
    return {k: v[1] for k, v in best.items()}


@torch.no_grad()
def get_model_outputs(model: TripletLRSelfSupModel, sample: dict, device: str) -> dict[str, torch.Tensor]:
    return model(
        sample["img_t"].unsqueeze(0).to(device),
        sample["img_t1"].unsqueeze(0).to(device),
        sample["img_t2"].unsqueeze(0).to(device),
        sample["tau1_x"].view(1).to(device),
        sample["tau2_x"].view(1).to(device),
        sample["K"].unsqueeze(0).to(device),
    )


def visualize_motion_response(model: TripletLRSelfSupModel, data_root: str, split: str, output_dir: Path, device: str) -> None:
    output_dir = ensure_dir(output_dir / "motion_response")
    dataset = TripletLRTwoBallDataset(data_root, split)
    chosen = select_extreme_samples(data_root, split)
    by_id = {dataset[i]["sample_id"]: dataset[i] for i in range(len(dataset))}
    for key in ["left", "right"]:
        sample = by_id[chosen[key]]
        out = get_model_outputs(model, sample, device)
        img = sample["img_t"].permute(1, 2, 0).numpy()
        mu = out["mu1_x"][0].detach().cpu().numpy()
        mu_up = np.asarray(Image.fromarray(mu.astype(np.float32), mode="F").resize((128, 128), Image.Resampling.BILINEAR))
        mu_norm = (mu_up - mu_up.min()) / max(float(mu_up.max() - mu_up.min()), 1e-6)
        overlay = 0.5 * img + 0.5 * plt.get_cmap("coolwarm")(mu_norm)[..., :3]
        Image.fromarray((overlay * 255).astype(np.uint8)).save(output_dir / f"{key}.png")


def visualize_correspondence(model: TripletLRSelfSupModel, data_root: str, split: str, output_dir: Path, device: str, num_points: int = 32) -> None:
    output_dir = ensure_dir(output_dir / "correspondence")
    dataset = TripletLRTwoBallDataset(data_root, split)
    chosen = select_extreme_samples(data_root, split)
    sample = next(dataset[i] for i in range(len(dataset)) if dataset[i]["sample_id"] == chosen["right"])
    sample_dir = Path(data_root) / split / sample["sample_id"]
    img_t = Image.open(sample_dir / "img_t.png").convert("RGB")
    img_t1 = Image.open(sample_dir / "img_t1.png").convert("RGB")
    out = get_model_outputs(model, sample, device)
    c1 = out["c1"][0].detach().cpu()
    radius = model.radius
    dx, dy = correlation_offsets(radius, "cpu")
    peak = c1.argmax(dim=0).numpy()
    offset_x = dx.numpy()[peak]
    offset_y = dy.numpy()[peak]
    conf = torch.softmax(c1, dim=0).max(dim=0).values.numpy()
    chosen_idx = np.argsort(conf.reshape(-1))[::-1][:num_points]

    canvas = Image.new("RGB", (img_t.width * 2 + 20, img_t.height), color=(255, 255, 255))
    canvas.paste(img_t, (0, 0))
    canvas.paste(img_t1, (img_t.width + 20, 0))
    draw = ImageDraw.Draw(canvas)
    scale = 4.0
    for idx in chosen_idx:
        y = idx // c1.shape[2]
        x = idx % c1.shape[2]
        p0 = (float((x + 0.5) * scale), float((y + 0.5) * scale))
        p1 = (float(img_t.width + 20 + (x + offset_x[y, x] + 0.5) * scale), float((y + offset_y[y, x] + 0.5) * scale))
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(40, 170, 255), width=1)
    canvas.save(output_dir / "overlay.png")


def evaluate_matching_error(model: TripletLRSelfSupModel, data_root: str, split: str, device: str) -> dict[str, float]:
    dataset = TripletValidationDataset(data_root, split)
    errs = []
    for i in range(min(len(dataset), 100)):
        sample = dataset[i]
        out = get_model_outputs(model, sample, device)
        c1 = out["c1"][0].detach().cpu()
        peak = c1.argmax(dim=0)
        dx, _ = correlation_offsets(model.radius, "cpu")
        pred_disp = dx[peak].numpy() * model.downsample
        depth = sample["depth_lr"].numpy()
        mask = sample["mask_lr"].numpy() > 0
        fx = float(sample["K"][0, 0])
        tau1 = float(sample["tau1_x"])
        gt_disp = fx * tau1 / np.maximum(depth, 1e-6)
        errs.append(float(np.abs(pred_disp[mask] - gt_disp[mask] / model.downsample).mean()))
    return {"mean_abs_disp_error_feat": float(np.mean(errs))}


def visualize_geometric_depth(model: TripletLRSelfSupModel, data_root: str, split: str, output_dir: Path, device: str) -> dict[str, float]:
    output_dir = ensure_dir(output_dir / "depth_recovery")
    dataset = TripletValidationDataset(data_root, split)
    chosen = select_extreme_samples(data_root, split)
    sample = next(dataset[i] for i in range(len(dataset)) if dataset[i]["sample_id"] == chosen["right"])
    out = get_model_outputs(model, sample, device)
    depth_pred = out["d_rel"][0].detach().cpu().numpy()
    depth_pred_up = np.asarray(Image.fromarray(depth_pred.astype(np.float32), mode="F").resize((128, 128), Image.Resampling.BILINEAR))
    depth_gt = sample["depth_full"].numpy()
    mask = sample["mask_full"].numpy() > 0
    err = np.abs(depth_pred_up - depth_gt)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(depth_pred_up, cmap="magma")
    axes[0].set_title("geometric depth")
    axes[0].axis("off")
    axes[1].imshow(np.where(mask, depth_gt, np.nan), cmap="magma")
    axes[1].set_title("ground-truth depth")
    axes[1].axis("off")
    axes[2].imshow(np.where(mask, err, np.nan), cmap="inferno")
    axes[2].set_title("absolute error")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "gt_comparison.png", dpi=180)
    plt.close(fig)
    Image.fromarray(colorize(depth_pred_up, "magma")).save(output_dir / "geometric.png")
    metrics = {"mae_m": float(err[mask].mean()), "rmse_m": float(np.sqrt((err[mask] ** 2).mean()))}
    save_json(metrics, output_dir / "metrics.json")
    return metrics


def colorize(array: np.ndarray, cmap_name: str) -> np.ndarray:
    vmin = np.nanpercentile(array, 5)
    vmax = np.nanpercentile(array, 95)
    norm = np.clip((array - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = plt.get_cmap(cmap_name)(norm)[..., :3]
    return (rgb * 255).astype(np.uint8)


def visualize_encoder_features(model: TripletLRSelfSupModel, data_root: str, split: str, output_dir: Path, device: str) -> None:
    output_dir = ensure_dir(output_dir / "encoder_features")
    dataset = TripletLRTwoBallDataset(data_root, split)
    sample = dataset[0]
    sample_dir = Path(data_root) / split / sample["sample_id"]
    img = np.asarray(Image.open(sample_dir / "img_t.png").convert("RGB"))
    with torch.no_grad():
        f = model.encoder(sample["img_t"].unsqueeze(0).to(device))[0].detach().cpu()
    mean_map = f.mean(dim=0).numpy()
    max_map = f.max(dim=0).values.numpy()
    top_channels = torch.topk(f.view(f.shape[0], -1).mean(dim=1), k=3).indices.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("img_t")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(mean_map, cmap="viridis")
    axes[0, 1].set_title("mean activation")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(max_map, cmap="viridis")
    axes[0, 2].set_title("max activation")
    axes[0, 2].axis("off")
    for ax, ch in zip(axes[1], top_channels):
        ax.imshow(f[ch].numpy(), cmap="viridis")
        ax.set_title(f"channel {ch}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "encoder_feature_maps.png", dpi=180)
    plt.close(fig)


class CorrFeatureDecoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = PointwiseDepthDecoder(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_decoder_epoch(model: nn.Module, feature_fn, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: str, depth_scale: float) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    maes = []
    for batch in loader:
        feats = feature_fn(batch, device)
        pred = model(feats)
        target = batch["depth_lr"].to(device)
        mask = batch["mask_lr"].to(device)
        loss = (torch.abs(pred / depth_scale - target / depth_scale) * mask).sum() / mask.sum().clamp_min(1.0)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        mae = (torch.abs(pred - target) * mask).sum() / mask.sum().clamp_min(1.0)
        maes.append(float(mae.item()))
    return float(np.mean(maes))


def train_depth_decoder(train_loader: DataLoader, val_loader: DataLoader, feature_fn, input_dim: int, device: str, output_dir: Path, epochs: int, lr: float) -> Path:
    model = CorrFeatureDecoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    depth_scale = 2.2
    history = []
    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train_mae = run_decoder_epoch(model, feature_fn, train_loader, optimizer, device, depth_scale)
        val_mae = run_decoder_epoch(model, feature_fn, val_loader, None, device, depth_scale)
        history.append({"epoch": epoch, "train_mae_m": train_mae, "val_mae_m": val_mae})
        save_json(history, output_dir / "history.json")
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            torch.save({"decoder_state": model.state_dict(), "input_dim": input_dim}, output_dir / "best.pt")
    plot_history(history, output_dir / "training_curve.png", output_dir.name, ["train_mae_m", "val_mae_m"])
    save_json({"best_val_mae_m": best_val, "best_epoch": best_epoch}, output_dir / "summary.json")
    return output_dir / "best.pt"


@torch.no_grad()
def eval_depth_decoder(loader: DataLoader, feature_fn, checkpoint: Path, device: str, output_dir: Path) -> dict[str, float]:
    payload = torch.load(checkpoint, map_location="cpu")
    model = CorrFeatureDecoder(payload["input_dim"]).to(device)
    model.load_state_dict(payload["decoder_state"])
    model.eval()
    maes = []
    first = None
    for batch in loader:
        feats = feature_fn(batch, device)
        pred = model(feats).cpu()
        target = batch["depth_lr"]
        mask = batch["mask_lr"]
        mae = (torch.abs(pred - target) * mask).sum() / mask.sum().clamp_min(1.0)
        maes.append(float(mae.item()))
        if first is None:
            first = (pred[0].numpy(), target[0].numpy())
    metrics = {"mae_m": float(np.mean(maes))}
    save_json(metrics, output_dir / "metrics.json")
    pred, target = first
    pred_up = np.asarray(Image.fromarray(pred.astype(np.float32), mode="F").resize((128, 128), Image.Resampling.BILINEAR))
    target_up = np.asarray(Image.fromarray(target.astype(np.float32), mode="F").resize((128, 128), Image.Resampling.NEAREST))
    err_up = np.abs(pred_up - target_up)
    Image.fromarray(colorize(pred_up, "magma")).save(output_dir / "pred_depth.png")
    Image.fromarray(colorize(target_up, "magma")).save(output_dir / "gt_depth.png")
    Image.fromarray(colorize(err_up, "inferno")).save(output_dir / "error_map.png")
    return metrics


def run_bundle(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    model = load_backbone(args.backbone_checkpoint, args.device)

    visualize_motion_response(model, args.data_root, "test", output_dir, args.device)
    visualize_correspondence(model, args.data_root, "test", output_dir, args.device)
    matching_metrics = evaluate_matching_error(model, args.data_root, "test", args.device)
    geom_metrics = visualize_geometric_depth(model, args.data_root, "test", output_dir, args.device)
    visualize_encoder_features(model, args.data_root, "test", output_dir, args.device)

    train_ds = TripletValidationDataset(args.data_root, "train")
    val_ds = TripletValidationDataset(args.data_root, "val")
    test_ds = TripletValidationDataset(args.data_root, "test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_keep_strings)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_keep_strings)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_keep_strings)

    def corr_feature_fn(batch, device):
        with torch.no_grad():
            out = model(
                batch["img_t"].to(device),
                batch["img_t1"].to(device),
                batch["img_t2"].to(device),
                batch["tau1_x"].to(device),
                batch["tau2_x"].to(device),
                batch["K"].to(device),
            )
        return out["c1"]

    random_model = TripletLRSelfSupModel(radius=model.radius).to(args.device).eval()
    for p in random_model.parameters():
        p.requires_grad = False

    def corr_random_feature_fn(batch, device):
        with torch.no_grad():
            out = random_model(
                batch["img_t"].to(device),
                batch["img_t1"].to(device),
                batch["img_t2"].to(device),
                batch["tau1_x"].to(device),
                batch["tau2_x"].to(device),
                batch["K"].to(device),
            )
        return out["c1"]

    def single_frame_feature_fn(batch, device):
        with torch.no_grad():
            feats = model.encoder(batch["img_t1"].to(device))
        return feats

    corr_dir = ensure_dir(output_dir / "depth_probe_corr")
    corr_best = train_depth_decoder(train_loader, val_loader, corr_feature_fn, (2 * model.radius + 1) ** 2, args.device, corr_dir, args.decoder_epochs, args.lr)
    corr_metrics = eval_depth_decoder(test_loader, corr_feature_fn, corr_best, args.device, corr_dir / "eval_test")

    rand_dir = ensure_dir(output_dir / "depth_probe_random")
    rand_best = train_depth_decoder(train_loader, val_loader, corr_random_feature_fn, (2 * model.radius + 1) ** 2, args.device, rand_dir, args.decoder_epochs, args.lr)
    rand_metrics = eval_depth_decoder(test_loader, corr_random_feature_fn, rand_best, args.device, rand_dir / "eval_test")

    sf_dir = ensure_dir(output_dir / "depth_probe_single_frame")
    sf_best = train_depth_decoder(train_loader, val_loader, single_frame_feature_fn, 128, args.device, sf_dir, args.decoder_epochs, args.lr)
    sf_metrics = eval_depth_decoder(test_loader, single_frame_feature_fn, sf_best, args.device, sf_dir / "eval_test")

    summary = {
        "matching": matching_metrics,
        "geometric_recovery": geom_metrics,
        "depth_probe_corr": corr_metrics,
        "depth_probe_random": rand_metrics,
        "depth_probe_single_frame": sf_metrics,
    }
    save_json(summary, output_dir / "summary_metrics.json")
    write_report(output_dir, summary)


def write_report(output_dir: Path, summary: dict) -> None:
    lines = [
        "# Three-Frame LR Validation Report",
        "",
        "## Summary Metrics",
        "",
        f"- mean matching error on feature map: `{summary['matching']['mean_abs_disp_error_feat']:.4f}`",
        f"- geometric recovery MAE: `{summary['geometric_recovery']['mae_m']:.4f} m`",
        f"- geometric recovery RMSE: `{summary['geometric_recovery']['rmse_m']:.4f} m`",
        f"- correlation depth probe MAE: `{summary['depth_probe_corr']['mae_m']:.4f} m`",
        f"- random-encoder correlation probe MAE: `{summary['depth_probe_random']['mae_m']:.4f} m`",
        f"- single-frame feature probe MAE: `{summary['depth_probe_single_frame']['mae_m']:.4f} m`",
        "",
        "## Interpretation",
        "",
        "### Objective 1: motion-sensitive local features",
        "",
        "Check `motion_response/left.png` and `motion_response/right.png`. A positive result should show spatially coherent horizontal displacement fields with larger magnitude on shallower regions.",
        "",
        "### Objective 2: temporal correspondence",
        "",
        "Check `correspondence/overlay.png` and the matching error metric. A positive result should show mostly horizontal, parallel matches with depth-dependent line length.",
        "",
        "### Objective 3: geometry-relevant representation",
        "",
        "Level A is the direct geometric test: `depth_recovery/geometric.png` versus `depth_recovery/gt_comparison.png`.",
        "Level B is the practical probe test: compare the depth-probe MAEs across trained correlation, random correlation, and single-frame feature controls.",
        "",
        "## Output Files",
        "",
        "- `motion_response/left.png`",
        "- `motion_response/right.png`",
        "- `correspondence/overlay.png`",
        "- `depth_recovery/geometric.png`",
        "- `depth_recovery/gt_comparison.png`",
        "- `encoder_features/encoder_feature_maps.png`",
        "- `depth_probe_corr/eval_test/pred_depth.png`",
        "- `depth_probe_random/eval_test/pred_depth.png`",
        "- `depth_probe_single_frame/eval_test/pred_depth.png`",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run validation for the three-frame left-right setting.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--backbone-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decoder-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_bundle(args)


if __name__ == "__main__":
    main()
