from __future__ import annotations

import json
import math
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

matplotlib.use("Agg")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class TripletLRTwoBallDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str) -> None:
        self.split_dir = Path(data_root) / split
        self.sample_dirs = sorted([p for p in self.split_dir.iterdir() if p.is_dir()])
        if not self.sample_dirs:
            raise FileNotFoundError(f"No samples found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict:
        sample_dir = self.sample_dirs[index]
        meta = load_json(sample_dir / "meta.json")
        return {
            "img_t": self._load_image(sample_dir / "img_t.png"),
            "img_t1": self._load_image(sample_dir / "img_t1.png"),
            "img_t2": self._load_image(sample_dir / "img_t2.png"),
            "tau1_x": torch.tensor(meta["T_t_to_t1"][0][3], dtype=torch.float32),
            "tau2_x": torch.tensor(meta["T_t1_to_t2"][0][3], dtype=torch.float32),
            "K": torch.tensor(meta["K"], dtype=torch.float32),
            "sample_id": sample_dir.name,
        }

    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


class RegionEncoder(nn.Module):
    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalCorrelation(nn.Module):
    def __init__(self, radius: int = 6, temperature: float = 0.07) -> None:
        super().__init__()
        self.radius = radius
        self.kernel_size = 2 * radius + 1
        self.temperature = temperature

    def forward(self, f_a: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = f_a.shape
        a = F.normalize(f_a, dim=1)
        b = F.normalize(f_b, dim=1)
        patches = F.unfold(b, kernel_size=self.kernel_size, padding=self.radius)
        patches = patches.view(bsz, channels, self.kernel_size * self.kernel_size, height, width)
        corr = (a.unsqueeze(2) * patches).sum(dim=1) / self.temperature
        return corr


def correlation_offsets(radius: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    dx, dy = torch.meshgrid(coords, coords, indexing="xy")
    return dx.reshape(-1), dy.reshape(-1)


class TripletLRSelfSupModel(nn.Module):
    def __init__(self, radius: int = 6, downsample: int = 4, depth_min: float = 0.2, depth_max: float = 2.2, eps: float = 1e-3, corr_temperature: float = 0.07) -> None:
        super().__init__()
        self.encoder = RegionEncoder(out_channels=128)
        self.correlation = LocalCorrelation(radius=radius, temperature=corr_temperature)
        self.radius = radius
        self.downsample = downsample
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.eps = eps
        self.corr_temperature = corr_temperature

    def forward(self, img_t: torch.Tensor, img_t1: torch.Tensor, img_t2: torch.Tensor, tau1_x: torch.Tensor, tau2_x: torch.Tensor, K: torch.Tensor) -> dict[str, torch.Tensor]:
        f_t = self.encoder(img_t)
        f_t1 = self.encoder(img_t1)
        f_t2 = self.encoder(img_t2)
        c1 = self.correlation(f_t, f_t1)
        c2 = self.correlation(f_t1, f_t2)

        dx_offsets, dy_offsets = correlation_offsets(self.radius, c1.device)
        a1 = torch.softmax(c1, dim=1)
        a2 = torch.softmax(c2, dim=1)
        mu1_x = (a1 * dx_offsets.view(1, -1, 1, 1)).sum(dim=1)
        mu2_x = (a2 * dx_offsets.view(1, -1, 1, 1)).sum(dim=1)
        mu2_y = (a2 * dy_offsets.view(1, -1, 1, 1)).sum(dim=1)

        fx = K[:, 0, 0].view(-1, 1, 1)
        delta_u = mu1_x * float(self.downsample)
        depth_raw = fx * tau1_x.view(-1, 1, 1) / (delta_u + self.eps * torch.sign(tau1_x).view(-1, 1, 1))
        d_rel = torch.clamp(depth_raw, min=self.depth_min, max=self.depth_max)

        bsz, _, h, w = c1.shape
        xs = torch.arange(w, device=c1.device, dtype=torch.float32).view(1, 1, w).expand(bsz, h, w)
        ys = torch.arange(h, device=c1.device, dtype=torch.float32).view(1, h, 1).expand(bsz, h, w)
        q_pred_x = xs + mu1_x * (tau2_x.view(-1, 1, 1) / (tau1_x.view(-1, 1, 1) + self.eps * torch.sign(tau1_x).view(-1, 1, 1)))
        q_pred_y = ys
        q_actual_x = xs + mu2_x
        q_actual_y = ys + mu2_y

        valid = (
            torch.isfinite(depth_raw)
            & (depth_raw > 0.0)
            & (q_pred_x >= 0.0)
            & (q_pred_x <= (w - 1))
            & (q_pred_y >= 0.0)
            & (q_pred_y <= (h - 1))
        )

        point_loss = (q_pred_x - q_actual_x) ** 2 + (q_pred_y - q_actual_y) ** 2
        loss = (point_loss * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        return {
            "loss": loss,
            "f_t": f_t,
            "f_t1": f_t1,
            "f_t2": f_t2,
            "c1": c1,
            "c2": c2,
            "a1": a1,
            "a2": a2,
            "mu1_x": mu1_x,
            "mu2_x": mu2_x,
            "mu2_y": mu2_y,
            "depth_raw": depth_raw,
            "d_rel": d_rel,
            "q_pred_x": q_pred_x,
            "q_actual_x": q_actual_x,
            "valid": valid,
        }


def collate_keep_strings(batch: list[dict]) -> dict:
    out = {"sample_id": [item["sample_id"] for item in batch]}
    for key in batch[0]:
        if key == "sample_id":
            continue
        out[key] = torch.stack([item[key] for item in batch])
    return out


def plot_history(history: list[dict[str, float]], path: str | Path, title: str, keys: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4.5))
    for key in keys:
        plt.plot(epochs, [row[key] for row in history], label=key)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
