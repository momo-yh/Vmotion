from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class TwoBallMotionDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str) -> None:
        self.split_dir = Path(data_root) / split
        self.sample_dirs = sorted([p for p in self.split_dir.iterdir() if p.is_dir()])
        if not self.sample_dirs:
            raise FileNotFoundError(f"No samples found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_dir = self.sample_dirs[index]
        meta = load_json(sample_dir / "meta.json")
        img_t = self._load_image(sample_dir / "img_t.png")
        img_t1 = self._load_image(sample_dir / "img_t1.png")
        translation = torch.tensor(meta["T_t_to_t1"], dtype=torch.float32)[:3, 3]
        item = {
            "img_t": img_t,
            "img_t1": img_t1,
            "translation_xy": translation[:2],
            "small_depth": torch.tensor(meta["small_ball_center_3d_t"][2], dtype=torch.float32),
            "large_depth": torch.tensor(meta["large_ball_center_3d_t"][2], dtype=torch.float32),
            "small_center_2d": torch.tensor(meta["small_ball_center_2d_t"], dtype=torch.float32),
            "large_center_2d": torch.tensor(meta["large_ball_center_2d_t"], dtype=torch.float32),
            "sample_id": sample_dir.name,
        }
        return item

    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)


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
    def __init__(self, radius: int = 6) -> None:
        super().__init__()
        self.radius = radius
        self.kernel_size = 2 * radius + 1

    def forward(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = z_t.shape
        patches = F.unfold(z_t1, kernel_size=self.kernel_size, padding=self.radius)
        patches = patches.view(bsz, channels, self.kernel_size * self.kernel_size, height, width)
        scores = (z_t.unsqueeze(2) * patches).sum(dim=1) / math.sqrt(channels)
        return scores


class TranslationModel(nn.Module):
    def __init__(self, radius: int = 6) -> None:
        super().__init__()
        self.encoder = RegionEncoder(out_channels=128)
        self.correlation = LocalCorrelation(radius=radius)
        self.head = nn.Linear((2 * radius + 1) ** 2, 2)

    def forward(self, img_t: torch.Tensor, img_t1: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z_t = self.encoder(img_t)
        z_t1 = self.encoder(img_t1)
        corr = self.correlation(z_t, z_t1)
        h_t = corr.mean(dim=(2, 3))
        pred = self.head(h_t)
        return pred, {"z_t": z_t, "z_t1": z_t1, "corr": corr, "h_t": h_t}


class DepthProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class MetricSummary:
    mae: float
    rmse: float


def collate_keep_strings(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list[str]]:
    sample_ids = [item["sample_id"] for item in batch]
    collated: dict[str, torch.Tensor | list[str]] = {"sample_id": sample_ids}
    tensor_keys = [k for k in batch[0].keys() if k != "sample_id"]
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch])
    return collated


def feature_coords_from_image(center_2d: torch.Tensor, feat_height: int, feat_width: int, img_height: int = 128, img_width: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.floor(center_2d[..., 0] * (feat_width / img_width)).long().clamp_(0, feat_width - 1)
    y = torch.floor(center_2d[..., 1] * (feat_height / img_height)).long().clamp_(0, feat_height - 1)
    return y, x


def translation_scales(data_root: str | Path) -> tuple[float, float]:
    train_dir = Path(data_root) / "train"
    tx = []
    ty = []
    for sample_dir in train_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        meta = load_json(sample_dir / "meta.json")
        t = np.asarray(meta["T_t_to_t1"], dtype=np.float32)[:3, 3]
        tx.append(abs(float(t[0])))
        ty.append(abs(float(t[1])))
    return max(max(tx), 1e-6), max(max(ty), 1e-6)


def compute_regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    mae = np.abs(err).mean()
    rmse = float(np.sqrt((err**2).mean()))
    result = {"mae": float(mae), "rmse": rmse}
    if target.ndim == 2 and target.shape[1] == 2:
        result["mae_x"] = float(np.abs(err[:, 0]).mean())
        result["mae_y"] = float(np.abs(err[:, 1]).mean())
        result["rmse_x"] = float(np.sqrt((err[:, 0] ** 2).mean()))
        result["rmse_y"] = float(np.sqrt((err[:, 1] ** 2).mean()))
    return result


def plot_history(history: list[dict[str, float]], path: str | Path, title: str, y_keys: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4.5))
    for key in y_keys:
        plt.plot(epochs, [row[key] for row in history], label=key)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

