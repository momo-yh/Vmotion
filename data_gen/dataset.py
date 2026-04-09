from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MotionPairDataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.sample_dirs = sorted((self.root / split).glob("sample_*"))
        if not self.sample_dirs:
            raise FileNotFoundError(f"No sample directories found in {(self.root / split)!s}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_dir = self.sample_dirs[index]
        img_t = np.asarray(Image.open(sample_dir / "img_t.png").convert("RGB"), dtype=np.float32) / 255.0
        img_t1 = np.asarray(Image.open(sample_dir / "img_t1.png").convert("RGB"), dtype=np.float32) / 255.0
        with (sample_dir / "meta.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        return {
            "img_t": torch.from_numpy(img_t).permute(2, 0, 1),
            "img_t1": torch.from_numpy(img_t1).permute(2, 0, 1),
            "K": torch.tensor(meta["K"], dtype=torch.float32),
            "T": torch.tensor(meta["T_t_to_t1"], dtype=torch.float32),
            "translation": torch.tensor(meta["T_t_to_t1"], dtype=torch.float32)[:3, 3],
            "ball_3d_t": torch.tensor(meta["ball_center_3d_t"], dtype=torch.float32),
            "ball_2d_t": torch.tensor(meta["ball_center_2d_t"], dtype=torch.float32),
            "ball_2d_t1": torch.tensor(meta["ball_center_2d_t1"], dtype=torch.float32),
        }
