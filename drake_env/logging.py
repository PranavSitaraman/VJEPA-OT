from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa, pyarrow.parquet as pq

@dataclass
class EpisodeLogger:
    root: str
    def start(self, eid: str):
        self.path = os.path.join(self.root, eid)
        os.makedirs(self.path, exist_ok=True)
        self.rows = []

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            # Drake camera format: (C, H, W) -> (H, W, C)
            arr = np.transpose(arr[:3], (1, 2, 0))
            # NO FLIPS: Meta's pretrained VJEPA2 expects unflipped images
            # Flipping would destroy the pretrained representations
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
            # NO FLIPS: Keep consistent with pretrained model expectations
        arr = np.clip(arr, 0, 255)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    def log(self, idx: int, img_front: np.ndarray, img_wrist: np.ndarray, state: Dict[str, Any], action: Dict[str, Any], contact: Dict[str, Any]):
        front = self._prepare_image(img_front)
        wrist = self._prepare_image(img_wrist)
        Image.fromarray(front).save(os.path.join(self.path, f"front_{idx:06d}.png"))
        Image.fromarray(wrist).save(os.path.join(self.path, f"wrist_{idx:06d}.png"))
        row = {"t": idx}
        for k,v in state.items(): row[f"state_{k}"] = np.asarray(v).reshape(-1).tolist()
        for k,v in action.items(): row[f"act_{k}"] = np.asarray(v).reshape(-1).tolist()
        for k,v in contact.items(): row[f"contact_{k}"] = float(v)
        self.rows.append(row)

    def finish(self, meta: Dict[str, Any]):
        table = pa.Table.from_pandas(pd.DataFrame(self.rows))
        pq.write_table(table, os.path.join(self.path, "episode.parquet"))
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
