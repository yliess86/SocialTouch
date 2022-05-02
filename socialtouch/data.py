from __future__ import annotations

from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import interpolate
from torchbooster.dataset import Split
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T


def load_data(
    path: Path,
    height: int,
    width: int,
    window_size: int,
    hop_size: int,
    split: Split,
) -> tuple(list(np.ndarray), list(int)):
    records, gestures = [], []
    
    for file in tqdm(list(path.glob("*.csv")), desc="Loading File"):
        df = pd.read_csv(str(file.resolve()))
        
        for gesture in df.gesture.unique():
            df_gesture = df[df.gesture == gesture]
            
            repetitions = list(df_gesture.repetition.unique())
            n = int(0.75 * len(repetitions))
            for repetition in repetitions[:n] if split == Split.TRAIN else repetitions[n:]:
                df_repetition = df_gesture[df_gesture.repetition == repetition]
                
                frames = df_gesture[filter(lambda name: "pixel" in name, df_repetition.columns)].values
                frames = frames.reshape(frames.shape[0], 1, height, width)
                
                windows = sliding_window_view(frames, window_size, axis=0)[::hop_size]
                windows = windows.transpose((0, 4, 1, 2, 3))
                
                for window in windows:
                    records.append(window)
                    gestures.append(gesture)
    
    return records, gestures


class SocialTouchDataset(Dataset):
    def __init__(
        self,
        path: Path,
        height: int,
        width: int,
        window_size: int,
        hop_size: int,
        split: Split,
        transform: Module,
    ) -> None:
        super().__init__()
        self.records, self.gestures = load_data(path, height, width, window_size, hop_size, split)
        self.transform = transform

    @property
    def n_class(self) -> int:
        return len(set(self.gestures))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple(Tensor, int):
        return self.transform(self.records[idx]), self.gestures[idx]


def scale_crop_or_pad(x: Tensor, factor: float) -> Tensor:
    t: Tensor = interpolate(x, scale_factor=factor, mode="bilinear")
    r = torch.zeros_like(x)
    *_, Ht, Wt = t.size()
    *_, Hr, Wr = r.size()
    if Ht < Hr:
        j = np.random.randint(0, Hr - Ht)
        i = np.random.randint(0, Wr - Wt)
        r[..., j:j + Ht, i:i + Wt] = t[..., :, :]
    else:
        j = np.random.randint(0, Ht - Hr) if Ht - Hr else 0
        i = np.random.randint(0, Wt - Wr) if Wt - Wr else 0
        r[..., :, :] = t[..., j:j + Hr, i:i + Wr]
    return r


ToTensor      = T.Lambda(lambda x: torch.from_numpy(x.copy()).float() / 255.0)
RandomReverse = T.Lambda(lambda x: x if np.random.rand() > 0.5 else x[::-1, :, :, :])
RandomFlipY   = T.Lambda(lambda x: x if np.random.rand() > 0.5 else x[:, :, ::-1, :])
RandomFlipX   = T.Lambda(lambda x: x if np.random.rand() > 0.5 else x[:, :, :, ::-1])
RandomScale   = T.Lambda(lambda x: scale_crop_or_pad(x, factor=0.8 + 0.4 * np.random.rand()))


if __name__ == "__main__":
    WINDOW_SIZE, WINDOW_HOP = 24, 3
    FRAME_WIDTH, FRAME_HEIGHT = 24, 19

    parameters = FRAME_HEIGHT, FRAME_WIDTH, WINDOW_SIZE, WINDOW_HOP
    transform = T.Compose([RandomReverse, RandomFlipY, RandomFlipX, ToTensor, RandomScale])
    dataset = SocialTouchDataset(Path("data"), *parameters, transform, Split.TEST)

    n_class = dataset.n_class
    with tqdm(dataset, desc="Checking Dataset") as pbar:
        for record, gesture in pbar:
            pbar.set_postfix(
                record=f"{tuple(record.size())}",
                gesture=f"{gesture:02d}/{n_class}",
            )