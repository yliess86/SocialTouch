from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from socialtouch.data import SocialTouchDataset
from socialtouch.nn import (RecurrentNet, RecurrentConvNet, RecurrentAttentionConvNet, TransformerAttentionConvNet)
from torch.nn import Module
from torchbooster.config import (
    BaseConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from torchbooster.dataset import Split


@dataclass
class SocialTouchDatasetConfig(BaseConfig):
    path: str
    height: int
    width: int
    window_size: int
    hop_size: int

    def make(self, split: Split, transform: Module) -> SocialTouchDataset:
        return SocialTouchDataset(
            Path(self.path),
            self.height,
            self.width,
            self.window_size,
            self.hop_size,
            split=split.value,
            transform=transform,
        )


@dataclass
class ModelConfig(BaseConfig):
    name: str
    h_dim: int = 256
    nhead: int = 8

    def make(self, height: int, width: int, n_class: int, max_len) -> Module:
        if self.name == "rnet"  : return RecurrentNet(height * width, self.h_dim, n_class)
        if self.name == "rcnet" : return RecurrentConvNet(self.h_dim, n_class)
        if self.name == "racnet": return RecurrentAttentionConvNet(self.h_dim, n_class)
        if self.name == "tacnet": return TransformerAttentionConvNet(self.h_dim, n_class, self.nhead, max_len)
        raise ValueError(f"Architecture {self.name} unkown. Available: 'recurrent_net'.")


@dataclass
class Config(BaseConfig):
    n_iter: int
    seed: int

    env: EnvironementConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig
    dataset: SocialTouchDatasetConfig
    model: ModelConfig