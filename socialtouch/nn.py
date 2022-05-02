from __future__ import annotations

from torch import Tensor
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Flatten,
    GELU,
    Linear,
    LSTM,
    Module,
    Sequential,
    TransformerEncoderLayer as Transformer,
)

import numpy as np
import torch


class Temporal(Module):
    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        B, S, *R = x.size()
        x = x.reshape(B * S, *R)
        x = self.module(x)
        _, *R = x.size()
        x = x.reshape(B, S, *R)
        return x


class BahdanauAttention(Module):
    def __init__(self, h_dim: int) -> None:
        super().__init__()
        self.key     = Linear(h_dim, h_dim, bias=True )
        self.query   = Linear(h_dim, h_dim, bias=False)
        self.context = Linear(h_dim,     1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        k = self.key(x)
        q = self.query(x.mean(dim=1, keepdim=True))
        s = self.context(torch.tanh(k + q)).view(x.size(0), -1)
        a = torch.softmax(s, dim=1)
        return (a[:, :, None] * x).sum(dim=1)


class PositionalEncoding(Module):
    def __init__(self, h_dim: int, max_len: int = 128) -> None:
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, h_dim, 2) * (-np.log(10000.0) / h_dim))
        pe = torch.zeros(1, max_len, h_dim)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1)]


class RecurrentNet(Module):
    def __init__(self, i_dim: int, h_dim: int, o_dim: int) -> None:
        super().__init__()
        self.temporal_features = Temporal(Sequential(
            Flatten(),
            Linear(i_dim, h_dim), GELU(),
            Linear(h_dim, h_dim), GELU(),
        ))
        self.lstm = LSTM(h_dim, h_dim, batch_first=True)
        self.classifier = Linear(h_dim, o_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.temporal_features(x)
        x, _ = self.lstm(x)
        x = self.classifier(x[:, -1])
        return x


class RecurrentConvNet(Module):
    def __init__(self, h_dim: int, o_dim: int) -> None:
        super().__init__()
        self.temporal_features = Temporal(Sequential(
            Conv2d( 1,  8, 5, 2, bias=False), BatchNorm2d( 8), GELU(),
            Conv2d( 8, 16, 3, 1, bias=False), BatchNorm2d(16), GELU(),
            Conv2d(16, 32, 3, 1, bias=False), BatchNorm2d(32), GELU(),
            Conv2d(32, 64, 3, 1, bias=False), BatchNorm2d(64), GELU(),
            Flatten(),
        ))
        self.lstm = LSTM(64 * 2 * 4, h_dim, batch_first=True)
        self.classifier = Linear(h_dim, o_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.temporal_features(x)
        x, _ = self.lstm(x)
        x = self.classifier(x[:, -1])
        return x


class RecurrentAttentionConvNet(Module):
    def __init__(self, h_dim: int, o_dim: int) -> None:
        super().__init__()
        self.temporal_features = Temporal(Sequential(
            Conv2d( 1,  8, 5, 2, bias=False), BatchNorm2d( 8), GELU(),
            Conv2d( 8, 16, 3, 1, bias=False), BatchNorm2d(16), GELU(),
            Conv2d(16, 32, 3, 1, bias=False), BatchNorm2d(32), GELU(),
            Conv2d(32, 64, 3, 1, bias=False), BatchNorm2d(64), GELU(),
            Flatten(),
        ))
        self.lstm = LSTM(64 * 2 * 4, h_dim, batch_first=True)
        self.attention = BahdanauAttention(h_dim)
        self.classifier = Linear(h_dim, o_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.temporal_features(x)
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x


class TransformerAttentionConvNet(Module):
    def __init__(self, h_dim: int, o_dim: int, nhead: int, max_len: int = 128) -> None:
        super().__init__()
        self.temporal_features = Temporal(Sequential(
            Conv2d( 1,  8, 5, 2, bias=False), BatchNorm2d( 8), GELU(),
            Conv2d( 8, 16, 3, 1, bias=False), BatchNorm2d(16), GELU(),
            Conv2d(16, 32, 3, 1, bias=False), BatchNorm2d(32), GELU(),
            Conv2d(32, 64, 3, 1, bias=False), BatchNorm2d(64), GELU(),
            Flatten(),
        ))
        self.posenc = PositionalEncoding(64 * 2 * 4, max_len=max_len)
        self.transformer = Transformer(64 * 2 * 4, nhead=nhead, dim_feedforward=h_dim, batch_first=True)
        self.attention = BahdanauAttention(64 * 2 * 4)
        self.classifier = Linear(64 * 2 * 4, o_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.temporal_features(x)
        x = self.posenc(x)
        x = self.transformer(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    WINDOW_SIZE = 24
    FRAME_WIDTH, FRAME_HEIGHT = 24, 19
    H_DIM, N_CLASS = 64, 32
    NHEAD = 8

    input = torch.rand((1, WINDOW_SIZE, 1, FRAME_HEIGHT, FRAME_WIDTH)).cuda()
    
    model = RecurrentNet(FRAME_HEIGHT * FRAME_WIDTH, H_DIM, N_CLASS).cuda()
    predi = model(input)
    print(input.size(), predi.size())

    model = RecurrentConvNet(H_DIM, N_CLASS).cuda()
    predi = model(input)
    print(input.size(), predi.size())

    model = RecurrentAttentionConvNet(H_DIM, N_CLASS).cuda()
    predi = model(input)
    print(input.size(), predi.size())

    model = TransformerAttentionConvNet(H_DIM, N_CLASS, NHEAD, WINDOW_SIZE).cuda()
    predi = model(input)
    print(input.size(), predi.size())