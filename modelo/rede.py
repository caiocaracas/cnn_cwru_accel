"""define a topologia da rede a partir dos hiperparametros."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

INPUT_LENGTH = 1024
JANELAS = (128, 256, 512, 1024, 2048)
INPUT_CHANNELS = 1
NUM_CLASSES = 4
POOL_SIZE = 2

PoolType = Literal["max", "avg", "none"]
HeadType = Literal["flatten", "gap"]

@dataclass
class ModelConfig:
    num_layers: int
    num_filters_first: int
    kernel_size: int
    pool_type: PoolType
    head: HeadType = "flatten"
    input_len: int = INPUT_LENGTH

    def __post_init__(self):
        if self.num_layers not in (2, 3, 4):
            raise ValueError(f"num_layers={self.num_layers} fora de {{2,3,4}}")
        if self.num_filters_first not in (8, 16, 32):
            raise ValueError(f"num_filters_first={self.num_filters_first} fora de {{8,16,32}}")
        if self.kernel_size not in (3, 5, 7):
            raise ValueError(f"kernel_size={self.kernel_size} fora de {{3,5,7}}")
        if self.pool_type not in ("max", "avg", "none"):
            raise ValueError(f"pool_type={self.pool_type} fora de {{max,avg,none}}")
        if self.head not in ("flatten", "gap"):
            raise ValueError(f"head={self.head} fora de {{flatten,gap}}")
        if self.input_len not in JANELAS:
            raise ValueError(f"input_len={self.input_len} fora de {JANELAS}")
        if self.input_len < 2 ** self.num_layers * self.kernel_size:
            raise ValueError(
                f"janela {self.input_len} curta demais para {self.num_layers} "
                f"camadas de kernel {self.kernel_size}")

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            num_layers=int(d["num_layers"]),
            num_filters_first=int(d["num_filters_first"]),
            kernel_size=int(d["kernel_size"]),
            pool_type=str(d["pool_type"]),
            head=str(d.get("head", "flatten")),
            input_len=int(d.get("input_len", INPUT_LENGTH)),
        )

def geometria_saida(cfg: ModelConfig) -> tuple[int, int]:
    ch = cfg.num_filters_first
    comp = cfg.input_len
    for _ in range(cfg.num_layers - 1):
        if cfg.pool_type in ("max", "avg"):
            comp //= POOL_SIZE
        ch *= 2
    if cfg.pool_type in ("max", "avg"):
        comp //= POOL_SIZE
    return ch, comp

def dim_entrada_fc(cfg: ModelConfig) -> int:
    ch, comp = geometria_saida(cfg)
    return ch if cfg.head == "gap" else ch * comp

class MediaGlobal(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2)

class CNN1D(nn.Module):

    def __init__(self, cfg: ModelConfig, dropout: float = 0.0,
                 batchnorm: bool = True):
        super().__init__()
        self.cfg = cfg

        pad = (cfg.kernel_size - 1) // 2

        layers = []
        in_ch = INPUT_CHANNELS
        out_ch = cfg.num_filters_first
        current_length = cfg.input_len

        for _ in range(cfg.num_layers):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=cfg.kernel_size,
                                    stride=1, padding=pad, bias=True))
            layers.append(nn.BatchNorm1d(out_ch) if batchnorm else nn.Identity())
            layers.append(nn.ReLU(inplace=True))

            if cfg.pool_type == "max":
                layers.append(nn.MaxPool1d(kernel_size=POOL_SIZE, stride=POOL_SIZE))
                current_length //= POOL_SIZE
            elif cfg.pool_type == "avg":
                layers.append(nn.AvgPool1d(kernel_size=POOL_SIZE, stride=POOL_SIZE))
                current_length //= POOL_SIZE

            in_ch = out_ch
            out_ch *= 2

        self.features = nn.Sequential(*layers)
        self.head = MediaGlobal() if cfg.head == "gap" else nn.Flatten(start_dim=1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        flat_dim = in_ch if cfg.head == "gap" else in_ch * current_length
        self.classifier = nn.Linear(flat_dim, NUM_CLASSES)

        self._init_weights()

        self._final_channels = in_ch
        self._final_length = current_length
        self._flat_dim = flat_dim

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def build_model(cfg: ModelConfig, dropout: float = 0.0,
                batchnorm: bool = True) -> CNN1D:
    return CNN1D(cfg, dropout=dropout, batchnorm=batchnorm)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_macs(cfg: ModelConfig) -> int:
    macs = 0
    in_ch = INPUT_CHANNELS
    out_ch = cfg.num_filters_first
    current_length = cfg.input_len

    for _ in range(cfg.num_layers):

        macs += out_ch * in_ch * cfg.kernel_size * current_length

        if cfg.pool_type in ("max", "avg"):
            current_length //= POOL_SIZE

        in_ch = out_ch
        out_ch *= 2

    macs += dim_entrada_fc(cfg) * NUM_CLASSES

    return macs

def model_info(cfg: ModelConfig) -> dict:
    model = build_model(cfg, batchnorm=False)
    return {
        "num_layers": cfg.num_layers,
        "num_filters_first": cfg.num_filters_first,
        "kernel_size": cfg.kernel_size,
        "pool_type": cfg.pool_type,
        "head": cfg.head,
        "num_params": count_parameters(model),
        "weight_bytes_fp32": count_parameters(model) * 4,
        "macs_per_inference": estimate_macs(cfg),
        "final_channels": model._final_channels,
        "final_length": model._final_length,
        "flat_dim": model._flat_dim,
    }
