"""
model.py

cnn 1d parametrizavel para classificacao de falhas no cwru.

topologia:
    input (1, 1024)
    -> [conv1d(filtros, kernel) -> relu -> pool] x num_layers
    -> flatten
    -> linear(num_classes)

hiperparametros expostos no config:
    num_layers          {2, 3, 4}
    num_filters_first   {4, 8, 16}    (dobra a cada camada subsequente)
    kernel_size         {3, 5, 7}
    pool_type           {"max", "avg", "none"}

fixos:
    stride = 1
    padding = "same"
    ativacao = relu
    pool_size = 2
    num_classes = 4
    input_length = 1024
    input_channels = 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


# constantes do escopo
INPUT_LENGTH = 1024
INPUT_CHANNELS = 1
NUM_CLASSES = 4
POOL_SIZE = 2


# config
PoolType = Literal["max", "avg", "none"]


@dataclass
class ModelConfig:
    """hiperparametros que definem uma cnn 1d especifica."""
    num_layers: int             # 2, 3 ou 4
    num_filters_first: int      # 4, 8 ou 16
    kernel_size: int            # 3, 5 ou 7
    pool_type: PoolType         # "max", "avg" ou "none"

    def __post_init__(self):
        if self.num_layers not in (2, 3, 4):
            raise ValueError(f"num_layers={self.num_layers} fora de {{2,3,4}}")
        if self.num_filters_first not in (4, 8, 16):
            raise ValueError(f"num_filters_first={self.num_filters_first} fora de {{4,8,16}}")
        if self.kernel_size not in (3, 5, 7):
            raise ValueError(f"kernel_size={self.kernel_size} fora de {{3,5,7}}")
        if self.pool_type not in ("max", "avg", "none"):
            raise ValueError(f"pool_type={self.pool_type} fora de {{max,avg,none}}")

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            num_layers=int(d["num_layers"]),
            num_filters_first=int(d["num_filters_first"]),
            kernel_size=int(d["kernel_size"]),
            pool_type=str(d["pool_type"]),
        )

# modelo
class CNN1D(nn.Module):
    """
    cnn 1d com topologia parametrica.

    nao usa batchnorm. razao: a fase de geracao de hdl fica mais limpa
    sem batchnorm pq evita ter que dobrar a normalizacao em pesos durante
    a quantizacao.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # padding "same" em conv1d com stride=1 e (kernel-1)//2 quando kernel impar
        pad = (cfg.kernel_size - 1) // 2

        layers = []
        in_ch = INPUT_CHANNELS
        out_ch = cfg.num_filters_first
        current_length = INPUT_LENGTH

        for _ in range(cfg.num_layers):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=cfg.kernel_size,
                                    stride=1, padding=pad, bias=True))
            layers.append(nn.ReLU(inplace=True))

            if cfg.pool_type == "max":
                layers.append(nn.MaxPool1d(kernel_size=POOL_SIZE, stride=POOL_SIZE))
                current_length //= POOL_SIZE
            elif cfg.pool_type == "avg":
                layers.append(nn.AvgPool1d(kernel_size=POOL_SIZE, stride=POOL_SIZE))
                current_length //= POOL_SIZE
            # "none": nao acrescenta pool

            in_ch = out_ch
            out_ch *= 2  # dobra a cada camada

        self.features = nn.Sequential(*layers)

        # cabeca: flatten + linear -> num_classes
        # in_ch aqui e o numero de filtros da ultima camada
        flat_dim = in_ch * current_length
        self.classifier = nn.Linear(flat_dim, NUM_CLASSES)

        # guarda dimensoes pra debug e calculo de macs
        self._final_channels = in_ch
        self._final_length = current_length
        self._flat_dim = flat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

# fabrica
def build_model(cfg: ModelConfig) -> CNN1D:
    """instancia o modelo a partir do config."""
    return CNN1D(cfg)

# metricas analiticas
def count_parameters(model: nn.Module) -> int:
    """numero total de parametros treinaveis."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_macs(cfg: ModelConfig) -> int:
    """
    estima multiply-accumulates por inferencia (1 amostra de entrada).

    para conv1d:
        macs_camada = filtros_out * filtros_in * kernel * comprimento_saida

    para linear:
        macs_camada = in_features * out_features

    pool e relu nao tem mac (so comparacoes/somas e nao linearidades).

   cruzar com utilizacao de dsp na fpga.
    """
    macs = 0
    in_ch = INPUT_CHANNELS
    out_ch = cfg.num_filters_first
    current_length = INPUT_LENGTH

    for _ in range(cfg.num_layers):
        # padding "same" mantem comprimento, pool divide por 2 (se houver pool)
        macs += out_ch * in_ch * cfg.kernel_size * current_length

        if cfg.pool_type in ("max", "avg"):
            current_length //= POOL_SIZE

        in_ch = out_ch
        out_ch *= 2

    # camada linear final
    flat_dim = in_ch * current_length
    macs += flat_dim * NUM_CLASSES

    return macs

def model_info(cfg: ModelConfig) -> dict:
    """resumo do modelo: usado pra logging."""
    model = build_model(cfg)
    return {
        "num_layers": cfg.num_layers,
        "num_filters_first": cfg.num_filters_first,
        "kernel_size": cfg.kernel_size,
        "pool_type": cfg.pool_type,
        "num_params": count_parameters(model),
        "weight_bytes_fp32": count_parameters(model) * 4,
        "macs_per_inference": estimate_macs(cfg),
        "final_channels": model._final_channels,
        "final_length": model._final_length,
        "flat_dim": model._flat_dim,
    }