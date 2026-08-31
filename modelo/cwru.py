"""le o dataset de rolamentos e recorta as janelas."""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

TARGET_RATE_HZ = 12_000
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 512

NOMINAL_RPM_BY_LOAD_HP = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

CLASS_NORMAL = 0
CLASS_INNER_RACE = 1
CLASS_BALL = 2
CLASS_OUTER_RACE = 3

CLASS_NAMES = {
    CLASS_NORMAL: "normal",
    CLASS_INNER_RACE: "inner_race",
    CLASS_BALL: "ball",
    CLASS_OUTER_RACE: "outer_race",
}

MULT_FALHA = {CLASS_OUTER_RACE: 3.5848,
              CLASS_INNER_RACE: 5.4152,
              CLASS_BALL:       2.3567}

def periodos_por_janela(window_size: int, load_hp: int = 3,
                        fs: int = TARGET_RATE_HZ) -> dict:
    fr = NOMINAL_RPM_BY_LOAD_HP[load_hp] / 60.0
    return {CLASS_NAMES[c]: round(window_size / fs * (m * fr), 2)
            for c, m in MULT_FALHA.items()}

GROUPS = {
    "12k Drive End Bearing Fault Data": (12_000, ("DE",)),
    "48k Drive End Bearking Fault Data": (48_000, ("DE",)),
    "12k Fan End Bearing Fault Data": (12_000, ("FE",)),
    "Normal Baseline Data": (48_000, ("DE", "FE")),
}

_G12DE = "12k Drive End Bearing Fault Data"
_G48DE = "48k Drive End Bearking Fault Data"
_G12FE = "12k Fan End Bearing Fault Data"
_GNORM = "Normal Baseline Data"

_TABELA = [
    (_GNORM, CLASS_NORMAL, None, None, [97, 98, 99, 100]),

    (_G12DE, CLASS_INNER_RACE, 0.007, None, [105, 106, 107, 108]),
    (_G12DE, CLASS_BALL,       0.007, None, [118, 119, 120, 121]),
    (_G12DE, CLASS_OUTER_RACE, 0.007, "@6", [130, 131, 132, 133]),
    (_G12DE, CLASS_OUTER_RACE, 0.007, "@3", [144, 145, 146, 147]),
    (_G12DE, CLASS_OUTER_RACE, 0.007, "@12", [156, 158, 159, 160]),
    (_G12DE, CLASS_INNER_RACE, 0.014, None, [169, 170, 171, 172]),
    (_G12DE, CLASS_BALL,       0.014, None, [185, 186, 187, 188]),
    (_G12DE, CLASS_OUTER_RACE, 0.014, "@6", [197, 198, 199, 200]),
    (_G12DE, CLASS_INNER_RACE, 0.021, None, [209, 210, 211, 212]),
    (_G12DE, CLASS_BALL,       0.021, None, [222, 223, 224, 225]),
    (_G12DE, CLASS_OUTER_RACE, 0.021, "@6", [234, 235, 236, 237]),
    (_G12DE, CLASS_OUTER_RACE, 0.021, "@3", [246, 247, 248, 249]),
    (_G12DE, CLASS_OUTER_RACE, 0.021, "@12", [258, 259, 260, 261]),
    (_G12DE, CLASS_INNER_RACE, 0.028, None, [3001, 3002, 3003, 3004]),
    (_G12DE, CLASS_BALL,       0.028, None, [3005, 3006, 3007, 3008]),

    (_G48DE, CLASS_INNER_RACE, 0.007, None, [109, 110, 111, 112]),
    (_G48DE, CLASS_BALL,       0.007, None, [122, 123, 124, 125]),
    (_G48DE, CLASS_OUTER_RACE, 0.007, "@6", [135, 136, 137, 138]),
    (_G48DE, CLASS_OUTER_RACE, 0.007, "@3", [148, 149, 150, 151]),
    (_G48DE, CLASS_OUTER_RACE, 0.007, "@12", [161, 162, 163, 164]),
    (_G48DE, CLASS_INNER_RACE, 0.014, None, [174, 175, 176, 177]),
    (_G48DE, CLASS_BALL,       0.014, None, [189, 190, 191, 192]),
    (_G48DE, CLASS_OUTER_RACE, 0.014, "@6", [201, 202, 203, 204]),
    (_G48DE, CLASS_INNER_RACE, 0.021, None, [213, 214, 215, 217]),
    (_G48DE, CLASS_BALL,       0.021, None, [226, 227, 228, 229]),
    (_G48DE, CLASS_OUTER_RACE, 0.021, "@6", [238, 239, 240, 241]),
    (_G48DE, CLASS_OUTER_RACE, 0.021, "@3", [250, 251, 252, 253]),
    (_G48DE, CLASS_OUTER_RACE, 0.021, "@12", [262, 263, 264, 265]),

    (_G12FE, CLASS_INNER_RACE, 0.007, None, [278, 279, 280, 281]),
    (_G12FE, CLASS_BALL,       0.007, None, [282, 283, 284, 285]),
    (_G12FE, CLASS_OUTER_RACE, 0.007, "@6", [294, 295, 296, 297]),
    (_G12FE, CLASS_OUTER_RACE, 0.007, "@3", [298, 299, 300, 301]),
    (_G12FE, CLASS_OUTER_RACE, 0.007, "@12", [302, 305, 306, 307]),
    (_G12FE, CLASS_INNER_RACE, 0.014, None, [274, 275, 276, 277]),
    (_G12FE, CLASS_BALL,       0.014, None, [286, 287, 288, 289]),
    (_G12FE, CLASS_OUTER_RACE, 0.014, "@6", [313, None, None, None]),
    (_G12FE, CLASS_OUTER_RACE, 0.014, "@3", [310, 309, 311, 312]),
    (_G12FE, CLASS_INNER_RACE, 0.021, None, [270, 271, 272, 273]),
    (_G12FE, CLASS_BALL,       0.021, None, [290, 291, 292, 293]),
    (_G12FE, CLASS_OUTER_RACE, 0.021, "@6", [315, None, None, None]),
    (_G12FE, CLASS_OUTER_RACE, 0.021, "@3", [None, 316, 317, 318]),
]

@dataclass(frozen=True)
class FileSpec:
    group: str
    class_id: int
    fault_diameter_inches: Optional[float]
    position: Optional[str]
    load_hp: int
    rate_hz: int
    channels: tuple

def _build_index() -> dict[int, FileSpec]:
    idx = {}
    for group, cls, diam, pos, nums in _TABELA:
        rate, channels = GROUPS[group]
        for load, num in enumerate(nums):
            if num is None:
                continue
            if num in idx:
                raise ValueError(f"arquivo {num} duplicado na tabela canonica")
            idx[num] = FileSpec(group, cls, diam, pos, load, rate, channels)
    return idx

CWRU_FILES: dict[int, FileSpec] = _build_index()

@dataclass
class FileMetadata:
    path: Path
    class_id: int
    fault_diameter_inches: Optional[float]
    load_hp: int
    cwru_file_number: int
    position: Optional[str]
    channel: str
    rate_hz: int

_CH_PATTERN = {
    "DE": re.compile(r"^X(\d+)_DE_time$"),
    "FE": re.compile(r"^X(\d+)_FE_time$"),
}
_RPM_PATTERN = re.compile(r"^X(\d+)RPM$")

def _pick(mat, pattern, file_number):
    cand = [(int(m.group(1)), k) for k in mat if (m := pattern.match(k))]
    if not cand:
        return None
    exact = [k for num, k in cand if num == file_number]
    if exact:
        return exact[0]
    if len(cand) == 1:
        return cand[0][1]
    raise ValueError(f"multiplas chaves {[k for _, k in cand]} e nenhuma casa com {file_number}")

def load_channel(mat_path: Path, file_number: int, channel: str,
                 rate_hz: int) -> tuple[Optional[np.ndarray], Optional[float]]:
    mat = loadmat(str(mat_path))

    key = _pick(mat, _CH_PATTERN[channel], file_number)
    if key is None:
        return None, None

    signal = np.asarray(mat[key], dtype=np.float64).squeeze()
    if signal.ndim != 1:
        raise ValueError(f"esperava sinal 1d, recebi {signal.shape} em {mat_path.name}")

    if rate_hz != TARGET_RATE_HZ:
        if rate_hz % TARGET_RATE_HZ:
            raise ValueError(f"taxa {rate_hz} nao e multiplo de {TARGET_RATE_HZ}")
        signal = resample_poly(signal, 1, rate_hz // TARGET_RATE_HZ)

    rpm = None
    rpm_key = _pick(mat, _RPM_PATTERN, file_number)
    if rpm_key is not None:
        v = np.asarray(mat[rpm_key]).squeeze()
        rpm = float(v.item() if v.ndim == 0 else v[0])

    return signal, rpm

def windowize(signal: np.ndarray, window_size: int = DEFAULT_WINDOW_SIZE,
              stride: int = DEFAULT_STRIDE) -> np.ndarray:
    if signal.ndim != 1:
        raise ValueError(f"esperava sinal 1d, recebi {signal.ndim}d")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size e stride devem ser positivos")
    if len(signal) < window_size:
        return np.empty((0, window_size), dtype=signal.dtype)
    n = (len(signal) - window_size) // stride + 1
    return np.stack([signal[i * stride: i * stride + window_size] for i in range(n)])

def zscore_per_window(windows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if windows.ndim != 2:
        raise ValueError(f"esperava (n, w), recebi {windows.shape}")
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    return (windows - mean) / (std + eps)

ENTRADA_BITS = 8

def escala_int8(X_z: np.ndarray, bits: int = ENTRADA_BITS) -> float:
    qmax = 2 ** (bits - 1) - 1
    m = float(np.abs(X_z).max())
    return (m / qmax) if m > 0 else 1.0

def para_int8(X_z: np.ndarray, escala: float,
              bits: int = ENTRADA_BITS) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    return np.clip(np.rint(np.asarray(X_z) / escala),
                   -qmax - 1, qmax).astype(np.int8)

def janela_int8(bruto: np.ndarray, escala: float,
                bits: int = ENTRADA_BITS) -> np.ndarray:
    return para_int8(zscore_per_window(np.atleast_2d(bruto)), escala, bits)

def escala_por_condicao(bruto: np.ndarray, fonte) -> tuple[np.ndarray, np.ndarray]:
    fonte = np.asarray(fonte)
    mu = np.empty(len(fonte), dtype=np.float64)
    sd = np.empty(len(fonte), dtype=np.float64)
    for f in np.unique(fonte):
        m = fonte == f
        x = bruto[m]
        mu[m] = float(x.mean())
        sd[m] = float(x.std())
    return mu, sd

def chave_condicao(class_id: int, diam: Optional[float], pos: Optional[str],
                   load_hp: int) -> str:
    d = "na" if diam is None else f"{diam:.3f}"
    return f"c{class_id}_d{d}_p{pos or '-'}_l{load_hp}"

@dataclass
class IngestionResult:
    X: np.ndarray
    X_bruto: np.ndarray
    y: np.ndarray
    rpm: np.ndarray
    load_hp: np.ndarray
    fault_diameter: np.ndarray
    source_file: list[str]
    metadata: list[FileMetadata]
    channel: np.ndarray
    position: np.ndarray
    grupo: np.ndarray

def ingest_directory(data_root: Path, window_size: int = DEFAULT_WINDOW_SIZE,
                     stride: int = DEFAULT_STRIDE,
                     skip_unrecognized: bool = False) -> IngestionResult:
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise NotADirectoryError(f"{data_root} nao e diretorio")

    files = sorted(data_root.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"nenhum *.mat sob {data_root}")

    cols = {k: [] for k in ("win", "bruto", "y", "rpm", "load", "diam", "src",
                            "ch", "pos", "grp")}
    meta = []

    for path in files:
        try:
            num = int(path.stem)
        except ValueError:
            if skip_unrecognized:
                continue
            raise ValueError(f"nome nao numerico: {path.name}")

        spec = CWRU_FILES.get(num)
        if spec is None:
            if skip_unrecognized:
                print(f"[skip] {path.name}: fora da tabela canonica")
                continue
            raise ValueError(f"arquivo {num} fora da tabela canonica CWRU")

        for ch in spec.channels:
            signal, rpm = load_channel(path, num, ch, spec.rate_hz)
            if signal is None:
                continue
            w = windowize(signal, window_size, stride)
            if len(w) == 0:
                continue
            if rpm is None:
                rpm = NOMINAL_RPM_BY_LOAD_HP[spec.load_hp]

            n = len(w)
            cols["bruto"].append(w.astype(np.float64))
            cols["win"].append(zscore_per_window(w))
            cols["y"].append(np.full(n, spec.class_id, dtype=np.int64))
            cols["rpm"].append(np.full(n, rpm, dtype=np.float64))
            cols["load"].append(np.full(n, spec.load_hp, dtype=np.int64))
            cols["diam"].append(np.full(
                n, spec.fault_diameter_inches if spec.fault_diameter_inches else np.nan))
            cols["src"].extend([f"{path.name}:{ch}"] * n)
            cols["ch"].extend([ch] * n)
            cols["pos"].extend([spec.position or "-"] * n)
            cols["grp"].extend([chave_condicao(spec.class_id,
                                               spec.fault_diameter_inches,
                                               spec.position, spec.load_hp)] * n)
            meta.append(FileMetadata(path, spec.class_id, spec.fault_diameter_inches,
                                     spec.load_hp, num, spec.position, ch, spec.rate_hz))

    if not cols["win"]:
        raise RuntimeError(f"nenhuma janela extraida de {data_root}")

    X = np.concatenate(cols["win"], axis=0)[:, np.newaxis, :].astype(np.float32)
    return IngestionResult(
        X=X,
        X_bruto=np.concatenate(cols["bruto"], axis=0),
        y=np.concatenate(cols["y"], axis=0),
        rpm=np.concatenate(cols["rpm"], axis=0),
        load_hp=np.concatenate(cols["load"], axis=0),
        fault_diameter=np.concatenate(cols["diam"], axis=0),
        source_file=cols["src"],
        metadata=meta,
        channel=np.asarray(cols["ch"]),
        position=np.asarray(cols["pos"]),
        grupo=np.asarray(cols["grp"]),
    )
