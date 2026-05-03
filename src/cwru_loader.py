"""
cwru_loader.py

ingestao do dataset cwru bearing (12 khz drive end + normal baseline)

        normal_baseline_data/
            97.mat   98.mat   99.mat   100.mat        (cargas 0, 1, 2, 3 hp)
        12k_drive_end_bearing_fault_data/
            0.007/
                inner_race/  105.mat 106.mat 107.mat 108.mat
                ball/        118.mat 119.mat 120.mat 121.mat
                centered/    130.mat 131.mat 132.mat 133.mat
            0.014/
                inner_race/  169.mat 170.mat 171.mat 172.mat
                ball/        185.mat 186.mat 187.mat 188.mat
                centered/    197.mat 198.mat 199.mat 200.mat
            0.021/
                inner_race/  209.mat 210.mat 211.mat 212.mat
                ball/        222.mat 223.mat 224.mat 225.mat
                centered/    234.mat 235.mat 236.mat 237.mat

- 4 classes: 0=normal, 1=inner_race, 2=ball, 3=outer_race@6:00 (centered)
- classe vem da pasta. carga vem do mapeamento hardcoded do numero
  do arquivo cwru. numeros nao mapeados causam erro.
- so o canal _de_time (acelerometro do drive end - padrao da literatura)
- janela 1024 amostras, stride 512 (50% overlap)
- z-score por janela
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.io import loadmat

# constantes
SAMPLING_RATE_HZ = 12_000

DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 512

# rpm nominal por carga em hp.
# fonte: tabelas oficiais do cwru. usado como fallback quando o .mat
# nao traz a variavel <prefixo>RPM (alguns arquivos nao tem).
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

# numero do arquivo cwru -> carga em hp.
# fonte: tabela oficial do cwru bearing data center.
# regra: dentro de cada bloco, numero menor = carga menor.
# se vier um numero fora dessa lista, parse_path() levanta erro
CWRU_FILE_TO_LOAD_HP: dict[int, int] = {
    # normal baseline
    97: 0, 98: 1, 99: 2, 100: 3,

    # 12k de - 0.007"
    105: 0, 106: 1, 107: 2, 108: 3,   # inner race
    118: 0, 119: 1, 120: 2, 121: 3,   # ball
    130: 0, 131: 1, 132: 2, 133: 3,   # outer race @6:00 (centered)

    # 12k de - 0.014"
    169: 0, 170: 1, 171: 2, 172: 3,   # inner race
    185: 0, 186: 1, 187: 2, 188: 3,   # ball
    197: 0, 198: 1, 199: 2, 200: 3,   # outer race @6:00 (centered)

    # 12k de - 0.021"
    209: 0, 210: 1, 211: 2, 212: 3,   # inner race
    222: 0, 223: 1, 224: 2, 225: 3,   # ball
    234: 0, 235: 1, 236: 2, 237: 3,   # outer race @6:00 (centered)
}

SUBFOLDER_TO_CLASS = {
    "inner_race": CLASS_INNER_RACE,
    "ball": CLASS_BALL,
    "centered": CLASS_OUTER_RACE,
}

@dataclass
class FileMetadata:
    path: Path
    class_id: int
    fault_diameter_inches: Optional[float]  # none para normal
    load_hp: int
    cwru_file_number: int

# 1. carregamento do .mat (canal _de_time + rpm)

# nomes das variaveis dentro do .mat seguem padrao x<numero>_de_time,
# x<numero>rpm, etc. o <numero> e um id de sessao do cwru e nem sempre
# bate com o nome do arquivo - alguns .mat trazem variaveis de mais de
# uma sessao empacotadas juntas (ex: 99.mat tem X098_DE_time e X099_DE_time).
# regra: pega a variavel cujo numero casa com o nome do arquivo.
_DE_TIME_PATTERN = re.compile(r"^X(\d+)_DE_time$")
_RPM_PATTERN = re.compile(r"^X(\d+)RPM$")


def load_de_signal(mat_path: Path, file_number: int) -> tuple[np.ndarray, Optional[float]]:
    """carrega o canal _de_time correspondente ao file_number.
    retorna (sinal_1d, rpm_ou_none)."""
    mat = loadmat(str(mat_path))

    # acha todas as chaves _de_time e pega a que casa com o numero do arquivo.
    de_candidates = []
    for k in mat.keys():
        m = _DE_TIME_PATTERN.match(k)
        if m:
            de_candidates.append((int(m.group(1)), k))

    if not de_candidates:
        raise ValueError(
            f"nenhuma variavel _DE_time em {mat_path.name}. "
            f"chaves: {[k for k in mat.keys() if not k.startswith('__')]}"
        )

    # prefere a chave cujo numero bate com o nome do arquivo.
    matching = [k for num, k in de_candidates if num == file_number]
    if matching:
        de_key = matching[0]
    elif len(de_candidates) == 1:
        # so uma chave e ela nao bate com o nome - usa assim mesmo, mas avisa.
        de_key = de_candidates[0][1]
        print(f"[warn] {mat_path.name}: chave _DE_time {de_key} nao casa com numero do arquivo")
    else:
        raise ValueError(
            f"{mat_path.name} tem multiplas chaves _DE_time {[k for _, k in de_candidates]} "
            f"e nenhuma bate com o numero do arquivo ({file_number})"
        )

    signal = np.asarray(mat[de_key], dtype=np.float64).squeeze()
    if signal.ndim != 1:
        raise ValueError(
            f"esperava sinal 1d, recebi shape {signal.shape} em {mat_path.name}"
        )

    # mesma logica para rpm.
    rpm = None
    rpm_candidates = []
    for k in mat.keys():
        m = _RPM_PATTERN.match(k)
        if m:
            rpm_candidates.append((int(m.group(1)), k))
    if rpm_candidates:
        matching_rpm = [k for num, k in rpm_candidates if num == file_number]
        rpm_key = matching_rpm[0] if matching_rpm else rpm_candidates[0][1]
        rpm_value = np.asarray(mat[rpm_key]).squeeze()
        rpm = float(rpm_value.item() if rpm_value.ndim == 0 else rpm_value[0])

    return signal, rpm

# 2. classe / carga / diametro a partir do caminho
def parse_path(mat_path: Path, data_root: Path) -> FileMetadata:
    """
    extrai metadados a partir do caminho relativo.

    padroes aceitos:
        normal_baseline_data/<nn>.mat
        12k_drive_end_bearing_fault_data/<diam>/<subfolder>/<nn>.mat

    onde:
        <nn>        = numero do arquivo cwru (deve estar em CWRU_FILE_TO_LOAD_HP)
        <diam>      = "0.007", "0.014" ou "0.021"
        <subfolder> = "inner_race", "ball" ou "centered"

    levanta valueerror em qualquer desvio.
    """
    mat_path = mat_path.resolve()
    data_root = data_root.resolve()
    try:
        rel = mat_path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"{mat_path} nao esta sob {data_root}") from exc

    parts = rel.parts

    # numero do arquivo cwru
    try:
        file_number = int(mat_path.stem)
    except ValueError as exc:
        raise ValueError(
            f"stem nao e numero inteiro: {mat_path.name} (esperado <nn>.mat)"
        ) from exc

    if file_number not in CWRU_FILE_TO_LOAD_HP:
        raise ValueError(
            f"numero {file_number} nao esta no mapeamento canonico cwru. "
        )
    load_hp = CWRU_FILE_TO_LOAD_HP[file_number]

    # caso 1: normal baseline.
    # estrutura: normal_baseline_data/<nn>.mat   -> 2 partes
    if parts[0] == "normal_baseline_data":
        if len(parts) != 2:
            raise ValueError(f"estrutura inesperada em normal_baseline_data: {rel}")
        return FileMetadata(
            path=mat_path,
            class_id=CLASS_NORMAL,
            fault_diameter_inches=None,
            load_hp=load_hp,
            cwru_file_number=file_number,
        )

    # caso 2: 12k drive end fault.
    # estrutura: 12k_drive_end_bearing_fault_data/<diam>/<subfolder>/<nn>.mat
    # -> 4 partes
    if parts[0] == "12k_drive_end_bearing_fault_data":
        if len(parts) != 4:
            raise ValueError(
                f"estrutura inesperada em 12k_drive_end: {rel} "
                f"(esperado <diam>/<subfolder>/<nn>.mat)"
            )
        diam_str, subfolder, _ = parts[1], parts[2], parts[3]
        try:
            diameter = float(diam_str)
        except ValueError as exc:
            raise ValueError(f"diametro nao reconhecido: {diam_str}") from exc

        if subfolder not in SUBFOLDER_TO_CLASS:
            raise ValueError(
                f"subpasta nao reconhecida: {subfolder} "
                f"(esperado {list(SUBFOLDER_TO_CLASS.keys())})"
            )
        class_id = SUBFOLDER_TO_CLASS[subfolder]

        return FileMetadata(
            path=mat_path,
            class_id=class_id,
            fault_diameter_inches=diameter,
            load_hp=load_hp,
            cwru_file_number=file_number,
        )
    raise ValueError(
        f"pasta raiz nao reconhecida: {parts[0]}. "
        "esperado 'normal_baseline_data' ou '12k_drive_end_bearing_fault_data'."
    )

# 3. janelamento
def windowize(
    signal: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> np.ndarray:
    """
    quebra o sinal 1d em janelas com stride configuravel.

    1024 a 12 khz = 85.3 ms ≈ 2.5 rotacoes do eixo a 30 hz.
    cobre varios eventos de impacto (bpfi, bpfo, bsf sao multiplas
    da rotacao) sem inflar custo de memoria na fpga.

    stride 512 = overlap 50%, convencao comum no cwru.
    janelas incompletas no fim do sinal sao descartadas.
    """
    if signal.ndim != 1:
        raise ValueError(f"esperava sinal 1d, recebi {signal.ndim}d")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size e stride devem ser positivos")
    if len(signal) < window_size:
        return np.empty((0, window_size), dtype=signal.dtype)

    n_windows = (len(signal) - window_size) // stride + 1
    windows = np.stack([
        signal[i * stride : i * stride + window_size]
        for i in range(n_windows)
    ])
    return windows

# 4. z-score por janela
def zscore_per_window(windows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    normaliza cada janela: subtrai media, divide por desvio.

    por janela (e nao global):
    - remove offset dc e diferencas de ganho entre arquivos/cargas.
    - pode ser replicada bit-a-bit no programa c que vai alimentar a PL
    """
    if windows.ndim != 2:
        raise ValueError(f"esperava (n, w), recebi shape {windows.shape}")
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    return (windows - mean) / (std + eps)

# 5. pipeline completo
@dataclass
class IngestionResult:
    X: np.ndarray             # (n, 1, window_size) float32
    y: np.ndarray             # (n,) int64
    rpm: np.ndarray           # (n,) float64; usa rpm nominal se ausente no .mat
    load_hp: np.ndarray       # (n,) int64
    fault_diameter: np.ndarray  # (n,) float64; nan para normal
    source_file: list[str]    # nome do arquivo por janela
    metadata: list[FileMetadata]  # 1 entrada por arquivo


def ingest_directory(
    data_root: Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    skip_unrecognized: bool = False,
) -> IngestionResult:
    """
    processa recursivamente todos os .mat.

    skip_unrecognized=False (default): falha em estruturas
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise NotADirectoryError(f"{data_root} nao e diretorio")

    files = sorted(data_root.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"nenhum *.mat encontrado sob {data_root}")

    all_windows = []
    all_labels = []
    all_rpm = []
    all_load = []
    all_diam = []
    all_source = []
    all_meta = []

    for mat_path in files:
        try:
            meta = parse_path(mat_path, data_root)
        except ValueError as exc:
            if skip_unrecognized:
                print(f"[skip] {mat_path.name}: {exc}")
                continue
            raise

        signal, rpm = load_de_signal(mat_path, meta.cwru_file_number)
        windows = windowize(signal, window_size=window_size, stride=stride)
        if len(windows) == 0:
            print(f"[warn] {mat_path.name}: sinal mais curto que window_size, ignorado")
            continue
        windows = zscore_per_window(windows)

        # se o .mat nao trouxe rpm, usa o nominal da carga.
        # alguns arquivos do cwru (ex: 98.mat, 99.mat) nao gravaram esse campo.
        if rpm is None:
            rpm = NOMINAL_RPM_BY_LOAD_HP[meta.load_hp]

        all_windows.append(windows)
        n = len(windows)
        all_labels.append(np.full(n, meta.class_id, dtype=np.int64))
        all_rpm.append(np.full(n, rpm, dtype=np.float64))
        all_load.append(np.full(n, meta.load_hp, dtype=np.int64))
        all_diam.append(np.full(
            n,
            meta.fault_diameter_inches if meta.fault_diameter_inches is not None else np.nan,
            dtype=np.float64,
        ))
        all_source.extend([mat_path.name] * n)
        all_meta.append(meta)

    X = np.concatenate(all_windows, axis=0)
    X = X[:, np.newaxis, :].astype(np.float32)
    y = np.concatenate(all_labels, axis=0)
    rpm = np.concatenate(all_rpm, axis=0)
    load = np.concatenate(all_load, axis=0)
    diam = np.concatenate(all_diam, axis=0)

    return IngestionResult(
        X=X,
        y=y,
        rpm=rpm,
        load_hp=load,
        fault_diameter=diam,
        source_file=all_source,
        metadata=all_meta,
    )