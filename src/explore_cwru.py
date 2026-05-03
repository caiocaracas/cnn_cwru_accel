"""
explore_cwru.py

exploracao do dataset apos ingestao. imprime estatisticas e gera
figuras para sanidade.

uso:
    python explore_cwru.py --data-dir ../data/raw --out-dir ../results/exploration

saida:
    exploration/
        windows_per_class.txt        (texto)
        sample_window_per_class.png  (4 subplots, dominio do tempo)
        rpm_histogram_per_class.png  (4 subplots, histograma de rpm)
        fft_per_class.png            (4 subplots, espectro de magnitude)
"""

from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cwru_loader import (
    CLASS_NAMES,
    SAMPLING_RATE_HZ,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STRIDE,
    ingest_directory,
)

def print_summary(result, out_path: Path) -> None:
    """imprime e grava contagens por classe e por carga."""
    counts = Counter(result.y.tolist())
    n_total = len(result.y)

    lines = []
    lines.append(f"total de janelas: {n_total}")
    lines.append(f"total de arquivos: {len(result.metadata)}")
    lines.append(f"window size: {result.X.shape[-1]}, sampling: {SAMPLING_RATE_HZ} Hz")
    lines.append(f"duracao por janela: {result.X.shape[-1]/SAMPLING_RATE_HZ*1000:.2f} ms")
    lines.append("")
    lines.append("janelas por classe:")
    for cid in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES[cid]
        n = counts.get(cid, 0)
        pct = 100 * n / n_total if n_total else 0
        lines.append(f"  {cid} ({name:12s}): {n:6d}  ({pct:5.2f}%)")

    lines.append("")
    lines.append("janelas por classe x carga (hp):")
    lines.append("  classe        | 0 hp | 1 hp | 2 hp | 3 hp")
    for cid in sorted(CLASS_NAMES.keys()):
        row = [CLASS_NAMES[cid].ljust(13)]
        for load in (0, 1, 2, 3):
            mask = (result.y == cid) & (result.load_hp == load)
            row.append(f"{int(mask.sum()):4d}")
        lines.append("  " + " | ".join(row))

    text = "\n".join(lines)
    print(text)
    out_path.write_text(text + "\n", encoding="utf-8")

def plot_sample_window_per_class(result, out_path: Path) -> None:
    """uma janela por classe no dominio do tempo."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    t_ms = np.arange(result.X.shape[-1]) / SAMPLING_RATE_HZ * 1000.0

    for ax, cid in zip(axes, sorted(CLASS_NAMES.keys())):
        idx = np.where(result.y == cid)[0]
        if len(idx) == 0:
            ax.set_title(f"classe {cid} ({CLASS_NAMES[cid]}) - sem amostras")
            continue
        # primeira janela da classe (reproducibilidade)
        w = result.X[idx[0], 0]
        src = result.source_file[idx[0]]
        ax.plot(t_ms, w, linewidth=0.7)
        ax.set_title(f"classe {cid} ({CLASS_NAMES[cid]}) - {src}")
        ax.set_ylabel("amplitude (z-score)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("tempo (ms)")
    fig.suptitle(f"janela por classe (w={result.X.shape[-1]}, fs={SAMPLING_RATE_HZ} Hz)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def plot_rpm_histogram_per_class(result, out_path: Path) -> None:
    """
    histograma de rpm por classe.

    confirma que cada classe tem amostras espalhadas pelas 4 cargas
    (0/1/2/3 hp -> ~1797/1772/1750/1730 rpm). se uma classe estiver
    concentrada em um rpm so, a cnn pode aprender rpm em vez da falha.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()

    valid_rpm = result.rpm[~np.isnan(result.rpm)]
    if len(valid_rpm) == 0:
        print("[warn] nenhum rpm disponivel - pulando histograma de rpm")
        return
    bins = np.linspace(valid_rpm.min() - 5, valid_rpm.max() + 5, 30)

    for ax, cid in zip(axes, sorted(CLASS_NAMES.keys())):
        mask = (result.y == cid) & ~np.isnan(result.rpm)
        rpms = result.rpm[mask]
        ax.hist(rpms, bins=bins, edgecolor="black", linewidth=0.5)
        ax.set_title(f"classe {cid} ({CLASS_NAMES[cid]}) - n={len(rpms)}")
        ax.set_xlabel("rpm")
        ax.set_ylabel("# janelas")
        ax.grid(True, alpha=0.3)

    fig.suptitle("distribuicao de rpm por classe")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def plot_fft_per_class(result, out_path: Path) -> None:
    """
    espectro de magnitude (fft) de uma janela por classe.

    fft do sinal raw (antes do z-score), com janela de hann para reduzir
    leakage. escala db relativa ao pico da janela, para enfatizar
    estrutura espectral em vez de amplitude absoluta.

    - 1024 amostras a 12 khz dao resolucao de 12000/1024 ≈ 11.7 hz/bin,
      suficiente para separar bpfi/bpfo/bsf que ficam tipicamente
      em 100-300 hz.
    - nyquist = 6 khz. foi plotado so ate 3 khz pra focar na regiao de
      interesse; a banda ressonante (2-4 khz) e onde os impactos
      excitam modos mecanicos do mancal.
    """
    from cwru_loader import load_de_signal

    n = result.X.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLING_RATE_HZ)
    hann = np.hanning(n)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for ax, cid in zip(axes, sorted(CLASS_NAMES.keys())):
        idx = np.where(result.y == cid)[0]
        if len(idx) == 0:
            ax.set_title(f"classe {cid} ({CLASS_NAMES[cid]}) - sem amostras")
            continue

        # recarrega o sinal raw do arquivo de origem (sem z-score).
        # z-score por janela achata diferencas de amplitude que sao
        # justamente o que torna o espectro discriminante.
        meta = next(m for m in result.metadata
                    if m.path.name == result.source_file[idx[0]])
        raw_signal, _ = load_de_signal(meta.path, meta.cwru_file_number)
        w_raw = raw_signal[:n]  # primeira janela do arquivo

        # janela de hann reduz leakage espectral
        windowed = w_raw * hann
        mag = np.abs(np.fft.rfft(windowed))

        # db relativo ao pico, evita mostrar diferencas de escala absoluta
        mag_db = 20 * np.log10(mag / (mag.max() + 1e-12) + 1e-12)

        ax.plot(freqs, mag_db, linewidth=0.7)
        ax.set_xlim(0, 3000)
        ax.set_ylim(-80, 5)
        ax.set_title(f"classe {cid} ({CLASS_NAMES[cid]}) - {result.source_file[idx[0]]}")
        ax.set_ylabel("|fft| (db rel. pico)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("frequencia (hz)")
    fig.suptitle(
        f"espectro por classe (raw + hann, resolucao = {SAMPLING_RATE_HZ/n:.2f} hz/bin)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="exploracao do dataset cwru")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="diretorio raiz com normal_baseline_data/ e 12k_drive_end_bearing_fault_data/")
    parser.add_argument("--out-dir", type=Path, default=Path("./exploration"),
                        help="diretorio de saida para figuras e relatorio")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"lendo {args.data_dir}...")
    result = ingest_directory(
        args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
    )
    print(f"ok. X.shape={result.X.shape}  y.shape={result.y.shape}")
    print()

    print_summary(result, args.out_dir / "windows_per_class.txt")
    plot_sample_window_per_class(result, args.out_dir / "sample_window_per_class.png")
    plot_rpm_histogram_per_class(result, args.out_dir / "rpm_histogram_per_class.png")
    plot_fft_per_class(result, args.out_dir / "fft_per_class.png")

    print(f"\nfiguras salvas em {args.out_dir}/")

if __name__ == "__main__":
    main()