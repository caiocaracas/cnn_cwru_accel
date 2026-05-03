"""
run_experiment.py

executa um experimento ponta a ponta a partir de um yaml.

uso:
    python run_experiment.py --config configs/baseline.yaml --run-id baseline_v1

estrutura esperada do yaml:

    model:
      num_layers: 3
      num_filters_first: 8
      kernel_size: 5
      pool_type: max
    training:
      seed: 42
      batch_size: 64
      max_epochs: 50
      learning_rate: 0.001
      val_fraction: 0.15
      test_fraction: 0.15
      early_stop_patience: 10
    data:
      data_dir: ../data/raw

saida em runs/<run_id>/:
    config.yaml             (copia exata da config de entrada)
    best_checkpoint.pt
    metrics.json
    history.json
    curves.png
    confusion_matrix.png
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

from cwru_loader import ingest_directory, CLASS_NAMES
from model import ModelConfig
from train import TrainConfig, train_and_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="caminho do yaml com a config do experimento")
    parser.add_argument("--run-id", type=str, required=True,
                        help="identificador unico do experimento")
    parser.add_argument("--runs-dir", type=Path, default=Path("../runs"),
                        help="diretorio raiz onde criar runs/<run_id>")
    args = parser.parse_args()

    # carrega yaml
    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)

    for section in ("model", "training", "data"):
        if section not in cfg:
            print(f"erro: secao '{section}' faltando no yaml", file=sys.stderr)
            sys.exit(1)

    model_cfg = ModelConfig.from_dict(cfg["model"])
    train_cfg = TrainConfig.from_dict(cfg["training"])
    data_dir = Path(cfg["data"]["data_dir"])

    run_dir = args.runs_dir / args.run_id
    if run_dir.exists():
        print(f"aviso: {run_dir} ja existe, sera sobrescrito")
    run_dir.mkdir(parents=True, exist_ok=True)

    # copia o yaml de entrada para o run_dir (auditavel)
    shutil.copy(args.config, run_dir / "config.yaml")

    # carrega o dataset (usa defaults do loader: window=1024, stride=512)
    print(f"ingerindo {data_dir}...")
    data = ingest_directory(data_dir)
    print(f"X.shape={data.X.shape}  y.shape={data.y.shape}")

    class_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

    metrics = train_and_eval(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data=data,
        run_dir=run_dir,
        class_names=class_names,
    )

    print()
    print(f"=== resumo {args.run_id} ===")
    print(f"best_val_acc:    {metrics['best_val_acc']:.4f}")
    print(f"test_acc:        {metrics['test_acc']:.4f}")
    print(f"test_f1_macro:   {metrics['test_f1_macro']:.4f}")
    print(f"epocas:          {metrics['epochs_run']}")
    print(f"tempo treino:    {metrics['train_time_s']:.1f}s")
    print(f"params:          {metrics['model_info']['num_params']}")
    print(f"macs/inferencia: {metrics['model_info']['macs_per_inference']}")
    print(f"saida em:        {run_dir}")


if __name__ == "__main__":
    main()
