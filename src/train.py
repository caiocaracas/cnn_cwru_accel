"""
train.py

loop de treino + avaliacao para uma config especifica.
chamado pelo run_experiment.py via train_and_eval(cfg, data, run_dir).

- split por janela, estratificado por classe, seed fixa (simples,
  determinismo entre experimentos). overlap de 50% pode inflar acuracia
  marginalmente; aceitavel pq o foco da ic e sensibilidade a hiperparametros
  e nao acuracia absoluta.
- adam lr=1e-3, reduce_on_plateau, early stopping em val_acc (patience=10)
- sem augmentation
- salva: best_checkpoint.pt, metrics.json, curves.png, confusion_matrix.png
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

from model import ModelConfig, build_model, model_info
from cwru_loader import IngestionResult

# config de treino
@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    max_epochs: int = 50
    learning_rate: float = 1e-3
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    early_stop_patience: int = 10
    lr_reduce_patience: int = 5
    lr_reduce_factor: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        # so aceita chaves conhecidas, evita typo silencioso
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"chaves desconhecidas em training: {unknown}")
        return cls(**d)


# split
def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    split estratificado em train/val/test.

    primeiro corta test, depois divide o restante em train/val.
    estratificacao garante mesma proporcao de classes em cada split.
    """
    # 1) test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_fraction,
        stratify=y,
        random_state=seed,
    )
    # 2) val (ajustado relativo ao tamanho ja reduzido)
    rel_val = val_fraction / (1.0 - test_fraction)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=rel_val,
        stratify=y_trainval,
        random_state=seed,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# determinismo
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# loops
def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle, drop_last=False)

def _run_epoch(model, loader, optimizer, device, train_mode: bool):
    model.train(train_mode)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if train_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_n += xb.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
        "preds": np.concatenate(all_preds),
        "targets": np.concatenate(all_targets),
    }

# plots
def _plot_curves(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_xlabel("epoca")
    axes[0].set_ylabel("loss")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_xlabel("epoca")
    axes[1].set_ylabel("acuracia")
    axes[1].set_title("acuracia")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_confusion(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")
    ax.set_title("matriz de confusao (test)")
    # anota valores
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# entry point
def train_and_eval(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data: IngestionResult,
    run_dir: Path,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    treina e avalia um modelo. salva tudo em run_dir.

    retorna dict com metricas finais (json-serializavel).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # split
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = make_splits(
        data.X, data.y,
        val_fraction=train_cfg.val_fraction,
        test_fraction=train_cfg.test_fraction,
        seed=train_cfg.seed,
    )

    train_loader = _to_loader(X_tr, y_tr, train_cfg.batch_size, shuffle=True)
    val_loader = _to_loader(X_val, y_val, train_cfg.batch_size, shuffle=False)
    test_loader = _to_loader(X_te, y_te, train_cfg.batch_size, shuffle=False)

    # modelo
    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # maximiza val_acc
        factor=train_cfg.lr_reduce_factor,
        patience=train_cfg.lr_reduce_patience,
    )

    # treino
    history = []
    best_val_acc = -1.0
    best_state = None
    epochs_since_best = 0
    t0 = time.time()

    for epoch in range(1, train_cfg.max_epochs + 1):
        tr = _run_epoch(model, train_loader, optimizer, device, train_mode=True)
        with torch.no_grad():
            va = _run_epoch(model, val_loader, optimizer, device, train_mode=False)

        scheduler.step(va["acc"])

        history.append({
            "epoch": epoch,
            "train_loss": tr["loss"], "train_acc": tr["acc"],
            "val_loss": va["loss"], "val_acc": va["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            # state_dict via .cpu() pra checkpoint portatil
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        print(f"epoca {epoch:3d}  train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f}  "
              f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if epochs_since_best >= train_cfg.early_stop_patience:
            print(f"early stop na epoca {epoch} (sem melhora ha {epochs_since_best} epocas)")
            break

    train_time_s = time.time() - t0

    # restaura melhor checkpoint e avalia em test
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best_checkpoint.pt")

    with torch.no_grad():
        te = _run_epoch(model, test_loader, optimizer, device, train_mode=False)

    test_acc = te["acc"]
    test_f1 = f1_score(te["targets"], te["preds"], average="macro")
    cm = confusion_matrix(te["targets"], te["preds"])

    # plots
    _plot_curves(history, run_dir / "curves.png")
    if class_names is None:
        class_names = [f"c{i}" for i in range(int(te["targets"].max()) + 1)]
    _plot_confusion(cm, class_names, run_dir / "confusion_matrix.png")

    # info do modelo (params, macs)
    info = model_info(model_cfg)

    metrics = {
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1_macro": test_f1,
        "confusion_matrix": cm.tolist(),
        "epochs_run": len(history),
        "train_time_s": train_time_s,
        "model_info": info,
        "n_train": len(y_tr),
        "n_val": len(y_val),
        "n_test": len(y_te),
        "device": str(device),
    }

    with (run_dir / "metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)

    # historico completo (uma linha por epoca) tambem util pra debug
    with (run_dir / "history.json").open("w") as fh:
        json.dump(history, fh, indent=2)

    return metrics