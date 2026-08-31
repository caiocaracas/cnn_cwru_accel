"""treina a rede e grava o modelo com a config que o gerou."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modelo.rede import ModelConfig, build_model, model_info, INPUT_LENGTH
from modelo.cwru import (IngestionResult, DEFAULT_STRIDE,
                         escala_int8, zscore_per_window)

VAL_FRACTION = 0.25

DEFAULT_PROTOCOL = "loso_carga"
DEFAULT_HELD_OUT_LOAD = 3

BALANCEAMENTO = ("nenhum", "perda", "amostrador")

@dataclass
class TrainConfig:
    seed: int = 42

    normalizacao: str = "janela"

    protocol: str = DEFAULT_PROTOCOL
    held_out_load: int = DEFAULT_HELD_OUT_LOAD
    val_load: int = -1

    batchnorm: bool = True
    balanceamento: str = "perda"

    weight_decay: float = 1e-4
    dropout: float = 0.0
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    sam_rho: float = 0.0
    ema_decay: float = 0.0
    grad_clip: float = 1.0
    aug_shift: float = 0.0
    aug_noise_snr_db: float = float("inf")
    aug_noise_prob: float = 0.0

    entrada_bits: int = 8

    qat_epocas: int = 0
    qat_lr: float = 1e-4
    qat_bits: int = 8

    batch_size: int = 128
    max_epochs: int = 120
    base_lr: float = 3e-3
    warmup_epochs: int = 5
    min_lr_factor: float = 0.01
    early_stop_patience: int = 25

    def __post_init__(self):
        if self.balanceamento not in BALANCEAMENTO:
            raise ValueError(f"balanceamento={self.balanceamento} fora de {BALANCEAMENTO}")

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(d) - known
        if unknown:
            print(f"[aviso] ignorando chaves desconhecidas em training: {unknown}")
        return cls(**{k: v for k, v in d.items() if k in known})

def _split_por_grupo(idx, y, groups, val_frac, seed):
    g = np.asarray(groups)[idx]
    yy = np.asarray(y)[idx]
    por_classe = {int(c): len(set(g[yy == c])) for c in np.unique(yy)}
    k_max = min(por_classe.values())
    if k_max < 2:
        raise ValueError(f"classe com menos de 2 grupos, particao impossivel: "
                         f"{por_classe}")
    k = int(np.clip(round(1.0 / val_frac), 2, k_max))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    a, b = next(sgkf.split(np.zeros(len(idx)), yy, g))
    print(f"      grupos por classe no pool: {por_classe}")
    print(f"      {k} dobras (validacao = 1/{k} dos grupos)")
    return idx[a], idx[b]

def make_splits(X, y, seed: int, groups=None, loads=None,
                protocol: str = DEFAULT_PROTOCOL,
                held_out_load: int = DEFAULT_HELD_OUT_LOAD,
                val_load: int | None = None):
    y = np.asarray(y)
    n = len(y)
    all_idx = np.arange(n)

    if protocol == "random":
        tv, te = train_test_split(all_idx, test_size=0.15,
                                  stratify=y, random_state=seed)
        tr, va = train_test_split(tv, test_size=VAL_FRACTION,
                                  stratify=y[tv], random_state=seed)
        return tr, va, te

    if protocol == "grupo":
        if groups is None:
            raise ValueError("grupo exige groups")
        tv, te = _split_por_grupo(all_idx, y, groups, 0.15, seed)
        tr, va = _split_por_grupo(tv, y, groups, VAL_FRACTION, seed)
        return tr, va, te

    if protocol == "loso_carga":
        if loads is None or groups is None:
            raise ValueError("loso_carga exige loads e groups")
        loads_arr = np.asarray(loads)
        te = all_idx[loads_arr == held_out_load]
        rem = all_idx[loads_arr != held_out_load]
        if len(te) == 0:
            raise ValueError(f"nenhuma janela com carga={held_out_load}")
        tr, va = _split_por_grupo(rem, y, groups, VAL_FRACTION, seed)
        return tr, va, te

    if protocol == "carga_dupla":
        if loads is None:
            raise ValueError("carga_dupla exige loads")
        if val_load is None:
            raise ValueError("carga_dupla exige val_load")
        if val_load == held_out_load:
            raise ValueError(f"val_load={val_load} igual a held_out_load")
        loads_arr = np.asarray(loads)
        te = all_idx[loads_arr == held_out_load]
        va = all_idx[loads_arr == val_load]
        tr = all_idx[(loads_arr != held_out_load) & (loads_arr != val_load)]
        for nome, sel, carga in (("teste", te, held_out_load),
                                 ("validacao", va, val_load)):
            if len(sel) == 0:
                raise ValueError(f"nenhuma janela com carga={carga} ({nome})")
        print(f"      treino nas cargas "
              f"{sorted(set(loads_arr[tr].tolist()))}, "
              f"validacao na carga {val_load}, teste na carga {held_out_load}")
        return tr, va, te

    raise ValueError(f"protocol desconhecido: {protocol}")

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class JanelaDataset(Dataset):

    def __init__(self, X_bruto: np.ndarray, y: np.ndarray, cfg: TrainConfig,
                 treino: bool, escala_global=None, escala_int8=None):
        self.escala_global = escala_global
        self.escala_int8 = escala_int8
        self.X = np.ascontiguousarray(X_bruto, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        self.cfg = cfg
        self.treino = treino
        self.rng = np.random.default_rng(cfg.seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.treino:
            c = self.cfg
            if c.aug_shift > 0 and self.rng.random() < c.aug_shift:
                x = np.roll(x, int(self.rng.integers(0, len(x))))
            if math.isfinite(c.aug_noise_snr_db) and self.rng.random() < c.aug_noise_prob:
                p_sig = float(np.mean(x.astype(np.float64) ** 2))
                p_ruido = p_sig / (10.0 ** (c.aug_noise_snr_db / 10.0))
                x = x + self.rng.normal(0.0, math.sqrt(p_ruido), size=x.shape).astype(np.float32)
        if self.cfg.normalizacao == "global":
            m, s = self.escala_global
        elif self.cfg.normalizacao == "condicao":
            m, s = self.escala_global[0][i], self.escala_global[1][i]
        else:
            m = x.mean()
            s = x.std()
        x = (x - m) / (s + 1e-8)
        if self.escala_int8:
            q = 2 ** (self.cfg.entrada_bits - 1) - 1
            x = (np.clip(np.rint(x / self.escala_int8), -q - 1, q)
                 * self.escala_int8).astype(np.float32)
        return torch.from_numpy(x[np.newaxis, :].copy()), int(self.y[i])

class LoteGPU:

    def __init__(self, X_bruto: np.ndarray, y: np.ndarray, cfg: TrainConfig,
                 treino: bool, device, escala_global=None, escala_int8=None):
        self.escala_global = escala_global
        self.escala_int8 = escala_int8
        self.X = torch.as_tensor(np.ascontiguousarray(X_bruto, dtype=np.float32),
                                 device=device)
        self.y = torch.as_tensor(np.ascontiguousarray(y, dtype=np.int64), device=device)
        self.cfg = cfg
        self.treino = treino
        self.device = device
        self.L = self.X.shape[1]
        self._ar = torch.arange(self.L, device=device)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(cfg.seed)
        self.pesos_amostra = None

    def __len__(self) -> int:
        return int(self.y.numel())

    def _prepara(self, xb: torch.Tensor, sel=None) -> torch.Tensor:
        c = self.cfg
        if self.treino:
            n = xb.shape[0]
            if c.aug_shift > 0:
                usa = torch.rand(n, device=self.device, generator=self.g) < c.aug_shift
                desl = torch.randint(0, self.L, (n,), device=self.device,
                                     generator=self.g)
                desl = torch.where(usa, desl, torch.zeros_like(desl))
                idx = (self._ar.unsqueeze(0) - desl.unsqueeze(1)) % self.L
                xb = torch.gather(xb, 1, idx)
            if math.isfinite(c.aug_noise_snr_db) and c.aug_noise_prob > 0:
                usa = (torch.rand(n, 1, device=self.device, generator=self.g)
                       < c.aug_noise_prob).to(xb.dtype)
                p_sig = xb.pow(2).mean(dim=1, keepdim=True)
                sigma = (p_sig / (10.0 ** (c.aug_noise_snr_db / 10.0))).sqrt()
                ruido = torch.randn(xb.shape, device=self.device, generator=self.g)
                xb = xb + usa * sigma * ruido
        if self.cfg.normalizacao == "global":
            m, s = self.escala_global
        elif self.cfg.normalizacao == "condicao":
            m, s = self.escala_global
            if sel is not None:
                m, s = m[sel], s[sel]
        else:
            m = xb.mean(dim=1, keepdim=True)
            s = xb.std(dim=1, keepdim=True, unbiased=False)
        xb = (xb - m) / (s + 1e-8)
        if self.escala_int8:
            q = 2 ** (self.cfg.entrada_bits - 1) - 1
            xb = torch.clamp(torch.round(xb / self.escala_int8),
                             -q - 1, q) * self.escala_int8
        return xb.unsqueeze(1)

    def define_amostrador(self, pesos: np.ndarray) -> None:
        self.pesos_amostra = torch.as_tensor(pesos, dtype=torch.double,
                                             device=self.device)

    def lotes(self, batch_size: int, embaralha: bool, descarta_resto: bool):
        n = len(self)
        if self.pesos_amostra is not None:
            ordem = torch.multinomial(self.pesos_amostra, n, replacement=True,
                                      generator=self.g)
        elif embaralha:
            ordem = torch.randperm(n, device=self.device, generator=self.g)
        else:
            ordem = self._ar.new_tensor(range(n)) if False else torch.arange(
                n, device=self.device)
        limite = (n // batch_size) * batch_size if descarta_resto else n
        for i in range(0, limite, batch_size):
            sel = ordem[i:i + batch_size]
            yield self._prepara(self.X[sel], sel), self.y[sel]

class SAM(torch.optim.Optimizer):

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        if rho < 0:
            raise ValueError(f"rho invalido: {rho}")
        super().__init__(params, dict(rho=rho, **kwargs))
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def _grad_norm(self):
        dev = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([
            p.grad.norm(p=2).to(dev)
            for g in self.param_groups for p in g["params"] if p.grad is not None
        ]), p=2)

    @torch.no_grad()
    def first_step(self):
        gn = self._grad_norm()
        for g in self.param_groups:
            scale = g["rho"] / (gn + 1e-12)
            for p in g["params"]:
                if p.grad is None:
                    continue
                e = p.grad * scale.to(p)
                p.add_(e)
                self.state[p]["e_w"] = e
        self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self):
        for g in self.param_groups:
            for p in g["params"]:
                e = self.state[p].get("e_w")
                if e is not None:
                    p.sub_(e)
                    self.state[p]["e_w"] = None
        self.base_optimizer.step()
        self.zero_grad(set_to_none=True)

    def step(self, closure=None):
        raise RuntimeError("SAM usa first_step/second_step")

class EMA:

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.n = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.shadow[k] = v.detach().clone().float()

    def state_dict(self, ref: nn.Module) -> dict:
        return {k: self.shadow[k].to(v.dtype) for k, v in ref.state_dict().items()}

def _lr_at(epoch: int, cfg: "TrainConfig") -> float:
    if epoch < cfg.warmup_epochs:
        return cfg.base_lr * (epoch + 1) / cfg.warmup_epochs
    t = (epoch - cfg.warmup_epochs) / max(1, cfg.max_epochs - cfg.warmup_epochs)
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return cfg.base_lr * (cfg.min_lr_factor + (1.0 - cfg.min_lr_factor) * cos)

def _mixup(x, alpha):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], perm, lam

@torch.no_grad()
def _evaluate(model, lote: "LoteGPU", device):
    model.eval()
    loss_sum, n = 0.0, 0
    preds, targets = [], []
    for xb, yb in lote.lotes(4096, embaralha=False, descarta_resto=False):
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        n += xb.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(yb.cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    return {
        "loss": loss_sum / n,
        "acc": float((p == t).mean()),
        "f1": float(f1_score(t, p, average="macro")),
        "preds": p,
        "targets": t,
    }

def _plot_curves(history, out_path: Path) -> None:
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(ep, [h["train_loss"] for h in history], label="treino")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validacao")
    ax[0].set_title("entropia cruzada (mesma medida nos dois)")
    ax[0].set_ylabel("perda")

    ax[1].plot(ep, [h["train_acc"] for h in history], label="acc treino")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="acc validacao")
    ax[1].plot(ep, [h["val_f1"] for h in history], label="f1 validacao")
    ax[1].set_title("acuracia / f1")

    ax[2].plot(ep, [h["objetivo"] for h in history], color="tab:red",
               label="objetivo minimizado")
    ax[2].set_ylabel("objetivo")
    ax2 = ax[2].twinx()
    ax2.plot(ep, [h["lr"] for h in history], color="tab:gray", ls="--", label="lr")
    ax2.set_ylabel("lr")
    ax[2].set_title("objetivo de treino e taxa de aprendizado")

    for a in ax:
        a.set_xlabel("epoca")
        a.grid(True, alpha=0.3)
    ax[0].legend(); ax[1].legend(); ax[2].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def _plot_confusion(cm, class_names, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("previsto"); ax.set_ylabel("real")
    ax.set_title("matriz de confusao (test)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def train_and_eval(model_cfg: ModelConfig, train_cfg: TrainConfig,
                   data: IngestionResult, run_dir: Path,
                   class_names: Optional[list[str]] = None) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_i, va_i, te_i = make_splits(data.X_bruto, data.y, train_cfg.seed,
                                   groups=data.grupo, loads=data.load_hp,
                                   protocol=train_cfg.protocol,
                                   held_out_load=train_cfg.held_out_load,
                                   val_load=(None if train_cfg.val_load < 0
                                             else train_cfg.val_load))

    n_classes = int(data.y.max()) + 1

    esc = None
    esc_por_split = {}
    if train_cfg.normalizacao == "global":
        xt = data.X_bruto[tr_i]
        esc = (torch.tensor(float(xt.mean()), device=device),
               torch.tensor(float(xt.std()), device=device))
        print(f"      escala global do treino: media {float(esc[0]):.4f}, "
              f"desvio {float(esc[1]):.4f}")
    elif train_cfg.normalizacao == "condicao":
        from modelo.cwru import escala_por_condicao
        mu_a, sd_a = escala_por_condicao(data.X_bruto, data.source_file)
        def _esc(idx):
            return (torch.tensor(mu_a[idx], dtype=torch.float32,
                                 device=device).unsqueeze(1),
                    torch.tensor(sd_a[idx], dtype=torch.float32,
                                 device=device).unsqueeze(1))
        esc_por_split = {k: _esc(v) for k, v in
                         (("tr", tr_i), ("va", va_i), ("te", te_i))}
        print(f"      escala por condicao: {len(set(data.source_file))} "
              f"gravacoes, desvio de {sd_a.min():.4f} a {sd_a.max():.4f}")

    q_ent = None
    if train_cfg.entrada_bits:
        if train_cfg.normalizacao == "condicao":
            xt = data.X_bruto[tr_i]
            zt = (xt - mu_a[tr_i][:, None]) / (sd_a[tr_i][:, None] + 1e-8)
        elif train_cfg.normalizacao == "global":
            zt = (data.X_bruto[tr_i] - float(esc[0])) / (float(esc[1]) + 1e-8)
        else:
            zt = zscore_per_window(data.X_bruto[tr_i])
        q_ent = escala_int8(zt, train_cfg.entrada_bits)
        print(f"      entrada em int{train_cfg.entrada_bits}: escala "
              f"{q_ent:.6e} (do conjunto de treino, fixa)")
        (run_dir / "entrada.json").write_text(json.dumps(
            {"bits": train_cfg.entrada_bits, "escala": q_ent,
             "janela": int(data.X_bruto.shape[1]),
             "normalizacao": train_cfg.normalizacao}, indent=2))

    e_tr = esc_por_split.get("tr", esc)
    e_va = esc_por_split.get("va", esc)
    e_te = esc_por_split.get("te", esc)
    lote_tr = LoteGPU(data.X_bruto[tr_i], data.y[tr_i], train_cfg, True, device, e_tr, q_ent)
    lote_tr_lim = LoteGPU(data.X_bruto[tr_i], data.y[tr_i], train_cfg, False, device, e_tr, q_ent)
    lote_va = LoteGPU(data.X_bruto[va_i], data.y[va_i], train_cfg, False, device, e_va, q_ent)
    lote_te = LoteGPU(data.X_bruto[te_i], data.y[te_i], train_cfg, False, device, e_te, q_ent)

    cnt = np.bincount(data.y[tr_i], minlength=n_classes).astype(np.float64)

    if train_cfg.balanceamento == "amostrador":
        lote_tr.define_amostrador((1.0 / np.maximum(cnt, 1))[data.y[tr_i]])

    if train_cfg.balanceamento == "perda":
        peso_classe = torch.as_tensor(cnt.sum() / (n_classes * np.maximum(cnt, 1)),
                                      dtype=torch.float32, device=device)
    else:
        peso_classe = None

    model = build_model(model_cfg, dropout=train_cfg.dropout,
                        batchnorm=train_cfg.batchnorm).to(device)

    usa_sam = train_cfg.sam_rho > 0
    if usa_sam:
        opt = SAM(model.parameters(), torch.optim.AdamW, rho=train_cfg.sam_rho,
                  lr=train_cfg.base_lr, weight_decay=train_cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.base_lr,
                                weight_decay=train_cfg.weight_decay)

    ema = EMA(model, train_cfg.ema_decay) if train_cfg.ema_decay > 0 else None
    crit = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing,
                               weight=peso_classe)

    history = []
    best_f1, best_state, best_epoch, best_fonte = -1.0, None, 0, "raw"
    since_best = 0
    t0 = time.time()

    for epoch in range(train_cfg.max_epochs):
        lr = _lr_at(epoch, train_cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        obj_sum, total = 0.0, 0
        for xb, yb in lote_tr.lotes(train_cfg.batch_size, embaralha=True,
                                    descarta_resto=True):
            if train_cfg.mixup_alpha > 0:
                xin, perm, lam = _mixup(xb, train_cfg.mixup_alpha)
                fn = lambda o: lam * crit(o, yb) + (1.0 - lam) * crit(o, yb[perm])
            else:
                xin, fn = xb, lambda o: crit(o, yb)

            loss = fn(model(xin))
            loss.backward()
            if train_cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

            if usa_sam:
                opt.first_step()
                fn(model(xin)).backward()
                if train_cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                opt.second_step()
            else:
                opt.step()
                opt.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

            obj_sum += loss.item() * xb.size(0)
            total += xb.size(0)

        tr_raw = _evaluate(model, lote_tr_lim, device)
        va = _evaluate(model, lote_va, device)
        cand = [("raw", tr_raw, va, {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()})]

        if ema is not None:
            shadow = ema.state_dict(model)
            backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(shadow)
            tr_ema = _evaluate(model, lote_tr_lim, device)
            va_ema = _evaluate(model, lote_va, device)
            cand.append(("ema", tr_ema, va_ema,
                         {k: v.cpu().clone() for k, v in shadow.items()}))
            model.load_state_dict(backup)
        else:
            va_ema = None

        fonte, tr_m, va_sel, state_ep = max(cand, key=lambda c: c[2]["f1"])

        history.append({
            "epoch": epoch + 1, "lr": lr,
            "objetivo": obj_sum / total,
            "train_loss": tr_m["loss"], "train_acc": tr_m["acc"], "train_f1": tr_m["f1"],
            "val_loss": va_sel["loss"], "val_acc": va_sel["acc"], "val_f1": va_sel["f1"],
            "val_f1_raw": va["f1"],
            "val_f1_ema": (va_ema["f1"] if va_ema else None),
            "fonte": fonte,
        })

        if va_sel["f1"] > best_f1:
            best_f1, best_state = va_sel["f1"], state_ep
            best_epoch, best_fonte = epoch + 1, fonte
            since_best = 0
        else:
            since_best += 1

        print(f"epoca {epoch+1:3d}  lr={lr:.2e}  obj={obj_sum/total:.4f}  "
              f"treino: loss={tr_m['loss']:.4f} acc={tr_m['acc']:.4f}  "
              f"val: loss={va_sel['loss']:.4f} acc={va_sel['acc']:.4f} "
              f"f1={va_sel['f1']:.4f}"
              + (f" [{fonte}]" if ema else ""))

        if since_best >= train_cfg.early_stop_patience:
            print(f"early stop na epoca {epoch+1}")
            break

    train_time_s = time.time() - t0

    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best_checkpoint.pt")

    te = _evaluate(model, lote_te, device)
    tr_final = _evaluate(model, lote_tr_lim, device)
    cm = confusion_matrix(te["targets"], te["preds"])

    _plot_curves(history, run_dir / "curves.png")
    if class_names is None:
        class_names = [f"c{i}" for i in range(n_classes)]
    _plot_confusion(cm, class_names, run_dir / "confusion_matrix.png")

    metrics = {
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
        "receita": {
            "optimizer": "SAM(AdamW)" if usa_sam else "AdamW",
            "lr_schedule": f"warmup {train_cfg.warmup_epochs} + cosine",
            "base_lr": train_cfg.base_lr, "batch_size": train_cfg.batch_size,
            "max_epochs": train_cfg.max_epochs,
            "selecao": "macro-F1 de validacao",
            "peso_escolhido": best_fonte, "melhor_epoca": best_epoch,
        },
        "entrada": {"bits": train_cfg.entrada_bits, "escala": q_ent},
        "best_val_f1": best_f1,
        "train_acc": tr_final["acc"],
        "train_f1_macro": tr_final["f1"],
        "test_acc": te["acc"],
        "test_f1_macro": te["f1"],
        "lacuna_treino_teste": tr_final["acc"] - te["acc"],
        "confusion_matrix": cm.tolist(),
        "per_class_f1": f1_score(te["targets"], te["preds"], average=None).tolist(),
        "epochs_run": len(history),
        "train_time_s": train_time_s,
        "model_info": model_info(model_cfg),
        "n_train": len(tr_i), "n_val": len(va_i), "n_test": len(te_i),
        "grupos_treino": int(len(set(data.grupo[tr_i]))),
        "grupos_val": int(len(set(data.grupo[va_i]))),
        "grupos_teste": int(len(set(data.grupo[te_i]))),
        "device": str(device),
    }
    with (run_dir / "metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)
    with (run_dir / "history.json").open("w") as fh:
        json.dump(history, fh, indent=2)

    return metrics

def nome_run(model_cfg: ModelConfig) -> str:
    jan = "" if model_cfg.input_len == INPUT_LENGTH else f"_W{model_cfg.input_len}"
    return (f"L{model_cfg.num_layers}_F{model_cfg.num_filters_first:02d}"
            f"_K{model_cfg.kernel_size}_P{model_cfg.pool_type}"
            f"_H{model_cfg.head}{jan}")

def main() -> None:
    import argparse
    import yaml
    from modelo.cwru import ingest_directory, CLASS_NAMES

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="spec/baseline.yaml")
    ap.add_argument("--data-dir", default="data/full")
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--filters-first", type=int, default=None)
    ap.add_argument("--kernel-size", type=int, default=None)
    ap.add_argument("--pool-type", default=None)
    ap.add_argument("--head", default=None)
    ap.add_argument("--janela", type=int, default=None,
                    help="amostras por janela; fixa o piso da latencia de decisao")
    ap.add_argument("--passo", type=int, default=None,
                    help="amostras entre janelas; e ele, nao a janela, que fixa "
                         "o prazo. Mantenha fixo ao varrer a janela")
    ap.add_argument("--treino", nargs="*", default=None,
                    help="sobrepoe training, ex: mixup_alpha=0.2 sam_rho=0.05")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    m = dict(cfg.get("model", {}))
    m.setdefault("kernel_size", 5)
    m.setdefault("pool_type", "max")
    for chave, valor in (("num_layers", args.layers),
                         ("num_filters_first", args.filters_first),
                         ("kernel_size", args.kernel_size),
                         ("pool_type", args.pool_type),
                         ("head", args.head),
                         ("input_len", args.janela)):
        if valor is not None:
            m[chave] = valor

    t = dict(cfg.get("training", {}))
    for par in (args.treino or []):
        k, v = par.split("=", 1)
        t[k] = yaml.safe_load(v)

    model_cfg = ModelConfig.from_dict(m)
    train_cfg = TrainConfig.from_dict(t)
    nome = nome_run(model_cfg)
    run_dir = Path(args.out) if args.out else Path("runs") / nome

    passo = args.passo if args.passo is not None else DEFAULT_STRIDE

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(
        {"model": asdict(model_cfg), "training": asdict(train_cfg),
         "data": {"data_dir": args.data_dir,
                  "window_size": model_cfg.input_len,
                  "stride": passo}}, sort_keys=False))

    print(f"ingerindo {args.data_dir} "
          f"(janela {model_cfg.input_len}, passo {passo})")
    data = ingest_directory(Path(args.data_dir),
                            window_size=model_cfg.input_len, stride=passo)
    print(f"{len(data.y)} janelas, {len(data.metadata)} arquivos-canal, "
          f"{len(set(data.grupo))} condicoes")

    r = train_and_eval(model_cfg, train_cfg, data, run_dir,
                       class_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)])
    print(f"\n{nome}  treino_acc={r['train_acc']:.4f}  test_acc={r['test_acc']:.4f}  "
          f"test_f1={r['test_f1_macro']:.4f}  "
          f"(lacuna treino-teste {r['lacuna_treino_teste']:+.4f})")
    print(f"por classe (f1): {[round(v, 4) for v in r['per_class_f1']]}")
    print(f"salvo em {run_dir}")

if __name__ == "__main__":
    main()
