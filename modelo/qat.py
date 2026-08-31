"""reajusta o modelo com a quantizacao simulada dentro do laco de treino."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from modelo.rede import ModelConfig, build_model
from modelo.treina import (TrainConfig, LoteGPU, make_splits, set_seed,
                           _evaluate, _lr_at)
from modelo.quantiza_pesos import fold_batchnorm, carrega_estado, evaluate

RAIZ = Path(__file__).resolve().parent.parent

class Arredonda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g

def _arredonda(x: torch.Tensor) -> torch.Tensor:
    return Arredonda.apply(x)

def quantiza_peso(w: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    escala = w.detach().abs().max() / qmax
    if escala == 0:
        return w
    return torch.clamp(_arredonda(w / escala), -qmax, qmax) * escala

class QuantAtiv(nn.Module):

    def __init__(self, bits: int = 8, momento: float = 0.99):
        super().__init__()
        self.bits = bits
        self.momento = momento
        self.register_buffer("pico", torch.zeros(1))
        self.register_buffer("pronto", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qmax = 2 ** (self.bits - 1) - 1
        if self.training:
            atual = x.detach().abs().max()
            if self.pronto.item() == 0:
                self.pico.fill_(float(atual))
                self.pronto.fill_(1)
            else:
                self.pico.mul_(self.momento).add_((1 - self.momento) * atual)
        if self.pico.item() <= 0:
            return x
        escala = self.pico / qmax
        return torch.clamp(_arredonda(x / escala), -qmax - 1, qmax) * escala

class RedeQAT(nn.Module):

    def __init__(self, base: nn.Module, bits: int = 8,
                 bits_peso: int | None = None,
                 quantiza_entrada: bool = False):
        super().__init__()
        self.cfg = base.cfg
        self.bits = bits
        self.bits_peso = bits if bits_peso is None else bits_peso
        self.features = base.features
        self.head = base.head
        self.dropout = base.dropout
        self.classifier = base.classifier

        self.q_entrada = QuantAtiv(bits) if quantiza_entrada else nn.Identity()
        n_blocos = sum(1 for m in self.features if isinstance(m, nn.Conv1d))
        self.q_ativ = nn.ModuleList([QuantAtiv(bits) for _ in range(n_blocos)])

    def _peso_q(self, mod: nn.Conv1d | nn.Linear) -> torch.Tensor:
        return quantiza_peso(mod.weight, self.bits_peso)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.q_entrada(x)
        bloco = 0
        for mod in self.features:
            if isinstance(mod, nn.Conv1d):
                x = nn.functional.conv1d(x, self._peso_q(mod), mod.bias,
                                         stride=mod.stride, padding=mod.padding)
            elif isinstance(mod, (nn.MaxPool1d, nn.AvgPool1d)):
                x = mod(x)
                x = self.q_ativ[bloco](x)
                bloco += 1
            elif isinstance(mod, nn.Identity):
                continue
            else:
                x = mod(x)
                if isinstance(mod, nn.ReLU) and not self._tem_pool():
                    x = self.q_ativ[bloco](x)
                    bloco += 1
        x = self.head(x)
        x = self.dropout(x)
        return nn.functional.linear(x, self._peso_q(self.classifier),
                                    self.classifier.bias)

    def _tem_pool(self) -> bool:
        return self.cfg.pool_type in ("max", "avg")

    def para_inferencia(self) -> nn.Module:
        m = build_model(self.cfg, batchnorm=False)
        m.load_state_dict({k: v for k, v in self.state_dict().items()
                           if not k.startswith(("q_entrada", "q_ativ"))})
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, (nn.Conv1d, nn.Linear)):
                    mod.weight.data = quantiza_peso(mod.weight.data,
                                                    self.bits_peso)
        return m.eval()

def _laco(modelo, tcfg, dados, splits, epocas, lr, disp,
          verboso: bool = True, q_ent=None):
    tr_i, va_i, te_i = splits
    def _esc(idx):
        if tcfg.normalizacao == "global":
            xt = dados.X_bruto[tr_i]
            return (torch.tensor(float(xt.mean()), device=disp),
                    torch.tensor(float(xt.std()), device=disp))
        if tcfg.normalizacao == "condicao":
            from modelo.cwru import escala_por_condicao
            mu, sd = escala_por_condicao(dados.X_bruto, dados.source_file)
            return (torch.tensor(mu[idx], dtype=torch.float32,
                                 device=disp).unsqueeze(1),
                    torch.tensor(sd[idx], dtype=torch.float32,
                                 device=disp).unsqueeze(1))
        return None

    lote_tr = LoteGPU(dados.X_bruto[tr_i], dados.y[tr_i], tcfg, True, disp, _esc(tr_i), q_ent)
    lote_va = LoteGPU(dados.X_bruto[va_i], dados.y[va_i], tcfg, False, disp, _esc(va_i), q_ent)
    lote_te = LoteGPU(dados.X_bruto[te_i], dados.y[te_i], tcfg, False, disp, _esc(te_i), q_ent)
    cnt = np.bincount(dados.y[tr_i], minlength=int(dados.y.max()) + 1)
    if tcfg.balanceamento == "amostrador":
        lote_tr.define_amostrador((1.0 / np.maximum(cnt, 1))[dados.y[tr_i]])

    modelo.train()
    with torch.no_grad():
        vistos = 0
        for xb, _ in lote_tr.lotes(tcfg.batch_size, embaralha=False,
                                   descarta_resto=False):
            modelo(xb)
            vistos += xb.size(0)
            if vistos >= 4096:
                break

    modelo.eval()
    antes = _evaluate(modelo, lote_te, disp)

    opt = torch.optim.AdamW(modelo.parameters(), lr=lr,
                            weight_decay=tcfg.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=tcfg.label_smoothing)

    melhor_f1, melhor_estado, melhor_ep = -1.0, None, 0
    hist = []
    t0 = time.time()
    for ep in range(epocas):
        modelo.train()
        for xb, yb in lote_tr.lotes(tcfg.batch_size, embaralha=True,
                                    descarta_resto=True):
            loss = crit(modelo(xb), yb)
            loss.backward()
            if tcfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(modelo.parameters(), tcfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        va = _evaluate(modelo, lote_va, disp)
        hist.append({"epoca": ep + 1, "val_f1": va["f1"], "val_acc": va["acc"]})
        if va["f1"] > melhor_f1:
            melhor_f1 = va["f1"]
            melhor_estado = {k: v.detach().cpu().clone()
                             for k, v in modelo.state_dict().items()}
            melhor_ep = ep + 1
        if verboso:
            print(f"  epoca {ep+1:2d}  val f1={va['f1']:.4f} "
                  f"acc={va['acc']:.4f}", flush=True)

    modelo.load_state_dict(melhor_estado)
    modelo.eval()
    depois = _evaluate(modelo, lote_te, disp)

    return modelo, {"antes_do_reajuste": {"acc": antes["acc"], "f1": antes["f1"]},
                    "depois_do_reajuste": {"acc": depois["acc"], "f1": depois["f1"]},
                    "melhor_epoca": melhor_ep, "epocas": epocas,
                    "segundos": time.time() - t0, "historico": hist}

def particiona(dados, tcfg: TrainConfig):
    return make_splits(
        dados.X_bruto, dados.y, tcfg.seed, groups=dados.grupo,
        loads=dados.load_hp, protocol=tcfg.protocol,
        held_out_load=tcfg.held_out_load,
        val_load=(None if tcfg.val_load < 0 else tcfg.val_load))

def reajusta_modelo(base: nn.Module, mcfg: ModelConfig, tcfg: TrainConfig,
                    dados, splits, epocas: int, lr: float, bits: int,
                    disp=None, verboso: bool = True,
                    bits_peso: int | None = None, q_ent=None):
    disp = disp or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_i, va_i, te_i = splits
    fold_batchnorm(base)
    modelo = RedeQAT(base, bits, bits_peso).to(disp)
    modelo, m = _laco(modelo, tcfg, dados, (tr_i, va_i, te_i), epocas, lr,
                      disp, verboso, q_ent)
    return modelo.para_inferencia(), m

def reajusta(run: Path, epocas: int, lr: float, bits: int, dados,
             saida: Path, bits_peso: int | None = None) -> dict:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    mcfg = ModelConfig.from_dict(cfg["model"])
    tcfg = TrainConfig.from_dict(cfg["training"])
    set_seed(tcfg.seed)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_i, va_i, te_i = particiona(dados, tcfg)

    base = build_model(mcfg, batchnorm=tcfg.batchnorm)
    carrega_estado(base, torch.load(run / "best_checkpoint.pt",
                                    map_location="cpu"),
                   run / "best_checkpoint.pt")
    fold_batchnorm(base)

    q_ent = None
    ent = run / "entrada.json"
    if ent.exists():
        q_ent = float(json.loads(ent.read_text())["escala"])
    elif tcfg.entrada_bits:
        from modelo.cwru import escala_int8, zscore_per_window
        q_ent = escala_int8(zscore_per_window(dados.X_bruto[tr_i]),
                            tcfg.entrada_bits)

    modelo = RedeQAT(base, bits, bits_peso).to(disp)
    modelo, m = _laco(modelo, tcfg, dados, (tr_i, va_i, te_i), epocas, lr, disp,
                      True, q_ent)
    m["bits"] = bits
    m["entrada_escala"] = q_ent

    saida.mkdir(parents=True, exist_ok=True)
    pronto = modelo.para_inferencia()
    torch.save(pronto.state_dict(), saida / "best_checkpoint.pt")
    cfg_qat = deepcopy(cfg)
    cfg_qat["training"]["batchnorm"] = False
    cfg_qat["training"]["qat_bits"] = bits
    cfg_qat["training"]["qat_epocas"] = epocas
    (saida / "config.yaml").write_text(yaml.safe_dump(cfg_qat, sort_keys=False))
    if q_ent is not None:
        (saida / "entrada.json").write_text(json.dumps(
            {"bits": tcfg.entrada_bits, "escala": q_ent,
             "janela": int(dados.X_bruto.shape[1]),
             "normalizacao": tcfg.normalizacao}, indent=2))

    (saida / "metrics_qat.json").write_text(json.dumps(m, indent=2))
    return m

def main() -> int:
    ap = argparse.ArgumentParser(
        description="reajusta o modelo treinado com quantizacao simulada")
    ap.add_argument("--run", required=True, help="pasta em runs/")
    ap.add_argument("--epocas", type=int, default=15,
                    help="Wu et al.: ~10%% do treino original basta")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="passo pequeno: e' reajuste, nao treino do zero")
    ap.add_argument("--bits", type=int, default=8,
                    help="bits da ativacao")
    ap.add_argument("--bits-peso", type=int, default=None,
                    help="bits do peso; menos que a ativacao permite dois "
                         "produtos por DSP48E1")
    ap.add_argument("--dados", type=Path, default=Path("data/full"))
    ap.add_argument("--saida", type=Path, default=None)
    a = ap.parse_args()

    import yaml
    from modelo.cwru import ingest_directory, DEFAULT_STRIDE

    run = RAIZ / "runs" / a.run
    if not (run / "best_checkpoint.pt").exists():
        raise SystemExit(f"sem checkpoint em {run}")
    saida = a.saida or (RAIZ / "runs" / f"{a.run}_qat")

    cfg = yaml.safe_load((run / "config.yaml").read_text())
    janela = int(cfg.get("model", {}).get("input_len", 1024))
    passo = int(cfg.get("data", {}).get("stride", DEFAULT_STRIDE))

    print(f"reajustando {a.run} com {a.bits} bits simulados, "
          f"{a.epocas} epocas, janela {janela}, passo {passo}")
    dados = ingest_directory(a.dados, window_size=janela, stride=passo)
    m = reajusta(run, a.epocas, a.lr, a.bits, dados, saida, a.bits_peso)

    print(f"\nso' arredondando, sem reajuste: acc={m['antes_do_reajuste']['acc']:.4f} "
          f"f1={m['antes_do_reajuste']['f1']:.4f}")
    print(f"depois do reajuste           : acc={m['depois_do_reajuste']['acc']:.4f} "
          f"f1={m['depois_do_reajuste']['f1']:.4f}")
    print(f"\nmodelo em {saida}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
