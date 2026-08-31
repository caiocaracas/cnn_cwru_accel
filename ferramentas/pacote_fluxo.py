"""monta o pacote de FLUXO CONTINUO que a placa consome Na arquitetura por janela o pacote leva."""

from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import yaml

from modelo.cwru import ingest_directory, escala_por_condicao, para_int8
from modelo.rede import ModelConfig
from modelo.treina import TrainConfig, make_splits
from modelo.quantiza import (resolve_layer_order, pooling_da_spec,
                             decisoes_em_fluxo, escreve_pacote_fluxo)

def sinal_por_gravacao(bruto: np.ndarray, passo_do_corte: int) -> np.ndarray:
    if len(bruto) == 0:
        return np.empty(0)
    return np.concatenate([bruto[0]] + [w[-passo_do_corte:] for w in bruto[1:]])

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--plano", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/full"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-amostras", type=int, default=0,
                    help="corta o fluxo; 0 usa o conjunto de teste inteiro")
    a = ap.parse_args()

    plano = json.loads(a.plano.read_text())
    if not plano.get("fluxo"):
        raise SystemExit(f"{a.plano} nao e' plano de fluxo continuo")
    u = plano["camadas"][-1]
    npg = u["pox"] // min(u["pox"], u["pool"])
    npos = u["comp"] // u["pool"]
    prod_pool = plano["entrada"] // npos
    intervalo = npg * prod_pool
    if plano.get("dec_intervalo") and plano["dec_intervalo"] != intervalo:
        raise SystemExit(f"o plano diz decisao a cada {plano['dec_intervalo']} "
                         f"amostras e a geometria da {intervalo}")
    print(f"decisao a cada {intervalo} amostras "
          f"({prod_pool} do pooling x {npg} do dobramento); "
          f"cada decisao ve as ultimas {plano['entrada']} amostras")

    cfg = yaml.safe_load(a.config.read_text())
    mcfg = ModelConfig.from_dict(cfg["model"])
    tcfg = TrainConfig.from_dict(cfg["training"])
    npz = np.load(a.npz)
    if "entrada_escala" not in npz.files:
        raise SystemExit(f"{a.npz} nao carrega a escala da entrada")
    escala = float(npz["entrada_escala"])
    ordem = resolve_layer_order(npz)
    pool, modo = pooling_da_spec(mcfg)

    dados = ingest_directory(a.data_dir, window_size=mcfg.input_len)
    _, _, te = make_splits(dados.X_bruto, dados.y, tcfg.seed,
                           groups=dados.grupo, loads=dados.load_hp,
                           protocol=tcfg.protocol,
                           held_out_load=tcfg.held_out_load)
    passo_do_corte = mcfg.input_len // 2

    mu, sd = escala_por_condicao(dados.X_bruto, dados.source_file)
    fonte = np.asarray(dados.source_file)

    trechos, rotulos, limites, n = [], [], [], 0
    for f in dict.fromkeys(fonte[te]):
        sel = te[fonte[te] == f]
        sel = sel[np.argsort(sel)]
        sinal = sinal_por_gravacao(dados.X_bruto[sel], passo_do_corte)
        z = (sinal - mu[sel[0]]) / (sd[sel[0]] + 1e-8)
        trechos.append(para_int8(z, escala))
        rotulos.append(int(dados.y[sel[0]]))
        limites.append((n, n + len(sinal)))
        n += len(sinal)
        if a.max_amostras and n >= a.max_amostras:
            break
    fluxo = np.concatenate(trechos).astype(np.int8)
    if a.max_amostras:
        fluxo = fluxo[:a.max_amostras]
    print(f"fluxo: {len(fluxo)} amostras de {len(trechos)} gravacoes")

    man_f = a.out.parent / "manifest.json"
    if not man_f.exists():
        raise SystemExit(
            f"nao achei o manifest com as escalas de saida perto de {a.out}.\n"
            f"  Ele e' escrito pela etapa 'quantiza' e e' de onde saem as "
            f"constantes de requantizacao que a placa usa.")
    man = json.loads(man_f.read_text())
    out_scales = {L["name"]: L["requant"]["out_scale"]
                  for L in man["layers"] if L["name"].startswith("conv")}
    print(f"escalas de saida lidas de {man_f.name}: "
          + ", ".join(f"{k}={v:.5g}" for k, v in out_scales.items()))

    print("calculando as decisoes de referencia sobre o fluxo...")
    _, classes, pos = decisoes_em_fluxo(fluxo, npz, ordem, escala, out_scales,
                                        pool=pool, modo=modo, npg=npg, npos=npos)

    fim = (pos + 1) * prod_pool - 1
    ini = fim - plano["entrada"] + 1
    verdade = np.full(len(classes), -1, dtype=np.int32)
    for (a0, b0), rot in zip(limites, rotulos):
        dentro = (ini >= a0) & (fim < b0)
        verdade[dentro] = rot
    validas = int((verdade >= 0).sum())
    acerto = float((classes[verdade >= 0] == verdade[verdade >= 0]).mean())
    print(f"{len(classes)} decisoes, {validas} com janela inteira numa "
          f"gravacao; acuracia da referencia {acerto:.2%}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    escreve_pacote_fluxo(a.out, fluxo, classes, verdade, escala)
    print(f"pacote em {a.out} ({a.out.stat().st_size/1e6:.1f} MB)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
