"""Treina a rede da topologia pedida e exporta o modelo em inteiro."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from modelo.rede import ModelConfig, JANELAS
from modelo.treina import TrainConfig
from ferramentas.gerador import planeja

def valida(mcfg: ModelConfig, orcamento: int, taxa: float | None,
           clock: float) -> dict:
    prazo = int(clock * 1e6 / taxa) if taxa else None
    p = planeja(mcfg.num_layers, mcfg.num_filters_first, mcfg.kernel_size,
                mcfg.pool_type, mcfg.head, orcamento=orcamento,
                entrada=mcfg.input_len, prazo=prazo)
    print(f"  topologia            : {mcfg.num_layers} camadas, "
          f"{mcfg.num_filters_first} filtros na primeira, kernel "
          f"{mcfg.kernel_size}, pooling {mcfg.pool_type}, saida {mcfg.head}")
    print(f"  janela               : {mcfg.input_len} amostras "
          f"({mcfg.input_len / 12000 * 1e3:.1f} ms a 12 kHz)")
    if taxa:
        print(f"  requisito            : {taxa:g} janelas/s a {clock:g} MHz "
              f"= {p.prazo} ciclos por janela")
    print(f"  circuito previsto    : {p.dsp_total} multiplicadores "
          f"({p.dsp_blocos} em bloco DSP, {p.mac_logica} em logica), "
          f"{p.ii} ciclos por inferencia, {p.eficiencia_dsp:.1%} de ocupacao")
    print(f"  pesos                : {p.pesos_total}")
    if taxa:
        print(f"  reserva              : {p.dsp_reserva} DSP, "
              f"{p.ciclos_ociosos} ciclos por janela, "
              f"{p.bram_reserva} blocos BRAM")
    if not p.cabe:
        raise SystemExit(
            f"\nESTA TOPOLOGIA NAO CABE NA PLACA:\n  {p.motivo}\n\n"
            f"  Nada foi treinado. Reduza filtros, camadas ou kernel, ou "
            f"aumente --orcamento se a peca for outra.")
    print("  cabe na placa        : sim")
    return asdict(p)

def sobrepoe(base: dict, pares: list[str]) -> dict:
    padroes = asdict(TrainConfig())
    tre = dict(base)
    for item in pares:
        if "=" not in item:
            raise SystemExit(f"--treino espera chave=valor, veio '{item}'")
        chave, valor = item.split("=", 1)
        if chave not in padroes:
            raise SystemExit(f"hiperparametro de treino desconhecido: {chave}\n"
                             f"  disponiveis: {', '.join(sorted(padroes))}")
        p = padroes[chave]
        v = yaml.safe_load(valor)
        tre[chave] = (bool(v) if isinstance(p, bool)
                      else int(v) if isinstance(p, int)
                      else float(v) if isinstance(p, float) else v)
    return tre

def roda(cmd: list, etapa: str) -> None:
    r = subprocess.run(cmd, cwd=RAIZ)
    if r.returncode:
        raise SystemExit(f"\nPAROU em '{etapa}' (codigo {r.returncode})")

def main() -> int:
    ap = argparse.ArgumentParser(
        description="treina a rede para a topologia pedida",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exemplo:\n"
               "  python3 treinar.py --camadas 3 --filtros 8 --kernel 7 "
               "--pooling max --saida gap")
    g = ap.add_argument_group("topologia")
    g.add_argument("--camadas", type=int, choices=(2, 3, 4), default=None,
                   help="numero de camadas de convolucao")
    g.add_argument("--filtros", type=int, choices=(8, 16, 32), default=None,
                   help="filtros na primeira camada; dobra a cada camada")
    g.add_argument("--kernel", type=int, choices=(3, 5, 7), default=None,
                   help="tamanho do filtro")
    g.add_argument("--pooling", choices=("max", "avg", "none"), default=None)
    g.add_argument("--janela", type=int, choices=JANELAS, default=None,
                   help="amostras por janela; fixa quanto sinal cada decisao "
                        "ve e escala o custo linearmente")
    g.add_argument("--saida", choices=("flatten", "gap"), default=None,
                   help="como a ultima convolucao entra na camada de decisao")

    ap.add_argument("--base", type=Path, default=RAIZ / "spec/baseline.yaml",
                   help="receita de partida; o que vier acima sobrepoe")
    ap.add_argument("--treino", action="append", default=[],
                   metavar="CHAVE=VALOR",
                   help="hiperparametro de treino, repetivel")
    ap.add_argument("--qat-epocas", type=int, default=None,
                   help="epocas de reajuste com quantizacao simulada; "
                        "0 desliga (padrao: 15)")
    ap.add_argument("--orcamento", type=int, default=220,
                   help="multiplicadores disponiveis na peca")
    ap.add_argument("--taxa", type=float, default=None, metavar="JANELAS/S",
                   help="taxa a sustentar; com ela o circuito e dimensionado "
                        "pelo requisito e o que sobra vira reserva declarada")
    ap.add_argument("--clock", type=float, default=100.0, metavar="MHZ",
                   help="clock previsto da PL, para converter taxa em ciclos")
    ap.add_argument("--dados", default="data/full")
    ap.add_argument("--etiqueta", default=None,
                   help="sufixo do nome, para guardar receitas lado a lado")
    ap.add_argument("--refaz", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.base.read_text())
    m = dict(cfg.get("model", {}))
    for chave, valor in (("num_layers", a.camadas),
                         ("num_filters_first", a.filtros),
                         ("kernel_size", a.kernel),
                         ("pool_type", a.pooling),
                         ("head", a.saida),
                         ("input_len", a.janela)):
        if valor is not None:
            m[chave] = valor
    m.setdefault("head", "flatten")

    treino = sobrepoe(cfg.get("training", {}), a.treino)
    treino.setdefault("qat_epocas", 15)
    if a.qat_epocas is not None:
        treino["qat_epocas"] = a.qat_epocas

    print("=" * 68)
    print("TREINO")
    print("=" * 68)

    try:
        mcfg = ModelConfig.from_dict(m)
    except ValueError as e:
        raise SystemExit(f"topologia nao suportada: {e}\n  Nada foi treinado.")
    valida(mcfg, a.orcamento, a.taxa, a.clock)
    qat = int(treino["qat_epocas"])
    print(f"  reajuste em int8     : "
          + (f"{qat} epocas" if qat else "desligado"))
    print()

    cmd = [sys.executable, "-u", "-m", "modelo.prepara",
           "--config", str(a.base), "--data-dir", a.dados,
           "--layers", str(mcfg.num_layers),
           "--filters-first", str(mcfg.num_filters_first),
           "--kernel-size", str(mcfg.kernel_size),
           "--pool-type", mcfg.pool_type, "--head", mcfg.head,
           "--janela", str(mcfg.input_len)]
    for chave, valor in treino.items():
        cmd += ["--treino", f"{chave}={valor}"]
    if a.etiqueta:
        cmd += ["--tag", a.etiqueta]
    if a.refaz:
        cmd += ["--refaz"]
    roda(cmd, "treino")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
