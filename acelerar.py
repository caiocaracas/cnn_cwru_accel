"""Deriva o acelerador da rede treinada, sintetiza e mede na placa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from ferramentas.acelera import ETAPAS

O_QUE_FAZ = {
    "valida":    "le o modelo e confere que a topologia e' suportada",
    "quantiza":  "converte os pesos para inteiro e prepara os vetores de teste",
    "recursos":  "mede area e clock do acelerador por sintese fora de contexto",
    "modela":    "calcula o circuito e escreve o Verilog desta rede",
    "confere":   "simula e exige resultado identico ao modelo, bit a bit",
    "sintetiza": "roda o Vivado por TCL ate' o bitstream",
    "sistema":   "monta a imagem do Linux com o programa do ARM",
    "placa":     "grava na Arty e mede o conjunto de teste inteiro",
    "relatorio": "junta acuracia, tempo, area e consumo num arquivo",
}

def main() -> int:
    ap = argparse.ArgumentParser(
        description="da rede treinada ao circuito medido na placa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="etapas:\n" + "\n".join(f"  {k:10s} {v}"
                                       for k, v in O_QUE_FAZ.items()))
    ap.add_argument("--rede", required=True,
                    help="pasta em runs/ deixada pelo treinar.py")
    ap.add_argument("--orcamento", type=int, default=220,
                    help="multiplicadores disponiveis na peca")
    ap.add_argument("--taxa", type=float, default=None, metavar="JANELAS/S",
                    help="taxa a sustentar; o hardware e dimensionado para "
                         "cumprir o prazo em vez de encher o chip")
    ap.add_argument("--clock", type=int, default=133,
                    help="frequencia alvo da PL em MHz; o fluxo baixa sozinho "
                         "se nao fechar tempo")
    ap.add_argument("--dados", default="data/full")
    ap.add_argument("--n-teste", type=int, default=0,
                    help="janelas levadas a placa; 0 e' o conjunto inteiro")
    ap.add_argument("--ate", choices=ETAPAS, default="relatorio")
    ap.add_argument("--de", choices=ETAPAS, default=None)
    ap.add_argument("--fluxo", action="store_true",
                    help="acelerador em fluxo continuo: o motor nunca para e "
                         "nao recomputa a janela entre decisoes (exige gap)")
    ap.add_argument("--refaz-sintese", action="store_true")
    ap.add_argument("--vivado", default="/home/caiocv/2025.2/Vivado")
    ap.add_argument("--petalinux", default="/home/caiocv/petalinux/2025.2")
    a = ap.parse_args()

    if not (RAIZ / "runs" / a.rede / "config.yaml").exists():
        disponiveis = sorted(q.parent.name
                             for q in (RAIZ / "runs").glob("*/config.yaml"))
        raise SystemExit(
            f"nao achei runs/{a.rede}.\n"
            f"  Treine antes:  python3 treinar.py --camadas 3 --filtros 8 ...\n"
            f"  Ou escolha uma das {len(disponiveis)} redes ja' treinadas, "
            f"por exemplo: {', '.join(disponiveis[:3])}")

    argv = ["--run", a.rede, "--dados", a.dados]
    if a.taxa:
        argv += ["--taxa", str(a.taxa)]
    argv += [
            "--orcamento", str(a.orcamento), "--clock", str(a.clock),
            "--n-teste", str(a.n_teste), "--ate", a.ate,
            "--vivado", a.vivado, "--petalinux", a.petalinux]
    if a.de:
        argv += ["--de", a.de]
    if a.fluxo:
        argv += ["--fluxo"]
    if a.refaz_sintese:
        argv += ["--refaz-sintese"]

    from ferramentas import acelera
    sys.argv = ["acelera"] + argv
    return acelera.main()

if __name__ == "__main__":
    raise SystemExit(main())
