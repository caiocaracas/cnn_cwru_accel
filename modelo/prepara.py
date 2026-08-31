"""treina a topologia pedida e exporta o modelo para o fluxo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from modelo.rede import INPUT_LENGTH
from modelo.treina import TrainConfig

RAIZ = Path(__file__).resolve().parent.parent
EIXOS = ("num_layers", "num_filters_first", "kernel_size", "pool_type",
         "head", "input_len")

def nome_da(m: dict) -> str:
    jan = m.get("input_len", INPUT_LENGTH)
    return (f"L{m['num_layers']}_F{m['num_filters_first']:02d}"
            f"_K{m['kernel_size']}_P{m['pool_type']}_H{m['head']}"
            + ("" if jan == INPUT_LENGTH else f"_W{jan}"))

def receita(d: dict | None) -> dict:
    return asdict(TrainConfig.from_dict(d or {}))

def converte(chave: str, texto: str, padrao):
    if isinstance(padrao, bool):
        if texto.lower() in ("1", "true", "sim"):
            return True
        if texto.lower() in ("0", "false", "nao"):
            return False
        raise SystemExit(f"--treino {chave}: '{texto}' nao e' booleano")
    try:
        return type(padrao)(texto)
    except ValueError:
        raise SystemExit(f"--treino {chave}: '{texto}' nao e' "
                         f"{type(padrao).__name__}")

def sobrepoe_treino(base: dict, pares: list[str]) -> dict:
    padroes = asdict(TrainConfig())
    tre = dict(base)
    for item in pares:
        if "=" not in item:
            raise SystemExit(f"--treino espera chave=valor, veio '{item}'")
        chave, valor = item.split("=", 1)
        if chave not in padroes:
            raise SystemExit(
                f"hiperparametro de treino desconhecido: {chave}\n"
                f"  disponiveis: {', '.join(sorted(padroes))}")
        tre[chave] = converte(chave, valor, padroes[chave])
    return tre

def roda(cmd: list) -> None:
    r = subprocess.run(cmd, cwd=RAIZ)
    if r.returncode:
        raise SystemExit(r.returncode)

def main() -> int:
    ap = argparse.ArgumentParser(
        description="treina a topologia e exporta o modelo para o fluxo")
    ap.add_argument("--config", type=Path, default=RAIZ / "spec/baseline.yaml",
                    help="config base; os eixos abaixo sobrepoem o que vier dela")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--filters-first", type=int)
    ap.add_argument("--kernel-size", type=int)
    ap.add_argument("--pool-type", choices=("max", "avg", "none"))
    ap.add_argument("--head", choices=("flatten", "gap"))
    ap.add_argument("--janela", type=int, default=None,
                    help="amostras por janela")
    ap.add_argument("--treino", action="append", default=[], metavar="CHAVE=VALOR",
                    help="hiperparametro de treino, repetivel: "
                         "--treino max_epochs=40 --treino base_lr=1e-3")
    ap.add_argument("--tag", default=None,
                    help="sufixo do nome da rodada, para guardar duas receitas "
                         "da mesma topologia lado a lado")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--refaz", action="store_true",
                    help="treina de novo mesmo se ja houver checkpoint")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    m = dict(cfg["model"])
    for chave, valor in (("num_layers", a.layers),
                         ("num_filters_first", a.filters_first),
                         ("kernel_size", a.kernel_size),
                         ("pool_type", a.pool_type),
                         ("head", a.head),
                         ("input_len", a.janela)):
        if valor is not None:
            m[chave] = valor
    m.setdefault("head", "flatten")
    m.setdefault("input_len", INPUT_LENGTH)
    cfg["model"] = {k: m[k] for k in EIXOS}
    cfg["training"] = sobrepoe_treino(cfg.get("training", {}), a.treino)
    if a.data_dir:
        cfg.setdefault("data", {})["data_dir"] = a.data_dir

    nome = nome_da(cfg["model"]) + (f"_{a.tag}" if a.tag else "")
    run = RAIZ / "runs" / nome
    run.mkdir(parents=True, exist_ok=True)
    efetiva = run / "config.yaml"

    anterior = yaml.safe_load(efetiva.read_text()) if efetiva.exists() else None
    mudou = (anterior is not None
             and receita(anterior.get("training")) != receita(cfg["training"]))

    efetiva.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("=" * 62)
    print(f"prepara - {nome}")
    print("=" * 62)
    print(f"config efetiva em {efetiva.relative_to(RAIZ)}")
    if a.treino:
        base = asdict(TrainConfig())
        for chave in sorted({p.split("=", 1)[0] for p in a.treino}):
            print(f"  {chave}: {base[chave]} -> {cfg['training'][chave]}")

    ckpt = run / "best_checkpoint.pt"
    if ckpt.exists() and not a.refaz and not mudou:
        print(f"\n[1/2] checkpoint ja existe, treino pulado (--refaz forca)")
    else:
        if mudou:
            print("\n      a receita de treino mudou desde o checkpoint gravado;"
                  " treinando de novo")
        print(f"\n[1/2] treinando {nome}")
        cmd = [sys.executable, "-m", "modelo.treina",
               "--config", str(efetiva), "--out", str(run)]
        if a.data_dir:
            cmd += ["--data-dir", a.data_dir]
        roda(cmd)

    qat_ep = int(cfg["training"].get("qat_epocas", 0))
    if qat_ep > 0:
        print(f"\n[2/3] reajustando com quantizacao simulada ({qat_ep} epocas)")
        cmd = [sys.executable, "-u", "-m", "modelo.qat", "--run", nome,
               "--epocas", str(qat_ep),
               "--lr", str(cfg["training"].get("qat_lr", 1e-4)),
               "--bits", str(cfg["training"].get("qat_bits", 8))]
        if a.data_dir:
            cmd += ["--dados", a.data_dir]
        roda(cmd)
        nome = f"{nome}_qat"
        run = RAIZ / "runs" / nome
        efetiva = run / "config.yaml"
        ckpt = run / "best_checkpoint.pt"

    passo = "[3/3]" if qat_ep > 0 else "[2/2]"
    print(f"\n{passo} quantizando para int{a.bits} e exportando o modelo")
    cmd = [sys.executable, "-m", "modelo.quantiza_pesos",
           "--checkpoint", str(ckpt), "--config", str(efetiva),
           "--bits", str(a.bits), "--export-weights",
           "--out", str(run / f"quant_int{a.bits}")]
    if qat_ep > 0:
        cmd += ["--escala-por-camada"]
    if a.data_dir:
        cmd += ["--data-dir", a.data_dir]
    roda(cmd)

    print(f"\nRUN {nome}")
    print(f"modelo pronto. Para levar a placa:")
    print(f"  python3 -m ferramentas.acelera --run {nome}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
