"""prova cada camada em FLUXO CONTINUO contra a referencia em inteiro Diferenca para."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from modelo.quantiza import conv1d_int, relu_int, pool_int, requantize, write_mem

LIMITE_S = 14400
LIMITE_COMPILA_S = 3600
NBLOCO = 4

TB = """`timescale 1ns/1ps
module tb_gerado;
    tb_camada_fluxo #(
        .NOME("{nome}"), .NIF({nif}), .NOF({nof}), .K({k}), .LEN({comp}),
        .POOL({pool}), .POOL_AVG({pavg}), .POF({pof}), .SHIFT(16),
        .RQ_W({rq_w}), .ACC_W({accw}), .POX({pox}), .PK({pk}), .NRQU({nrqu}),
        .NBLOCO({nbloco}), .TESTA_BOLHA(1), .RQ_POR_CANAL({rqc}),
        .LIMITE_NS({limite}),
        .W_FILE   ("{d}/pesos.mem"),
        .B_FILE   ("{d}/bias.mem"),
        .M_FILE   ("{d}/mult.mem"),
        .IN_FILE  ("{d}/entrada.mem"),
        .GOLD_FILE("{d}/gold.mem")
    ) u ();
endmodule
"""

def gera_camada(c: dict, rng, dest: Path) -> np.ndarray:
    nif, nof, k = c["nif"], c["nof"], c["k"]
    comp, pool = c["comp"], c["pool"]
    nam = NBLOCO * comp

    w = rng.integers(-127, 128, size=(nof, nif, k), dtype=np.int64)
    b = rng.integers(-(nif * k * 64), nif * k * 64 + 1, size=(nof,), dtype=np.int64)

    x = rng.integers(-128, 128, size=(nif, nam), dtype=np.int64)
    modo = "avg" if c.get("pool_avg") else "max"
    acc = pool_int(relu_int(conv1d_int(x, w, b)), pool=pool, modo=modo)

    if c.get("rq_por_canal"):
        pico = np.maximum(np.rint(np.percentile(acc, 99.9, axis=1)).astype(np.int64), 1)
    else:
        pico = np.array([max(int(np.percentile(acc, 99.9)), 1)])
    mult = np.clip(np.rint(127 * (1 << 16) / pico), 1, (1 << 17) - 1).astype(np.int64)
    gold = requantize(acc, mult, 16)

    dest.mkdir(parents=True, exist_ok=True)
    write_mem(w, 8, dest / "pesos.mem")
    write_mem(b, 32, dest / "bias.mem")
    write_mem(mult, 18, dest / "mult.mem")
    write_mem(x.T.reshape(-1), 8, dest / "entrada.mem")
    olen = comp // pool
    blocos = [gold[:, i*olen:(i+1)*olen].reshape(-1) for i in range(NBLOCO)]
    write_mem(np.concatenate(blocos), 8, dest / "gold.mem")
    return mult

def roda(c: dict, dest: Path, raiz: Path) -> tuple:
    ciclos = (NBLOCO * c["comp"] * (c["nof"] // c.get("pof", c["nof"]))
              * ((c["nif"] * c["k"]) // c.get("pk", 1)) // max(1, c["pox"]))
    limite = int(max(4e8, 40 * ciclos))
    tb = TB.format(nome=c["nome"], nif=c["nif"], nof=c["nof"], k=c["k"],
                   comp=c["comp"], pool=c["pool"], pavg=c.get("pool_avg", 0),
                   rqc=1 if c.get("rq_por_canal") else 0, rq_w=c["rq_w"],
                   pox=c["pox"], pof=c.get("pof", c["nof"]), pk=c.get("pk", 1),
                   nrqu=c["nrqu"], accw=c.get("acc_w", 32), nbloco=NBLOCO,
                   limite=limite, d=dest)
    (dest / "tb_gerado.v").write_text(tb)
    vvp = dest / "sim.vvp"

    r = subprocess.run(
        ["timeout", "-k", "10", str(LIMITE_COMPILA_S),
         "iverilog", "-g2005-sv", "-s", "tb_gerado", "-o", str(vvp),
         str(raiz / "rtl/mac_lane.v"), str(raiz / "rtl/requant.v"),
         str(raiz / "rtl/conv1d_engine.v"), str(raiz / "sim/tb_camada_fluxo.v"),
         str(dest / "tb_gerado.v")],
        capture_output=True, text=True)
    if r.returncode == 124:
        return None, f"nao compilou em {LIMITE_COMPILA_S}s"
    if r.returncode:
        return False, (r.stderr.strip().splitlines() or ["erro de compilacao"])[-1]

    r = subprocess.run(["timeout", "-k", "10", str(LIMITE_S), "vvp", str(vvp)],
                       capture_output=True, text=True, cwd=raiz)
    if r.returncode == 124:
        return None, f"nao concluiu em {LIMITE_S}s"
    saida = r.stdout
    passo = [l.split(":", 1)[1].strip() for l in saida.splitlines()
             if "ciclos por grupo" in l or "ciclos por amostra" in l]
    if "FLUXO BIT-EXATO" in saida:
        return True, "  ".join(passo) if passo else ""
    ruim = [l.strip() for l in saida.splitlines()
            if "FALHOU" in l or "obtido" in l]
    return False, ruim[0] if ruim else "sem FLUXO BIT-EXATO na saida"

def chave(c: dict) -> str:
    return (f"NIF{c['nif']}_NOF{c['nof']}_K{c['k']}_LEN{c['comp']}"
            f"_P{c['pool']}{'a' if c.get('pool_avg') else 'm'}"
            f"_POX{c['pox']}_POF{c.get('pof', c['nof'])}_PK{c.get('pk', 1)}"
            f"_V{c['nrqu']}_RQ{c['rq_w']}"
            + ("_C" if c.get("rq_por_canal") else ""))

def prova(args) -> tuple:
    k, c, raiz = args
    rng = np.random.default_rng(zlib.crc32(k.encode()))
    with tempfile.TemporaryDirectory(prefix="fluxo_") as td:
        d = Path(td) / c["nome"]
        gera_camada(c, rng, d)
        ok, msg = roda(c, d, Path(raiz))
    return k, ok, msg

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plano", type=Path)
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--gen", type=Path, default=Path("results/gen"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--saida", type=Path, default=Path("results/cobertura_fluxo.json"))
    a = ap.parse_args()
    raiz = Path(__file__).resolve().parent.parent

    planos = ([a.plano] if a.plano else sorted(a.gen.glob("*/plano.json")))
    if a.todos:
        planos = [q for q in planos if json.loads(q.read_text())["cabe"]]

    tarefas, de_quem = {}, {}
    for q in planos:
        p = json.loads(q.read_text())
        for c in p["camadas"]:
            c = dict(c, rq_por_canal=bool(p.get("rq_por_canal")))
            k = chave(c)
            tarefas.setdefault(k, c)
            de_quem.setdefault(k, []).append(p["nome"])

    feito = json.loads(a.saida.read_text()) if a.saida.exists() else {}
    pend = [k for k in tarefas if not feito.get(k, {}).get("ok")]
    pend.sort(key=lambda k: -tarefas[k]["comp"] * tarefas[k]["nof"])
    print(f"{len(planos)} topologias -> {len(tarefas)} camadas distintas; "
          f"{len(tarefas)-len(pend)} ja' provadas, {len(pend)} a rodar em "
          f"{a.jobs} processos", flush=True)

    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(prova, (k, tarefas[k], str(raiz))): k for k in pend}
        for i, f in enumerate(as_completed(futs), 1):
            k = futs[f]
            try:
                k, ok, msg = f.result()
            except Exception as e:
                ok, msg = None, f"{type(e).__name__}: {e}"
            feito[k] = {"ok": bool(ok), "concluiu": ok is not None, "msg": msg,
                        "topologias": len(de_quem[k])}
            a.saida.write_text(json.dumps(feito, indent=2))
            marca = "ok" if ok else ("PAROU" if ok is None else "FALHOU")
            print(f"  [{i}/{len(pend)}] {k:52s} {marca}  {msg}", flush=True)

    ruins = sorted(k for k, v in feito.items()
                   if k in tarefas and not v["ok"] and v.get("concluiu", True))
    parou = sorted(k for k, v in feito.items()
                   if k in tarefas and not v.get("concluiu", True))
    print(f"\n{len(tarefas)-len(ruins)-len(parou)}/{len(tarefas)} camadas "
          f"bit-exatas em fluxo continuo")
    for k in ruins: print(f"  FALHOU {k}: {feito[k]['msg']}")
    for k in parou: print(f"  NAO CONCLUIU {k}: {feito[k]['msg']}")
    return 0 if not (ruins or parou) else 1

if __name__ == "__main__":
    raise SystemExit(main())
