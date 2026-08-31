"""prova cada camada gerada contra a referencia em inteiro."""

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

TB = """`timescale 1ns/1ps
module tb_gerado;
    tb_camada #(
        .NOME("{nome}"), .NIF({nif}), .NOF({nof}), .K({k}), .LEN({comp}),
        .POOL({pool}), .POOL_AVG({pavg}), .POF({pof}), .SHIFT(16), .ACC_BITS({rq_w}),
        .RQ_W({rq_w}), .ACC_W({accw}), .POX({pox}), .PK({pk}), .NRQU({nrqu}),
        .NVEC({nvec}), .NVEC_ARQ({nvec}), .TESTA_BOLHA(1),
        .W_FILE   ("{d}/pesos.mem"),
        .B_FILE   ("{d}/bias.mem"),
        .M_FILE   ("{d}/mult.mem"), .RQ_POR_CANAL({rqc}),
        .IN_FILE  ("{d}/entrada.mem"),
        .GOLD_FILE("{d}/gold.mem")
    ) u ();
endmodule
"""

LIMITE_S = 14400
LIMITE_COMPILA_S = 3600

def gera_camada(c: dict, nvec: int, rng, dest: Path) -> int:
    nif, nof, k = c["nif"], c["nof"], c["k"]
    comp, pool = c["comp"], c["pool"]

    w = rng.integers(-127, 128, size=(nof, nif, k), dtype=np.int64)

    b = rng.integers(-(nif * k * 64), nif * k * 64 + 1, size=(nof,), dtype=np.int64)

    entradas, golds, accs = [], [], []
    for _ in range(nvec):
        x = rng.integers(-127, 128, size=(nif, comp), dtype=np.int64)
        modo = "avg" if c.get("pool_avg") else "max"
        a = pool_int(relu_int(conv1d_int(x, w, b)), pool=pool, modo=modo)
        entradas.append(x)
        accs.append(a)

    empilhado = np.concatenate(accs, axis=1)
    if c.get("rq_por_canal"):
        pico = np.maximum(np.rint(np.percentile(empilhado, 99.9, axis=1)).astype(np.int64), 1)
    else:
        pico = np.array([max(int(np.percentile(empilhado, 99.9)), 1)])
    mult = np.clip(np.rint(127 * (1 << 16) / pico), 1, (1 << 17) - 1).astype(np.int64)
    for a in accs:
        golds.append(requantize(a, mult, 16))

    dest.mkdir(parents=True, exist_ok=True)
    write_mem(w, 8, dest / "pesos.mem")
    write_mem(b, 32, dest / "bias.mem")
    write_mem(mult, 18, dest / "mult.mem")
    write_mem(np.concatenate([x.reshape(-1) for x in entradas]), 8,
              dest / "entrada.mem")
    write_mem(np.concatenate([g.reshape(-1) for g in golds]), 8,
              dest / "gold.mem")
    return mult

def roda(c: dict, mult: int, nvec: int, dest: Path, raiz: Path) -> tuple[bool, str]:
    tb = TB.format(nome=c["nome"], nif=c["nif"], nof=c["nof"], k=c["k"],
                   comp=c["comp"], pool=c["pool"], pavg=c.get("pool_avg",0),
                   rqc=1 if c.get("rq_por_canal") else 0, rq_w=c["rq_w"],
                   pox=c["pox"], pof=c.get("pof", c["nof"]), pk=c.get("pk",1), nrqu=c["nrqu"], accw=c.get("acc_w",32), nvec=nvec, d=dest)
    (dest / "tb_gerado.v").write_text(tb)
    vvp = dest / "sim.vvp"

    r = subprocess.run(
        ["timeout", "-k", "10", str(LIMITE_COMPILA_S),
         "iverilog", "-g2005-sv", "-s", "tb_gerado", "-o", str(vvp),
         str(raiz / "rtl/mac_lane.v"), str(raiz / "rtl/requant.v"),
         str(raiz / "rtl/conv1d_engine.v"), str(raiz / "sim/tb_camada.v"),
         str(dest / "tb_gerado.v")],
        capture_output=True, text=True)
    if r.returncode == 124:
        return None, f"nao compilou em {LIMITE_COMPILA_S}s"
    if r.returncode:
        return False, r.stderr.strip().splitlines()[-1] if r.stderr else "erro de compilacao"

    r = subprocess.run(["timeout", "-k", "10", str(LIMITE_S), "vvp", str(vvp)],
                       capture_output=True, text=True, cwd=raiz)
    if r.returncode == 124:
        return None, f"nao concluiu em {LIMITE_S}s"
    saida = r.stdout
    if "BIT-EXATO" in saida:
        cic = [l for l in saida.splitlines() if "ciclos/inferencia" in l]
        return True, cic[0].split(":")[1].strip() if cic else ""
    ruim = [l.strip() for l in saida.splitlines()
            if "FALHOU" in l or "diverge" in l or "obtido" in l]
    return False, ruim[0] if ruim else "sem BIT-EXATO na saida"

def chave(c: dict) -> str:
    return (f"NIF{c['nif']}_NOF{c['nof']}_K{c['k']}_LEN{c['comp']}"
            f"_P{c['pool']}{'a' if c.get('pool_avg') else 'm'}"
            f"_POX{c['pox']}_POF{c.get('pof', c['nof'])}_PK{c.get('pk', 1)}"
            f"_V{c['nrqu']}_RQ{c['rq_w']}"
            + ("_C" if c.get("rq_por_canal") else ""))

def prova(args) -> tuple:
    k, c, nvec, raiz = args
    import numpy as np
    rng = np.random.default_rng(zlib.crc32(k.encode()))
    with tempfile.TemporaryDirectory(prefix="verif_") as td:
        d = Path(td) / c["nome"]
        mult = gera_camada(c, nvec, rng, d)
        ok, msg = roda(c, mult, nvec, d, Path(raiz))
    return k, ok, msg, int(c["ciclos"])

def verifica(plano: Path, nvec: int, raiz: Path) -> bool:
    p = json.loads(plano.read_text())
    rng = np.random.default_rng(zlib.crc32(p["nome"].encode()))
    print(f"=== {p['nome']}  ({p['dsp_total']} DSP, II {p['ii']})")
    tudo_ok = True
    with tempfile.TemporaryDirectory(prefix="verif_") as td:
        for c in p["camadas"]:
            c["rq_por_canal"] = bool(p.get("rq_por_canal"))
            d = Path(td) / c["nome"]
            mult = gera_camada(c, nvec, rng, d)
            ok, msg = roda(c, mult, nvec, d, raiz)
            marca = "ok  " if ok else ("PAROU " if ok is None else "FALHOU")
            print(f"  {c['nome']:6s} NIF={c['nif']:3d} NOF={c['nof']:3d} "
                  f"POX={c['pox']:2d} PK={c.get('pk',1):2d} vias={c['nrqu']:2d} "
                  f"MULT={int(mult.min())}..{int(mult.max())}  "
                  f"{marca} {msg}")
            tudo_ok &= bool(ok)
    return tudo_ok

def cobertura(planos: list[Path], nvec: int, raiz: Path, jobs: int,
              saida: Path) -> int:
    tarefas, de_quem = {}, {}
    for q in planos:
        p = json.loads(q.read_text())
        for c in p["camadas"]:
            c = dict(c, rq_por_canal=bool(p.get("rq_por_canal")))
            k = chave(c)
            tarefas.setdefault(k, c)
            de_quem.setdefault(k, []).append(p["nome"])

    feito = json.loads(saida.read_text()) if saida.exists() else {}
    pendentes = [k for k in tarefas if not feito.get(k, {}).get("ok")]
    pendentes.sort(key=lambda k: -tarefas[k]["ciclos"])
    print(f"{len(planos)} topologias -> {len(tarefas)} configuracoes de camada "
          f"distintas; {len(tarefas) - len(pendentes)} ja' provadas, "
          f"{len(pendentes)} a rodar em {jobs} processos")

    ciclos = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(prova, (k, tarefas[k], nvec, str(raiz))): k
                for k in pendentes}
        for i, f in enumerate(as_completed(futs), 1):
            k = futs[f]
            try:
                k, ok, msg, cic = f.result()
            except Exception as e:
                ok, msg, cic = None, f"{type(e).__name__}: {e}", 0
            feito[k] = {"ok": bool(ok), "concluiu": ok is not None,
                        "msg": msg, "ciclos": cic,
                        "topologias": len(de_quem[k])}
            ciclos += cic * nvec
            saida.write_text(json.dumps(feito, indent=2))
            marca = "ok" if ok else ("PAROU" if ok is None else "FALHOU")
            print(f"  [{i}/{len(pendentes)}] {k:52s} {marca}  {msg}",
                  flush=True)

    ruins = sorted(k for k, v in feito.items()
                   if k in tarefas and not v["ok"] and v.get("concluiu", True))
    parou = sorted(k for k, v in feito.items()
                   if k in tarefas and not v.get("concluiu", True))
    bons = len(tarefas) - len(ruins) - len(parou)
    print(f"\n{bons}/{len(tarefas)} configuracoes de camada bit-exatas "
          f"({ciclos} ciclos simulados nesta rodada)")
    for k in ruins:
        print(f"  FALHOU {k}: {feito[k]['msg']}  "
              f"(usada por {feito[k]['topologias']} topologias)")
    for k in parou:
        print(f"  NAO CONCLUIU {k}: {feito[k]['msg']}  "
              f"(usada por {feito[k]['topologias']} topologias)")
    return 0 if not (ruins or parou) else 1

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plano", type=Path)
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--gen", type=Path, default=Path("results/gen"))
    ap.add_argument("--nvec", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1,
                    help="processos em paralelo; so' vale com --todos")
    ap.add_argument("--saida", type=Path, default=Path("results/cobertura.json"),
                    help="onde guardar o veredito por configuracao; a rodada "
                         "e' retomavel a partir dele")
    a = ap.parse_args()
    raiz = Path(__file__).resolve().parent.parent

    planos = ([a.plano] if a.plano else
              sorted(a.gen.glob("*/plano.json")))
    if a.todos:
        planos = [q for q in planos if json.loads(q.read_text())["cabe"]]
        return cobertura(planos, a.nvec, raiz, a.jobs, a.saida)

    n_ok = 0
    for q in planos:
        n_ok += verifica(q, a.nvec, raiz)
    print(f"\n{n_ok}/{len(planos)} configuracoes com hardware bit-exato")
    return 0 if n_ok == len(planos) else 1

if __name__ == "__main__":
    raise SystemExit(main())
