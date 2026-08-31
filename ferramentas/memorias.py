"""reparte os tensores nos bancos de memoria que a sintese le."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MULT_W = 18

def le_mem(p: Path, bits: int) -> np.ndarray:
    v = [int(l, 16) for l in p.read_text().split() if not l.startswith("//")]
    a = np.array(v, dtype=np.int64)
    return np.where(a >= (1 << (bits - 1)), a - (1 << bits), a)

def grava(vals: np.ndarray, bits: int, destino: Path) -> None:
    m = (1 << bits) - 1
    largura = bits // 4
    destino.write_text("\n".join(f"{int(v) & m:0{largura}X}" for v in vals) + "\n")

def bancos_conv(c: dict, w: np.ndarray, b: np.ndarray, m: np.ndarray,
                prefixo: Path) -> int:
    nof, nif, k = c["nof"], c["nif"], c["k"]
    pof = c["pof"]
    og = nof // pof
    w = w.reshape(nof, nif, k)

    for l in range(pof):
        canais = [g * pof + l for g in range(og)]
        grava(np.concatenate([w[oc].reshape(-1) for oc in canais]), 8,
              Path(f"{prefixo}_w{l}.mem"))
        grava(np.array([b[oc] for oc in canais]), 32,
              Path(f"{prefixo}_b{l}.mem"))

    grava(m if c.get("rq_por_canal") else m[:1], MULT_W,
          Path(f"{prefixo}_m.mem"))
    return pof

def bancos_fc(nflat: int, ncls: int, nvia: int, npos: int,
              w: np.ndarray, b: np.ndarray, prefixo: Path) -> int:
    w = w.reshape(ncls, nflat)
    nch = nflat // npos
    for c in range(ncls):
        for v in range(nvia):
            canais = [ch for ch in range(nch) if ch % nvia == v]
            grava(np.concatenate([w[c, ch * npos:(ch + 1) * npos]
                                  for ch in canais]), 8,
                  Path(f"{prefixo}_c{c}_v{v}.mem"))
        grava(np.array([b[c]]), 32, Path(f"{prefixo}_b{c}.mem"))
    return ncls * nvia

def digesto(hw_data: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(hw_data.glob("*.mem")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]

def emite(plano: dict, hw_data: Path, destino: Path, run: str = "") -> dict:
    destino.mkdir(parents=True, exist_ok=True)
    for f in destino.glob("*.mem"):
        f.unlink()

    n = 0
    for i, c in enumerate(plano["camadas"], start=1):
        w = le_mem(hw_data / f"pesos_conv{i}.mem", 8)
        b = le_mem(hw_data / f"bias_conv{i}.mem", 32)
        m = le_mem(hw_data / f"mult_conv{i}.mem", MULT_W)
        esperado = c["nof"] * c["nif"] * c["k"]
        if w.size != esperado:
            raise ValueError(f"conv{i}: {w.size} pesos, o grafo pede {esperado}")
        c["rq_por_canal"] = bool(plano.get("rq_por_canal"))
        if c["rq_por_canal"] and m.size != c["nof"]:
            raise ValueError(f"conv{i}: {m.size} multiplicadores de "
                             f"requantizacao, o plano pede {c['nof']} "
                             f"(um por canal de saida)")
        if not c["rq_por_canal"] and m.size > 1 and len(set(m.tolist())) > 1:
            raise ValueError(
                f"conv{i}: o plano pede uma constante de requantizacao por "
                f"camada, mas o modelo trouxe {len(set(m.tolist()))} valores "
                f"diferentes. Ou gere com --rq-por-canal, ou quantize com "
                f"--escala-por-camada")
        n += bancos_conv(c, w, b, m, destino / f"c{i}")

    ult = plano["camadas"][-1]
    npos = 1 if plano.get("head") == "gap" else ult["comp"] // ult["pool"]
    w = le_mem(hw_data / "pesos_fc.mem", 8)
    b = le_mem(hw_data / "bias_fc.mem", 32)
    n += bancos_fc(plano["fc_nflat"], 4, plano["fc_vias"], npos, w, b,
                   destino / "fc")

    if plano.get("fluxo"):
        grava(w, 8,  destino / "fc_w.mem")
        grava(b, 32, destino / "fc_b.mem")
        n += 2

    total = sum(f.stat().st_size for f in destino.glob("*.mem"))
    proc = {"run": run, "hw": plano["nome"], "pesos_sha": digesto(hw_data),
            "hw_data": str(hw_data)}
    (destino / "procedencia.json").write_text(json.dumps(proc, indent=2))
    return {"bancos": n, "arquivos": len(list(destino.glob("*.mem"))),
            "bytes": total, **proc}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plano", type=Path, required=True)
    ap.add_argument("--hw-data", type=Path, default=Path("results/hw_data"))
    ap.add_argument("--destino", type=Path, default=None)
    ap.add_argument("--run", default="", help="rodada dona destes pesos")
    a = ap.parse_args()

    plano = json.loads(a.plano.read_text())
    destino = a.destino or (a.plano.parent / "mem")
    r = emite(plano, a.hw_data, destino, a.run)
    print(f"{r['bancos']} bancos, {r['arquivos']} arquivos, "
          f"{r['bytes']/1e3:.0f} kB em {destino} "
          f"(rodada {r['run'] or '?'}, pesos {r['pesos_sha']})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
