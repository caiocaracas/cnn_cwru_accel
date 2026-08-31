"""mede area e clock de qualquer ponto do dominio, sem precisar treinar O modelo de LUT do."""

from __future__ import annotations

import argparse, json, itertools, subprocess, os
from pathlib import Path
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
NCLS = 4

def hw_data_sintetico(plano: dict, dest: Path) -> None:
    from ferramentas.memorias import grava
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, c in enumerate(plano["camadas"], start=1):
        n = c["nof"] * c["nif"] * c["k"]
        grava(rng.integers(-127, 128, n, dtype=np.int64), 8, dest / f"pesos_conv{i}.mem")
        grava(rng.integers(-1000, 1000, c["nof"], dtype=np.int64), 32, dest / f"bias_conv{i}.mem")
        nm = c["nof"] if plano.get("rq_por_canal") else 1
        grava(np.full(nm, 1000, dtype=np.int64), 18, dest / f"mult_conv{i}.mem")
    grava(rng.integers(-127, 128, plano["fc_nflat"] * NCLS, dtype=np.int64), 8,
          dest / "pesos_fc.mem")
    grava(rng.integers(-1000, 1000, NCLS, dtype=np.int64), 32, dest / "bias_fc.mem")

TCL = """
set nome [lindex $argv 0]
set per  [lindex $argv 1]
set raiz {raiz}
create_project -in_memory -part xc7z020clg400-1
add_files -norecurse [concat [glob $raiz/rtl/*.v] [list $raiz/results/gen/$nome/acelerador_gen.v]]
set_property file_type SystemVerilog [get_files -quiet */conv1d_engine.v]
set_property file_type SystemVerilog [get_files -quiet */fc_engine.v]
synth_design -top acelerador_gen -part xc7z020clg400-1 -mode out_of_context
create_clock -period $per -name clk [get_ports clk]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set b36 [llength [get_cells -hier -filter {{REF_NAME =~ RAMB36*}}]]
set b18 [llength [get_cells -hier -filter {{REF_NAME =~ RAMB18*}}]]
puts "MEDE lut [llength [get_cells -hier -filter {{REF_NAME =~ LUT*}}]]"
puts "MEDE ff [llength [get_cells -hier -filter {{REF_NAME =~ FD*}}]]"
puts "MEDE dsp [llength [get_cells -hier -filter {{REF_NAME =~ DSP48*}}]]"
puts "MEDE bram [expr {{$b36 + int(ceil($b18/2.0))}}]"
puts "MEDE wns $wns"
puts "MEDE fmax [format %.1f [expr {{1000.0/($per-$wns)}}]]"
exit 0
"""

def mede(nome: str, vivado: str, per: float, tcl: Path) -> dict | None:
    r = subprocess.run(
        ["bash", "-lc",
         f"source {vivado}/settings64.sh && vivado -mode batch -nojournal "
         f"-nolog -notrace -source {tcl} -tclargs {nome} {per}"],
        capture_output=True, text=True, cwd=RAIZ, env=dict(os.environ))
    med = {}
    for ln in r.stdout.splitlines():
        if ln.startswith("MEDE "):
            _, k, v = ln.split()
            med[k] = float(v) if "." in v or k in ("wns", "fmax") else int(v)
    return med or None

def main() -> int:
    import ferramentas.gerador as g
    from ferramentas.memorias import emite

    ap = argparse.ArgumentParser()
    ap.add_argument("--pontos", type=int, default=12,
                    help="topologias amostradas do dominio")
    ap.add_argument("--fluxo", action="store_true")
    ap.add_argument("--clock", type=float, default=8.0, metavar="NS")
    ap.add_argument("--saida", type=Path, default=RAIZ / "results/area_medida.json")
    ap.add_argument("--vivado", default="/home/caiocv/2025.2/Vivado")
    a = ap.parse_args()

    tcl = RAIZ / "results" / "mede_area.tcl"
    tcl.parent.mkdir(parents=True, exist_ok=True)
    tcl.write_text(TCL.format(raiz=RAIZ))

    dom = [q for q in itertools.product((2, 3, 4), (8, 16, 32), (3, 5, 7),
                                        ("max", "avg", "none"),
                                        ("gap",) if a.fluxo else ("gap", "flatten"))]
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(dom))[:a.pontos]

    feito = json.loads(a.saida.read_text()) if a.saida.exists() else {}
    for k in idx:
        L, F, K, P, H = dom[k]
        p = g.planeja(L, F, K, P, H, fluxo=a.fluxo)
        if not p.cabe and "acumulador" in p.motivo:
            continue
        from dataclasses import asdict
        d = json.loads(json.dumps(asdict(p)))
        if p.nome in feito:
            continue
        g.salva(p, RAIZ / "results/gen")
        hw = RAIZ / "results/hw_sintetico" / p.nome
        hw_data_sintetico(d, hw)
        emite(d, hw, RAIZ / "results/gen" / p.nome / "mem", run="sintetico")
        m = mede(p.nome, a.vivado, a.clock, tcl)
        if not m:
            print(f"  {p.nome}: sintese falhou", flush=True)
            feito[p.nome] = {"erro": "sintese", "plano_cabe": p.cabe}
        else:
            feito[p.nome] = {**m, "plano": d}
            print(f"  {p.nome}: lut {m['lut']} ff {m['ff']} dsp {m['dsp']} "
                  f"bram {m['bram']} fmax {m['fmax']}", flush=True)
        a.saida.write_text(json.dumps(feito, indent=1))
    print(f"\n{len(feito)} pontos em {a.saida}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
