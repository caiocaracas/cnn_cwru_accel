"""prova a rede montada em FLUXO CONTINUO contra a referencia em inteiro O sinal entra amostra a."""

from __future__ import annotations

import argparse, json, os, subprocess, tempfile, zlib
from pathlib import Path
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from modelo.quantiza import conv1d_int, relu_int, pool_int, requantize, write_mem

NCLS = 4

TB = """`timescale 1ns/1ps
`default_nettype none
module tb_fluxo;
    localparam LEN  = {comp0};
    localparam NAM  = {nam};
    localparam NDEC = {ndec};
    reg clk = 0;  always #5 clk = ~clk;
    reg rst_n = 0;
    reg [3:0] ld_sel = 0;
    reg ld_w_en=0, ld_w_valid=0;  reg signed [7:0]  ld_w_data=0;
    reg ld_b_en=0, ld_b_valid=0;  reg signed [31:0] ld_b_data=0;
    reg ld_m_en=0, ld_m_valid=0;  reg signed [17:0] ld_m_data=0;
    reg start=0, alimenta=0;
    wire busy, done, in_ready, classe_valid;
    wire [7:0] trunc_err;
    wire [{ncls}*32-1:0] logits;
    wire [3:0] classe;

    integer fed = 0;
    wire signed [7:0] in_data = vin[fed];
    wire in_valid = alimenta && (fed < NAM);

    acelerador_gen #(.NCAM(8), .NCLS({ncls}), .MEMDIR("{d}/mem")) dut (
        .clk(clk), .rst_n(rst_n), .ld_sel(ld_sel),
        .ld_w_en(ld_w_en), .ld_w_valid(ld_w_valid), .ld_w_data(ld_w_data),
        .ld_b_en(ld_b_en), .ld_b_valid(ld_b_valid), .ld_b_data(ld_b_data),
        .ld_m_en(ld_m_en), .ld_m_valid(ld_m_valid), .ld_m_data(ld_m_data),
        .start(start), .busy(busy), .done(done), .trunc_err(trunc_err),
        .in_valid(in_valid), .in_data(in_data), .in_ready(in_ready),
        .logits(logits), .classe(classe), .classe_valid(classe_valid),
        .ativ_reinicia(1'b0), .ativ_avanca(1'b0), .ativ_data()
    );

{decl}
    reg [7:0]  vin  [0:NAM-1];
    reg [31:0] gold [0:NDEC*{ncls}-1];

    integer i, v, d, erros, esp, obt, ndec_vistas, fh;
    integer ciclo = 0, ult = 0, dt_min = 1000000000, dt_max = 0, n_dt = 0;
    reg signed [31:0] vistos [0:NDEC*{ncls}-1];

    // despejo da saida da ultima conv: separa cadeia de cabeca quando a
    // rede montada diverge mas as camadas isoladas passam
    integer fc3, iv;
    initial fc3 = $fopen("{d}/c_ult.txt", "w");
    always @(posedge clk)
        if (dut.c{lult}_ov)
            for (iv = 0; iv < {nrqu}; iv = iv + 1)
                $fdisplay(fc3, "%0d %0d %0d", dut.c{lult}_op,
                          dut.c{lult}_oc + iv,
                          $signed(dut.c{lult}_od[iv*8 +: 8]));

    always @(posedge clk) if (in_ready && in_valid) fed <= fed + 1;
    always @(posedge clk) ciclo <= ciclo + 1;

    integer ic;
    always @(posedge clk)
        if (classe_valid) begin
            if (ndec_vistas < NDEC)
                for (ic = 0; ic < {ncls}; ic = ic + 1)
                    vistos[ndec_vistas*{ncls} + ic] <= $signed(logits[ic*32 +: 32]);
            if (ndec_vistas > 0) begin
                if (ciclo - ult < dt_min) dt_min = ciclo - ult;
                if (ciclo - ult > dt_max) dt_max = ciclo - ult;
                n_dt = n_dt + 1;
            end
            ult = ciclo;
            ndec_vistas = ndec_vistas + 1;
        end

    task carrega(input [3:0] sel, input integer nw, input integer nb, input integer nm);
        begin
            ld_sel = sel;
            @(posedge clk); ld_w_en <= 1'b1;
            @(posedge clk); ld_w_en <= 1'b0;
            for (i = 0; i < nw; i = i + 1) begin
                @(posedge clk); ld_w_valid <= 1'b1;
                case (sel)
{casew}
                endcase
            end
            @(posedge clk); ld_w_valid <= 1'b0;
            @(posedge clk); ld_b_en <= 1'b1;
            @(posedge clk); ld_b_en <= 1'b0;
            for (i = 0; i < nb; i = i + 1) begin
                @(posedge clk); ld_b_valid <= 1'b1;
                case (sel)
{caseb}
                endcase
            end
            @(posedge clk); ld_b_valid <= 1'b0;
            if (nm > 0) begin
                @(posedge clk); ld_m_en <= 1'b1;
                @(posedge clk); ld_m_en <= 1'b0;
                for (i = 0; i < nm; i = i + 1) begin
                    @(posedge clk); ld_m_valid <= 1'b1;
                    case (sel)
{casem}
                    endcase
                end
                @(posedge clk); ld_m_valid <= 1'b0;
            end
        end
    endtask

    initial begin
        ndec_vistas = 0;
{leituras}
        $readmemh("{d}/entrada.mem", vin);
        $readmemh("{d}/gold_logits.mem", gold);
        repeat (4) @(posedge clk); rst_n <= 1'b1;
        repeat (2) @(posedge clk);
{cargas}
        @(posedge clk); start <= 1'b1;
        @(posedge clk); start <= 1'b0;
        @(posedge clk); alimenta <= 1'b1;
        wait (fed >= NAM);
        repeat (20000) @(posedge clk);

        fh = $fopen("{d}/vistos.txt", "w");
        for (d = 0; d < ndec_vistas && d < NDEC; d = d + 1) begin
            for (i = 0; i < {ncls}; i = i + 1)
                $fwrite(fh, "%0d ", vistos[d*{ncls} + i]);
            $fdisplay(fh, "");
        end
        $fclose(fh);
        erros = 0;
        $display("  decisoes: %0d (referencia tem %0d)", ndec_vistas, NDEC);
        $display("  ciclos entre decisoes: %0d..%0d   jitter %0d   (%0d medidas)",
                 dt_min, dt_max, dt_max - dt_min, n_dt);
        if (trunc_err !== 8'd0) begin
            $display("  FALHOU: estouro de acumulador %b", trunc_err);
            erros = erros + 1;
        end
        $display("  decisoes gravadas");
        $finish;
    end
    initial begin #900_000_000; $display("  FALHOU: timeout"); $finish; end
endmodule
`default_nettype wire
"""

TB_BUS = """`timescale 1ns/1ps
`default_nettype none
module tb_fbus;
    localparam NAM  = {nam};
    localparam NDEC = {ndec};
    reg clk = 0;  always #5 clk = ~clk;
    reg rstn = 0;

    reg  [7:0]  awaddr=0; reg awvalid=0; wire awready;
    reg  [31:0] wdata=0;  reg wvalid=0;  wire wready;
    wire [1:0]  bresp; wire bvalid; reg bready=1;
    reg  [7:0]  araddr=0; reg arvalid=0; wire arready;
    wire [31:0] rdata; wire [1:0] rresp; wire rvalid; reg rready=1;

    reg  [31:0] sdata=0;  reg svalid=0;  wire sready;

    axi_acelerador #(.LEN({comp0}), .NCLS({ncls}), .FLUXO(1)) dut (
        .s_axi_aclk(clk), .s_axi_aresetn(rstn),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .s_axis_tdata(sdata), .s_axis_tvalid(svalid), .s_axis_tready(sready),
        .s_axis_tlast(1'b0), .leds());

    task wr(input [7:0] a, input [31:0] d);
        begin
            @(posedge clk); awaddr<=a; awvalid<=1; wdata<=d; wvalid<=1;
            @(posedge clk); while(!(awready&&wready)) @(posedge clk);
            awvalid<=0; wvalid<=0; @(posedge clk);
        end
    endtask
    reg [31:0] rv;
    task rd(input [7:0] a);
        begin
            @(posedge clk); araddr<=a; arvalid<=1;
            @(posedge clk); while(!arready) @(posedge clk);
            arvalid<=0;
            while(!rvalid) @(posedge clk);
            rv = rdata; @(posedge clk);
        end
    endtask

{decl}
    reg [7:0]  vin  [0:NAM-1];
    reg [31:0] gold [0:NDEC-1];
    integer i, k, erros, vistas, fed, fh;
    reg [31:0] wdata_tmp, bdata_tmp, mdata_tmp;

    task carrega(input integer sel, input integer nw, input integer nb, input integer nm);
        begin
            wr(8'h20, sel | 32'h70);
            for (i = 0; i < nw; i = i + 1) begin
                case (sel)
{casew}
                endcase
                wr(8'h24, wdata_tmp);
            end
            for (i = 0; i < nb; i = i + 1) begin
                case (sel)
{caseb}
                endcase
                wr(8'h28, bdata_tmp);
            end
            for (i = 0; i < nm; i = i + 1) begin
                case (sel)
{casem}
                endcase
                wr(8'h40, mdata_tmp);
            end
        end
    endtask

    // alimenta pelo AXI-Stream, como o DMA faz: quatro amostras por palavra e
    // contrapressao de verdade
    initial begin
        fed = 0; svalid = 0;
        wait (rstn && alimenta);
        while (fed + 4 <= NAM) begin
            @(posedge clk);
            sdata  <= {{vin[fed+3], vin[fed+2], vin[fed+1], vin[fed]}};
            svalid <= 1'b1;
            @(posedge clk);
            while (!sready) @(posedge clk);
            fed = fed + 4;
        end
        @(posedge clk); svalid <= 1'b0;
    end

    reg alimenta = 0;
    initial begin
        vistas = 0; erros = 0;
{leituras}
        $readmemh("{d}/entrada.mem", vin);
        $readmemh("{d}/gold_classes.mem", gold);
        repeat (8) @(posedge clk); rstn <= 1'b1;
        repeat (4) @(posedge clk);
{cargas}
        wr(8'h00, 32'h2);
        alimenta <= 1'b1;
        wr(8'h00, 32'h1);

        fh = $fopen("{d}/vistos_bus.txt", "w");
        while (vistas < NDEC && fed < NAM) begin
            rd(8'h48);
            if (rv[8]) begin
                $fwrite(fh, "%0d", rv[3:0]); $fdisplay(fh, "");
                vistas = vistas + 1;
            end
        end
        repeat (20000) @(posedge clk);
        for (k = 0; k < 200; k = k + 1) begin
            rd(8'h48);
            if (rv[8]) begin
                $fwrite(fh, "%0d", rv[3:0]); $fdisplay(fh, "");
                vistas = vistas + 1;
            end
        end
        $fclose(fh);
        rd(8'h5C);
        $display("  decisoes pelo barramento: %0d, perdidas %0d", vistas, rv);
        $finish;
    end
    initial begin #900_000_000; $display("  FALHOU: timeout no barramento"); $finish; end
endmodule
`default_nettype wire
"""

def referencia(p, nam, rng, dest: Path):
    cams = p["camadas"]
    pesos, bias, mults = [], [], []
    x = rng.integers(-128, 128, size=(1, nam), dtype=np.int64)
    cur = x
    for c in cams:
        w = rng.integers(-127, 128, size=(c["nof"], c["nif"], c["k"]), dtype=np.int64)
        lim = c["nif"] * c["k"] * 64
        b = rng.integers(-lim, lim + 1, size=(c["nof"],), dtype=np.int64)
        modo = "avg" if c.get("pool_avg") else "max"
        acc = pool_int(relu_int(conv1d_int(cur, w, b)), pool=c["pool"], modo=modo)
        if p.get("rq_por_canal"):
            pico = np.maximum(np.rint(np.percentile(acc, 99.9, axis=1)).astype(np.int64), 1)
        else:
            pico = np.array([max(int(np.percentile(acc, 99.9)), 1)])
        m = np.clip(np.rint(127 * (1 << 16) / pico), 1, (1 << 17) - 1).astype(np.int64)
        cur = requantize(acc, m, 16).astype(np.int64)
        pesos.append(w); bias.append(b); mults.append(m)

    nch, npos_tot = cur.shape
    npos = cams[-1]["comp"] // cams[-1]["pool"]
    w_fc = rng.integers(-127, 128, size=(NCLS, nch), dtype=np.int64)
    b_fc = rng.integers(-(1 << 16), 1 << 16, size=(NCLS,), dtype=np.int64)

    csum = np.cumsum(np.pad(cur, ((0, 0), (1, 0))), axis=1)
    logits = []
    for pos in range(npos_tot):
        ini = max(0, pos - npos + 1)
        S = csum[:, pos + 1] - csum[:, ini]
        logits.append(w_fc @ S + b_fc)
    logits = np.stack(logits)

    dest.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cams, start=1):
        write_mem(pesos[i-1], 8, dest / f"pesos_conv{i}.mem")
        write_mem(bias[i-1], 32, dest / f"bias_conv{i}.mem")
        write_mem(mults[i-1], 18, dest / f"mult_conv{i}.mem")
    write_mem(w_fc, 8, dest / "pesos_fc.mem")
    write_mem(b_fc, 32, dest / "bias_fc.mem")
    write_mem(x.reshape(-1), 8, dest / "entrada.mem")
    write_mem(logits.reshape(-1), 32, dest / "gold_logits.mem")
    np.savetxt(dest / "gold_ref.txt", logits.astype(np.int64), fmt="%d")
    write_mem(np.argmax(logits, axis=1).astype(np.int64), 32,
              dest / "gold_classes.mem")
    np.savetxt(dest / "gold_cur.txt", cur.astype(np.int64), fmt="%d")
    from ferramentas.memorias import emite
    emite(p, dest, dest / "mem")
    return pesos + [w_fc], bias + [b_fc], mults, npos, npos_tot

def monta_bus(p, dest, nam, ndec, pesos, bias, mults):
    L = len(p["camadas"])
    decl, casew, caseb, casem, leituras, cargas = [], [], [], [], [], []
    for i in range(L + 1):
        nw, nb = pesos[i].size, bias[i].size
        nm = mults[i].size if i < L else 0
        nome = f"conv{i+1}" if i < L else "fc"
        decl.append(f"    reg [7:0]  w{i} [0:{nw-1}];")
        decl.append(f"    reg [31:0] b{i} [0:{nb-1}];")
        if nm:
            decl.append(f"    reg [17:0] m{i} [0:{nm-1}];")
        leituras.append(f'        $readmemh("{dest}/pesos_{nome}.mem", w{i});')
        leituras.append(f'        $readmemh("{dest}/bias_{nome}.mem", b{i});')
        if nm:
            leituras.append(f'        $readmemh("{dest}/mult_{nome}.mem", m{i});')
        casew.append(f"                    {i}: wdata_tmp = w{i}[i];")
        caseb.append(f"                    {i}: bdata_tmp = b{i}[i];")
        if nm:
            casem.append(f"                    {i}: mdata_tmp = m{i}[i];")
        cargas.append(f"        carrega({i}, {nw}, {nb}, {nm});")
    return TB_BUS.format(comp0=p["camadas"][0]["comp"], nam=nam, ndec=ndec,
                         ncls=NCLS, d=dest, decl="\n".join(decl),
                         casew="\n".join(casew), caseb="\n".join(caseb),
                         casem="\n".join(casem) or "                    default: ;",
                         leituras="\n".join(leituras), cargas="\n".join(cargas))

def monta(p, dest, nam, ndec, pula, pesos, bias, mults):
    L = len(p["camadas"])
    decl, casew, caseb, casem, leituras, cargas = [], [], [], [], [], []
    for i in range(L + 1):
        nw, nb = pesos[i].size, bias[i].size
        nm = mults[i].size if i < L else 0
        nome = f"conv{i+1}" if i < L else "fc"
        decl.append(f"    reg [7:0]  w{i} [0:{nw-1}];")
        decl.append(f"    reg [31:0] b{i} [0:{nb-1}];")
        if nm:
            decl.append(f"    reg [17:0] m{i} [0:{nm-1}];")
        leituras.append(f'        $readmemh("{dest}/pesos_{nome}.mem", w{i});')
        leituras.append(f'        $readmemh("{dest}/bias_{nome}.mem", b{i});')
        if nm:
            leituras.append(f'        $readmemh("{dest}/mult_{nome}.mem", m{i});')
        casew.append(f"                    4'd{i}: ld_w_data <= w{i}[i];")
        caseb.append(f"                    4'd{i}: ld_b_data <= b{i}[i];")
        if nm:
            casem.append(f"                        4'd{i}: ld_m_data <= m{i}[i];")
        cargas.append(f"        carrega(4'd{i}, {nw}, {nb}, {nm});")
    return TB.format(comp0=p["camadas"][0]["comp"], nam=nam, ndec=ndec, pula=pula,
                     lult=len(p["camadas"]), nrqu=p["camadas"][-1]["nrqu"],
                     ncls=NCLS, d=dest, decl="\n".join(decl),
                     casew="\n".join(casew), caseb="\n".join(caseb),
                     casem="\n".join(casem) or "                        default: ;",
                     leituras="\n".join(leituras), cargas="\n".join(cargas))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plano", type=Path, required=True)
    ap.add_argument("--guarda", type=Path, default=None,
                    help="guarda estimulo, referencia e saida vista para analise")
    ap.add_argument("--barramento", action="store_true",
                    help="prova tambem o caminho AXI: alimenta pelo stream e "
                         "colhe as decisoes pela fila, como a placa faz")
    ap.add_argument("--janelas", type=int, default=4,
                    help="comprimento do fluxo, em multiplos da janela")
    a = ap.parse_args()
    raiz = Path(__file__).resolve().parent.parent
    p = json.loads(a.plano.read_text())
    if not p.get("fluxo"):
        raise SystemExit(f"{a.plano} nao e' um plano de fluxo continuo "
                         f"(gere com ferramentas.gerador --fluxo)")
    rng = np.random.default_rng(zlib.crc32(p["nome"].encode()))
    nam = a.janelas * p["entrada"]
    print(f"=== {p['nome']}  ({p['dsp_total']} DSP, {p['ii']//p['entrada']} "
          f"ciclos por amostra)", flush=True)

    import contextlib
    ctx = (contextlib.nullcontext(str(a.guarda)) if a.guarda
           else tempfile.TemporaryDirectory(prefix="tfluxo_"))
    with ctx as td:
        dest = Path(td); dest.mkdir(parents=True, exist_ok=True)
        pesos, bias, mults, npos, ndec = referencia(p, nam, rng, dest)
        (dest / "tb.v").write_text(monta(p, dest, nam, ndec, npos, pesos, bias, mults))
        fontes = [str(x) for x in sorted((raiz / "rtl").glob("*.v"))
                  if x.name not in ("acelerador.v", "axi_acelerador.v")]
        vvp = dest / "t.vvp"
        r = subprocess.run(["iverilog", "-g2005-sv", "-s", "tb_fluxo", "-o", str(vvp),
                            *fontes, str(a.plano.parent / "acelerador_gen.v"),
                            str(dest / "tb.v")],
                           capture_output=True, text=True, timeout=900)
        if r.returncode:
            print("  FALHOU na compilacao:",
                  (r.stderr.strip().splitlines() or ["?"])[-1])
            return 1
        r = subprocess.run(["vvp", str(vvp)], capture_output=True, text=True,
                           timeout=7200, cwd=raiz)
        for l in r.stdout.splitlines():
            if l.startswith("  "):
                print(l)
        vf = dest / "vistos.txt"
        if not vf.exists():
            print("  FALHOU: o banco nao gravou decisao nenhuma")
            return 1
        vis = np.array([[int(x) for x in ln.split()]
                        for ln in vf.read_text().split("\n") if ln.strip()])
        ref = np.loadtxt(dest / "gold_ref.txt", dtype=np.int64).reshape(-1, NCLS)
        u = p["camadas"][-1]
        npg = u["pox"] // min(u["pox"], u["pool"])
        print(f"  decisao a cada {npg} posicoes (fim de grupo da ultima conv)")
        n = len(vis)
        casou = None
        for fase in range(npg):
            sub_ref = ref[fase::npg]
            for off in range(0, max(1, len(sub_ref) - n + 1)):
                if np.array_equal(vis, sub_ref[off:off+n]):
                    casou = (fase, off)
                    break
            if casou:
                ref = sub_ref
                break
        if casou is None:
            melhor, ruim = None, None
            for fase in range(npg):
                sr = ref[fase::npg]
                for o in range(0, max(1, len(sr) - n + 1)):
                    d = np.abs(vis - sr[o:o+n]).sum()
                    if ruim is None or d < ruim:
                        ruim, melhor = d, (fase, o, sr)
            fase, o, sr = melhor
            d = np.abs(vis - sr[o:o+n])
            print(f"  FALHOU: nenhuma fase/deslocamento casa. melhor "
                  f"fase={fase} off={o}, {int((d != 0).sum())} logits "
                  f"diferentes de {d.size}, pior diferenca {int(d.max())}")
            return 1
        print(f"  FLUXO BIT-EXATO  ({n} decisoes, fase {casou[0]} de {npg}, "
              f"{casou[1]} decisoes de enchimento do cano)")
        if not a.barramento:
            return 0

        (dest / "tb_bus.v").write_text(
            monta_bus(p, dest, nam, len(ref), pesos, bias, mults))
        fbus = [str(x) for x in sorted((raiz / "rtl").glob("*.v"))
                if x.name != "acelerador.v"]
        r = subprocess.run(["iverilog", "-g2005-sv", "-s", "tb_fbus", "-o",
                            str(dest / "b.vvp"), *fbus,
                            str(a.plano.parent / "acelerador_gen.v"),
                            str(dest / "tb_bus.v")],
                           capture_output=True, text=True, timeout=900)
        if r.returncode:
            print("  FALHOU na compilacao do barramento:",
                  (r.stderr.strip().splitlines() or ["?"])[-1])
            return 1
        r = subprocess.run(["vvp", str(dest / "b.vvp")], capture_output=True,
                           text=True, timeout=7200, cwd=raiz)
        for l in r.stdout.splitlines():
            if l.startswith("  "):
                print(l)
        vb = dest / "vistos_bus.txt"
        if not vb.exists():
            print("  FALHOU: o barramento nao entregou decisao nenhuma")
            return 1
        cls = np.array([int(x) for x in vb.read_text().split() if x.strip()])
        esp = np.argmax(ref, axis=1)
        m = min(len(cls), len(esp))
        if m == 0 or not np.array_equal(cls[:m], esp[:m]):
            dif = int((cls[:m] != esp[:m]).sum()) if m else -1
            print(f"  FALHOU: {dif} de {m} classes diferentes pelo barramento")
            return 1
        print(f"  BARRAMENTO BIT-EXATO  ({m} decisoes colhidas pela fila)")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
