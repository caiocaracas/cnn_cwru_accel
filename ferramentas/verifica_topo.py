"""prova a rede montada inteira contra a referencia em inteiro."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np

from modelo.quantiza import (conv1d_int, relu_int, pool_int, requantize,
                             write_mem)

TB = """`timescale 1ns/1ps
`default_nettype none

module tb_topo;
    localparam REINICIA = {reinicia};
    localparam LEN  = {comp0};
    localparam NCLS = {ncls};
    localparam NCAM = 8;

    reg clk = 1'b0;  always #5 clk = ~clk;
    reg                rst_n = 1'b0;
    reg  [3:0]         ld_sel = 0;
    reg                ld_w_en = 0, ld_w_valid = 0;
    reg  signed [7:0]  ld_w_data = 0;
    reg                ld_b_en = 0, ld_b_valid = 0;
    reg  signed [31:0] ld_b_data = 0;
    reg                start = 0, in_valid = 0, alimenta = 0;
    reg  [15:0]        vec = 0, fed = 0;
    reg                ld_m_en = 0, ld_m_valid = 0;
    reg  signed [17:0] ld_m_data = 0;

    wire               busy, done, in_ready, classe_valid;
    wire [NCAM-1:0]    trunc_err;
    wire [NCLS*32-1:0] logits;
    wire [3:0]         classe;

    // o vetor que a camada final consome, guardado para o passo de atualizacao
    reg                ativ_rein = 0, ativ_av = 0;
    wire [31:0]        ativ_data;

    // a janela entra ja' normalizada e em int8, como sai do preparo dos dados
    wire signed [7:0] in_data = vin[vec*LEN + fed];

    acelerador_gen #(.NCAM(NCAM), .NCLS(NCLS),
                     .MEMDIR("{d}/mem")) dut (
        .clk(clk), .rst_n(rst_n), .ld_sel(ld_sel),
        .ld_w_en(ld_w_en), .ld_w_valid(ld_w_valid), .ld_w_data(ld_w_data),
        .ld_b_en(ld_b_en), .ld_b_valid(ld_b_valid), .ld_b_data(ld_b_data),
        .ld_m_en(ld_m_en), .ld_m_valid(ld_m_valid), .ld_m_data(ld_m_data),
        .start(start), .busy(busy), .done(done), .trunc_err(trunc_err),
        .in_valid(in_valid), .in_data(in_data), .in_ready(in_ready),
        .logits(logits), .classe(classe), .classe_valid(classe_valid),
        .ativ_reinicia(ativ_rein), .ativ_avanca(ativ_av), .ativ_data(ativ_data)
    );

{decl}
    reg [7:0]  vin  [0:{nvec}*LEN-1];
    reg [31:0] gold [0:{nvec}*NCLS-1];
    reg [7:0]  g_ativ  [0:{nvec}*{nflat}-1];
    integer    cls_gold [0:{nvec}-1];

    integer i, j, v, erros, erros_a, ciclos, esp, obt, acertos, lat_min, lat_max;

    always @(posedge clk) if (busy) ciclos = ciclos + 1;
    always @(posedge clk) if (in_ready && in_valid) fed <= fed + 1'b1;
    always @(posedge clk) in_valid <= (alimenta && fed < LEN);

    task carrega(input [3:0] sel, input integer nw, input integer nb,
                 input integer nm);
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

    integer fh, r;
    initial begin
{leituras}
        $readmemh("{d}/entrada.mem", vin);
        $readmemh("{d}/gold_logits.mem", gold);
        $readmemh("{d}/gold_ativ.mem", g_ativ);
        fh = $fopen("{d}/gold_classes.txt", "r");
        for (i = 0; i < {nvec}; i = i + 1) r = $fscanf(fh, "%d\\n", cls_gold[i]);
        $fclose(fh);

        repeat (4) @(posedge clk);
        rst_n <= 1'b1;
        repeat (2) @(posedge clk);

{cargas}
        erros = 0; erros_a = 0; acertos = 0;
        for (v = 0; v < {nvec}; v = v + 1) begin
            vec = v[15:0]; fed = 0; ciclos = 0;
            // a placa so' reinicia uma vez, antes da campanha: com REINICIA=0
            // o teste exercita o fluxo continuo, que e' o modo real
            if (v == 0 || REINICIA) begin
                @(posedge clk); start <= 1'b1;
                @(posedge clk); start <= 1'b0;
            end
            @(posedge clk); alimenta <= 1'b1;
            wait (fed == LEN);
            @(posedge clk); alimenta <= 1'b0;
            wait (done === 1'b1);
            @(posedge clk);
            for (i = 0; i < NCLS; i = i + 1) begin
                esp = $signed(gold[v*NCLS + i]);
                obt = $signed(logits[i*32 +: 32]);
                if (esp !== obt) begin
                    if (erros < 4)
                        $display("  vetor %0d logit %0d: obtido %0d, esperado %0d",
                                 v, i, obt, esp);
                    erros = erros + 1;
                end
            end
            // o que a inferencia deixou pronto para o aprendizado tem de ser
            // exatamente o vetor da referencia, e nao algo parecido
            @(posedge clk); ativ_rein <= 1'b1;
            @(posedge clk); ativ_rein <= 1'b0;
            @(posedge clk);
            for (i = 0; i < {nflat}/4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    esp = $signed(g_ativ[v*{nflat} + i*4 + j]);
                    obt = $signed(ativ_data[j*8 +: 8]);
                    if (esp !== obt) begin
                        if (erros_a < 4)
                            $display("  vetor %0d ativacao %0d: obtido %0d, esperado %0d",
                                     v, i*4 + j, obt, esp);
                        erros_a = erros_a + 1;
                    end
                end
                @(posedge clk); ativ_av <= 1'b1;
                @(posedge clk); ativ_av <= 1'b0;
                // o ponteiro so' vale depois da regiao nao-bloqueante: ler no
                // mesmo flanco devolve a palavra anterior
                @(posedge clk);
            end

            if (classe === cls_gold[v][3:0]) acertos = acertos + 1;
            if (v == 0) begin lat_min = ciclos; lat_max = ciclos; end
            else begin
                if (ciclos < lat_min) lat_min = ciclos;
                if (ciclos > lat_max) lat_max = ciclos;
            end
        end

        $display("  ciclos %0d..%0d, jitter %0d, trunc %b",
                 lat_min, lat_max, lat_max - lat_min, trunc_err);
        if (erros == 0 && erros_a == 0 && acertos == {nvec})
            $display("  BIT-EXATO  (logits, classes e a ativacao da camada final)");
        else
            $display("  FALHOU: %0d logits, %0d/%0d classes, %0d ativacoes",
                     erros, acertos, {nvec}, erros_a);
        $finish;
    end

    initial begin
        #400_000_000;
        $display("  FALHOU: timeout");
        $finish;
    end

endmodule

`default_nettype wire
"""

def gera_estimulo(p: dict, nvec: int, rng) -> dict:
    camadas = p["camadas"]
    ncls = 4

    pesos, bias = [], []
    for c in camadas:
        pesos.append(rng.integers(-127, 128, size=(c["nof"], c["nif"], c["k"]),
                                  dtype=np.int64))
        lim = c["nif"] * c["k"] * 64
        bias.append(rng.integers(-lim, lim + 1, size=(c["nof"],), dtype=np.int64))
    nflat = p["fc_nflat"]
    w_fc = rng.integers(-127, 128, size=(ncls, nflat), dtype=np.int64)
    b_fc = rng.integers(-(1 << 16), 1 << 16, size=(ncls,), dtype=np.int64)

    entradas = [rng.integers(-128, 128, size=(1, camadas[0]["comp"]),
                             dtype=np.int64)
                for _ in range(nvec)]

    mults, x_cam = [], list(entradas)
    for i, c in enumerate(camadas):
        accs = []
        for x in x_cam:
            modo = "avg" if c.get("pool_avg") else "max"
            accs.append(pool_int(relu_int(conv1d_int(x, pesos[i], bias[i])),
                                 pool=c["pool"], modo=modo))
        empilhado = np.concatenate(accs, axis=1)
        if p.get("rq_por_canal"):
            pico = np.percentile(empilhado, 99.9, axis=1)
            pico = np.maximum(np.rint(pico).astype(np.int64), 1)
        else:
            pico = np.array([max(int(np.percentile(empilhado, 99.9)), 1)])
        m = np.clip(np.rint(127 * (1 << 16) / pico), 1,
                    (1 << 17) - 1).astype(np.int64)
        mults.append(m)
        x_cam = [requantize(a, m, 16).astype(np.int64) for a in accs]

    gap = p.get("head") == "gap"
    logits, classes = [], []
    for x in x_cam:
        plano = x.sum(axis=1) if gap else x.reshape(-1)
        lg = np.array([int(np.sum(plano * w_fc[o])) + int(b_fc[o])
                       for o in range(ncls)], dtype=np.int64)
        logits.append(lg)
        classes.append(int(np.argmax(lg)))

    return {"pesos": pesos + [w_fc], "bias": bias + [b_fc], "mults": mults,
            "entradas": entradas,
            "ativ": [x.reshape(-1) for x in x_cam],
            "logits": logits, "classes": classes, "ncls": ncls}

def referencia(p: dict, nvec: int, rng, dest: Path) -> list[int]:
    from ferramentas.memorias import emite

    dest.mkdir(parents=True, exist_ok=True)
    e = gera_estimulo(p, nvec, rng)
    for i, w in enumerate(e["pesos"]):
        write_mem(w, 8, dest / f"pesos_L{i}.mem")
        write_mem(e["bias"][i], 32, dest / f"bias_L{i}.mem")

    for i, c in enumerate(p["camadas"], start=1):
        write_mem(e["pesos"][i - 1], 8, dest / f"pesos_conv{i}.mem")
        write_mem(e["bias"][i - 1], 32, dest / f"bias_conv{i}.mem")
        write_mem(e["mults"][i - 1], 18, dest / f"mult_conv{i}.mem")
    write_mem(e["pesos"][-1], 8, dest / "pesos_fc.mem")
    write_mem(e["bias"][-1], 32, dest / "bias_fc.mem")
    emite(p, dest, dest / "mem")
    write_mem(np.concatenate([x.reshape(-1) for x in e["entradas"]]), 8,
              dest / "entrada.mem")
    write_mem(np.concatenate(e["logits"]), 32, dest / "gold_logits.mem")
    write_mem(np.concatenate(e["ativ"]), 8, dest / "gold_ativ.mem")
    (dest / "gold_classes.txt").write_text(
        "\n".join(str(c) for c in e["classes"]) + "\n")
    return e["mults"]

TB_BUS = """`timescale 1ns/1ps
`default_nettype none

module tb_bus;
    localparam REINICIA = {reinicia};
    localparam LEN  = {comp0};
    localparam NCLS = {ncls};

    reg clk = 0;  always #5 clk = ~clk;
    reg rstn = 0;

    reg  [7:0]  awaddr = 0;  reg awvalid = 0;  wire awready;
    reg  [31:0] wdata  = 0;  reg wvalid  = 0;  wire wready;
    wire [1:0]  bresp;       wire bvalid;      reg  bready = 1;
    reg  [7:0]  araddr = 0;  reg arvalid = 0;  wire arready;
    wire [31:0] rdata;       wire [1:0] rresp; wire rvalid;  reg rready = 1;

    axi_acelerador #(.LEN(LEN), .NCLS(NCLS)) dut (
        .s_axi_aclk(clk), .s_axi_aresetn(rstn),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid),
        .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid),
        .s_axi_rready(rready)
    );

    task wr(input [7:0] a, input [31:0] d);
        begin
            @(posedge clk);
            awaddr <= a; awvalid <= 1'b1; wdata <= d; wvalid <= 1'b1;
            @(posedge clk);
            while (!(awready && wready)) @(posedge clk);
            awvalid <= 1'b0; wvalid <= 1'b0;
            @(posedge clk);
        end
    endtask

    reg [31:0] rv;
    task rd(input [7:0] a);
        begin
            @(posedge clk);
            araddr <= a; arvalid <= 1'b1;
            @(posedge clk);
            while (!arready) @(posedge clk);
            arvalid <= 1'b0;
            while (!rvalid) @(posedge clk);
            rv = rdata;
            @(posedge clk);
        end
    endtask

{decl}
    reg [7:0]  vin  [0:{nvec}*LEN-1];
    reg [31:0] gold [0:{nvec}*NCLS-1];
    reg [7:0]  g_ativ  [0:{nvec}*{nflat}-1];
    integer    cls_gold [0:{nvec}-1];

    integer i, v, s, erros, fh, r, t0, t_carga, t_env, t_calc;
    integer ciclos_bus = 0;
    always @(posedge clk) ciclos_bus <= ciclos_bus + 1;
    reg [31:0] wdata_tmp, bdata_tmp, mdata_tmp;
    // cao de guarda: travar calado esconde defeito, e ja' escondeu
    localparam GUARDA = 200000;
    integer espera;

    // carrega uma camada na ordem exata que a aplicacao do ARM usa
    task carrega(input integer sel, input integer nw, input integer nb,
                 input integer nm);
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

    initial begin
{leituras}
        $readmemh("{d}/entrada.mem", vin);
        $readmemh("{d}/gold_logits.mem", gold);
        $readmemh("{d}/gold_ativ.mem", g_ativ);
        fh = $fopen("{d}/gold_classes.txt", "r");
        for (i = 0; i < {nvec}; i = i + 1) r = $fscanf(fh, "%d\\n", cls_gold[i]);
        $fclose(fh);

        repeat (8) @(posedge clk);
        rstn <= 1'b1;
        repeat (4) @(posedge clk);

        t0 = 0;
{cargas}
        erros = 0;
        for (v = 0; v < {nvec}; v = v + 1) begin
            if (v == 0 || REINICIA) wr(8'h00, 32'h2);
            // o registrador 0x30 leva QUATRO amostras de 8 bits por escrita:
            // a janela ja' vem em int8 e o barramento e' de 32 bits
            t0 = ciclos_bus;
            for (i = 0; i < LEN; i = i + 4)
                wr(8'h30, {{vin[v*LEN+i+3], vin[v*LEN+i+2],
                            vin[v*LEN+i+1], vin[v*LEN+i]}});
            t_env = ciclos_bus - t0;
            // a janela ja' esta' inteira na fila: a partida vale de imediato
            wr(8'h00, 32'h1);

            espera = 0;
            rd(8'h04);
            while (!rv[1] && espera < GUARDA) begin
                espera = espera + 1;
                rd(8'h04);
            end
            if (!rv[1]) begin
                $display("  FALHOU: done nao chegou em %0d leituras", GUARDA);
                $finish;
            end

            for (i = 0; i < NCLS; i = i + 1) begin
                rd(8'h10 + i*4);
                if ($signed(rv) !== $signed(gold[v*NCLS + i])) begin
                    if (erros < 4)
                        $display("  vetor %0d logit %0d: axi %0d, gold %0d",
                                 v, i, $signed(rv), $signed(gold[v*NCLS+i]));
                    erros = erros + 1;
                end
            end
            rd(8'h08);
            if (rv[3:0] !== cls_gold[v][3:0]) begin
                $display("  vetor %0d classe: axi %0d, gold %0d",
                         v, rv, cls_gold[v]);
                erros = erros + 1;
            end
            rd(8'h04);
            if (rv[15:8] !== 0) begin
                $display("  vetor %0d: estouro de acumulador %b", v, rv[15:8]);
                erros = erros + 1;
            end
        end

        $display("  envio de %0d amostras: %0d ciclos de barramento", LEN, t_env);
        if (erros == 0) $display("  BARRAMENTO BIT-EXATO");
        else            $display("  FALHOU: %0d divergencias pelo barramento", erros);
        $finish;
    end

    initial begin
        #900_000_000;
        $display("  FALHOU: timeout no barramento");
        $finish;
    end

endmodule

`default_nettype wire
"""

def monta_tb_bus(p: dict, nvec: int, mults: list[int], dest: Path,
                 reinicia: bool = True) -> str:
    camadas = p["camadas"]
    nl = len(camadas) + 1
    tam_w = [c["nof"] * c["nif"] * c["k"] for c in camadas] + [4 * p["fc_nflat"]]
    tam_b = [c["nof"] for c in camadas] + [4]

    n_m = (lambda c: c["nof"]) if p.get("rq_por_canal") else (lambda c: 1)
    tam_m = [n_m(c) for c in camadas] + [0]
    decl = "\n".join(
        f"    reg [7:0]  w{i} [0:{tam_w[i]-1}];\n"
        f"    reg [31:0] b{i} [0:{tam_b[i]-1}];"
        + (f"\n    reg [17:0] m{i} [0:{tam_m[i]-1}];" if tam_m[i] else "")
        for i in range(nl))
    casew = "\n".join(f"                    {i}: wdata_tmp = w{i}[i];"
                      for i in range(nl))
    caseb = "\n".join(f"                    {i}: bdata_tmp = b{i}[i];"
                      for i in range(nl))
    leituras = "\n".join(
        f'        $readmemh("{dest}/pesos_L{i}.mem", w{i});\n'
        f'        $readmemh("{dest}/bias_L{i}.mem", b{i});'
        + (f'\n        $readmemh("{dest}/mult_conv{i+1}.mem", m{i});'
           if tam_m[i] else "")
        for i in range(nl))
    casem = "\n".join(f"                    {i}: mdata_tmp = m{i}[i];"
                      for i in range(nl) if tam_m[i])
    cargas = "\n".join(
        f"        carrega({i}, {tam_w[i]}, {tam_b[i]}, {tam_m[i]});"
        for i in range(nl))
    return TB_BUS.format(comp0=camadas[0]["comp"], ncls=4, nvec=nvec, d=dest,
                         nflat=p["fc_nstream"],
                         decl=decl, casew=casew, caseb=caseb, casem=casem,
                         leituras=leituras, cargas=cargas,
                         reinicia=1 if reinicia else 0)

def monta_tb(p: dict, nvec: int, mults: list[int], dest: Path,
             reinicia: bool = True) -> str:
    camadas = p["camadas"]
    nl = len(camadas) + 1
    tam_w = [c["nof"] * c["nif"] * c["k"] for c in camadas] + [4 * p["fc_nflat"]]
    tam_b = [c["nof"] for c in camadas] + [4]

    n_m = (lambda c: c["nof"]) if p.get("rq_por_canal") else (lambda c: 1)
    tam_m = [n_m(c) for c in camadas] + [0]

    decl = "\n".join(
        f"    reg [7:0]  w{i} [0:{tam_w[i]-1}];\n"
        f"    reg [31:0] b{i} [0:{tam_b[i]-1}];"
        + (f"\n    reg [17:0] m{i} [0:{tam_m[i]-1}];" if tam_m[i] else "")
        for i in range(nl))
    casew = "\n".join(f"                    4'd{i}: ld_w_data <= w{i}[i];"
                      for i in range(nl))
    caseb = "\n".join(f"                    4'd{i}: ld_b_data <= b{i}[i];"
                      for i in range(nl))
    casem = "\n".join(f"                        4'd{i}: ld_m_data <= m{i}[i];"
                      for i in range(nl) if tam_m[i])
    leituras = "\n".join(
        f'        $readmemh("{dest}/pesos_L{i}.mem", w{i});\n'
        f'        $readmemh("{dest}/bias_L{i}.mem", b{i});'
        + (f'\n        $readmemh("{dest}/mult_conv{i+1}.mem", m{i});'
           if tam_m[i] else "")
        for i in range(nl))
    cargas = "\n".join(
        f"        carrega(4'd{i}, {tam_w[i]}, {tam_b[i]}, {tam_m[i]});"
        for i in range(nl))
    return TB.format(comp0=camadas[0]["comp"], ncls=4, nvec=nvec, d=dest,
                     nflat=p["fc_nstream"],
                     decl=decl, casew=casew, caseb=caseb, casem=casem,
                     leituras=leituras, cargas=cargas,
                     reinicia=1 if reinicia else 0)

def verifica(plano: Path, nvec: int, raiz: Path, barramento: bool = False,
             reinicia: bool = True) -> bool:
    p = json.loads(plano.read_text())
    rng = np.random.default_rng(zlib.crc32(p["nome"].encode()))
    print(f"=== {p['nome']}  ({p['dsp_total']} DSP, II {p['ii']})", flush=True)

    with tempfile.TemporaryDirectory(prefix="topo_") as td:
        dest = Path(td)
        mults = referencia(p, nvec, rng, dest)
        (dest / "tb_topo.v").write_text(
            monta_tb(p, nvec, mults, dest, reinicia))
        gen = plano.parent / "acelerador_gen.v"
        if not gen.exists():
            raise SystemExit(f"sem acelerador_gen.v ao lado de {plano}; "
                             f"rode ferramentas.gerador para este plano")
        fontes = [str(x) for x in sorted((raiz / "rtl").glob("*.v"))
                  if x.name not in ("acelerador.v", "axi_acelerador.v")]
        vvp = dest / "topo.vvp"
        r = subprocess.run(["iverilog", "-g2005-sv", "-s", "tb_topo", "-o", str(vvp),
                            *fontes, str(gen), str(dest / "tb_topo.v")],
                           capture_output=True, text=True, timeout=600)
        if r.returncode:
            print("  FALHOU na compilacao:",
                  (r.stderr.strip().splitlines() or ["?"])[-1])
            return False
        r = subprocess.run(["vvp", str(vvp)], capture_output=True, text=True,
                           timeout=21600, cwd=raiz)
        for linha in r.stdout.splitlines():
            if linha.startswith("  "):
                print(linha)
        ok = "BIT-EXATO" in r.stdout
        if ok and barramento:
            ok = pelo_barramento(p, nvec, mults, dest, raiz, gen, reinicia)
        return ok

def pelo_barramento(p: dict, nvec: int, mults: list[int], dest: Path,
                    raiz: Path, gen: Path, reinicia: bool = True) -> bool:
    (dest / "tb_bus.v").write_text(
        monta_tb_bus(p, nvec, mults, dest, reinicia))
    fontes = [str(x) for x in sorted((raiz / "rtl").glob("*.v"))
              if x.name != "acelerador.v"]
    vvp = dest / "bus.vvp"
    r = subprocess.run(["iverilog", "-g2005-sv", "-s", "tb_bus", "-o", str(vvp),
                        *fontes, str(gen), str(dest / "tb_bus.v")],
                       capture_output=True, text=True, timeout=900)
    if r.returncode:
        print("  FALHOU na compilacao do caminho AXI:",
              (r.stderr.strip().splitlines() or ["?"])[-1])
        return False
    r = subprocess.run(["vvp", str(vvp)], capture_output=True, text=True,
                       timeout=7200, cwd=raiz)
    for linha in r.stdout.splitlines():
        if linha.startswith("  "):
            print(linha)
    return "BARRAMENTO BIT-EXATO" in r.stdout

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plano", type=Path)
    ap.add_argument("--gen", type=Path, default=Path("results/gen"))
    ap.add_argument("--nvec", type=int, default=2)
    ap.add_argument("--barramento", action="store_true",
                    help="prova tambem o caminho AXI, com os pesos reais")
    ap.add_argument("--continuo", action="store_true",
                    help="nao reinicia entre janelas, como a placa faz. O modo "
                         "padrao reinicia a cada janela e por isso nao "
                         "exercita o fluxo continuo, que e' o real")
    a = ap.parse_args()
    raiz = Path(__file__).resolve().parent.parent

    planos = ([a.plano] if a.plano else
              [q for q in sorted(a.gen.glob("*/plano.json"))
               if json.loads(q.read_text())["cabe"]])
    n_ok = sum(verifica(q, a.nvec, raiz, a.barramento, not a.continuo)
               for q in planos)
    alvo = "topo e barramento" if a.barramento else "topos gerados"
    print(f"\n{n_ok}/{len(planos)} {alvo} bit-exatos")
    return 0 if n_ok == len(planos) else 1

if __name__ == "__main__":
    raise SystemExit(main())
