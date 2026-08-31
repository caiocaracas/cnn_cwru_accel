// banco de teste da camada em FLUXO CONTINUO.

`timescale 1ns/1ps
`default_nettype none

module tb_camada_fluxo #(
    parameter NOME     = "conv",
    parameter NIF      = 1,
    parameter NOF      = 8,
    parameter K        = 5,
    parameter LEN      = 1024,
    parameter POOL     = 2,
    parameter POOL_AVG = 0,
    parameter M_FILE   = "",
    parameter RQ_POR_CANAL = 0,
    parameter SHIFT    = 16,
    parameter RQ_W     = 18,
    parameter ACC_W    = 32,
    parameter POX      = 1,
    parameter POF      = NOF,
    parameter NRQU     = 1,
    parameter PK       = 1,
    parameter NBLOCO   = 3,
    parameter LIMITE_NS = 400000000,
    parameter TESTA_BOLHA = 1,
    parameter W_FILE    = "",
    parameter B_FILE    = "",
    parameter IN_FILE   = "",
    parameter GOLD_FILE = ""
);
    localparam OLEN    = LEN/POOL;
    localparam POOL_IN = (POX < POOL) ? POX : POOL;
    localparam NQ      = POX / POOL_IN;
    localparam CH_W = $clog2(NOF);
    localparam PO_W = $clog2(OLEN);
    localparam NSAI = OLEN*NOF;
    localparam NAM  = NBLOCO*LEN;

    reg clk = 1'b0;  always #5 clk = ~clk;
    reg rst_n = 1'b0;
    reg ld_w_en = 0, ld_w_valid = 0;  reg signed [7:0]  ld_w_data = 0;
    reg ld_b_en = 0, ld_b_valid = 0;  reg signed [31:0] ld_b_data = 0;
    reg ld_m_en = 0, ld_m_valid = 0;  reg signed [17:0] ld_m_data = 0;
    reg start = 0, alimenta = 0;

    wire busy, done, in_ready, out_valid, trunc_err;
    wire [NRQU*8-1:0]  out_data;
    wire [CH_W-1:0]    out_ch;
    wire [PO_W-1:0]    out_pos;
    wire [POF*POX*ACC_W-1:0] dbg_acc;

    reg  [7:0]  wfile [0:NOF*NIF*K-1];
    reg  [31:0] bfile [0:NOF-1];
    reg  [7:0]  vin   [0:NAM*NIF-1];
    reg  [7:0]  gold  [0:NBLOCO*NSAI-1];
    reg  [7:0]  got   [0:NBLOCO*NSAI-1];

    reg  gap = 1'b0, gap_ok = 1'b1;
    reg  trava = 1'b0, trava_ok = 1'b1;
    always @(posedge clk) gap_ok   <= (!gap)   || ($random % 3 != 0);
    always @(posedge clk) trava_ok <= (!trava) || ($random % 4 != 0);

    integer fed = 0;
    wire in_valid = alimenta && (fed < NAM) && gap_ok;

    reg [NIF*8-1:0] in_data;
    integer q;
    always @* begin
        for (q = 0; q < NIF; q = q + 1)
            in_data[q*8 +: 8] = vin[fed*NIF + q];
    end

    localparam NMULT = (RQ_POR_CANAL != 0) ? NOF : 1;
    reg [17:0] mfile [0:NMULT-1];

    conv1d_engine #(
        .NIF(NIF), .NOF(NOF), .K(K), .LEN(LEN), .POOL(POOL), .POOL_AVG(POOL_AVG),
        .POX(POX), .POF(POF), .PK(PK), .NRQU(NRQU), .DATA_W(8), .ACC_W(ACC_W),
        .RQ_W(RQ_W), .MULT_W(18), .SHIFT(SHIFT),
        .RQ_POR_CANAL(RQ_POR_CANAL), .FLUXO(1), .WR_PESO(1)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .ld_w_en(ld_w_en), .ld_w_valid(ld_w_valid), .ld_w_data(ld_w_data),
        .ld_b_en(ld_b_en), .ld_b_valid(ld_b_valid), .ld_b_data(ld_b_data),
        .ld_m_en(ld_m_en), .ld_m_valid(ld_m_valid), .ld_m_data(ld_m_data),
        .start(start), .busy(busy), .done(done), .trunc_err(trunc_err),
        .out_ready(trava_ok),
        .in_valid(in_valid), .in_data(in_data), .in_ready(in_ready),
        .out_valid(out_valid), .out_data(out_data),
        .out_ch(out_ch), .out_pos(out_pos),
        .dbg_acc(dbg_acc)
    );

    always @(posedge clk) if (in_ready && in_valid) fed <= fed + 1;

    integer gi, bloco = 0, pos_ant = -1, pos_i;
    always @(posedge clk)
        if (out_valid) begin
            pos_i = out_pos;
            if (pos_ant >= 0 && (pos_ant - pos_i) >= NQ) bloco = bloco + 1;
            pos_ant = pos_i;
            for (gi = 0; gi < NRQU; gi = gi + 1)
                if (bloco < NBLOCO)
                    got[bloco*NSAI + (out_ch + gi)*OLEN + out_pos] <= out_data[gi*8 +: 8];
        end

    reg mede = 1'b1;
    integer ciclo = 0, ult_grp = -1, ult_ciclo = 0;
    integer dt_min = 1000000000, dt_max = 0, n_dt = 0, grp_i;
    always @(posedge clk) begin
        ciclo <= ciclo + 1;
        if (mede && out_valid) begin
            grp_i = out_pos / NQ;
            if (grp_i !== ult_grp) begin
                if (ult_grp >= 0) begin
                    if (ciclo - ult_ciclo < dt_min) dt_min = ciclo - ult_ciclo;
                    if (ciclo - ult_ciclo > dt_max) dt_max = ciclo - ult_ciclo;
                    n_dt = n_dt + 1;
                end
                ult_grp   = grp_i;
                ult_ciclo = ciclo;
            end
        end
    end

    integer i, b, idx, erros, erros_b, esperado, obtido;

    task carrega;
        begin
            @(posedge clk); ld_w_en <= 1'b1;
            @(posedge clk); ld_w_en <= 1'b0;
            for (i = 0; i < NOF*NIF*K; i = i + 1) begin
                @(posedge clk); ld_w_valid <= 1'b1; ld_w_data <= wfile[i];
            end
            @(posedge clk); ld_w_valid <= 1'b0;
            @(posedge clk); ld_b_en <= 1'b1;
            @(posedge clk); ld_b_en <= 1'b0;
            for (i = 0; i < NOF; i = i + 1) begin
                @(posedge clk); ld_b_valid <= 1'b1; ld_b_data <= bfile[i];
            end
            @(posedge clk); ld_b_valid <= 1'b0;
            @(posedge clk); ld_m_en <= 1'b1;
            @(posedge clk); ld_m_en <= 1'b0;
            for (i = 0; i < NMULT; i = i + 1) begin
                @(posedge clk); ld_m_valid <= 1'b1; ld_m_data <= mfile[i];
            end
            @(posedge clk); ld_m_valid <= 1'b0;
        end
    endtask

    initial begin
        $readmemh(W_FILE, wfile);  $readmemh(B_FILE, bfile);
        $readmemh(M_FILE, mfile);  $readmemh(IN_FILE, vin);
        $readmemh(GOLD_FILE, gold);

        repeat (4) @(posedge clk);
        rst_n <= 1'b1;
        repeat (2) @(posedge clk);
        carrega;

        @(posedge clk); start <= 1'b1;
        @(posedge clk); start <= 1'b0;
        @(posedge clk); alimenta <= 1'b1;
        // o primeiro bloco corre limpo: e' nele que se mede o intervalo entre
        wait (fed >= LEN);
        mede = 1'b0;
        if (TESTA_BOLHA) begin gap <= 1'b1; trava <= 1'b1; end

        // deixa o fluxo correr ate' o fim do sinal e drenar o cano
        wait (fed >= NAM);
        repeat (8*(LEN/POX) + 256) @(posedge clk);
        alimenta <= 1'b0;

        erros = 0;
        // bloco 0 carrega o aquecimento do cano; o ultimo depende de amostras
        // que nunca chegaram. Confere-se o miolo, que e' o regime permanente
        for (b = 1; b < NBLOCO-1; b = b + 1) begin
            erros_b = 0;
            for (idx = 0; idx < NSAI; idx = idx + 1) begin
                esperado = gold[b*NSAI + idx];
                obtido   = got[b*NSAI + idx];
                if (esperado !== obtido) begin
                    if (erros_b < 4)
                        $display("  bloco %0d ch %0d pos %0d: obtido %02h, gold %02h",
                                 b, idx/OLEN, idx%OLEN, obtido, esperado);
                    erros_b = erros_b + 1;
                end
            end
            erros = erros + erros_b;
        end
        $display("  blocos completos observados: %0d", bloco);

        $display("=== %0s (fluxo): %0d blocos de %0d valores ===", NOME, NBLOCO-1, NSAI);
        $display("  ciclos por grupo (%0d amostras): %0d..%0d   jitter %0d   (%0d medidas)",
                 POX, dt_min, dt_max, dt_max - dt_min, n_dt);
        $display("  ciclos por amostra de entrada: %0d   derivado: %0d",
                 dt_min / POX, (NOF/POF)*((NIF*K)/PK)/POX);
        $display("  amostras consumidas : %0d de %0d", fed, NAM);
        if (trunc_err !== 1'b0) begin
            $display("  FALHOU: trunc_err ativo, RQ_W=%0d insuficiente", RQ_W);
            erros = erros + 1;
        end
        if (erros == 0) $display("  FLUXO BIT-EXATO");
        else            $display("  FALHOU: %0d divergencias", erros);
        $display("");
        $finish;
    end

    initial begin
        #LIMITE_NS;
        $display("  FALHOU: timeout apos %0d ns simulados", LIMITE_NS);
        $finish;
    end
endmodule

`default_nettype wire
