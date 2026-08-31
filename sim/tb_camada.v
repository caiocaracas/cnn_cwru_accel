// banco de teste de uma camada de convolucao.

`timescale 1ns/1ps
`default_nettype none

module tb_camada #(
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
    parameter ACC_BITS = 18,
    parameter RQ_W     = 18,
    parameter ACC_W    = 32,
    parameter POX      = 1,
    parameter POF      = NOF,
    parameter NRQU     = 1,
    parameter PK       = 1,
    parameter NVEC     = 20,
    parameter NVEC_ARQ = 20,
    parameter TESTA_BOLHA = 1,
    parameter W_FILE    = "",
    parameter B_FILE    = "",
    parameter IN_FILE   = "",
    parameter GOLD_FILE = ""
);

    localparam OLEN   = LEN/POOL;
    localparam CH_W   = $clog2(NOF);
    localparam PO_W   = $clog2(OLEN);
    localparam NSAI   = OLEN*NOF;

    localparam JAN     = POX + K - 1;
    localparam POOL_IN = (POX < POOL) ? POX : POOL;
    localparam NQ      = POX / POOL_IN;
    localparam NRQ     = POF * NQ;

    localparam OG     = NOF/POF;

    localparam PIPE_END = 1;
    localparam CICLOS = (JAN + 1) + (LEN/POX)*OG*(NIF*K/PK)
                        + (4 + PIPE_END) + NRQ/NRQU + 4 + 1;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg                     rst_n = 1'b0;
    reg                     ld_w_en = 1'b0, ld_w_valid = 1'b0;
    reg  signed [7:0]       ld_w_data = 0;
    reg                     ld_b_en = 1'b0, ld_b_valid = 1'b0;
    reg  signed [31:0]      ld_b_data = 0;
    reg                     start = 1'b0;
    reg                     feeding = 1'b0;
    reg  [15:0]             vec = 0;
    reg  [15:0]             fed = 0;

    wire                    busy, done, in_ready, out_valid, trunc_err;
    wire [NRQU*8-1:0]       out_data;
    wire [CH_W-1:0]         out_ch;
    wire [PO_W-1:0]         out_pos;
    wire [POF*POX*ACC_W-1:0] dbg_acc;

    reg  [7:0]  wfile [0:NOF*NIF*K-1];
    reg  [31:0] bfile [0:NOF-1];
    reg  [7:0]  vin   [0:NVEC_ARQ*NIF*LEN-1];
    reg  [7:0]  gold  [0:NVEC_ARQ*NSAI-1];
    reg  [7:0]  got   [0:NSAI-1];

    reg  gap = 1'b0, gap_ok = 1'b1;
    always @(posedge clk) gap_ok <= (!gap) || ($random % 3 != 0);

    reg  trava = 1'b0, trava_ok = 1'b1;
    always @(posedge clk) trava_ok <= (!trava) || ($random % 4 != 0);

    wire in_valid = feeding && (fed < LEN) && gap_ok;

    reg [NIF*8-1:0] in_data;
    integer q;
    always @* begin
        for (q = 0; q < NIF; q = q + 1)
            in_data[q*8 +: 8] = vin[vec*NIF*LEN + q*LEN + fed];
    end

    localparam NMULT = (RQ_POR_CANAL != 0) ? NOF : 1;
    reg  [17:0] mfile [0:NMULT-1];
    reg         ld_m_en = 0, ld_m_valid = 0;
    reg  signed [17:0] ld_m_data = 0;

    conv1d_engine #(
        .NIF(NIF), .NOF(NOF), .K(K), .LEN(LEN), .POOL(POOL), .POOL_AVG(POOL_AVG),
        .POX(POX), .POF(POF), .PK(PK), .NRQU(NRQU), .DATA_W(8), .ACC_W(ACC_W), .RQ_W(RQ_W), .MULT_W(18), .SHIFT(SHIFT),
        .RQ_POR_CANAL(RQ_POR_CANAL),

        .WR_PESO(1)
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

    always @(posedge clk)
        if (in_ready && in_valid) fed <= fed + 1'b1;

    integer n_got = 0, gi;
    always @(posedge clk)
        if (out_valid)
            for (gi = 0; gi < NRQU; gi = gi + 1) begin
                got[(out_ch + gi)*OLEN + out_pos] <= out_data[gi*8 +: 8];
                n_got <= n_got + gi + 1;
            end

    integer li;
    reg signed [ACC_W-1:0] a_i;
    reg [ACC_W-1:0] acc_pico = 0;
    always @(posedge clk)
        if (busy)
            for (li = 0; li < NOF*POX; li = li + 1) begin
                a_i = dbg_acc[li*ACC_W +: ACC_W];
                if (a_i < 0) a_i = -a_i;
                if (a_i > acc_pico) acc_pico = a_i;
            end

    integer ciclos = 0;
    always @(posedge clk) if (busy) ciclos <= ciclos + 1;

    integer i, erros, erros_v, esperado, obtido, idx;
    integer ciclos_min = 0, ciclos_max = 0, ciclos_gap = 0, erros_gap = 0;
    integer ciclos_carga = 0;

    task carrega_parametros;
        begin
            ciclos_carga = 0;
            @(posedge clk); ld_w_en <= 1'b1;
            @(posedge clk); ld_w_en <= 1'b0;
            for (i = 0; i < NOF*NIF*K; i = i + 1) begin
                @(posedge clk);
                ld_w_valid   <= 1'b1;
                ld_w_data    <= wfile[i];
                ciclos_carga  = ciclos_carga + 1;
            end
            @(posedge clk); ld_w_valid <= 1'b0;

            @(posedge clk); ld_b_en <= 1'b1;
            @(posedge clk); ld_b_en <= 1'b0;
            for (i = 0; i < NOF; i = i + 1) begin
                @(posedge clk);
                ld_b_valid   <= 1'b1;
                ld_b_data    <= bfile[i];
                ciclos_carga  = ciclos_carga + 1;
            end
            @(posedge clk); ld_b_valid <= 1'b0;
            @(posedge clk); ld_m_en <= 1'b1;
            @(posedge clk); ld_m_en <= 1'b0;
            for (i = 0; i < NMULT; i = i + 1) begin
                @(posedge clk);
                ld_m_valid   <= 1'b1;
                ld_m_data    <= mfile[i];
                ciclos_carga  = ciclos_carga + 1;
            end
            @(posedge clk); ld_m_valid <= 1'b0;
        end
    endtask

    task roda_vetor(input integer v);
        begin
            vec    = v[15:0];
            fed    = 0;
            n_got  = 0;
            ciclos = 0;
            @(posedge clk);
            start   <= 1'b1;
            feeding <= 1'b1;
            @(posedge clk);
            start   <= 1'b0;
            wait (done === 1'b1);
            @(posedge clk);
            feeding <= 1'b0;
        end
    endtask

    task confere(input integer v);
        begin
            erros_v = 0;
            if (trunc_err !== 1'b0) begin
                $display("  vetor %0d: trunc_err ativo, RQ_W=%0d insuficiente", v, RQ_W);
                erros_v = erros_v + 1;
            end
            if (n_got !== NSAI) begin
                $display("  vetor %0d: %0d saidas, esperado %0d", v, n_got, NSAI);
                erros_v = erros_v + 1;
            end
            for (idx = 0; idx < NSAI; idx = idx + 1) begin
                esperado = gold[v*NSAI + idx];
                obtido   = got[idx];
                if (esperado !== obtido) begin
                    if (erros_v < 4)
                        $display("  vetor %0d ch %0d pos %0d: obtido %02h, gold %02h",
                                 v, idx/OLEN, idx%OLEN, obtido, esperado);
                    erros_v = erros_v + 1;
                end
            end
            erros = erros + erros_v;
        end
    endtask

    initial begin
        $readmemh(W_FILE,    wfile);
        $readmemh(B_FILE,    bfile);
        $readmemh(M_FILE,    mfile);
        $readmemh(IN_FILE,   vin);
        $readmemh(GOLD_FILE, gold);

        repeat (4) @(posedge clk);
        rst_n <= 1'b1;
        repeat (2) @(posedge clk);

        carrega_parametros;

        erros = 0;
        for (i = 0; i < NVEC; i = i + 1) begin
            roda_vetor(i);
            confere(i);
            if (i == 0) begin
                ciclos_min = ciclos;
                ciclos_max = ciclos;
            end else begin
                if (ciclos < ciclos_min) ciclos_min = ciclos;
                if (ciclos > ciclos_max) ciclos_max = ciclos;
            end
        end

        $display("=== %0s: %0d vetores x %0d valores ===", NOME, NVEC, NSAI);
        $display("  ciclos/inferencia   : %0d..%0d   derivado: %0d",
                 ciclos_min, ciclos_max, CICLOS);
        $display("  jitter              : %0d ciclos", ciclos_max - ciclos_min);
        $display("  ciclos de carga     : %0d (pesos %0d + bias %0d)",
                 ciclos_carga, NOF*NIF*K, NOF);
        $display("  pico |acumulador|   : %0d   limite de %0d bits: %0d",
                 acc_pico, ACC_BITS, (1 << (ACC_BITS-1)) - 1);
        $display("  RQ_W / POX          : %0d bits / %0d posicoes em paralelo", RQ_W, POX);
        $display("  lanes MAC (DSP)     : %0d   vias de saida: %0d   taps/ciclo: %0d",
                 POF*POX*PK, NRQU, PK);

        if (ciclos_min !== CICLOS || ciclos_max !== CICLOS) begin
            $display("  FALHOU: ciclos medidos divergem da derivacao");
            erros = erros + 1;
        end
        if (acc_pico > ((1 << (ACC_BITS-1)) - 1)) begin
            $display("  FALHOU: acumulador excede acc_bits do manifest");
            erros = erros + 1;
        end

        if (TESTA_BOLHA) begin
            gap       = 1'b1;
            trava     = 1'b1;
            erros_gap = erros;
            for (i = 0; i < (NVEC < 4 ? NVEC : 4); i = i + 1) begin
                roda_vetor(i);
                confere(i);
                if (i == 0) ciclos_gap = ciclos;
            end
            gap   = 1'b0;
            trava = 1'b0;
            $display("  bolha na entrada e contrapressao na saida: %0d ciclos, %0d erros", ciclos_gap, erros - erros_gap);
        end

        if (erros == 0) $display("  BIT-EXATO");
        else            $display("  FALHOU: %0d divergencias", erros);
        $display("");
        $finish;
    end

    initial begin
        #200_000_000;
        $display("  FALHOU: timeout");
        $finish;
    end

endmodule

`default_nettype wire
