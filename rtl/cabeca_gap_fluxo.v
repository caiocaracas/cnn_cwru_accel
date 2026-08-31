// cabeca de media global DESLIZANTE, para o motor em fluxo continuo.

`default_nettype none

module cabeca_gap_fluxo #(
    parameter NCLS   = 4,
    parameter NCH    = 32,
    parameter NPOS   = 128,
    parameter DATA_W = 8,
    parameter ACC_W  = 32,
    parameter NVIA   = 1,
    parameter NPG    = 1,
    parameter MEM    = "",
    parameter WR_PESO = 0
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        limpa,

    input  wire                        ld_w_en,
    input  wire                        ld_w_valid,
    input  wire signed [DATA_W-1:0]    ld_w_data,
    input  wire                        ld_b_en,
    input  wire                        ld_b_valid,
    input  wire signed [31:0]          ld_b_data,

    input  wire                        in_valid,
    input  wire [NVIA*DATA_W-1:0]      in_data,
    input  wire [$clog2(NCH)-1:0]      in_ch,
    input  wire [$clog2(NPOS)-1:0]     in_pos,

    output wire [NCLS*ACC_W-1:0]       logits,
    output reg  [$clog2(NCLS)-1:0]     classe,
    output reg                         classe_valid,
    output reg                         desvio
);
    localparam PW  = $clog2(NPOS);
    localparam CHW = $clog2(NCH);
    localparam CW  = $clog2(NCLS);
    localparam POR_GRUPO = NPG * NCH;
    localparam NVW = $clog2(POR_GRUPO+1);

    localparam CPB  = NCH / NVIA;
    localparam AW_B = $clog2(NPOS*CPB);
    reg signed [DATA_W-1:0] wrom [0:NCLS*NCH-1];
    reg signed [31:0]       brom [0:NCLS-1];
    initial if (MEM != "") begin
        $readmemh({MEM, "_w.mem"}, wrom);
        $readmemh({MEM, "_b.mem"}, brom);
    end

    reg [$clog2(NCLS*NCH+1)-1:0] ld_w_p;
    reg [CW:0]                   ld_b_p;
    always @(posedge clk) begin
        if (!rst_n)                       ld_w_p <= 0;
        else if (ld_w_en)                 ld_w_p <= 0;
        else if (WR_PESO && ld_w_valid) begin
            wrom[ld_w_p] <= ld_w_data;
            ld_w_p <= ld_w_p + 1'b1;
        end
        if (!rst_n)                       ld_b_p <= 0;
        else if (ld_b_en)                 ld_b_p <= 0;
        else if (WR_PESO && ld_b_valid) begin
            brom[ld_b_p] <= ld_b_data;
            ld_b_p <= ld_b_p + 1'b1;
        end
    end

    reg [PW:0]    n_pos;
    wire          cheio = (n_pos >= NPOS);
    reg [NVW-1:0] n_val;
    wire          fecha_ja = in_valid && (n_val + NVIA >= POR_GRUPO);

    reg                    v1, f1;
    reg [CHW-1:0]          ch1;
    reg signed [DATA_W:0]  d1 [0:NVIA-1];

    always @(posedge clk) begin
        if (!rst_n || limpa) begin
            v1 <= 1'b0; f1 <= 1'b0; n_val <= 0; n_pos <= 0;
        end else begin
            v1 <= in_valid;
            f1 <= fecha_ja;
            if (in_valid) begin
                if (fecha_ja) begin
                    n_val <= 0;
                    if (!cheio) n_pos <= n_pos + NPG[PW:0];
                end else begin
                    n_val <= n_val + NVIA[NVW-1:0];
                end
            end
        end
    end

    wire [AW_B-1:0] end_b = in_pos * CPB + (in_ch / NVIA);

    reg signed [DATA_W-1:0] w1 [0:NCLS*NVIA-1];
    integer a0, u0;
    always @(posedge clk)
        if (in_valid) begin
            ch1 <= in_ch;
            for (a0 = 0; a0 < NCLS; a0 = a0 + 1)
                for (u0 = 0; u0 < NVIA; u0 = u0 + 1)
                    w1[a0*NVIA + u0] <= wrom[a0*NCH + in_ch + u0];
        end

    genvar b;
    generate
        for (b = 0; b < NVIA; b = b + 1) begin : banco
            reg signed [DATA_W-1:0] anel [0:NPOS*CPB-1];
            wire signed [DATA_W-1:0] novo =
                $signed(in_data[b*DATA_W +: DATA_W]);
            always @(posedge clk) begin
                d1[b] <= in_valid
                       ? (novo - (cheio ? anel[end_b] : {DATA_W{1'b0}}))
                       : {(DATA_W+1){1'b0}};
                if (in_valid) anel[end_b] <= novo;
            end
        end
    endgenerate

    localparam MW_H = DATA_W + 1;

    wire signed [ACC_W-1:0] acc [0:NCLS*NVIA-1];
    reg  signed [ACC_W-1:0] saida [0:NCLS-1];
    reg  f2, f3, f4, zera;

    always @(posedge clk) begin
        if (!rst_n || limpa) begin
            f2 <= 1'b0; f3 <= 1'b0; f4 <= 1'b0; zera <= 1'b1;
        end else begin
            f2 <= f1; f3 <= f2; f4 <= f3;
            if (v1) zera <= 1'b0;
        end
    end

    // LIMITE ANALITICO DO ACUMULADOR, e por que ele existe
    //
    // O acumulador e' telescopico: `acc += w*(novo - velho)`. Isso e' o que o
    // torna barato - uma multiplicacao por chegada em vez de NCH por decisao -
    // e tambem o que o torna fragil: ele nunca esquece. Uma unica chegada mal
    // contabilizada desloca TODAS as decisoes seguintes, para sempre.
    //
    // Por telescopagem, o acumulador de uma via vale
    //     acc = SOMA_ch w[classe][ch] * S[ch]
    // sobre os CPB canais que passam por ela, onde S e' a soma da janela
    localparam signed [ACC_W-1:0] LIM_ACC = CPB * NPOS * 127 * 127;

    integer ad;
    always @(posedge clk) begin
        if (!rst_n || limpa) begin
            desvio <= 1'b0;
        end else begin
            for (ad = 0; ad < NCLS*NVIA; ad = ad + 1)
                if (acc[ad] > LIM_ACC || acc[ad] < -LIM_ACC)
                    desvio <= 1'b1;
        end
    end

    genvar gc, gv;
    generate
        for (gc = 0; gc < NCLS; gc = gc + 1) begin : classe_mac
            for (gv = 0; gv < NVIA; gv = gv + 1) begin : via
                wire signed [MW_H-1:0] w_ext =
                    {{(MW_H-DATA_W){w1[gc*NVIA+gv][DATA_W-1]}},
                     w1[gc*NVIA+gv]};
                mac_lane #(.DATA_W(MW_H), .ACC_W(ACC_W), .PK(1)) u_mac (
                    .clk(clk), .en(1'b1),
                    .amostra(d1[gv]), .wt(w_ext),
                    .bias({ACC_W{1'b0}}), .load(zera || limpa),
                    .acc(acc[gc*NVIA+gv])
                );
            end
        end
    endgenerate

    reg publica;
    always @(posedge clk)
        if (!rst_n || limpa) publica <= 1'b0;
        else        publica <= f4;

    // as vias sao somadas so' aqui, uma vez por decisao: fora do caminho
    integer a4, u4;
    reg signed [ACC_W-1:0] junta;
    always @(posedge clk)
        if (f4)
            for (a4 = 0; a4 < NCLS; a4 = a4 + 1) begin
                junta = brom[a4];
                for (u4 = 0; u4 < NVIA; u4 = u4 + 1)
                    junta = junta + acc[a4*NVIA + u4];
                saida[a4] <= junta;
            end

    localparam NPAR = (NCLS + 1) / 2;

    reg signed [ACC_W-1:0] mpar [0:NPAR-1];
    reg [CW-1:0]           ipar [0:NPAR-1];
    reg                    pub2;

    always @(posedge clk)
        if (!rst_n || limpa) pub2 <= 1'b0;
        else        pub2 <= publica;

    integer a5;
    always @(posedge clk) begin
        begin
            for (a5 = 0; a5 < NPAR; a5 = a5 + 1) begin
                if ((2*a5 + 1 < NCLS) && (saida[2*a5+1] > saida[2*a5])) begin
                    mpar[a5] <= saida[2*a5+1];
                    ipar[a5] <= (2*a5+1);
                end else begin
                    mpar[a5] <= saida[2*a5];
                    ipar[a5] <= (2*a5);
                end
            end
        end
    end

    integer a6;
    reg signed [ACC_W-1:0] melhor;
    always @(posedge clk) begin
        classe_valid <= 1'b0;
        if (!rst_n) begin
            classe <= 0;
        end else if (pub2) begin
            melhor = mpar[0];
            classe = ipar[0];
            for (a6 = 1; a6 < NPAR; a6 = a6 + 1)
                if (mpar[a6] > melhor) begin
                    melhor = mpar[a6];
                    classe = ipar[a6];
                end
            classe_valid <= 1'b1;
        end
    end

    genvar g;
    generate
        for (g = 0; g < NCLS; g = g + 1) begin : pub
            assign logits[g*ACC_W +: ACC_W] = saida[g];
        end
    endgenerate
endmodule

`default_nettype wire
