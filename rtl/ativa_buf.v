// guarda o vetor da ultima camada para o passo de aprendizado.

`default_nettype none

module ativa_buf #(
    parameter NCH    = 32,
    parameter NPOS   = 128,
    parameter NVIA   = 1,
    parameter DATA_W = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     in_valid,
    input  wire [NVIA*DATA_W-1:0]   in_data,
    input  wire [$clog2(NCH)-1:0]   in_ch,
    input  wire [$clog2(NPOS)-1:0]  in_pos,
    input  wire                     rd_reinicia,
    input  wire                     rd_avanca,
    output wire [31:0]              rd_data
);

    localparam CHW   = (NCH  <= 1) ? 1 : $clog2(NCH);
    localparam POW   = (NPOS <= 1) ? 1 : $clog2(NPOS);
    localparam NFLAT = NCH * NPOS;
    localparam AW    = $clog2(NFLAT);

    localparam VW    = (NVIA <= 1) ? 1 : $clog2(NVIA);
    localparam CPS   = NCH / NVIA;
    localparam PROF  = CPS * (NPOS / 4);
    localparam PW    = (PROF <= 1) ? 1 : $clog2(PROF);

    reg [AW-1:0] ptr;
    always @(posedge clk) begin
        if (!rst_n || rd_reinicia) ptr <= {AW{1'b0}};
        else if (rd_avanca)        ptr <= ptr + 3'd4;
    end

    wire [CHW-1:0] ch_rd  = ptr / NPOS;
    wire [POW-1:0] pos_rd = ptr % NPOS;
    wire [VW-1:0]  s_rd   = (NVIA <= 1) ? {VW{1'b0}} : ch_rd[VW-1:0];
    wire [PW-1:0]  a_rd   = (ch_rd / NVIA) * (NPOS/4) + (pos_rd / 4);

    genvar b, s;
    generate
    for (b = 0; b < 4; b = b + 1) begin : faixa
        wire [NVIA*DATA_W-1:0] lido;

        for (s = 0; s < NVIA; s = s + 1) begin : sub
            reg [DATA_W-1:0] mem [0:PROF-1];

            // a via s escreve o canal in_ch+s, que por construcao tem
            // (in_ch+s) % NVIA == s: uma porta de escrita por sub-banco
            wire [CHW-1:0] ch_wr  = in_ch + s[CHW-1:0];
            wire [PW-1:0]  a_wr   = (ch_wr / NVIA) * (NPOS/4) + (in_pos / 4);
            wire           casa_b = (in_pos[1:0] == b[1:0]);

            always @(posedge clk)
                if (in_valid && casa_b)
                    mem[a_wr] <= in_data[s*DATA_W +: DATA_W];

            assign lido[s*DATA_W +: DATA_W] = mem[a_rd];
        end

        assign rd_data[b*8 +: 8] = lido[s_rd*DATA_W +: DATA_W];
    end
    endgenerate

endmodule

`default_nettype wire
