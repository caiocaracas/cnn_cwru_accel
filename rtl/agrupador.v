// junta as saidas das vias num vetor de canais.

`default_nettype none

module agrupador #(
    parameter NCAN   = 8,
    parameter DATA_W = 8,
    parameter NVIA   = 1,
    parameter NPOSG  = 1,
    parameter NPOS   = 2
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        in_valid,
    input  wire [NVIA*DATA_W-1:0]      in_data,
    input  wire [$clog2(NCAN)-1:0]     in_ch,
    input  wire [$clog2(NPOS)-1:0]     in_pos,
    output reg                         out_valid,
    output wire [NCAN*DATA_W-1:0]      out_data
);
    localparam SW = (NPOSG <= 1) ? 1 : $clog2(NPOSG);

    reg [NCAN*DATA_W-1:0] buf_r [0:NPOSG-1];
    reg [SW-1:0]          sel_d;

    wire [SW-1:0] sel = (NPOSG <= 1) ? {SW{1'b0}} : in_pos[SW-1:0];

    assign out_data = buf_r[sel_d];

    integer v;
    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
        end else begin
            if (in_valid)
                for (v = 0; v < NVIA; v = v + 1)
                    buf_r[sel][(in_ch + v)*DATA_W +: DATA_W] <=
                        in_data[v*DATA_W +: DATA_W];
            out_valid <= in_valid && (in_ch + NVIA >= NCAN);
            if (in_valid) sel_d <= sel;
        end
    end
endmodule

`default_nettype wire
