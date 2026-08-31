// requantizacao em ponto fixo com saturacao.

`default_nettype none

module requant #(
    parameter ACC_W  = 18,
    parameter MULT_W = 18,
    parameter SHIFT  = 16,
    parameter OUT_W  = 8,
    parameter TAG_W  = 16
)(
    input  wire                      clk,
    input  wire                      en,
    input  wire                      in_valid,
    input  wire signed [ACC_W-1:0]   in_acc,
    input  wire [TAG_W-1:0]          in_tag,
    input  wire signed [MULT_W-1:0]  mult,
    output reg                       out_valid,
    output reg  signed [OUT_W-1:0]   out_data,
    output reg  [TAG_W-1:0]          out_tag
);

    localparam PROD_W = ACC_W + MULT_W;
    localparam signed [OUT_W-1:0] QMAX =  (1 << (OUT_W-1)) - 1;
    localparam signed [OUT_W-1:0] QMIN = -(1 << (OUT_W-1));

    wire signed [PROD_W-1:0] round_c = {{(PROD_W-1){1'b0}}, 1'b1} << (SHIFT-1);

    reg signed [ACC_W-1:0]   a_r;
    reg signed [MULT_W-1:0]  b_r;
    reg signed [PROD_W-1:0]  m_r, p_r;
    reg [TAG_W-1:0]          t0, t1, t2;
    reg                      v0, v1, v2;

    wire signed [PROD_W-1:0] desl = p_r >>> SHIFT;

    always @(posedge clk) begin
        if (en) begin
            a_r <= in_acc;  b_r <= mult;  t0 <= in_tag;  v0 <= in_valid;

            m_r <= a_r * b_r;             t1 <= t0;      v1 <= v0;

            p_r <= m_r + round_c;         t2 <= t1;      v2 <= v1;

            out_valid <= v2;
            out_tag   <= t2;
            if      (desl > QMAX) out_data <= QMAX;
            else if (desl < QMIN) out_data <= QMIN;
            else                  out_data <= desl[OUT_W-1:0];
        end
    end

endmodule

`default_nettype wire
