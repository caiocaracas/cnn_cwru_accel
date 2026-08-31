// atrasa o barramento de carga de peso em varios estagios registrados.

`default_nettype none

module carga_pipe #(
    parameter EST  = 3,
    parameter W_W  = 8,
    parameter B_W  = 32,
    parameter M_W  = 18
)(
    input  wire                  clk,

    input  wire                  i_w_en,
    input  wire                  i_w_valid,
    input  wire signed [W_W-1:0] i_w_data,
    input  wire                  i_b_en,
    input  wire                  i_b_valid,
    input  wire signed [B_W-1:0] i_b_data,
    input  wire                  i_m_en,
    input  wire                  i_m_valid,
    input  wire signed [M_W-1:0] i_m_data,

    output wire                  o_w_en,
    output wire                  o_w_valid,
    output wire signed [W_W-1:0] o_w_data,
    output wire                  o_b_en,
    output wire                  o_b_valid,
    output wire signed [B_W-1:0] o_b_data,
    output wire                  o_m_en,
    output wire                  o_m_valid,
    output wire signed [M_W-1:0] o_m_data
);

    localparam LARG = 3 + 3 + W_W + B_W + M_W;

    wire [LARG-1:0] ent = {i_w_en, i_w_valid, i_w_data,
                           i_b_en, i_b_valid, i_b_data,
                           i_m_en, i_m_valid, i_m_data};

    (* shreg_extract = "no" *) reg [LARG-1:0] pipe [0:EST-1];

    integer k;
    initial for (k = 0; k < EST; k = k + 1) pipe[k] = {LARG{1'b0}};

    always @(posedge clk) begin
        pipe[0] <= ent;
        for (k = 1; k < EST; k = k + 1) pipe[k] <= pipe[k-1];
    end

    wire [LARG-1:0] sai = pipe[EST-1];

    assign {o_w_en, o_w_valid, o_w_data,
            o_b_en, o_b_valid, o_b_data,
            o_m_en, o_m_valid, o_m_data} = sai;

endmodule

`default_nettype wire
