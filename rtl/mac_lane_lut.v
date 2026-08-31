// via de multiplicacao e acumulacao em logica.

`default_nettype none

(* use_dsp = "no" *)
module mac_lane_lut #(
    parameter DATA_W = 8,
    parameter ACC_W  = 32,

    parameter PK     = 1
)(
    input  wire                      clk,
    input  wire                      en,
    input  wire [PK*DATA_W-1:0]      amostra,
    input  wire [PK*DATA_W-1:0]      wt,
    input  wire signed [ACC_W-1:0]   bias,
    input  wire                      load,
    output wire signed [ACC_W-1:0]   acc
);

    localparam SOMA_W = 2*DATA_W + ((PK <= 1) ? 0 : $clog2(PK));

    reg  [PK*DATA_W-1:0]      a_r, b_r;
    reg signed [SOMA_W-1:0]   m_r;
    reg signed [ACC_W-1:0]    c_r1, c_r2;
    reg signed [ACC_W-1:0]    p_r;
    reg                       ld_r1, ld_r2;

    integer t;
    reg signed [SOMA_W-1:0] soma;
    always @* begin
        soma = {SOMA_W{1'b0}};
        for (t = 0; t < PK; t = t + 1)
            soma = soma + $signed(a_r[t*DATA_W +: DATA_W]) *
                          $signed(b_r[t*DATA_W +: DATA_W]);
    end

    always @(posedge clk) begin
        if (en) begin
            a_r   <= amostra;
            b_r   <= wt;
            c_r1  <= bias;
            ld_r1 <= load;

            m_r   <= soma;
            c_r2  <= c_r1;
            ld_r2 <= ld_r1;

            p_r   <= (ld_r2 ? c_r2 : p_r) + m_r;
        end
    end

    assign acc = p_r;

endmodule

`default_nettype wire
