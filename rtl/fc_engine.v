// camada final densa com acumulacao por classe.

`default_nettype none

module fc_engine #(
    parameter NCLS   = 4,
    parameter NCH    = 32,
    parameter NPOS   = 128,
    parameter GAP    = 0,
    parameter DATA_W = 8,
    parameter ACC_W  = 32,
    parameter NVIA   = 1,
    parameter MEM = "",
    parameter WR_PESO = 0
)(
    input  wire                        clk,
    input  wire                        rst_n,

    input  wire                        ld_w_en,
    input  wire                        ld_w_valid,
    input  wire signed [DATA_W-1:0]    ld_w_data,
    input  wire                        ld_b_en,
    input  wire                        ld_b_valid,
    input  wire signed [31:0]          ld_b_data,

    input  wire                        start,
    output wire                        busy,
    output reg                         done,

    input  wire                        in_valid,
    input  wire [NVIA*DATA_W-1:0]      in_data,
    input  wire [$clog2(NCH)-1:0]      in_ch,
    input  wire [$clog2(NPOS)-1:0]     in_pos,

    output wire [NCLS*ACC_W-1:0]       logits,
    output reg  [$clog2(NCLS)-1:0]     classe,
    output reg                         classe_valid
);

    localparam NFLUXO = NCH*NPOS;
    localparam NPESO  = GAP ? NCH : NCH*NPOS;
    localparam NPAR   = NPESO/NVIA;
    localparam FW     = (NPESO <= 1) ? 1 : $clog2(NPESO);
    localparam PW     = (NPAR  <= 1) ? 1 : $clog2(NPAR);
    localparam VW     = (NVIA  <= 1) ? 1 : $clog2(NVIA);
    localparam CW     = $clog2(NCLS);
    localparam NW     = $clog2(NFLUXO+1);

    localparam S_IDLE = 2'd0, S_RUN = 2'd1, S_DRAIN = 2'd2;

    reg [1:0]    state;
    reg [NW-1:0] n_in;
    reg [2:0]    dreno;

    assign busy = (state != S_IDLE);

    reg [CW-1:0] ld_cls;
    reg [FW-1:0] ld_a;
    reg [CW-1:0] ld_b_p;

    always @(posedge clk) begin
        if (ld_w_en) begin
            ld_cls <= 0;
            ld_a   <= 0;
        end else if (ld_w_valid) begin
            if (ld_a == NPESO-1) begin
                ld_a   <= 0;
                ld_cls <= ld_cls + 1'b1;
            end else begin
                ld_a <= ld_a + 1'b1;
            end
        end
        if (ld_b_en)         ld_b_p <= 0;
        else if (ld_b_valid) ld_b_p <= ld_b_p + 1'b1;
    end

    wire [PW-1:0] flat = GAP ? (in_ch / NVIA)
                             : ((in_ch / NVIA) * NPOS + in_pos);

    reg [NVIA*DATA_W-1:0] d0;
    reg                   first0;

    always @(posedge clk) begin
        d0     <= in_valid ? in_data : {(NVIA*DATA_W){1'b0}};
        first0 <= in_valid && (n_in == 0);
    end

    wire [VW-1:0] ld_via  = (NVIA <= 1) ? {VW{1'b0}}
                          : (GAP ? (ld_a % NVIA) : ((ld_a / NPOS) % NVIA));
    wire [PW-1:0] ld_addr = GAP ? (ld_a / NVIA)
                                : (((ld_a / NPOS) / NVIA) * NPOS + (ld_a % NPOS));

    genvar c, v;
    generate
    for (c = 0; c < NCLS; c = c + 1) begin : classe_lane

        reg signed [ACC_W-1:0]  brom [0:0];
        wire signed [ACC_W-1:0] parcial [0:NVIA-1];

        initial if (MEM != "")
            $readmemh($sformatf("%s_b%0d.mem", MEM, c), brom);

        always @(posedge clk)
            if (WR_PESO && ld_b_valid && ld_b_p == c)
                brom[0] <= ld_b_data[ACC_W-1:0];

        for (v = 0; v < NVIA; v = v + 1) begin : via
            reg signed [DATA_W-1:0] wmem [0:NPAR-1];
            reg signed [DATA_W-1:0] w_r;

            initial if (MEM != "")
                $readmemh($sformatf("%s_c%0d_v%0d.mem", MEM, c, v), wmem);

            always @(posedge clk) begin
                if (WR_PESO && ld_w_valid && ld_cls == c && ld_via == v)
                    wmem[ld_addr] <= ld_w_data;
                w_r <= wmem[flat];
            end

            mac_lane #(.DATA_W(DATA_W), .ACC_W(ACC_W)) u_mac (
                .clk     (clk),
                .en      (1'b1),
                .amostra (d0[v*DATA_W +: DATA_W]),
                .wt      (w_r),
                .bias    ((v == 0) ? brom[0] : {ACC_W{1'b0}}),
                .load    (first0),
                .acc     (parcial[v])
            );
        end

        reg signed [ACC_W-1:0] soma;
        integer sv;
        always @* begin
            soma = {ACC_W{1'b0}};
            for (sv = 0; sv < NVIA; sv = sv + 1) soma = soma + parcial[sv];
        end

        assign logits[(c+1)*ACC_W-1 : c*ACC_W] = soma;
    end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE:
                    if (start) begin
                        state <= S_RUN;
                        n_in  <= 0;
                        dreno <= 0;
                    end
                S_RUN:
                    if (in_valid) begin
                        n_in <= n_in + NVIA[NW-1:0];
                        if (n_in + NVIA >= NFLUXO) state <= S_DRAIN;
                    end
                S_DRAIN: begin

                    dreno <= dreno + 1'b1;
                    if (dreno == 3'd6) begin
                        state <= S_IDLE;
                        done  <= 1'b1;
                    end
                end
            endcase
        end
    end

    integer m;
    reg signed [ACC_W-1:0] log_r [0:NCLS-1];
    reg signed [ACC_W-1:0] semi  [0:(NCLS/2)-1];
    reg [CW-1:0]           semi_i[0:(NCLS/2)-1];
    reg                    v_a, v_b;

    always @(posedge clk) begin
        for (m = 0; m < NCLS; m = m + 1)
            log_r[m] <= $signed(logits[m*ACC_W +: ACC_W]);
        v_a <= (state == S_DRAIN) && (dreno == 3'd3);

        for (m = 0; m < NCLS/2; m = m + 1) begin
            semi[m]   <= (log_r[2*m+1] > log_r[2*m]) ? log_r[2*m+1] : log_r[2*m];
            semi_i[m] <= (log_r[2*m+1] > log_r[2*m]) ? (2*m+1) : (2*m);
        end
        v_b <= v_a;

        classe       <= (semi[1] > semi[0]) ? semi_i[1] : semi_i[0];
        classe_valid <= v_b;
    end

endmodule

`default_nettype wire
