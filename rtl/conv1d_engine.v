// motor de convolucao unidimensional com dobramento parametrizado.

`default_nettype none

module conv1d_engine #(
    parameter NIF    = 1,
    parameter NOF    = 8,
    parameter K      = 5,
    parameter LEN    = 1024,
    parameter POOL   = 2,

    parameter POOL_AVG = 0,
    parameter POX    = 1,

    parameter POF    = NOF,

    parameter PK     = 1,

    parameter NRQU   = 1,

    parameter NDSP   = 0,
    parameter DATA_W = 8,

    parameter ACC_W  = 32,

    parameter RQ_W   = 18,
    parameter MULT_W = 18,

    parameter RQ_POR_CANAL = 0,
    parameter SHIFT  = 16,
    parameter MEM = "",

    parameter FLUXO = 0,

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
    input  wire                        ld_m_en,
    input  wire                        ld_m_valid,
    input  wire signed [MULT_W-1:0]    ld_m_data,

    input  wire                        start,
    output wire                        busy,
    output reg                         done,

    output reg                         trunc_err,

    input  wire                        in_valid,
    input  wire [NIF*DATA_W-1:0]       in_data,
    output wire                        in_ready,

    input  wire                        out_ready,

    output wire                        out_valid,
    output wire [NRQU*DATA_W-1:0]      out_data,
    output wire [$clog2(NOF)-1:0]      out_ch,
    output wire [$clog2(LEN/POOL)-1:0] out_pos,

    output wire [POF*POX*ACC_W-1:0]    dbg_acc
);

    localparam PAD     = (K-1)/2;
    localparam JAN     = POX + K - 1;
    localparam GRP     = (NIF*K)/PK;
    localparam NG      = LEN/POX;
    localparam POOL_IN = (POX < POOL) ? POX : POOL;
    localparam NQ      = POX / POOL_IN;
    localparam GPP     = POOL / POOL_IN;
    localparam OG      = NOF/POF;
    localparam NLANE   = POF*POX;
    localparam NRQ     = POF*NQ;
    localparam OLEN    = LEN/POOL;
    localparam LOG_POOL = (POOL <= 1) ? 0 : $clog2(POOL);

    localparam CH_W  = $clog2(NOF);
    localparam NMULT = (RQ_POR_CANAL != 0) ? NOF : 1;
    localparam MW    = (NMULT <= 1) ? 1 : $clog2(NMULT);
    localparam LW    = (POF <= 1) ? 1 : $clog2(POF);
    localparam OGW   = (OG  <= 1) ? 1 : $clog2(OG);
    localparam PO_W  = $clog2(OLEN);
    localparam Q_W   = (NQ <= 1) ? 1 : $clog2(NQ);
    localparam TAG_W = CH_W + PO_W;

    localparam PIPE_END = 1;
    localparam AW    = $clog2(NIF*K);
    localparam WAW   = $clog2(OG*NIF*K);
    localparam JW    = $clog2(NIF*JAN);
    localparam KW    = $clog2(K+1);
    localparam IW    = $clog2(NIF+1);
    localparam PLW   = $clog2(GPP+1);
    localparam NW    = $clog2(LEN+JAN+2);
    localparam RQW   = $clog2(NRQ+1);
    localparam OCW   = $clog2(OLEN*NOF+1);

    localparam S_IDLE = 2'd0, S_PRIME = 2'd1, S_RUN = 2'd2, S_DRAIN = 2'd3;

    initial begin
        if (LEN % POX != 0) begin
            $display("conv1d_engine: LEN=%0d nao divisivel por POX=%0d", LEN, POX);
            $finish;
        end
        if (POX % POOL_IN != 0 || POOL % POOL_IN != 0) begin
            $display("conv1d_engine: POX=%0d e POOL=%0d incompativeis", POX, POOL);
            $finish;
        end

        if (NRQ > NRQU*((OG > 1) ? GRP : GPP*GRP)) begin
            $display("conv1d_engine: NRQ=%0d nao cabe nas vias de saida", NRQ);
            $finish;
        end
        if (POF % NRQU != 0) begin
            $display("conv1d_engine: NRQU=%0d nao divide POF=%0d", NRQU, POF);
            $finish;
        end
        if (NOF % POF != 0 || (POF & (POF-1)) != 0) begin
            $display("conv1d_engine: POF=%0d deve dividir NOF=%0d e ser potencia de 2",
                     POF, NOF);
            $finish;
        end
        if (POX > GRP) begin
            $display("conv1d_engine: POX=%0d nao cabe nos %0d ciclos do grupo",
                     POX, GRP);
            $finish;
        end
        if (ACC_W < RQ_W + 1) begin
            $display("conv1d_engine: ACC_W=%0d sem bit de guarda sobre RQ_W=%0d",
                     ACC_W, RQ_W);
            $finish;
        end
        if (K % PK != 0) begin
            $display("conv1d_engine: PK=%0d nao divide K=%0d", PK, K);
            $finish;
        end
        if (RQ_W > 25)
            $display("conv1d_engine: RQ_W=%0d > 25, requant cascateia 2 DSP48E1",
                     RQ_W);
    end

    reg [1:0]      state;
    reg [JW+1:0]   prime_cnt;
    reg [NW-1:0]   n_push;
    reg            n_push_ok;
    reg [KW-1:0]   cnt_k;
    reg [IW-1:0]   cnt_ic;
    reg [NW-1:0]   g;
    reg            g_ok;
    reg [3+PIPE_END:0] dly;
    reg [OGW-1:0]  og;
    reg [OGW-1:0]  og_d [0:3+PIPE_END];
    reg [PLW-1:0]  pool_ph;
    reg [PO_W-1:0] opos, rq_pos;
    reg            rq_go;
    reg [RQW-1:0]  rq_cnt;
    reg [OGW-1:0]  rq_og;
    reg [OCW-1:0]  out_cnt;

    reg signed [DATA_W-1:0] sr_fill [0:NIF*JAN-1];
    reg signed [DATA_W-1:0] sr_rd   [0:NIF*JAN-1];

    wire [NRQ*ACC_W-1:0] pooled_flat;

    wire sub_last  = (cnt_k == (K/PK)-1) && (cnt_ic == NIF-1);
    wire tap_last  = sub_last && (og == OG-1);
    wire tap_first = (cnt_k == 0)        && (cnt_ic == 0) && (state == S_RUN);
    wire [AW-1:0] cnt = cnt_ic*(K/PK) + cnt_k;

    wire push_run  = (state == S_RUN) && (cnt < POX) && (og == 0) && g_ok;

    wire zera_ini  = (state == S_PRIME) && (prime_cnt < PAD);
    wire do_push   = ((state == S_PRIME) && (prime_cnt < JAN)) || push_run;
    wire copia     = ((state == S_PRIME) && (prime_cnt == JAN)) ||
                     ((state == S_RUN)   && tap_last);

    wire [KW-1:0]  cnt_k_p  = sub_last ? {KW{1'b0}}
                            : (cnt_k == (K/PK)-1) ? {KW{1'b0}} : cnt_k + 1'b1;
    wire [IW-1:0]  cnt_ic_p = sub_last ? {IW{1'b0}}
                            : (cnt_k == (K/PK)-1) ? cnt_ic + 1'b1 : cnt_ic;
    wire [OGW-1:0] og_p     = sub_last ? ((og == OG-1) ? {OGW{1'b0}} : og + 1'b1)
                                       : og;
    wire prox_tap_last = (cnt_k_p == (K/PK)-1) && (cnt_ic_p == NIF-1)
                         && (og_p == OG-1);
    localparam PRIMEIRO_JA_E_ULTIMO = (((K/PK) == 1) && (NIF == 1) && (OG == 1)) ? 1'b1 : 1'b0;

    wire prox_copia = ((state == S_PRIME) && copia) ? PRIMEIRO_JA_E_ULTIMO
                    : (state == S_PRIME) ? (prime_cnt + 1'b1 == JAN)
                    : (state == S_RUN)   ? prox_tap_last
                    : 1'b0;

    (* max_fanout = 32 *)
    reg copia_r;
    always @(posedge clk) begin
        if (!rst_n || start) copia_r <= 1'b0;
        else if (adv)        copia_r <= prox_copia;
    end
    wire need_in   = do_push && !zera_ini && n_push_ok;

    (* max_fanout = 64 *)
    wire   adv      = (!need_in || in_valid) && out_ready;

    assign in_ready = need_in && out_ready;

    assign busy = (state != S_IDLE);

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE:
                    if (start) begin
                        state     <= S_PRIME;
                        prime_cnt <= 0;
                        n_push    <= 0;
                        n_push_ok <= 1'b1;
                        cnt_k     <= 0;
                        cnt_ic    <= 0;
                        og        <= 0;
                        g         <= 0;
                        g_ok      <= (NG > 1);
                    end
                S_PRIME:
                    if (adv) begin
                        prime_cnt <= prime_cnt + 1'b1;
                        if (do_push && !zera_ini) begin
                            n_push    <= n_push + 1'b1;
                            n_push_ok <= FLUXO ? 1'b1 : (n_push + 1'b1 < LEN);
                        end
                        if (copia) state <= S_RUN;
                    end
                S_RUN:
                    if (adv) begin
                        if (push_run) begin
                            n_push    <= n_push + 1'b1;
                            n_push_ok <= FLUXO ? 1'b1 : (n_push + 1'b1 < LEN);
                        end
                        if (sub_last) begin
                            cnt_k  <= 0;
                            cnt_ic <= 0;
                            if (og == OG-1) begin
                                og <= 0;
                                if (FLUXO) begin
                                    g    <= (g == NG-1) ? {NW{1'b0}} : g + 1'b1;
                                    g_ok <= 1'b1;
                                end else if (g == NG-1) state <= S_DRAIN;
                                else begin
                                    g    <= g + 1'b1;
                                    g_ok <= (g + 1'b1 < NG-1);
                                end
                            end else begin
                                og <= og + 1'b1;
                            end
                        end else if (cnt_k == (K/PK)-1) begin
                            cnt_k  <= 0;
                            cnt_ic <= cnt_ic + 1'b1;
                        end else begin
                            cnt_k  <= cnt_k + 1'b1;
                        end
                    end
                S_DRAIN:
                    if (out_cnt == OLEN*NOF) begin
                        state <= S_IDLE;
                        done  <= 1'b1;
                    end
            endcase
        end
    end

    wire [NIF*DATA_W-1:0] push_data = (!zera_ini && (FLUXO || n_push < LEN))
                                      ? in_data : {(NIF*DATA_W){1'b0}};

    reg signed [DATA_W-1:0] sr_prox [0:NIF*JAN-1];
    integer ii, jj, kk;
    always @* begin
        for (ii = 0; ii < NIF; ii = ii + 1) begin
            if (do_push) begin
                for (jj = JAN-1; jj > 0; jj = jj - 1)
                    sr_prox[ii*JAN + jj] = sr_fill[ii*JAN + jj - 1];
                sr_prox[ii*JAN] = push_data[ii*DATA_W +: DATA_W];
            end else begin
                for (jj = 0; jj < JAN; jj = jj + 1)
                    sr_prox[ii*JAN + jj] = sr_fill[ii*JAN + jj];
            end
        end
    end

    always @(posedge clk)
        if (adv && do_push)
            for (ii = 0; ii < NIF*JAN; ii = ii + 1) sr_fill[ii] <= sr_prox[ii];

    always @(posedge clk)
        if (adv && copia_r)
            for (kk = 0; kk < NIF*JAN; kk = kk + 1) sr_rd[kk] <= sr_prox[kk];

    wire [JW-1:0] base_idx = cnt_ic*JAN + (JAN-1-cnt_k*PK);

    reg [CH_W-1:0] ld_w_oc, ld_b_p;
    reg [MW-1:0]   ld_m_p;
    reg [AW-1:0]   ld_w_a;
    wire [CH_W-1:0] ld_w_lane = ld_w_oc % POF;
    wire [CH_W-1:0] ld_w_og   = ld_w_oc / POF;
    wire [CH_W-1:0] ld_b_lane = ld_b_p  % POF;
    wire [CH_W-1:0] ld_b_og   = ld_b_p  / POF;

    always @(posedge clk) begin
        if (ld_w_en) begin
            ld_w_oc <= 0;
            ld_w_a  <= 0;
        end else if (ld_w_valid) begin
            if (ld_w_a == NIF*K-1) begin
                ld_w_a  <= 0;
                ld_w_oc <= ld_w_oc + 1'b1;
            end else begin
                ld_w_a  <= ld_w_a + 1'b1;
            end
        end
        if (ld_b_en)          ld_b_p <= 0;
        else if (ld_b_valid)  ld_b_p <= ld_b_p + 1'b1;

        if (ld_m_en)          ld_m_p <= 0;
        else if (ld_m_valid)  ld_m_p <= ld_m_p + 1'b1;
    end

    integer dd;
    always @(posedge clk) begin
        if (start) begin
            dly <= {(4+PIPE_END){1'b0}};
            for (dd = 0; dd < 4+PIPE_END; dd = dd + 1) og_d[dd] <= {OGW{1'b0}};
        end else if (adv) begin
            dly <= {dly[2+PIPE_END:0], (state == S_RUN) && sub_last};
            og_d[0] <= og;
            for (dd = 1; dd < 4+PIPE_END; dd = dd + 1) og_d[dd] <= og_d[dd-1];
        end
    end
    wire grp_done  = dly[3+PIPE_END];
    wire [OGW-1:0] og_q = og_d[3+PIPE_END];
    wire blk_done  = grp_done && (og_q == OG-1);
    wire pool_emit = grp_done && (pool_ph == GPP-1);

    always @(posedge clk) begin
        if (start) begin
            pool_ph <= 0;
            opos    <= 0;
        end else if (adv && blk_done) begin
            if (pool_ph == GPP-1) begin
                pool_ph <= 0;
                opos    <= opos + NQ[PO_W-1:0];
            end else begin
                pool_ph <= pool_ph + 1'b1;
            end
        end
    end

    wire signed [ACC_W-1:0] acc_w [0:NLANE-1];

    genvar l, p, q;
    generate

    for (l = 0; l < POF; l = l + 1) begin : canal
        reg signed [DATA_W-1:0] wrom [0:OG*NIF*K-1];
        reg signed [ACC_W-1:0]  brom [0:OG-1];
        reg signed [ACC_W-1:0]  bias_r;

        (* dont_touch = "true" *) reg [AW-1:0]  cnt_l;
        (* dont_touch = "true" *) reg [OGW-1:0] og_l;
        always @(posedge clk) if (adv) begin
            cnt_l <= cnt;
            og_l  <= og;
        end

        initial if (MEM != "") begin
            $readmemh($sformatf("%s_w%0d.mem", MEM, l), wrom);
            $readmemh($sformatf("%s_b%0d.mem", MEM, l), brom);
        end

        always @(posedge clk) begin
            if (WR_PESO && ld_w_valid && ld_w_lane == l)
                wrom[ld_w_og*(NIF*K) + ld_w_a] <= ld_w_data;
            if (WR_PESO && ld_b_valid && ld_b_lane == l)
                brom[ld_b_og] <= ld_b_data[ACC_W-1:0];
            if (adv) bias_r <= brom[og_l];
        end

        for (p = 0; p < POX; p = p + 1) begin : posicao
            reg [PK*DATA_W-1:0] amostra_r, wt_r;
            reg                     load_r;

            reg [PK*DATA_W-1:0] am_d;
            reg                 run_d, tapf_d;

            (* max_fanout = 16 *)
            wire [PK*DATA_W-1:0] am_sel;
            wire [PK*DATA_W-1:0] wt_sel;
            genvar tt;
            for (tt = 0; tt < PK; tt = tt + 1) begin : taps
                assign am_sel[tt*DATA_W +: DATA_W] = sr_rd[base_idx - p - tt];
                assign wt_sel[tt*DATA_W +: DATA_W] = wrom[og_l*(NIF*K) + cnt_l*PK + tt];
            end

            always @(posedge clk) begin
                if (adv) begin
                    am_d   <= am_sel;
                    run_d  <= (state == S_RUN);
                    tapf_d <= tap_first;

                    amostra_r <= run_d ? am_d : {(PK*DATA_W){1'b0}};
                    wt_r      <= wt_sel;
                    load_r    <= tapf_d;
                end
            end

            if (NDSP == 0 || (l*POX + p) < NDSP) begin : em_dsp
                mac_lane #(.DATA_W(DATA_W), .ACC_W(ACC_W), .PK(PK)) u_mac (
                    .clk(clk), .en(adv), .amostra(amostra_r), .wt(wt_r),
                    .bias(bias_r), .load(load_r), .acc(acc_w[l*POX + p])
                );
            end else begin : em_logica
                mac_lane_lut #(.DATA_W(DATA_W), .ACC_W(ACC_W), .PK(PK)) u_mac (
                    .clk(clk), .en(adv), .amostra(amostra_r), .wt(wt_r),
                    .bias(bias_r), .load(load_r), .acc(acc_w[l*POX + p])
                );
            end

            assign dbg_acc[(l*POX+p+1)*ACC_W-1 : (l*POX+p)*ACC_W] = acc_w[l*POX+p];
        end

        for (q = 0; q < NQ; q = q + 1) begin : saida
            wire signed [ACC_W-1:0] e0 = acc_w[l*POX + q*POOL_IN];
            wire signed [ACC_W-1:0] r0 = e0[ACC_W-1] ? {ACC_W{1'b0}} : e0;
            wire signed [ACC_W-1:0] e1 = acc_w[l*POX + q*POOL_IN + POOL_IN-1];
            wire signed [ACC_W-1:0] r1 = e1[ACC_W-1] ? {ACC_W{1'b0}} : e1;

            reg signed [ACC_W-1:0] pool_acc [0:OG-1];
            reg signed [ACC_W-1:0] pooled_r;

            wire signed [ACC_W-1:0] emax = (POOL_IN == 1) ? r0 :
                                           POOL_AVG       ? (r0 + r1) :
                                           (r1 > r0)      ? r1 : r0;

            wire signed [ACC_W-1:0] ant = pool_acc[og_q];
            wire signed [ACC_W-1:0] pmax = (pool_ph == 0) ? emax :
                                           POOL_AVG       ? (emax + ant) :
                                           (emax > ant)   ? emax : ant;

            always @(posedge clk) begin
                if (adv && grp_done) begin
                    pool_acc[og_q] <= pmax;

                    if (pool_ph == GPP-1)
                        pooled_r <= POOL_AVG ? (pmax >>> LOG_POOL) : pmax;
                end
            end

            assign pooled_flat[(q*POF+l+1)*ACC_W-1 : (q*POF+l)*ACC_W] = pooled_r;
        end
    end
    endgenerate

    always @(posedge clk) begin
        if (start) begin
            rq_go  <= 1'b0;
            rq_cnt <= 0;
            rq_pos <= 0;
            rq_og  <= 0;
        end else if (adv) begin
            if (pool_emit) begin
                rq_go  <= 1'b1;
                rq_cnt <= 0;
                rq_pos <= opos;
                rq_og  <= og_q;
            end else if (rq_go) begin
                if (rq_cnt + NRQU >= NRQ) rq_go <= 1'b0;
                rq_cnt <= rq_cnt + NRQU[RQW-1:0];
            end
        end
    end

    wire [CH_W-1:0] rq_ch = rq_og*POF + (rq_cnt % POF);
    wire [Q_W-1:0]  rq_q  = (NQ <= 1) ? {Q_W{1'b0}} : (rq_cnt / POF);
    wire [TAG_W-1:0] rq_tag = {rq_pos + rq_q, rq_ch};
    wire [TAG_W-1:0] out_tag;
    wire [NRQU-1:0]  rq_ovalid;
    wire [NRQU-1:0]  estoura;

    genvar u;
    generate
    for (u = 0; u < NRQU; u = u + 1) begin : saida
        wire [RQW-1:0] idx = rq_cnt + u[RQW-1:0];
        wire signed [ACC_W-1:0] cheio = pooled_flat[idx*ACC_W +: ACC_W];
        wire [TAG_W-1:0] tag_u;

        assign estoura[u] = rq_go && (|cheio[ACC_W-1:RQ_W-1]);

        reg signed [MULT_W-1:0] mrom [0:NMULT-1];
        initial if (MEM != "") $readmemh({MEM, "_m.mem"}, mrom);
        always @(posedge clk)
            if (WR_PESO && ld_m_valid) mrom[ld_m_p] <= ld_m_data;

        wire [CH_W-1:0] ch_u = rq_og*POF + (idx % POF);
        wire [MW-1:0] end_m = (RQ_POR_CANAL != 0) ? ch_u[MW-1:0] : {MW{1'b0}};
        wire signed [MULT_W-1:0] mult_u = mrom[end_m];

        requant #(
            .ACC_W (RQ_W), .MULT_W(MULT_W), .SHIFT(SHIFT),
            .OUT_W (DATA_W), .TAG_W(TAG_W)
        ) u_rq (
            .clk      (clk),
            .en       (adv),
            .in_valid (rq_go),
            .in_acc   (cheio[RQ_W-1:0]),
            .in_tag   (rq_tag),
            .mult     (mult_u),
            .out_valid(rq_ovalid[u]),
            .out_data (out_data[(u+1)*DATA_W-1 : u*DATA_W]),
            .out_tag  (tag_u)
        );

        if (u == 0) begin : etiqueta
            assign out_tag = tag_u;
        end
    end
    endgenerate

    always @(posedge clk) begin
        if (start)                  trunc_err <= 1'b0;
        else if (adv && |estoura)   trunc_err <= 1'b1;
    end

    assign out_valid = rq_ovalid[0] && adv;
    assign out_ch    = out_tag[CH_W-1:0];
    assign out_pos   = out_tag[TAG_W-1:CH_W];

    always @(posedge clk) begin
        if (start)             out_cnt <= 0;
        else if (out_valid)    out_cnt <= out_cnt + NRQU[OCW-1:0];
    end

endmodule

`default_nettype wire
