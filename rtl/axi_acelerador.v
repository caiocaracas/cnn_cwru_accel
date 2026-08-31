// envelope axi do acelerador com registradores de controle e resultado.

`default_nettype none

module axi_acelerador #(
    parameter LEN     = 1024,
    parameter PROF_IN = 2*LEN,
    parameter FLUXO   = 0,
    parameter PROF_RES = 1024,
    parameter NCLS    = 4,
    parameter NCAM    = 8,
    parameter ACC_W   = 32,
    parameter ADDR_W  = 8
)(

    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:s_axis, ASSOCIATED_RESET s_axi_aresetn" *)
    input  wire                  s_axi_aclk,
    input  wire                  s_axi_aresetn,

    input  wire [ADDR_W-1:0]     s_axi_awaddr,
    input  wire                  s_axi_awvalid,
    output wire                  s_axi_awready,
    input  wire [31:0]           s_axi_wdata,
    input  wire [3:0]            s_axi_wstrb,
    input  wire                  s_axi_wvalid,
    output wire                  s_axi_wready,
    output reg  [1:0]            s_axi_bresp,
    output reg                   s_axi_bvalid,
    input  wire                  s_axi_bready,

    input  wire [ADDR_W-1:0]     s_axi_araddr,
    input  wire                  s_axi_arvalid,
    output wire                  s_axi_arready,
    output reg  [31:0]           s_axi_rdata,
    output reg  [1:0]            s_axi_rresp,
    output reg                   s_axi_rvalid,
    input  wire                  s_axi_rready,

    input  wire [31:0]           s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,
    input  wire                  s_axis_tlast,

    output wire [3:0]            leds
);

    wire clk = s_axi_aclk;
    wire rst_n = s_axi_aresetn;

    reg         aw_hold, w_hold;
    reg [ADDR_W-1:0] aw_addr;
    reg [31:0]  w_data;

    wire escrita_amostra = (wa[7:2] == 6'h0B) || (wa[7:2] == 6'h0C);
    wire wr_ok = (aw_hold || s_axi_awvalid) && (w_hold || s_axi_wvalid) &&
                 !s_axi_bvalid && (push_n == 2'd0) &&
                 !(escrita_amostra && fi_cheia);
    assign s_axi_awready = !aw_hold;
    assign s_axi_wready  = !w_hold;

    wire [ADDR_W-1:0] wa = aw_hold ? aw_addr : s_axi_awaddr;
    wire [31:0]       wd = w_hold  ? w_data  : s_axi_wdata;

    (* max_fanout = 16 *) reg start_p;
    reg        limpa_p;
    reg [3:0]  ld_sel;
    reg        ld_w_en_p, ld_w_v_p, ld_b_en_p, ld_b_v_p;
    reg        ld_m_en_p, ld_m_v_p;
    reg signed [7:0]  ld_w_data;
    reg signed [31:0] ld_b_data;
    // um multiplicador de requantizacao por canal de saida (Wu et al. 2020):
    // sao NOF por camada e entram em rajada, como o peso
    reg signed [17:0] ld_m_data;
    reg        in_wr;
    reg [7:0]  in_wdata;

    // cada palavra do barramento traz quatro amostras int8; as tres seguintes
    // saem do registro de arrasto, uma por ciclo
    reg [1:0]  push_n;
    reg [23:0] push_d;
    reg        done_peg;
    // inferencias concluidas pela PL; o processador compara com as colhidas
    reg [31:0] n_done;
    // contador livre: duas leituras separadas por um intervalo medido no
    // processador dao o relogio da PL sem depender de sobreposicao
    reg [31:0] tique;
    // do instante em que a janela esta' na fila E o motor esta' livre ate' o
    reg [31:0] lat, lat_last;
    reg        livre_d;

    reg [3:0]  leds_sw;
    reg        leds_modo;

    reg        correndo;
    wire       res_cheia, res_vazia;
    reg  [3:0] res_seq;
    wire [7:0] res_saida;
    reg        res_rd;
    wire [$clog2(PROF_RES+1)-1:0] res_ocup;
    reg [31:0] n_perdida;

    reg [31:0] iv_cnt, iv_min, iv_max, n_iv, n_parada_ent;
    reg        iv_arm;

    reg         ativ_rein, ativ_av;
    wire [31:0] ativ_data;

    wire        busy, done, classe_valid, in_ready;
    wire [NCAM-1:0] trunc_err;
    wire [NCLS*ACC_W-1:0] logits;
    wire [3:0]  classe;

    wire        fi_cheia, fi_vazia;
    wire [7:0]  fi_rdata;
    wire [$clog2(PROF_IN+1)-1:0] fi_ocup;
    wire        fi_rd = in_ready && !fi_vazia;

    wire jan_pronta = FLUXO ? !fi_vazia : (fi_ocup >= LEN);
    wire pronta_livre = jan_pronta && !busy;
    reg  start_pend;
    reg  arranca;

    generate
        if (FLUXO) begin : fila_res
            fifo_sinc #(.W(8), .PROF(PROF_RES)) u_fres (
                .clk(clk), .rst_n(rst_n), .limpa(limpa_p),
                .wr(classe_valid && !res_cheia), .wdata({res_seq, classe}),
                .cheia(res_cheia),
                .rd(res_rd && !res_vazia), .rdata(res_saida),
                .vazia(res_vazia), .ocupacao(res_ocup)
            );
        end else begin : sem_fila
            assign res_cheia = 1'b0;
            assign res_vazia = 1'b1;
            assign res_saida = 8'd0;
            assign res_ocup  = 0;
        end
    endgenerate

    fifo_sinc #(.W(8), .PROF(PROF_IN)) u_fifo_in (
        .clk(clk), .rst_n(rst_n), .limpa(limpa_p),
        .wr(in_wr), .wdata(in_wdata), .cheia(fi_cheia),
        .rd(fi_rd), .rdata(fi_rdata), .vazia(fi_vazia), .ocupacao(fi_ocup)
    );

    // as filas entre camadas nao sao mais parametro do envelope: cada uma
    // sai dimensionada pela rajada da camada que consome
    acelerador_gen #(.NCAM(NCAM), .NCLS(NCLS)) u_acel (
        .clk(clk), .rst_n(rst_n),
        .ld_sel(ld_sel),
        .ld_w_en(ld_w_en_p), .ld_w_valid(ld_w_v_p), .ld_w_data(ld_w_data),
        .ld_b_en(ld_b_en_p), .ld_b_valid(ld_b_v_p), .ld_b_data(ld_b_data),
        .ld_m_en(ld_m_en_p), .ld_m_valid(ld_m_v_p), .ld_m_data(ld_m_data),

        .start(arranca), .busy(busy), .done(done), .trunc_err(trunc_err),
        .in_valid(!fi_vazia), .in_data(fi_rdata), .in_ready(in_ready),
        .logits(logits), .classe(classe), .classe_valid(classe_valid),
        .ativ_reinicia(ativ_rein), .ativ_avanca(ativ_av), .ativ_data(ativ_data)
    );

    // por padrao os quatro LEDs acendem quando a inferencia termina, e e' o
    assign leds = leds_modo ? leds_sw : {4{done_peg}};

    assign s_axis_tready = (push_n == 2'd0) && !wr_ok && (fi_ocup <= PROF_IN - 4);
    wire axis_ok = s_axis_tvalid && s_axis_tready;

    reg [NCLS*ACC_W-1:0] logits_r;
    reg [3:0]            classe_r;
    always @(posedge clk) begin
        if (done) begin
            logits_r <= logits;
            classe_r <= classe;
        end
    end

    reg [31:0] ciclos, ciclos_last;
    reg        busy_d;
    always @(posedge clk) begin
        if (!rst_n) begin
            ciclos      <= 0;
            ciclos_last <= 0;
            busy_d      <= 1'b0;
        end else begin
            busy_d <= busy;
            if      (busy && !busy_d) ciclos <= 32'd1;
            else if (busy)            ciclos <= ciclos + 1'b1;
            if (done) ciclos_last <= ciclos;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            aw_hold <= 1'b0; w_hold <= 1'b0;
            s_axi_bvalid <= 1'b0; s_axi_bresp <= 2'b00;
            start_p <= 0; limpa_p <= 0;
            ld_w_en_p <= 0; ld_w_v_p <= 0; ld_b_en_p <= 0; ld_b_v_p <= 0;
            ld_m_en_p <= 0; ld_m_v_p <= 0;
            in_wr <= 0; done_peg <= 0; n_done <= 32'd0; tique <= 32'd0;
            correndo <= 1'b0; n_perdida <= 32'd0; res_seq <= 4'd0;
            lat <= 32'd0; lat_last <= 32'd0; livre_d <= 1'b0;
            push_n <= 2'd0;
            start_pend <= 1'b0; arranca <= 1'b0;
            leds_sw <= 4'd0; leds_modo <= 1'b0;
            ativ_rein <= 1'b0;
            iv_cnt <= 32'd0; iv_min <= 32'hFFFFFFFF; iv_max <= 32'd0;
            n_iv <= 32'd0; n_parada_ent <= 32'd0; iv_arm <= 1'b0;
            ld_sel <= 0;
        end else begin
            start_p   <= 1'b0;
            limpa_p   <= 1'b0;
            ld_w_en_p <= 1'b0;
            ld_w_v_p  <= 1'b0;
            ld_b_en_p <= 1'b0;
            ld_b_v_p  <= 1'b0;
            ld_m_en_p <= 1'b0;
            ld_m_v_p  <= 1'b0;
            in_wr      <= 1'b0;
            ativ_rein  <= 1'b0;
            arranca    <= 1'b0;

            if (push_n != 2'd0) begin
                in_wdata <= push_d[7:0];
                in_wr    <= !fi_cheia;
                push_d   <= {8'd0, push_d[23:8]};
                push_n   <= push_n - 1'b1;
            end

            if (axis_ok) begin
                in_wdata <= s_axis_tdata[7:0];
                in_wr    <= 1'b1;
                push_d   <= s_axis_tdata[31:8];
                push_n   <= 2'd3;
            end

            if (s_axi_awvalid && s_axi_awready) begin
                aw_hold <= 1'b1; aw_addr <= s_axi_awaddr;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_hold <= 1'b1; w_data <= s_axi_wdata;
            end

            if (wr_ok) begin
                aw_hold <= 1'b0;
                w_hold  <= 1'b0;
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00;
                case (wa[7:2])
                    6'h00: begin
                        start_p  <= wd[0];
                        limpa_p  <= wd[1];
                        if (wd[0]) begin
                            done_peg   <= 1'b0;
                            start_pend <= 1'b1;
                        end
                        if (wd[1]) begin
                            n_done     <= 32'd0;
                            start_pend <= 1'b0;
                            correndo   <= 1'b0;
                            n_perdida  <= 32'd0;
                            res_seq    <= 4'd0;
                            iv_cnt     <= 32'd0;
                            iv_min     <= 32'hFFFFFFFF;
                            iv_max     <= 32'd0;
                            n_iv       <= 32'd0;
                            n_parada_ent     <= 32'd0;
                            iv_arm     <= 1'b0;
                        end
                    end
                    6'h08: begin
                        ld_sel    <= wd[3:0];
                        ld_w_en_p <= wd[4];
                        ld_b_en_p <= wd[5];
                        ld_m_en_p <= wd[6];
                    end
                    6'h09: begin ld_w_data <= wd[7:0];  ld_w_v_p <= 1'b1; end
                    6'h0A: begin ld_b_data <= wd;       ld_b_v_p <= 1'b1; end
                    6'h0B: begin in_wdata  <= wd[7:0]; in_wr    <= !fi_cheia; end
                    6'h0C: begin
                        in_wdata <= wd[7:0];
                        in_wr    <= !fi_cheia;
                        push_d   <= wd[31:8];
                        push_n   <= 2'd3;
                    end

                    6'h11: begin
                        leds_sw   <= wd[3:0];
                        leds_modo <= wd[4];
                    end

                    6'h31: ativ_rein <= 1'b1;
                    6'h10: begin ld_m_data <= wd[17:0]; ld_m_v_p <= 1'b1; end
                    default: ;
                endcase
            end

            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;

            if (start_pend && jan_pronta && (FLUXO ? !correndo : !busy)) begin
                arranca    <= 1'b1;
                start_pend <= 1'b0;
                correndo   <= 1'b1;
            end
            if (FLUXO && classe_valid) begin
                res_seq <= res_seq + 1'b1;
                if (res_cheia) n_perdida <= n_perdida + 1'b1;
            end

            if (FLUXO && correndo) begin
                iv_cnt <= iv_cnt + 1'b1;
                if (in_ready && fi_vazia) n_parada_ent <= n_parada_ent + 1'b1;
            end
            if (FLUXO && classe_valid) begin
                iv_cnt <= 32'd1;
                if (iv_arm) begin
                    n_iv <= n_iv + 1'b1;
                    if (iv_cnt < iv_min) iv_min <= iv_cnt;
                    if (iv_cnt > iv_max) iv_max <= iv_cnt;
                end
                iv_arm <= 1'b1;
            end

            if (done) begin
                done_peg   <= 1'b1;
                n_done     <= n_done + 1'b1;
                if (!FLUXO) start_pend <= 1'b0;
            end
            tique <= tique + 1'b1;

            livre_d <= pronta_livre;
            if (pronta_livre && !livre_d) lat <= 32'd1;
            else if (lat != 32'd0)        lat <= lat + 1'b1;
            if (done && lat != 32'd0) begin
                lat_last <= lat;
                lat      <= 32'd0;
            end
        end
    end

    reg ar_hold;
    reg [ADDR_W-1:0] ar_addr;
    // uma escrita pode ficar pendurada em wr_ok esperando o registro de
    // arrasto esvaziar (push_n). Sem segurar a leitura enquanto isso, um
    // 'escreve controle e le estado' do processador enxerga o estado ANTERIOR
    // a' escrita: foi assim que a marcacao de done sobreviveu ao pedido de
    wire escrita_pendente = aw_hold || w_hold;
    assign s_axi_arready = !ar_hold && !s_axi_rvalid && !escrita_pendente;

    wire [ADDR_W-1:0] ra = ar_hold ? ar_addr : s_axi_araddr;

    always @(posedge clk) begin
        if (!rst_n) begin
            ar_hold <= 1'b0; s_axi_rvalid <= 1'b0; s_axi_rresp <= 2'b00;
            ativ_av <= 1'b0; res_rd <= 1'b0;
        end else begin
            ativ_av <= 1'b0;
            res_rd  <= 1'b0;
            if (s_axi_arvalid && s_axi_arready) begin
                if (s_axi_araddr[7:2] == 6'h31) ativ_av <= 1'b1;
                // ler 0x48 tira uma decisao da fila - mas SO' se havia uma
                if (s_axi_araddr[7:2] == 6'h12) res_rd <= !res_vazia;
                ar_hold      <= 1'b1;
                ar_addr      <= s_axi_araddr;
                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= 2'b00;
                case (s_axi_araddr[7:2])
                    6'h01: s_axi_rdata <= {16'd0, trunc_err, 2'd0, jan_pronta,
                                           fi_vazia, fi_cheia, classe_valid,
                                           done_peg, busy};
                    6'h02: s_axi_rdata <= {28'd0, classe_r};
                    6'h03: s_axi_rdata <= ciclos_last;
                    6'h13: s_axi_rdata <= n_done;
                    6'h14: s_axi_rdata <= tique;
                    6'h15: s_axi_rdata <= lat_last;
                    // 0x48: bit 8 diz se havia decisao, bits 3:0 sao a classe.
                    // A leitura consome
                    // bits 3:0 classe, 7:4 sequencia, bit 8 havia decisao
                    6'h12: s_axi_rdata <= {23'd0, !res_vazia, res_saida};
                    6'h16: s_axi_rdata <= {{(32-$clog2(PROF_RES+1)){1'b0}}, res_ocup};
                    6'h17: s_axi_rdata <= n_perdida;
                    6'h18: s_axi_rdata <= iv_min;
                    6'h19: s_axi_rdata <= iv_max;
                    6'h1A: s_axi_rdata <= n_iv;
                    6'h1B: s_axi_rdata <= n_parada_ent;
                    6'h04: s_axi_rdata <= logits_r[0*ACC_W +: 32];
                    6'h05: s_axi_rdata <= logits_r[1*ACC_W +: 32];
                    6'h06: s_axi_rdata <= logits_r[2*ACC_W +: 32];
                    6'h07: s_axi_rdata <= logits_r[3*ACC_W +: 32];
                    6'h0F: s_axi_rdata <= {{(32-$clog2(PROF_IN+1)){1'b0}}, fi_ocup};

                    6'h31: s_axi_rdata <= ativ_data;
                                default: s_axi_rdata <= 32'hDEAD_0000;
                endcase
            end
            if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
                ar_hold      <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
