// fila sincrona entre camadas.

`default_nettype none

module fifo_sinc #(
    parameter W     = 8,
    parameter PROF  = 8
)(
    input  wire          clk,
    input  wire          rst_n,
    input  wire          limpa,

    input  wire          wr,
    input  wire [W-1:0]  wdata,
    output wire          cheia,

    input  wire          rd,
    output wire [W-1:0]  rdata,
    output wire          vazia,

    output reg  [$clog2(PROF+1)-1:0] ocupacao
);

    localparam AW = $clog2(PROF);

    reg [W-1:0]  mem [0:PROF-1];
    reg [AW-1:0] wp, rp;

    assign vazia = (ocupacao == 0);
    assign cheia = (ocupacao == PROF);
    assign rdata = mem[rp];

    always @(posedge clk) begin
        if (!rst_n || limpa) begin
            wp       <= 0;
            rp       <= 0;
            ocupacao <= 0;
        end else begin
            if (wr && !cheia) begin
                mem[wp] <= wdata;
                wp      <= wp + 1'b1;
            end
            if (rd && !vazia) rp <= rp + 1'b1;

            case ({wr && !cheia, rd && !vazia})
                2'b10:   ocupacao <= ocupacao + 1'b1;
                2'b01:   ocupacao <= ocupacao - 1'b1;
                default: ocupacao <= ocupacao;
            endcase
        end
    end

endmodule

`default_nettype wire
