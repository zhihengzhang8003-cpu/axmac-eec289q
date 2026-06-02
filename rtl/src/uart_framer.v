// uart_framer.v -- sends header + all 10 logits over UART on rising edge of done.
// Protocol: 0xAB | logit[0] LE | logit[1] LE | ... | logit[9] LE  (41 bytes total)
// result_flat layout: logit[i] = result_flat[i*32 +: 32] (little-endian byte order)
`default_nettype none
module uart_framer #(
    parameter integer NBYTES = 40  // 10 logits × 4 bytes
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              done,
    input  wire [NBYTES*8-1:0] result_flat,
    output reg  [7:0]        tx_data,
    output reg               tx_valid,
    input  wire              tx_ready
);
    localparam integer TOTAL  = NBYTES + 1;   // 41 bytes (header + data)
    localparam [7:0]   HEADER = 8'hAB;

    reg done_r;
    wire done_rise = done & ~done_r;

    reg [7:0]  buf_r [0:TOTAL-1];
    reg [5:0]  idx;
    reg        sending;

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done_r   <= 1'b0;
            sending  <= 1'b0;
            tx_valid <= 1'b0;
            idx      <= 6'd0;
        end else begin
            done_r <= done;

            if (!sending && done_rise) begin
                buf_r[0] <= HEADER;
                for (k = 0; k < NBYTES; k = k + 1)
                    buf_r[k+1] <= result_flat[k*8 +: 8];
                sending  <= 1'b1;
                idx      <= 6'd0;
                tx_valid <= 1'b0;
            end else if (sending) begin
                if (!tx_valid) begin
                    tx_data  <= buf_r[idx];
                    tx_valid <= 1'b1;
                end else if (tx_ready) begin
                    tx_valid <= 1'b0;
                    if (idx == TOTAL - 1)
                        sending <= 1'b0;
                    else
                        idx <= idx + 6'd1;
                end
            end
        end
    end
endmodule
`default_nettype wire
