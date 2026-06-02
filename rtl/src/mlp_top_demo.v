// ==========================================================================
// mlp_top_demo.v -- board-level wrapper for the on-chip demo (野火征途/EP4CE10).
//
// Runs the toy 64->16->10 MLP once after reset and displays the predicted
// class (argmax of the 10 output logits) on LEDs. Pure combinational argmax
// over the latched result; `start` is tied high so mlp_top runs a single
// inference and parks in its DONE state with the result held.
//
// Board hookup (征途 Pro has exactly 4 user LEDs, so the demo uses all 4 to
// show the predicted class and drops a separate done LED -- the class settles
// in ~320 cycles / 6.4 us, faster than the eye can see):
//   clk        <- 50 MHz onboard crystal (PIN_E1)
//   rst_n      <- reset button (PIN_M15, active low; held high = run)
//   led_class  -> 4 LEDs (N3/P3/M6/L7) showing predicted class 0..9 in binary
//
// 野火 LEDs are active-low (0 = lit). LED_ACTIVE_LOW=1 (default) inverts.
//
// Expected result with the shipped golden weights (K=2, trunc): the logits
// are [4345, 10718, -4421, -9026, -3228, -6422, 4780, -3136, 10420, 6344],
// so argmax = 1 -> led_class shows 0001.
// ==========================================================================

`default_nettype none

module mlp_top_demo #(
    parameter integer K_PARAM        = 2,
    parameter integer MODE           = 0,    // 0 = trunc
    parameter integer ACA_W          = 32,
    parameter integer LED_ACTIVE_LOW = 1,
    // Memory-init paths. Default depth (../../../../) targets the Quartus
    // build dir vendor/altera/build/<proj>; the sim testbench overrides these
    // with paths relative to rtl/tb/.
    parameter         X_FILE  = "../../../../golden/mlp_toy/x.mem",
    parameter         W0_FILE = "../../../../golden/mlp_toy/w0.mem",
    parameter         B0_FILE = "../../../../golden/mlp_toy/b0.mem",
    parameter         W1_FILE = "../../../../golden/mlp_toy/w1.mem",
    parameter         B1_FILE = "../../../../golden/mlp_toy/b1.mem"
) (
    input  wire       clk,
    input  wire       rst_n,
    output wire [3:0] led_class,
    output wire       uart_tx_pin    // 115200 8N1: connect to PIN_B3 (CH340 on 野火征途)
);

    localparam integer L1_OUT   = 10;
    localparam integer ACC_BITS = 32;

    wire                        done;
    wire [L1_OUT*ACC_BITS-1:0]  result_flat;

    // Hold start high: mlp_top runs one inference then parks in DONE with the
    // result latched (its S_DONE only leaves when start is deasserted).
    mlp_top #(
        .K_PARAM (K_PARAM),
        .MODE    (MODE),
        .ACA_W   (ACA_W),
        .X_FILE  (X_FILE),
        .W0_FILE (W0_FILE),
        .B0_FILE (B0_FILE),
        .W1_FILE (W1_FILE),
        .B1_FILE (B1_FILE)
    ) u_mlp (
        .clk(clk),
        .rst_n(rst_n),
        .start(1'b1),
        .done(done),
        .result_flat(result_flat)
    );

    // ----------------------------------------------------------------------
    // Combinational argmax over the 10 signed logits.
    // ----------------------------------------------------------------------
    reg  [3:0]            amax;
    reg  signed [ACC_BITS-1:0] best;
    reg  signed [ACC_BITS-1:0] cand;
    integer i;
    always @* begin
        amax = 4'd0;
        best = $signed(result_flat[0 +: ACC_BITS]);
        for (i = 1; i < L1_OUT; i = i + 1) begin
            cand = $signed(result_flat[i*ACC_BITS +: ACC_BITS]);
            if (cand > best) begin
                best = cand;
                amax = i[3:0];
            end
        end
    end

    assign led_class = (LED_ACTIVE_LOW != 0) ? ~amax : amax;

    // ------------------------------------------------------------------
    // UART output: send header + 10 logits (41 bytes) after inference.
    // ------------------------------------------------------------------
    wire [7:0] framer_data;
    wire       framer_valid;
    wire       framer_ready;

    uart_framer #(.NBYTES(L1_OUT * ACC_BITS / 8)) u_framer (
        .clk        (clk),
        .rst_n      (rst_n),
        .done       (done),
        .result_flat(result_flat),
        .tx_data    (framer_data),
        .tx_valid   (framer_valid),
        .tx_ready   (framer_ready)
    );

    uart_tx #(.CLK_FREQ(50_000_000), .BAUD(115_200)) u_uart (
        .clk   (clk),
        .rst_n (rst_n),
        .data  (framer_data),
        .valid (framer_valid),
        .ready (framer_ready),
        .tx    (uart_tx_pin)
    );

endmodule

`default_nettype wire
