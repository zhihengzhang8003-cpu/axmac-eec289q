// ==========================================================================
// tb_mlp_top_demo.sv -- simulate the board wrapper before burning to silicon.
//
// Drives clk + a reset pulse, waits for led_done, then checks led_class == 1
// (argmax of the K=2/trunc golden logits [4345, 10718, ...] is index 1).
//
// LED_ACTIVE_LOW=0 here so led_class reads the raw class directly; the board
// build flips it back to active-low for the physical LEDs.
// ==========================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_mlp_top_demo;

    localparam integer EXPECTED_CLASS = 1;

    logic       clk = 0;
    always #10 clk = ~clk;   // 50 MHz

    logic       rst_n;
    wire [3:0]  led_class;
    wire        uart_tx_sim;   // UART output -- not checked in sim, just connected

    mlp_top_demo #(
        .K_PARAM(2), .MODE(0), .ACA_W(32),
        .LED_ACTIVE_LOW(0),
        .X_FILE ("../golden/mlp_toy/x.mem"),
        .W0_FILE("../golden/mlp_toy/w0.mem"),
        .B0_FILE("../golden/mlp_toy/b0.mem"),
        .W1_FILE("../golden/mlp_toy/w1.mem"),
        .B1_FILE("../golden/mlp_toy/b1.mem")
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .led_class   (led_class),
        .uart_tx_pin (uart_tx_sim)
    );

    integer i;

    initial begin
        rst_n = 1'b0;
        @(posedge clk);
        @(posedge clk);
        rst_n = 1'b1;   // release reset; start is tied high inside the wrapper

        // Inference takes ~320 cycles; wait comfortably past that, then the
        // combinational argmax has settled on the final class.
        for (i = 0; i < 400; i = i + 1) @(posedge clk);

        $display("[mlp_top_demo] led_class=%0d (expected %0d)",
                 led_class, EXPECTED_CLASS);
        if (led_class !== EXPECTED_CLASS[3:0]) begin
            $display("  FAIL: predicted class %0d, expected %0d", led_class, EXPECTED_CLASS);
            $fatal(1);
        end

        $display("ALL TESTS PASSED");
        $finish;
    end

    initial begin
        #500000;
        $display("WALLCLOCK TIMEOUT");
        $fatal(1);
    end

endmodule

`default_nettype wire
