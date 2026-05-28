// ==========================================================================
// tb_mlp_top.sv -- end-to-end test for mlp_top.v on the toy 64 -> 16 -> 10 MLP.
//
// Builds the DUT at K=2, MODE=trunc (matches golden/mlp_toy/y_K2_trunc.csv),
// preloads the five .mem files via the DUT's parameter file paths, drives a
// start pulse, waits for `done`, and bit-exactly compares the 10 output
// activations against the Python golden.
//
// Why K=2 / trunc: it is the only configuration that:
//   * has a published golden (y_K2_trunc.csv from Phase 0 exporter),
//   * exercises non-trivial rounding (K > 0 -> bits are actually dropped),
//   * is deterministic (trunc / round are; stochastic isn't, see Phase 1 tb).
//
// Sweeping K or MODE happens by re-elaborating with different params; for
// PowerPlay PPA we'll build (trunc, round, stochastic) bitstreams in Phase 5b.
// ==========================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_mlp_top;

    // ----------------------------------------------------------------------
    // Design-under-test parameters (match the golden being checked)
    // ----------------------------------------------------------------------
    localparam integer A_BITS    = 8;
    localparam integer ACC_BITS  = 32;
    localparam integer L0_IN     = 64;
    localparam integer L0_OUT    = 16;
    localparam integer L1_OUT    = 10;
    localparam integer C         = 4;
    localparam integer K_PARAM   = 2;
    localparam integer MODE      = 0;        // 0 = trunc
    localparam integer ACA_W     = 32;       // exact accumulator

    localparam string GOLDEN_Y = "../golden/mlp_toy/y_K2_trunc.csv";

    // ----------------------------------------------------------------------
    // Clock & reset
    // ----------------------------------------------------------------------
    logic clk = 0;
    always #10 clk = ~clk;   // 50 MHz, matches 野火征途 onboard crystal

    logic rst_n;
    logic start;
    logic done;
    wire  [L1_OUT*ACC_BITS-1:0] result_flat;

    // ----------------------------------------------------------------------
    // DUT
    // ----------------------------------------------------------------------
    mlp_top #(
        .A_BITS  (A_BITS),
        .ACC_BITS(ACC_BITS),
        .L0_IN   (L0_IN),
        .L0_OUT  (L0_OUT),
        .L1_OUT  (L1_OUT),
        .C       (C),
        .K_PARAM (K_PARAM),
        .MODE    (MODE),
        .ACA_W   (ACA_W),
        .X_FILE  ("../golden/mlp_toy/x.mem"),
        .W0_FILE ("../golden/mlp_toy/w0.mem"),
        .B0_FILE ("../golden/mlp_toy/b0.mem"),
        .W1_FILE ("../golden/mlp_toy/w1.mem"),
        .B1_FILE ("../golden/mlp_toy/b1.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .done(done),
        .result_flat(result_flat)
    );

    // ----------------------------------------------------------------------
    // Test
    // ----------------------------------------------------------------------
    integer fd;
    string  line;
    integer expected [0:L1_OUT-1];
    integer got      [0:L1_OUT-1];
    integer scanned;
    integer i;
    integer n_fail;
    integer cycle_count;

    initial begin
        // ------------------------------------------------------------------
        // 1) Load the expected 10-vector from the golden CSV.
        // ------------------------------------------------------------------
        fd = $fopen(GOLDEN_Y, "r");
        if (fd == 0) begin
            $display("FATAL: cannot open %s", GOLDEN_Y);
            $fatal(1);
        end
        line = "";
        void'($fgets(line, fd));
        $fclose(fd);

        scanned = $sscanf(line, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
                          expected[0], expected[1], expected[2], expected[3],
                          expected[4], expected[5], expected[6], expected[7],
                          expected[8], expected[9]);
        if (scanned != L1_OUT) begin
            $display("FATAL: parsed %0d / %0d expected ints from golden",
                     scanned, L1_OUT);
            $fatal(1);
        end

        // ------------------------------------------------------------------
        // 2) Reset, then pulse start.
        // ------------------------------------------------------------------
        rst_n = 1'b0;
        start = 1'b0;
        @(posedge clk);
        @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        // ------------------------------------------------------------------
        // 3) Wait for done (with timeout).
        // ------------------------------------------------------------------
        cycle_count = 0;
        while (!done) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
            if (cycle_count > 2000) begin
                $display("FATAL: timeout waiting for done after %0d cycles",
                         cycle_count);
                $fatal(1);
            end
        end

        // ------------------------------------------------------------------
        // 4) Compare.
        // ------------------------------------------------------------------
        n_fail = 0;
        for (i = 0; i < L1_OUT; i = i + 1) begin
            got[i] = $signed(result_flat[i*ACC_BITS +: ACC_BITS]);
            if (got[i] !== expected[i]) begin
                $display("  y[%0d]: expected=%0d  got=%0d  diff=%0d",
                         i, expected[i], got[i], got[i] - expected[i]);
                n_fail = n_fail + 1;
            end
        end

        $display("[mlp_top] cycles=%0d  checked=%0d  failed=%0d",
                 cycle_count, L1_OUT, n_fail);
        if (n_fail > 0) $fatal(1);

        $display("ALL TESTS PASSED");
        $finish;
    end

    // Safety timeout.
    initial begin
        #500000;
        $display("WALLCLOCK TIMEOUT");
        $fatal(1);
    end

endmodule

`default_nettype wire
