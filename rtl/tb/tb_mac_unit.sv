// ==========================================================================
// tb_mac_unit.sv -- SystemVerilog testbench for src/mac_unit.v.
//
// Reads rtl/golden/mac_int8.csv line by line; for each (a, b, K, mode) row
// where mode is 'trunc' or 'round', applies inputs to the DUT and compares
// product_rounded bit-for-bit. Stochastic mode is exercised separately with
// internal random offsets and a mean-error check.
//
// Vendor-agnostic: works under Icarus Verilog, Vivado XSim,
// ModelSim-Altera Starter, Verilator and Questa. Run with run_tests.py.
//
// Exit code:
//   $finish(0)  -- all checks passed
//   $fatal(1)   -- at least one mismatch (first failure printed)
// ==========================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_mac_unit;

    localparam string GOLDEN_PATH = "../golden/mac_int8.csv";

    // ----------------------------------------------------------------------
    // DUT
    // ----------------------------------------------------------------------
    logic signed [7:0]  a;
    logic signed [7:0]  b;
    logic [3:0]         K;
    logic [1:0]         mode;
    logic [15:0]        rnd;
    wire  signed [15:0] product_rounded;

    mac_unit #(.A_BITS(8), .B_BITS(8)) dut (
        .a(a), .b(b), .K(K), .mode(mode), .rnd(rnd),
        .product_rounded(product_rounded)
    );

    // ----------------------------------------------------------------------
    // CSV mode encoding (numeric so plain %d works in ModelSim's $sscanf).
    // ----------------------------------------------------------------------
    localparam integer CSV_MODE_TRUNC = 0;
    localparam integer CSV_MODE_ROUND = 1;

    // ----------------------------------------------------------------------
    // Main
    // ----------------------------------------------------------------------
    integer fd;
    integer n_checked, n_fail;
    integer first_a, first_b, first_K, first_mode, first_exp, first_got;
    string  line;
    integer a_i, b_i, K_i, prod_full, prod_round;
    integer mode_i;
    integer scanned;

    // Stochastic-mode internal RNG (just $urandom seeded once; the production
    // LFSR module gets tested separately in Phase 1.5).
    int     i;
    real    mean_err, sigma, tol;
    longint sum_err;
    int     err_i;
    int     r;

    initial begin
        n_checked = 0;
        n_fail    = 0;

        fd = $fopen(GOLDEN_PATH, "r");
        if (fd == 0) begin
            $display("FATAL: cannot open %s -- run rtl/golden/export_golden.py first",
                     GOLDEN_PATH);
            $fatal(1);
        end

        // Skip header.
        void'($fgets(line, fd));

        while (!$feof(fd)) begin
            line = "";
            void'($fgets(line, fd));
            if (line.len() == 0) continue;

            scanned = $sscanf(line, "%d,%d,%d,%d,%d,%d",
                              a_i, b_i, K_i, mode_i, prod_full, prod_round);
            if (scanned < 6) continue;
            if (mode_i != CSV_MODE_TRUNC && mode_i != CSV_MODE_ROUND) continue;

            a    = a_i[7:0];
            b    = b_i[7:0];
            K    = K_i[3:0];
            // CSV mode codes happen to equal the DUT's mode encoding.
            mode = mode_i[1:0];
            rnd  = 16'h0;
            #1;

            n_checked = n_checked + 1;
            if ($signed(product_rounded) !== prod_round) begin
                if (n_fail == 0) begin
                    first_a    = a_i;
                    first_b    = b_i;
                    first_K    = K_i;
                    first_mode = mode_i;
                    first_exp  = prod_round;
                    first_got  = $signed(product_rounded);
                end
                n_fail = n_fail + 1;
            end
        end
        $fclose(fd);

        $display("[deterministic] checked=%0d  failed=%0d", n_checked, n_fail);
        if (n_fail > 0) begin
            $display("  first failure: a=%0d b=%0d K=%0d mode=%0d  expected=%0d got=%0d",
                     first_a, first_b, first_K, first_mode, first_exp, first_got);
            $fatal(1);
        end

        // --------------------------------------------------------------
        // Stochastic mean-error check: for each K, draw 2000 random
        // (a, b, rnd) triples, accumulate (got - a*b), verify the sample
        // mean is within ~5 sigma of zero (the Gupta 2015 unbiasedness
        // property the round mode does NOT have).
        // --------------------------------------------------------------
        // Seed once; subsequent calls advance the PRNG state.
        r = $urandom(32'hC0FFEE);
        for (K_i = 1; K_i <= 6; K_i = K_i + 1) begin
            sum_err = 0;
            for (i = 0; i < 2000; i = i + 1) begin
                // a, b in -128..127 by sampling the low byte and sign-mapping.
                a_i = $urandom() & 32'hFF;
                if (a_i >= 128) a_i = a_i - 256;
                b_i = $urandom() & 32'hFF;
                if (b_i >= 128) b_i = b_i - 256;

                a    = a_i[7:0];
                b    = b_i[7:0];
                K    = K_i[3:0];
                mode = 2'b10;  // stochastic
                rnd  = $urandom();
                #1;

                err_i = $signed(product_rounded) - (a_i * b_i);
                sum_err = sum_err + err_i;
            end
            mean_err = sum_err / 2000.0;
            sigma    = (1 << K_i) / $sqrt(12.0 * 2000.0);
            tol      = 5.0 * sigma;
            $display("[stochastic K=%0d] mean_err=%.3f  tol=%.3f", K_i, mean_err, tol);
            if ((mean_err > tol) || (mean_err < -tol)) begin
                $display("  FAIL: K=%0d biased beyond 5 sigma", K_i);
                $fatal(1);
            end
        end

        $display("ALL TESTS PASSED");
        $finish;
    end

endmodule

`default_nettype wire
