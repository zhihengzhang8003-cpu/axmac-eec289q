// ==========================================================================
// tb_drum_multiplier.sv -- SystemVerilog testbench for src/drum_multiplier.v
//
// Reads rtl/golden/drum_int8.csv line by line.  Each row contains
//   a, b, expected_product
// where a and b are signed INT8 operands and expected_product is the
// DRUM-4 approximate product as computed by axmac.approx_mac.drum_multiply.
//
// The testbench drives (a, b) into the DUT, waits 1 time unit for the
// combinational logic to settle, then compares product bit-for-bit with
// expected_product.  All 200 vectors must pass.
//
// Vendor-agnostic: compatible with ModelSim-Altera Starter, Icarus Verilog,
// Vivado XSim, and Questa.
//
// Exit:
//   $finish(0)  -- ALL TESTS PASSED
//   $fatal(1)   -- at least one mismatch
// ==========================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_drum_multiplier;

    // Path is relative to the working directory from which the simulator is
    // launched (the rtl/ directory in this project).
    localparam string GOLDEN_PATH = "../golden/drum_int8.csv";

    // ------------------------------------------------------------------
    // DUT instantiation
    // ------------------------------------------------------------------
    logic signed [7:0]  a;
    logic signed [7:0]  b;
    wire  signed [15:0] product;

    drum_multiplier #(
        .N_BITS(8),
        .K_DRUM(4)
    ) dut (
        .a(a),
        .b(b),
        .product(product)
    );

    // ------------------------------------------------------------------
    // Test variables
    // ------------------------------------------------------------------
    integer fd;
    integer n_checked, n_fail;
    string  line;
    integer a_i, b_i, exp_i;
    integer scanned;
    integer first_a, first_b, first_exp, first_got;

    // ------------------------------------------------------------------
    // Main
    // ------------------------------------------------------------------
    initial begin
        n_checked = 0;
        n_fail    = 0;

        fd = $fopen(GOLDEN_PATH, "r");
        if (fd == 0) begin
            $display("FATAL: cannot open %s", GOLDEN_PATH);
            $display("  Run rtl/golden/gen_drum_golden.py first.");
            $fatal(1);
        end

        // Skip header line.
        void'($fgets(line, fd));

        while (!$feof(fd)) begin
            line = "";
            void'($fgets(line, fd));
            if (line.len() == 0) continue;

            scanned = $sscanf(line, "%d,%d,%d", a_i, b_i, exp_i);
            if (scanned < 3) continue;

            a = a_i[7:0];
            b = b_i[7:0];
            #1;

            n_checked = n_checked + 1;

            if ($signed(product) !== exp_i) begin
                if (n_fail == 0) begin
                    first_a   = a_i;
                    first_b   = b_i;
                    first_exp = exp_i;
                    first_got = $signed(product);
                end
                $display("FAIL  a=%0d  b=%0d  expected=%0d  got=%0d",
                         a_i, b_i, exp_i, $signed(product));
                n_fail = n_fail + 1;
            end else begin
                $display("PASS  a=%0d  b=%0d  product=%0d", a_i, b_i, exp_i);
            end
        end
        $fclose(fd);

        $display("");
        $display("Checked %0d vectors, %0d failed.", n_checked, n_fail);

        if (n_fail > 0) begin
            $display("  First failure: a=%0d b=%0d expected=%0d got=%0d",
                     first_a, first_b, first_exp, first_got);
            $fatal(1);
        end

        $display("ALL TESTS PASSED");
        $finish;
    end

endmodule

`default_nettype wire
