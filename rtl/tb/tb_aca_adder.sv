// ==========================================================================
// tb_aca_adder.sv -- SystemVerilog testbench for src/aca_adder.v.
//
// Reads rtl/golden/aca.csv and checks each (a, b, bits=32, window, sum) row
// against the matching DUT instance. Since WINDOW is a compile-time parameter,
// we instantiate one DUT per window width in the golden set and dispatch by
// the row's `window` column.
// ==========================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_aca_adder;

    localparam string GOLDEN_PATH = "../golden/aca.csv";

    logic signed [31:0] a;
    logic signed [31:0] b;
    wire  signed [31:0] sum_w4, sum_w8, sum_w16, sum_w32;

    aca_adder #(.BITS(32), .WINDOW(4))  dut_w4  (.a(a), .b(b), .sum(sum_w4));
    aca_adder #(.BITS(32), .WINDOW(8))  dut_w8  (.a(a), .b(b), .sum(sum_w8));
    aca_adder #(.BITS(32), .WINDOW(16)) dut_w16 (.a(a), .b(b), .sum(sum_w16));
    aca_adder #(.BITS(32), .WINDOW(32)) dut_w32 (.a(a), .b(b), .sum(sum_w32));

    integer fd;
    integer n_checked, n_fail;
    integer a_i, b_i, bits_i, window_i, expected;
    integer scanned;
    integer got;
    integer first_a, first_b, first_w, first_exp, first_got;
    string  line;

    initial begin
        n_checked = 0;
        n_fail    = 0;

        fd = $fopen(GOLDEN_PATH, "r");
        if (fd == 0) begin
            $display("FATAL: cannot open %s", GOLDEN_PATH);
            $fatal(1);
        end
        void'($fgets(line, fd));  // skip header

        while (!$feof(fd)) begin
            line = "";
            void'($fgets(line, fd));
            if (line.len() == 0) continue;

            scanned = $sscanf(line, "%d,%d,%d,%d,%d",
                              a_i, b_i, bits_i, window_i, expected);
            if (scanned < 5) continue;

            a = a_i;
            b = b_i;
            #1;

            case (window_i)
                4:  got = sum_w4;
                8:  got = sum_w8;
                16: got = sum_w16;
                32: got = sum_w32;
                default: begin
                    $display("FATAL: unknown window %0d in golden CSV", window_i);
                    $fatal(1);
                end
            endcase

            n_checked = n_checked + 1;
            if (got !== expected) begin
                if (n_fail == 0) begin
                    first_a   = a_i;
                    first_b   = b_i;
                    first_w   = window_i;
                    first_exp = expected;
                    first_got = got;
                end
                n_fail = n_fail + 1;
            end
        end
        $fclose(fd);

        $display("[aca_adder] checked=%0d  failed=%0d", n_checked, n_fail);
        if (n_fail > 0) begin
            $display("  first failure: a=%0d b=%0d W=%0d  expected=%0d got=%0d",
                     first_a, first_b, first_w, first_exp, first_got);
            $fatal(1);
        end

        $display("ALL TESTS PASSED");
        $finish;
    end

endmodule

`default_nettype wire
