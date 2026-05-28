// ==========================================================================
// tb_mac_array.sv -- structural test for src/mac_array.v.
//
// Strategy: drive a known matmul with K=0 (exact mode), WINDOW=ACC_BITS
// (exact accumulator). Reference is computed inline in SV by plain $signed
// arithmetic. After K_DIM cycles every PE accumulator must equal the dot
// product of its row's activations with its column's weights.
//
// This is a *structural* test for the array (broadcast, accumulator wiring,
// generate-for connectivity). Rounding correctness is already validated by
// tb_mac_unit, and ACA correctness by tb_aca_adder.
// ==========================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_mac_array;

    localparam integer R        = 4;
    localparam integer C        = 4;
    localparam integer A_BITS   = 8;
    localparam integer B_BITS   = 8;
    localparam integer ACC_BITS = 32;
    localparam integer WINDOW   = ACC_BITS;   // exact accumulator
    localparam integer K_DIM    = 16;          // accumulation depth
    localparam integer P_BITS   = A_BITS + B_BITS;

    // ----------------------------------------------------------------------
    // Clock
    // ----------------------------------------------------------------------
    logic clk = 0;
    always #5 clk = ~clk;   // 100 MHz

    // ----------------------------------------------------------------------
    // DUT signals
    // ----------------------------------------------------------------------
    logic                          rst_n;
    logic                          clear;
    logic                          mac_en;
    logic [3:0]                    K;
    logic [1:0]                    mode;
    logic [R*A_BITS-1:0]           acts_flat;
    logic [C*B_BITS-1:0]           wgts_flat;
    logic [R*C*P_BITS-1:0]         rnd_flat;
    wire  [R*C*ACC_BITS-1:0]       acc_flat;

    mac_array #(
        .R(R), .C(C), .A_BITS(A_BITS), .B_BITS(B_BITS),
        .ACC_BITS(ACC_BITS), .WINDOW(WINDOW)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .clear(clear), .mac_en(mac_en),
        .K(K), .mode(mode),
        .acts_flat(acts_flat),
        .wgts_flat(wgts_flat),
        .rnd_flat(rnd_flat),
        .acc_flat(acc_flat)
    );

    // ----------------------------------------------------------------------
    // Stimulus storage
    // ----------------------------------------------------------------------
    // Acts[k][r], Wgts[k][c] -- streamed cycle by cycle.
    logic signed [A_BITS-1:0] acts_seq [K_DIM][R];
    logic signed [B_BITS-1:0] wgts_seq [K_DIM][C];
    longint ref_acc [R][C];

    int t, r_i, c_i;
    int n_fail;
    int first_r, first_c;
    longint first_exp, first_got, got_rc;

    initial begin
        // ------------------------------------------------------------------
        // 1) Generate random stimulus and compute the reference dot products.
        // ------------------------------------------------------------------
        n_fail = 0;
        for (r_i = 0; r_i < R; r_i = r_i + 1)
            for (c_i = 0; c_i < C; c_i = c_i + 1)
                ref_acc[r_i][c_i] = 0;

        // Seed once; advance the PRNG with every $urandom call.
        first_exp = $urandom(32'hCAFEBABE);
        for (t = 0; t < K_DIM; t = t + 1) begin
            for (r_i = 0; r_i < R; r_i = r_i + 1) begin
                acts_seq[t][r_i] = $urandom() % 256 - 128;  // -128..127
            end
            for (c_i = 0; c_i < C; c_i = c_i + 1) begin
                wgts_seq[t][c_i] = $urandom() % 256 - 128;
            end
            for (r_i = 0; r_i < R; r_i = r_i + 1)
                for (c_i = 0; c_i < C; c_i = c_i + 1)
                    ref_acc[r_i][c_i] = ref_acc[r_i][c_i]
                                     + $signed(acts_seq[t][r_i])
                                     * $signed(wgts_seq[t][c_i]);
        end

        // ------------------------------------------------------------------
        // 2) Reset + clear.
        // ------------------------------------------------------------------
        rst_n  = 0;
        clear  = 0;
        mac_en = 0;
        K      = 4'd0;
        mode   = 2'b00;     // trunc, but with K=0 it's a no-op
        rnd_flat  = '0;
        acts_flat = '0;
        wgts_flat = '0;
        @(posedge clk);
        @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // ------------------------------------------------------------------
        // 3) Stream K_DIM MAC cycles.
        // ------------------------------------------------------------------
        mac_en = 1;
        for (t = 0; t < K_DIM; t = t + 1) begin
            for (r_i = 0; r_i < R; r_i = r_i + 1)
                acts_flat[r_i*A_BITS +: A_BITS] = acts_seq[t][r_i];
            for (c_i = 0; c_i < C; c_i = c_i + 1)
                wgts_flat[c_i*B_BITS +: B_BITS] = wgts_seq[t][c_i];
            @(posedge clk);
        end
        mac_en = 0;
        @(posedge clk);

        // ------------------------------------------------------------------
        // 4) Compare each PE's acc to the inline reference.
        // ------------------------------------------------------------------
        for (r_i = 0; r_i < R; r_i = r_i + 1) begin
            for (c_i = 0; c_i < C; c_i = c_i + 1) begin
                got_rc = $signed(acc_flat[(r_i*C + c_i)*ACC_BITS +: ACC_BITS]);
                if (got_rc !== ref_acc[r_i][c_i]) begin
                    if (n_fail == 0) begin
                        first_r   = r_i;
                        first_c   = c_i;
                        first_exp = ref_acc[r_i][c_i];
                        first_got = got_rc;
                    end
                    n_fail = n_fail + 1;
                end
            end
        end

        $display("[mac_array] checked=%0d  failed=%0d", R*C, n_fail);
        if (n_fail > 0) begin
            $display("  first failure: PE(%0d,%0d)  expected=%0d got=%0d",
                     first_r, first_c, first_exp, first_got);
            $fatal(1);
        end
        $display("ALL TESTS PASSED");
        $finish;
    end

    // Safety timeout in case the FSM hangs.
    initial begin
        #10000;
        $display("TIMEOUT");
        $fatal(1);
    end

endmodule

`default_nettype wire
