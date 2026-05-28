// ==========================================================================
// mlp_top.v -- toy 64 -> 16 -> 10 MLP using one mac_array (R=1, C=4).
//
// Datapath:
//   * 5 on-chip memories preloaded via $readmemh from rtl/golden/mlp_toy/*.mem:
//       act_mem (64x8, reused for layer-1 input after layer-0 drain),
//       w0_mem  (64*16 = 1024x8, row-major: addr = in*L0_OUT + out),
//       b0_mem  (16x8), w1_mem (16*10 = 160x8), b1_mem (10x8).
//   * One R=1, C=4 mac_array. Each cycle in S_*_STR: broadcast act_mem[k]
//     to all 4 PEs, fan out 4 contiguous weights w[k, tile*C + 0..3].
//   * After accumulation: drain 4 PE accumulators in parallel, add bias,
//     ReLU + saturate at +127 for layer 0 (matches dnn_inference.tiny_mlp_forward),
//     leave layer-1 output at full ACC_BITS for argmax downstream.
//
// FSM (8 states):
//   IDLE -> L0_INIT -> L0_STR(L0_IN cycles) -> L0_DRN -> loop L0_NTILES
//        -> L1_INIT -> L1_STR(L1_IN cycles) -> L1_DRN -> loop L1_NTILES
//        -> DONE
//
// Cycle count per inference (defaults C=4, L0_IN=64, L0_OUT=16, L1_OUT=10):
//   L0: 4 tiles * (1+64+1) = 264 cycles
//   L1: 3 tiles * (1+16+1) =  54 cycles
//   total ~320 cycles, plus a few transition cycles. ~6.4 us @ 50 MHz.
//
// Rounding mode (K_PARAM, MODE, ACA_W) are compile-time parameters of the
// mac_array; per Contribution A, a bitstream is built per design point and
// the trio (trunc, round, stoch) is compared by Quartus PowerPlay.
//
// Stochastic mode (MODE=2) needs a per-cycle random offset on each PE's rnd
// input. v1 ties them to 0 (effectively trunc); a small LFSR module gets
// dropped in for the demo before bitstream gen.
// ==========================================================================

`default_nettype none

module mlp_top #(
    parameter integer A_BITS    = 8,
    parameter integer ACC_BITS  = 32,
    parameter integer L0_IN     = 64,
    parameter integer L0_OUT    = 16,
    parameter integer L1_OUT    = 10,
    parameter integer C         = 4,        // parallel output PEs (mac_array.C)
    parameter integer K_PARAM   = 0,        // 0..A_BITS+A_BITS-1
    parameter integer MODE      = 0,        // 0=trunc 1=round 2=stoch
    parameter integer ACA_W     = 32,       // ACA accumulator window
    parameter         X_FILE    = "",
    parameter         W0_FILE   = "",
    parameter         B0_FILE   = "",
    parameter         W1_FILE   = "",
    parameter         B1_FILE   = ""
) (
    input  wire                                clk,
    input  wire                                rst_n,
    input  wire                                start,
    output reg                                 done,
    output wire [L1_OUT*ACC_BITS-1:0]          result_flat
);

    localparam integer L1_IN     = L0_OUT;
    localparam integer L0_NTILES = (L0_OUT + C - 1) / C;   // = 4
    localparam integer L1_NTILES = (L1_OUT + C - 1) / C;   // = 3
    localparam integer P_BITS    = A_BITS + A_BITS;
    localparam signed [A_BITS-1:0] A_MAX_S = (1 << (A_BITS-1)) - 1;  // +127

    // ----------------------------------------------------------------------
    // On-chip memories preloaded via $readmemh.
    // act_mem (64 x 8): read-only after init, holds the layer-0 input X.
    // h_mem   (16 x 8): layer-0 -> layer-1 hidden activations. MUST be a
    //                   separate array; reusing act_mem corrupts later
    //                   layer-0 tiles which still need the original X.
    // ----------------------------------------------------------------------
    reg [A_BITS-1:0] act_mem [0:L0_IN-1];
    reg [A_BITS-1:0] h_mem   [0:L0_OUT-1];
    reg [A_BITS-1:0] w0_mem  [0:L0_IN*L0_OUT-1];
    reg [A_BITS-1:0] w1_mem  [0:L1_IN*L1_OUT-1];
    reg [A_BITS-1:0] b0_mem  [0:L0_OUT-1];
    reg [A_BITS-1:0] b1_mem  [0:L1_OUT-1];

    initial begin
        if (X_FILE  != "") $readmemh(X_FILE,  act_mem);
        if (W0_FILE != "") $readmemh(W0_FILE, w0_mem);
        if (B0_FILE != "") $readmemh(B0_FILE, b0_mem);
        if (W1_FILE != "") $readmemh(W1_FILE, w1_mem);
        if (B1_FILE != "") $readmemh(B1_FILE, b1_mem);
    end

    // ----------------------------------------------------------------------
    // Output registers (L1_OUT x ACC_BITS), packed into result_flat.
    // ----------------------------------------------------------------------
    reg [ACC_BITS-1:0] y_reg [0:L1_OUT-1];
    genvar gi;
    generate
        for (gi = 0; gi < L1_OUT; gi = gi + 1) begin : g_result_pack
            assign result_flat[gi*ACC_BITS +: ACC_BITS] = y_reg[gi];
        end
    endgenerate

    // ----------------------------------------------------------------------
    // FSM control signals
    // ----------------------------------------------------------------------
    localparam [2:0] S_IDLE    = 3'd0;
    localparam [2:0] S_L0_INIT = 3'd1;
    localparam [2:0] S_L0_STR  = 3'd2;
    localparam [2:0] S_L0_DRN  = 3'd3;
    localparam [2:0] S_L1_INIT = 3'd4;
    localparam [2:0] S_L1_STR  = 3'd5;
    localparam [2:0] S_L1_DRN  = 3'd6;
    localparam [2:0] S_DONE    = 3'd7;

    reg [2:0] state;
    reg [7:0] k_cnt;
    reg [3:0] tile_cnt;
    reg       layer1;            // 0 = layer 0, 1 = layer 1

    // Combinational control signals driven straight from `state` -- mac_array
    // sees them with NO 1-cycle delay. Registering these via <= caused k_cnt=0
    // to land in the same cycle as clear=1, dropping the first MAC per tile.
    wire array_clear  = (state == S_L0_INIT) || (state == S_L1_INIT);
    wire array_mac_en = (state == S_L0_STR)  || (state == S_L1_STR);

    // ----------------------------------------------------------------------
    // mac_array (R=1, C=C). Stochastic rnd inputs tied to 0 for v1; an LFSR
    // wrapper drops in before final bitstream.
    // ----------------------------------------------------------------------
    wire [A_BITS-1:0]              array_acts_flat;
    wire [C*A_BITS-1:0]            array_wgts_flat;
    wire [C*P_BITS-1:0]            array_rnd_flat;
    wire [C*ACC_BITS-1:0]          array_acc_flat;

    mac_array #(
        .R(1), .C(C),
        .A_BITS(A_BITS), .B_BITS(A_BITS),
        .ACC_BITS(ACC_BITS),
        .WINDOW(ACA_W)
    ) u_array (
        .clk(clk), .rst_n(rst_n),
        .clear(array_clear),
        .mac_en(array_mac_en),
        .K(K_PARAM[3:0]),
        .mode(MODE[1:0]),
        .acts_flat(array_acts_flat),
        .wgts_flat(array_wgts_flat),
        .rnd_flat(array_rnd_flat),
        .acc_flat(array_acc_flat)
    );

    // ----------------------------------------------------------------------
    // Stochastic random source. Only instantiated when MODE == 2 so the
    // trunc / round bitstreams don't pay for the LFSR -- this is exactly the
    // hardware-cost gap Contribution A quantifies. C*P_BITS = 64 bits feed
    // the four PEs (16 bits each); a 64-bit LFSR fills it in one slice.
    // ----------------------------------------------------------------------
    generate
        if (MODE == 2) begin : g_lfsr
            wire [63:0] lfsr_state;
            lfsr #(.WIDTH(64)) u_lfsr (
                .clk(clk), .rst_n(rst_n),
                .en(array_mac_en),
                .state(lfsr_state)
            );
            assign array_rnd_flat = lfsr_state[C*P_BITS-1:0];
        end else begin : g_no_lfsr
            assign array_rnd_flat = {C*P_BITS{1'b0}};
        end
    endgenerate

    // ----------------------------------------------------------------------
    // Combinational input/weight drive (selected by layer1).
    //   acts = act_mem[k_cnt]
    //   wgts[c] = w0_mem[k_cnt*L0_OUT + tile_cnt*C + c]   for layer 0
    //   wgts[c] = w1_mem[k_cnt*L1_OUT + tile_cnt*C + c]   for layer 1
    // ----------------------------------------------------------------------
    // Layer 0 streams X from act_mem; layer 1 streams the hidden activations
    // from h_mem (which was written during the layer-0 drains).
    assign array_acts_flat = layer1 ? h_mem[k_cnt] : act_mem[k_cnt];

    genvar gc;
    generate
        for (gc = 0; gc < C; gc = gc + 1) begin : g_weight_mux
            wire [$clog2(L0_IN*L0_OUT)-1:0] a0;
            wire [$clog2(L1_IN*L1_OUT)-1:0] a1;
            assign a0 = k_cnt * L0_OUT + tile_cnt * C + gc;
            assign a1 = k_cnt * L1_OUT + tile_cnt * C + gc;
            assign array_wgts_flat[gc*A_BITS +: A_BITS] =
                layer1 ? w1_mem[a1] : w0_mem[a0];
        end
    endgenerate

    // ----------------------------------------------------------------------
    // Per-PE drain: bias add, ReLU+saturate (layer 0) or pass-through (layer 1).
    // These are combinational; the FSM writes them into act_mem or y_reg.
    // ----------------------------------------------------------------------
    wire signed [ACC_BITS-1:0] drained_l0 [0:C-1];
    wire signed [ACC_BITS-1:0] drained_l1 [0:C-1];
    wire        [A_BITS-1:0]   drained_l0_relu [0:C-1];

    generate
        for (gc = 0; gc < C; gc = gc + 1) begin : g_drain
            wire signed [A_BITS-1:0] b0_v = b0_mem[tile_cnt*C + gc];
            wire signed [A_BITS-1:0] b1_v = b1_mem[tile_cnt*C + gc];
            wire signed [ACC_BITS-1:0] b0_ext = {{(ACC_BITS-A_BITS){b0_v[A_BITS-1]}}, b0_v};
            wire signed [ACC_BITS-1:0] b1_ext = {{(ACC_BITS-A_BITS){b1_v[A_BITS-1]}}, b1_v};
            wire signed [ACC_BITS-1:0] acc_v  = $signed(array_acc_flat[gc*ACC_BITS +: ACC_BITS]);
            assign drained_l0[gc] = acc_v + b0_ext;
            assign drained_l1[gc] = acc_v + b1_ext;
            assign drained_l0_relu[gc] =
                (drained_l0[gc] <= $signed({ACC_BITS{1'b0}}))  ? {A_BITS{1'b0}} :
                (drained_l0[gc] >  $signed({{(ACC_BITS-A_BITS){1'b0}}, A_MAX_S})) ? A_MAX_S :
                                                                drained_l0[gc][A_BITS-1:0];
        end
    endgenerate

    // ----------------------------------------------------------------------
    // FSM
    // ----------------------------------------------------------------------
    integer di;
    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            k_cnt    <= 8'd0;
            tile_cnt <= 4'd0;
            layer1   <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done     <= 1'b0;
                    tile_cnt <= 4'd0;
                    layer1   <= 1'b0;
                    if (start) state <= S_L0_INIT;
                end

                // Single-cycle clear: combinational array_clear=1 forces PE
                // acc_r<=0 at THIS posedge. Next cycle enters S_L0_STR with
                // a clean accumulator and mac_en=1 driving k_cnt=0's MAC.
                S_L0_INIT: begin
                    k_cnt <= 8'd0;
                    state <= S_L0_STR;
                end

                S_L0_STR: begin
                    k_cnt <= k_cnt + 8'd1;
                    if (k_cnt == L0_IN - 1) begin
                        state <= S_L0_DRN;
                    end
                end

                S_L0_DRN: begin
                    // Write 4 PE outputs (with bias + ReLU + sat) into h_mem.
                    // Keep act_mem intact -- later layer-0 tiles still need X.
                    for (di = 0; di < C; di = di + 1) begin
                        if (tile_cnt * C + di < L0_OUT) begin
                            h_mem[tile_cnt*C + di] <= drained_l0_relu[di];
                        end
                    end
                    if (tile_cnt == L0_NTILES - 1) begin
                        tile_cnt <= 4'd0;
                        layer1   <= 1'b1;
                        state    <= S_L1_INIT;
                    end else begin
                        tile_cnt <= tile_cnt + 4'd1;
                        state    <= S_L0_INIT;
                    end
                end

                S_L1_INIT: begin
                    k_cnt <= 8'd0;
                    state <= S_L1_STR;
                end

                S_L1_STR: begin
                    k_cnt <= k_cnt + 8'd1;
                    if (k_cnt == L1_IN - 1) begin
                        state <= S_L1_DRN;
                    end
                end

                S_L1_DRN: begin
                    // Write 4 PE outputs (with bias, no ReLU) into y_reg.
                    for (di = 0; di < C; di = di + 1) begin
                        if (tile_cnt * C + di < L1_OUT) begin
                            y_reg[tile_cnt*C + di] <= drained_l1[di];
                        end
                    end
                    if (tile_cnt == L1_NTILES - 1) begin
                        state <= S_DONE;
                    end else begin
                        tile_cnt <= tile_cnt + 4'd1;
                        state    <= S_L1_INIT;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    if (!start) state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
