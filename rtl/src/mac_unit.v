// ==========================================================================
// mac_unit.v -- approximate signed multiplier with K-bit low-end rounding.
//
// Hardware counterpart of axmac.approx_mac._apply_rounding(a*b, K, mode).
//
//   product_full     = $signed(a) * $signed(b)
//   product_rounded  = (product_full + offset) & ~((1<<K)-1)
//
// where
//   mode = TRUNC      : offset = 0                 (Mahdiani 2010 baseline)
//   mode = ROUND      : offset = (K==0) ? 0 : 1<<(K-1)
//                                                  (Schulte & Swartzlander 1993)
//   mode = STOCHASTIC : offset = rnd & ((1<<K)-1)  (Gupta 2015; rnd comes from
//                                                   an external LFSR so this
//                                                   block stays combinational
//                                                   and bit-exact testable.)
//
// K == 0 reduces to the exact product under all three modes (mask is zero).
//
// Two's-complement bitwise AND with ~mask matches Python's `int & ~mask` on
// arbitrary-precision ints, so the cocotb testbench can compare bit-for-bit
// against rtl/golden/mac_int8.csv.
// ==========================================================================

`default_nettype none

module mac_unit #(
    parameter integer A_BITS = 8,
    parameter integer B_BITS = 8
) (
    input  wire signed [A_BITS-1:0]            a,
    input  wire signed [B_BITS-1:0]            b,
    input  wire        [3:0]                   K,      // 0..A_BITS+B_BITS-1
    input  wire        [1:0]                   mode,   // 00 trunc, 01 round, 10 stoch
    input  wire        [A_BITS+B_BITS-1:0]     rnd,    // low K bits used in stoch
    output wire signed [A_BITS+B_BITS-1:0]     product_rounded
);

    localparam integer P_BITS = A_BITS + B_BITS;

    // ------------------------------------------------------------------
    // Mode encoding kept in one place so the testbench and any user RTL
    // share the same constants.
    // ------------------------------------------------------------------
    localparam [1:0] MODE_TRUNC      = 2'b00;
    localparam [1:0] MODE_ROUND      = 2'b01;
    localparam [1:0] MODE_STOCHASTIC = 2'b10;

    wire signed [P_BITS-1:0] product_full;
    assign product_full = $signed(a) * $signed(b);

    // Mask = (1<<K) - 1. For K==0 this is 0, so ~mask == all-ones and the
    // final AND is a no-op (exact product).
    wire [P_BITS-1:0] one_shifted;
    assign one_shifted = {{(P_BITS-1){1'b0}}, 1'b1} << K;
    wire [P_BITS-1:0] mask = one_shifted - {{(P_BITS-1){1'b0}}, 1'b1};

    // Round constant = (K==0) ? 0 : 1 << (K-1). Cheap: shift-left by (K-1).
    wire [P_BITS-1:0] round_const;
    assign round_const = (K == 4'd0)
                       ? {P_BITS{1'b0}}
                       : ({{(P_BITS-1){1'b0}}, 1'b1} << (K - 4'd1));

    // Stochastic offset = rnd masked to the low K bits.
    wire [P_BITS-1:0] stoch_offset = rnd & mask;

    reg [P_BITS-1:0] offset;
    always @* begin
        case (mode)
            MODE_TRUNC:      offset = {P_BITS{1'b0}};
            MODE_ROUND:      offset = round_const;
            MODE_STOCHASTIC: offset = stoch_offset;
            default:         offset = {P_BITS{1'b0}};
        endcase
    end

    wire signed [P_BITS-1:0] product_offset;
    assign product_offset = product_full + $signed({1'b0, offset[P_BITS-1:0]});

    // Clear low K bits in two's complement -- matches Python `int & ~mask`.
    assign product_rounded = product_offset & ~mask;

endmodule

`default_nettype wire
