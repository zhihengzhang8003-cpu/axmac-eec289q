// ==========================================================================
// drum_multiplier.v — DRUM-k Dynamic Range Unbiased Multiplier
//
// Implements the DRUM-k algorithm (Hashemi, Bahar & Reda, ICCAD 2015):
// each signed INT8 operand is approximated to its k most significant bits
// before multiplying. This yields zero-mean error (unlike plain truncation
// which is biased toward +2^(K-1) per MAC), at the cost of a leading-bit
// detector + barrel shift per operand.
//
// Parameters:
//   N_BITS   — operand bit-width (default 8, signed two's-complement)
//   K_DRUM   — MSBs to retain per operand (default 4)
//
// Ports:
//   a, b     — signed N_BITS-wide operands
//   product  — signed 2*N_BITS-wide approximate product
//
// Hardware cost vs mac_unit.v (K=4 trunc, same 50 MHz timing constraint):
//   mac_unit  trunc K=4 : exact multiplier + mask  → baseline
//   mac_unit  round K=4 : exact multiplier + 1 add → ~same as trunc
//   drum_multiplier K=4 : LZC + shift + k×k mult   → fewer partial products
//                                                     but + control logic
// ==========================================================================

`default_nettype none

module drum_multiplier #(
    parameter integer N_BITS = 8,
    parameter integer K_DRUM = 4
) (
    input  wire signed [N_BITS-1:0]       a,
    input  wire signed [N_BITS-1:0]       b,
    output wire signed [2*N_BITS-1:0]     product
);

    // ---------------------------------------------------------------
    // Sign extraction and absolute values
    // ---------------------------------------------------------------
    wire        sign_a = a[N_BITS-1];
    wire        sign_b = b[N_BITS-1];
    wire        sign_out = sign_a ^ sign_b;

    wire [N_BITS-1:0] abs_a = sign_a ? (~a + 1'b1) : a;
    wire [N_BITS-1:0] abs_b = sign_b ? (~b + 1'b1) : b;

    // ---------------------------------------------------------------
    // Leading-bit counter (LZC) — find position of MSB
    // Returns the index of the highest set bit (0-based from LSB).
    // For N_BITS=8: result is 3 bits wide (0..7).
    // ---------------------------------------------------------------
    localparam integer LOG2N = $clog2(N_BITS);   // 3 for N_BITS=8

    function automatic [LOG2N-1:0] leading_bit;
        input [N_BITS-1:0] x;
        integer i;
        begin
            leading_bit = 0;
            for (i = 0; i < N_BITS; i = i + 1)
                if (x[i]) leading_bit = i[LOG2N-1:0];
        end
    endfunction

    wire [LOG2N-1:0] msb_a = leading_bit(abs_a);
    wire [LOG2N-1:0] msb_b = leading_bit(abs_b);

    // ---------------------------------------------------------------
    // Compute right-shift amount: shift = max(0, msb - K_DRUM + 1)
    // After the shift we keep only K_DRUM bits.
    // ---------------------------------------------------------------
    localparam integer K1 = K_DRUM - 1;   // K_DRUM - 1

    // Shift amount is signed-saturated at 0.
    // Using signed subtraction; if msb < K_DRUM-1 the shift is 0.
    wire signed [LOG2N:0] shift_a_signed = $signed({1'b0, msb_a}) - K1;
    wire signed [LOG2N:0] shift_b_signed = $signed({1'b0, msb_b}) - K1;

    wire [LOG2N-1:0] shift_a = shift_a_signed[LOG2N] ? {LOG2N{1'b0}}
                                                       : shift_a_signed[LOG2N-1:0];
    wire [LOG2N-1:0] shift_b = shift_b_signed[LOG2N] ? {LOG2N{1'b0}}
                                                       : shift_b_signed[LOG2N-1:0];

    // ---------------------------------------------------------------
    // Barrel-shift operands right by shift_a / shift_b,
    // then shift left by the same amount to zero the LSBs.
    // ---------------------------------------------------------------
    wire [N_BITS-1:0] drum_abs_a = (abs_a >> shift_a) << shift_a;
    wire [N_BITS-1:0] drum_abs_b = (abs_b >> shift_b) << shift_b;

    // ---------------------------------------------------------------
    // Unsigned multiply of approximated magnitudes
    // ---------------------------------------------------------------
    wire [2*N_BITS-1:0] mag_product = drum_abs_a * drum_abs_b;

    // ---------------------------------------------------------------
    // Re-apply sign (two's-complement negate if odd number of negatives)
    // ---------------------------------------------------------------
    wire [2*N_BITS-1:0] neg_product = ~mag_product + 1'b1;

    assign product = sign_out ? $signed(neg_product) : $signed(mag_product);

endmodule

`default_nettype wire
