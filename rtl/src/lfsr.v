// ==========================================================================
// lfsr.v -- maximal-length Fibonacci LFSR, the per-MAC random source for
// stochastic rounding (Gupta et al. 2015) in mac_unit.
//
// A WIDTH-bit left-shifting LFSR; the feedback XORs the taps of a maximal
// polynomial so the state cycles through all 2^WIDTH-1 non-zero values.
// Default WIDTH=64 with taps {64,63,61,60} (a known maximal set), giving a
// period of 2^64-1 -- far longer than any inference run.
//
// `en` gates the shift so the sequence only advances on active MAC cycles,
// keeping the offset reproducible for a given schedule. The output `state`
// is sliced by the consumer (mlp_top feeds 16 bits to each of the C PEs).
//
// This is the hardware cost the project's Contribution A argues against:
// stochastic rounding removes bias but needs this LFSR + an adder per MAC,
// whereas deterministic `round` reaches near-zero bias with just the adder.
// ==========================================================================

`default_nettype none

module lfsr #(
    parameter integer WIDTH = 64,
    parameter [63:0]  SEED  = 64'hDEAD_BEEF_CAFE_BABE
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             en,
    output reg  [WIDTH-1:0] state
);

    // Feedback taps for the default 64-bit maximal polynomial
    // x^64 + x^63 + x^61 + x^60 + 1 (bits 63,62,60,59 zero-indexed).
    wire feedback = state[63] ^ state[62] ^ state[60] ^ state[59];

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= SEED[WIDTH-1:0];
        end else if (en) begin
            state <= {state[WIDTH-2:0], feedback};
        end
    end

endmodule

`default_nettype wire
