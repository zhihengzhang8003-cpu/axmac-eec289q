// ==========================================================================
// aca_adder.v -- Accuracy-Configurable Approximate Adder.
//
// Hardware counterpart of axmac.approx_mac.aca_add. Splits a BITS-wide
// two's-complement add into fixed-width segments of WINDOW bits each; carry-out
// of segment i never reaches segment i+1. WINDOW >= BITS reduces to an exact
// two's-complement add.
//
// Kahng & Kang, DAC 2012.
//
// The window width is a *compile-time parameter*, not a runtime port: every
// downstream module (mac_array, mlp_top) is parameterised by W and creates a
// distinct instance per setting. This matches how an ASIC/FPGA team would
// physically swap accumulator IP per design point in the Pareto sweep.
// ==========================================================================

`default_nettype none

module aca_adder #(
    parameter integer BITS   = 32,
    parameter integer WINDOW = 4
) (
    input  wire signed [BITS-1:0] a,
    input  wire signed [BITS-1:0] b,
    output wire signed [BITS-1:0] sum
);

    // Number of full WINDOW-wide segments + one possibly-narrow tail segment.
    localparam integer N_FULL = BITS / WINDOW;
    localparam integer TAIL   = BITS - N_FULL * WINDOW;

    wire [BITS-1:0] out;

    genvar i;
    generate
        if (WINDOW >= BITS) begin : g_exact
            // Exact two's-complement add; carries propagate normally.
            assign out = a + b;
        end else begin : g_segmented
            // Each WINDOW-wide segment computes (a_seg + b_seg) mod 2^WINDOW
            // with no carry-in from the segment below it.
            for (i = 0; i < N_FULL; i = i + 1) begin : g_full
                assign out[i*WINDOW +: WINDOW] =
                    a[i*WINDOW +: WINDOW] + b[i*WINDOW +: WINDOW];
            end
            if (TAIL > 0) begin : g_tail
                assign out[BITS-1 -: TAIL] =
                    a[BITS-1 -: TAIL] + b[BITS-1 -: TAIL];
            end
        end
    endgenerate

    assign sum = out;

endmodule

`default_nettype wire
