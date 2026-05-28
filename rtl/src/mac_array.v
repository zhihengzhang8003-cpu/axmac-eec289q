// ==========================================================================
// mac_array.v -- R x C output-stationary approximate-MAC array.
//
// Topology:
//
//   * R activation values broadcast along rows: PE(r, c) sees acts[r].
//   * C weight values broadcast along columns:  PE(r, c) sees wgts[c].
//   * Each PE = mac_unit (a*b with K-bit rounding) + aca_adder (W-window
//                 accumulator) + one ACC_BITS register.
//
// On every clk edge with mac_en=1 the array does one matmul step in parallel:
//
//   for (r, c):
//     acc[r, c] += round_K_mode(acts[r] * wgts[c])      via aca_adder(W)
//
// Driver streams the K dimension over time; after N_K cycles each PE holds
// one element of an R x C output tile. `clear` resets all PE accumulators.
//
// Vendor-agnostic. WINDOW (ACA carry-window) is a compile-time parameter so
// the design-space sweep instantiates a distinct array per W setting -- the
// way a real chip would tape out swapping accumulator IPs.
//
// Port arrays are flattened to wide buses for portability (some older
// simulators choke on unpacked-array ports). Internal generate-for re-slices
// them into per-PE wires.
// ==========================================================================

`default_nettype none

module mac_array #(
    parameter integer R        = 4,
    parameter integer C        = 4,
    parameter integer A_BITS   = 8,
    parameter integer B_BITS   = 8,
    parameter integer ACC_BITS = 32,
    parameter integer WINDOW   = 32      // ACA accumulator window
) (
    input  wire                            clk,
    input  wire                            rst_n,
    input  wire                            clear,      // sync clear all accs
    input  wire                            mac_en,     // accumulate this cycle
    input  wire [3:0]                      K,
    input  wire [1:0]                      mode,
    input  wire [R*A_BITS-1:0]             acts_flat,  // R activations
    input  wire [C*B_BITS-1:0]             wgts_flat,  // C weights
    input  wire [R*C*(A_BITS+B_BITS)-1:0]  rnd_flat,   // per-PE rnd (stoch)
    output wire [R*C*ACC_BITS-1:0]         acc_flat
);

    localparam integer P_BITS = A_BITS + B_BITS;

    genvar r, c;
    generate
        for (r = 0; r < R; r = r + 1) begin : g_row
            wire signed [A_BITS-1:0] a_r;
            assign a_r = acts_flat[r*A_BITS +: A_BITS];

            for (c = 0; c < C; c = c + 1) begin : g_col
                wire signed [B_BITS-1:0] b_c;
                assign b_c = wgts_flat[c*B_BITS +: B_BITS];

                wire [P_BITS-1:0] rnd_rc;
                assign rnd_rc = rnd_flat[(r*C + c)*P_BITS +: P_BITS];

                wire signed [P_BITS-1:0] prod;
                mac_unit #(.A_BITS(A_BITS), .B_BITS(B_BITS)) u_mac (
                    .a(a_r), .b(b_c),
                    .K(K), .mode(mode), .rnd(rnd_rc),
                    .product_rounded(prod)
                );

                // Sign-extend the P_BITS product up to ACC_BITS.
                wire signed [ACC_BITS-1:0] prod_ext;
                assign prod_ext = {{(ACC_BITS-P_BITS){prod[P_BITS-1]}}, prod};

                reg  signed [ACC_BITS-1:0] acc_r;
                wire signed [ACC_BITS-1:0] sum_w;

                aca_adder #(.BITS(ACC_BITS), .WINDOW(WINDOW)) u_add (
                    .a(acc_r), .b(prod_ext), .sum(sum_w)
                );

                always @(posedge clk) begin
                    if (!rst_n)       acc_r <= {ACC_BITS{1'b0}};
                    else if (clear)   acc_r <= {ACC_BITS{1'b0}};
                    else if (mac_en)  acc_r <= sum_w;
                end

                assign acc_flat[(r*C + c)*ACC_BITS +: ACC_BITS] = acc_r;
            end
        end
    endgenerate

endmodule

`default_nettype wire
