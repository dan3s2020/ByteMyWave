// Transit GPU — exact 64-element signed INT4 × INT8 bitplane dot core.
//
// This is a board-independent proof core. It deliberately does not contain
// PCIe, DDR3 or MXFP logic. Those wrappers feed it one 64-element block at a
// time as 4 weight bitplanes and 8 activation bitplanes.
//
// Arithmetic:
//   q = b0 + 2*b1 + 4*b2 - 8*b3
//   x = a0 + 2*a1 + 4*a2 + 8*a3 + 16*a4 + 32*a5 + 64*a6 - 128*a7
//
// For each weight plane Wi we calculate the signed sum of activations selected
// by Wi, then combine the four sums with [1, 2, 4, -8]. This is the same exact
// integer identity verified by the host reference; it is not yet MXFP4/MXFP8.

module transit_bitplane_dot64 (
    input  logic         in_valid,

    input  logic [63:0]  w0,
    input  logic [63:0]  w1,
    input  logic [63:0]  w2,
    input  logic [63:0]  w3,

    input  logic [63:0]  a0,
    input  logic [63:0]  a1,
    input  logic [63:0]  a2,
    input  logic [63:0]  a3,
    input  logic [63:0]  a4,
    input  logic [63:0]  a5,
    input  logic [63:0]  a6,
    input  logic [63:0]  a7,

    output logic         out_valid,
    output logic signed [63:0] dot
);

    function automatic logic [6:0] popcount64(input logic [63:0] x);
        integer i;
        logic [6:0] count;
        begin
            count = 7'd0;
            for (i = 0; i < 64; i = i + 1)
                count = count + x[i];
            return count;
        end
    endfunction

    // Sum the signed INT8 activations whose positions are selected by mask.
    function automatic logic signed [63:0] selected_int8_sum(
        input logic [63:0] mask,
        input logic [63:0] pa0,
        input logic [63:0] pa1,
        input logic [63:0] pa2,
        input logic [63:0] pa3,
        input logic [63:0] pa4,
        input logic [63:0] pa5,
        input logic [63:0] pa6,
        input logic [63:0] pa7
    );
        logic signed [63:0] c0, c1, c2, c3, c4, c5, c6, c7;
        logic signed [63:0] sum;
        begin
            c0 = $signed({57'd0, popcount64(mask & pa0)});
            c1 = $signed({57'd0, popcount64(mask & pa1)});
            c2 = $signed({57'd0, popcount64(mask & pa2)});
            c3 = $signed({57'd0, popcount64(mask & pa3)});
            c4 = $signed({57'd0, popcount64(mask & pa4)});
            c5 = $signed({57'd0, popcount64(mask & pa5)});
            c6 = $signed({57'd0, popcount64(mask & pa6)});
            c7 = $signed({57'd0, popcount64(mask & pa7)});

            sum =  c0
                 + (c1 <<< 1)
                 + (c2 <<< 2)
                 + (c3 <<< 3)
                 + (c4 <<< 4)
                 + (c5 <<< 5)
                 + (c6 <<< 6)
                 - (c7 <<< 7);
            return sum;
        end
    endfunction

    logic signed [63:0] s0, s1, s2, s3;

    always_comb begin
        s0 = selected_int8_sum(w0, a0, a1, a2, a3, a4, a5, a6, a7);
        s1 = selected_int8_sum(w1, a0, a1, a2, a3, a4, a5, a6, a7);
        s2 = selected_int8_sum(w2, a0, a1, a2, a3, a4, a5, a6, a7);
        s3 = selected_int8_sum(w3, a0, a1, a2, a3, a4, a5, a6, a7);

        dot = s0 + (s1 <<< 1) + (s2 <<< 2) - (s3 <<< 3);
        out_valid = in_valid;
    end

endmodule
