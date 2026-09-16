# V38 research notes

The external mechanism check used two current references:

- Microsoft BitNet CPU optimization uses configurable row/column tiling and weight/activation parallelism, reinforcing the direction of row-major compiled layouts rather than random DRAM LUTs.
- uops.info reports Ivy Bridge XMM VPMADDUBSW as 1 uop on p0, latency 5, throughput 1/cycle; VPSHUFB has latency 1 and measured throughput 0.5 cycles, making shuffle-based compact scale expansion attractive versus keeping PMULLD in the dot loop.

V38 therefore tests compact shuffle expansion, ILP2, and a pre-expanded RAM-heavy control separately.
