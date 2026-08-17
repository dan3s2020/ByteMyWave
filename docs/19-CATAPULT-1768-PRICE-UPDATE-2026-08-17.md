# 19 — Microsoft FPGA 1768 / Catapult V3 price update — 2026-08-17

This note records a materially better current price for the Microsoft FPGA 1768 / Catapult V3 family already tracked by Transit.

## Exact board / part numbers

Observed current listings identify the card as:

- Microsoft FPGA 1768
- `M1040125-001`
- `M1037382-001`
- `M1030299-001`
- `MSIP-REM-MSK-1768`

This is the same Catapult V3 / Longs Peak family previously investigated for Transit.

## Current price improvement

A current PIOSPARTS listing shows the tested/refurbished board at **EUR 49.99**, down from the approximately EUR 59 price level previously tracked.

A separate IT-Remarketing listing remains at EUR 59 and states 100 units available, confirming meaningful European surplus volume for the family even though the cheapest listing's exact stock count is not exposed in the search result.

A US eBay listing for a Microsoft 1768 / YF5GM unit is also active at **US$35.99 or best offer**, with 3 units available and 6 sold. The exact subassembly/variant should be confirmed from board photos before treating it as equivalent to the full `M1040125-001` Longs Peak card.

## Transit-relevant architecture

Previously documented reverse engineering for this family gives the key properties:

- Intel/Altera Arria 10 FPGA;
- two independent DDR4 memory interfaces on the Longs Peak board;
- dual FPGA-facing PCIe Gen3 x8 endpoint paths on the full card architecture;
- USB/JTAG/programming access;
- public community Quartus projects, pinout work, memory-controller bring-up and PCIe experiments.

This means the card remains one of the strongest fully programmable enterprise-surplus Transit prototypes, although it uses DDR4 rather than DDR3.

## Why the EUR 49.99 price matters

At EUR 49.99, the cost per local programmable memory path is materially better than at the earlier EUR 59 price and the board becomes more attractive as a multi-board lab platform.

It still does not beat very cheap Storey Peak cards on raw dollars per memory channel, but it offers substantially better integration per endpoint: a larger FPGA, two local memory paths and a much stronger documented PCIe architecture on one board.

## Current verdict

**Material price improvement; still buy/test-one or small-lot tier, not automatic bulk-buy tier.**

Before a bulk Transit deployment, the remaining gates are unchanged:

1. confirm exact populated memory capacity on the purchased revision;
2. validate R920 enumeration;
3. validate operation through the intended x1/mining-riser-style downstream topology if that topology is required;
4. measure sustained local-memory bandwidth with the Transit access pattern;
5. port and benchmark the local compute/reduction kernel;
6. compare total cost per sustained GB/s against Storey Peak and any future 4–8-channel surplus tile.
