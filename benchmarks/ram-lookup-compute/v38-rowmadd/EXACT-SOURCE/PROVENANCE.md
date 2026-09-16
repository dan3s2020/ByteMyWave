# V38 exact source provenance

This directory preserves the exact source artifact recovered from the original `TRANSIT_V38_ROWMADD.zip` that produced the recorded V38 benchmark. Do not reconstruct V38 from README/design notes when reproducing the result.

Original source SHA256: `61a1d4aa0364a1f3d8daa0b9d704b4b7339b25f97a86771fc4649d538714bf45`
Original runner SHA256: `c203bbb4d010528b17851614d78c9a353a905a0d62abc9660f1ca3b0b8ee2a07`
Original package SHA256: `4b6a717f5d30725246088a07783c754100183fe59b46c10023cf79b0709b8088`

The compressed/base64 source parts in this directory decode losslessly to `v38_rowmadd.cpp`; the restore script verifies the source SHA before accepting it.
