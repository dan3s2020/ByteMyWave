# V38 exact benchmark package

This directory preserves the exact `TRANSIT_V38_ROWMADD.zip` artifact that produced the recorded V38 result. It is **not** a reconstruction from design notes.

Authoritative hashes recovered from the original archive used in the benchmark:

- package ZIP SHA256: `4b6a717f5d30725246088a07783c754100183fe59b46c10023cf79b0709b8088`
- `v38_rowmadd.cpp` SHA256: `61a1d4aa0364a1f3d8daa0b9d704b4b7339b25f97a86771fc4649d538714bf45`
- `RUN.ps1` SHA256: `c203bbb4d010528b17851614d78c9a353a905a0d62abc9660f1ca3b0b8ee2a07`
- original `MANIFEST.txt` SHA256: `434969822df8f21af08fba113ce073d866d691f456d4a210fea4785e81a60f87`

The ZIP bytes are stored losslessly as six base64 text parts because the repository connector writes UTF-8 text. Run `RESTORE_EXACT_V38.ps1` in this directory. The script concatenates the parts, recreates the exact ZIP, verifies the package SHA256, extracts it, then verifies the exact C++ source and runner hashes.

**Rule:** never reproduce V38 by reimplementing the README. Restore this package and use the extracted `v38_rowmadd.cpp` and `RUN.ps1`.
