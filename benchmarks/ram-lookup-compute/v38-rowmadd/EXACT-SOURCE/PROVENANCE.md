# V38 exact source provenance

The authoritative V38 implementation is the exact original benchmark package preserved under `../EXACT-PACKAGE/`.

Do **not** reconstruct V38 from README/design notes and do not use older partial source fragments. Run:

```powershell
Set-Location ..\EXACT-PACKAGE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\RESTORE_EXACT_V38.ps1
```

The restore is accepted only when these hashes match:

- Original package SHA256: `4b6a717f5d30725246088a07783c754100183fe59b46c10023cf79b0709b8088`
- Original `v38_rowmadd.cpp` SHA256: `61a1d4aa0364a1f3d8daa0b9d704b4b7339b25f97a86771fc4649d538714bf45`
- Original `RUN.ps1` SHA256: `c203bbb4d010528b17851614d78c9a353a905a0d62abc9660f1ca3b0b8ee2a07`
- Original `MANIFEST.txt` SHA256: `434969822df8f21af08fba113ce073d866d691f456d4a210fea4785e81a60f87`

`RUN.ps1` is also preserved directly in this directory for inspection. The C++ file is restored byte-for-byte from the exact package; that restored file, identified by its SHA256 above, is the only source that may be called **exact V38**.
