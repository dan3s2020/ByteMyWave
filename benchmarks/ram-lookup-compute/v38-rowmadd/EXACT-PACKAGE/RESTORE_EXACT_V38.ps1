$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$parts = Get-ChildItem -LiteralPath $Here -Filter 'TRANSIT_V38_ROWMADD.zip.b64.part*' | Sort-Object Name
if ($parts.Count -ne 6) { throw "Expected 6 package parts, found $($parts.Count)." }
$b64 = ($parts | ForEach-Object { [IO.File]::ReadAllText($_.FullName) }) -join ''
$zip = Join-Path $Here 'TRANSIT_V38_ROWMADD.zip'
[IO.File]::WriteAllBytes($zip, [Convert]::FromBase64String($b64))
$zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
$expectedZip = '4b6a717f5d30725246088a07783c754100183fe59b46c10023cf79b0709b8088'
if ($zipHash -ne $expectedZip) { throw "ZIP SHA256 mismatch: $zipHash" }
$dst = Join-Path $Here 'RESTORED'
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
Expand-Archive -LiteralPath $zip -DestinationPath $dst -Force
$root = Join-Path $dst 'TRANSIT_V38_ROWMADD'
$cpp = Join-Path $root 'v38_rowmadd.cpp'
$run = Join-Path $root 'RUN.ps1'
$manifest = Join-Path $root 'MANIFEST.txt'
$checks = @(
  @{Path=$cpp; Expected='61a1d4aa0364a1f3d8daa0b9d704b4b7339b25f97a86771fc4649d538714bf45'},
  @{Path=$run; Expected='c203bbb4d010528b17851614d78c9a353a905a0d62abc9660f1ca3b0b8ee2a07'},
  @{Path=$manifest; Expected='434969822df8f21af08fba113ce073d866d691f456d4a210fea4785e81a60f87'}
)
foreach ($c in $checks) {
  $h=(Get-FileHash -Algorithm SHA256 -LiteralPath $c.Path).Hash.ToLowerInvariant()
  if ($h -ne $c.Expected) { throw "SHA256 mismatch for $($c.Path): $h" }
}
Write-Host 'V38 EXACT PACKAGE RESTORE: PASS' -ForegroundColor Green
Write-Host "ZIP SHA256: $zipHash"
Write-Host "CPP SHA256: $((Get-FileHash -Algorithm SHA256 -LiteralPath $cpp).Hash.ToLowerInvariant())"
Write-Host "RUN SHA256: $((Get-FileHash -Algorithm SHA256 -LiteralPath $run).Hash.ToLowerInvariant())"
Write-Host "Exact source: $cpp"
