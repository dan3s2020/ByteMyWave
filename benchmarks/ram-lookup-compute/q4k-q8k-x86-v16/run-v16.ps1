param(
    [Parameter(Mandatory = $true)]
    [string]$ModelBlob,
    [string]$Tensor = "blk.0.ffn_gate.weight",
    [ValidateSet("Native", "Ivy", "Both")]
    [string]$Target = "Both",
    [int]$Iters = 21,
    [int]$Threads = 8
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$source = Join-Path $PSScriptRoot "transit_q4k_q8k_v16.cpp"
if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
if (-not (Test-Path -LiteralPath $ModelBlob)) { throw "Model blob not found: $ModelBlob" }
$compiler = $null
foreach ($name in @("g++", "clang++")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { $compiler = $cmd.Source; break }
}
if (-not $compiler) {
    $candidate = "C:\Strawberry\c\bin\g++.exe"
    if (Test-Path $candidate) { $compiler = $candidate }
}
if (-not $compiler) { throw "No g++/clang++ found. Expected Strawberry g++ at C:\Strawberry\c\bin\g++.exe" }
Write-Host "Transit V16 production kernel" -ForegroundColor Cyan
Write-Host "Compiler : $compiler"
Write-Host "Model    : $ModelBlob"
Write-Host "Tensor   : $Tensor"
Write-Host "Iters    : $Iters"
Write-Host "Threads  : $Threads"
Write-Host ""
function Build-And-Run {
    param([Parameter(Mandatory = $true)][string]$Name,[Parameter(Mandatory = $true)][string[]]$ArchFlags)
    $exe = Join-Path $PSScriptRoot ("transit-v16-{0}.exe" -f $Name.ToLowerInvariant())
    $flags = @("-O3","-std=c++17","-fomit-frame-pointer","-pthread","-static-libgcc","-static-libstdc++") + $ArchFlags
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " BUILD $Name" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    & $compiler @flags $source -o $exe
    if ($LASTEXITCODE -ne 0) { throw "$Name compilation failed: $LASTEXITCODE" }
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host " RUN $Name" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    & $exe --model $ModelBlob --tensor $Tensor --iters $Iters --threads $Threads
    if ($LASTEXITCODE -ne 0) { throw "$Name benchmark failed: $LASTEXITCODE" }
    Write-Host ""
}
switch ($Target) {
    "Native" { Build-And-Run -Name "Native" -ArchFlags @("-march=native", "-mtune=native") }
    "Ivy" { Build-And-Run -Name "Ivy" -ArchFlags @("-march=ivybridge", "-mtune=ivybridge", "-mno-avx2") }
    "Both" {
        Build-And-Run -Name "Native" -ArchFlags @("-march=native", "-mtune=native")
        Build-And-Run -Name "Ivy" -ArchFlags @("-march=ivybridge", "-mtune=ivybridge", "-mno-avx2")
    }
}
