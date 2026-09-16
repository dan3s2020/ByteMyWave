param(
    [string]$Work = "$env:TEMP\q4k_exact_v11",
    [int]$Iterations = 60,
    [int]$Warmup = 8,
    [int]$Cpu = -1,
    [int]$EvictMiB = 64
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $Here "transit_v12_compare.cpp"

$Q4 = Join-Path $Work "tensor-q4k.bin"
$X = Join-Path $Work "activation-f32.bin"
$MetaPath = Join-Path $Work "meta.json"

foreach ($p in @($Source, $Q4, $X, $MetaPath)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "Missing required file: $p" }
}

$Meta = Get-Content $MetaPath -Raw | ConvertFrom-Json
if ($null -eq $Meta.PSObject.Properties["geometry_version"] -or [int]$Meta.geometry_version -lt 2) {
    throw "V12 refuses legacy fixture. Run the V11 geometry-correct preparation first."
}

$Rows = [int]$Meta.rows
$Cols = [int]$Meta.cols

$Gpp = Get-Command g++.exe, g++ -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $Gpp) { throw "g++ not found in PATH" }

Write-Host ""
Write-Host "============================================================"
Write-Host " TRANSIT V12 — APPLES-TO-APPLES"
Write-Host "============================================================"
Write-Host "Compiler        : $($Gpp.Source)"
Write-Host "Matrix          : $Rows rows x $Cols cols"
Write-Host "CPU affinity    : $Cpu (-1 = scheduler)"
Write-Host "Eviction buffer : $EvictMiB MiB"
Write-Host "Iterations      : $Iterations"
Write-Host "Warmup          : $Warmup"

$BaseFlags = @(
    $Source,
    "-O3",
    "-std=c++17",
    "-DNDEBUG",
    "-march=native",
    "-mtune=native",
    "-mavx2",
    "-mfma",
    "-mf16c",
    "-mssse3",
    "-msse4.1",
    "-Wall",
    "-Wextra"
)

function Build-And-Run {
    param([string]$Name,[string[]]$ExtraFlags)
    $Exe = Join-Path $Work ("transit_v12_{0}.exe" -f $Name)
    $Log = Join-Path $Work ("transit_v12_{0}-results.txt" -f $Name)
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " BUILD/RUN $Name" -ForegroundColor Cyan
    Write-Host "============================================================"
    $Flags = $BaseFlags + $ExtraFlags + @("-o", $Exe)
    & $Gpp.Source @Flags
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed for $Name" }
    $Args = @("--q4",$Q4,"--x",$X,"--rows",([string]$Rows),"--cols",([string]$Cols),"--iters",([string]$Iterations),"--warmup",([string]$Warmup),"--cpu",([string]$Cpu),"--evict-mb",([string]$EvictMiB))
    & $Exe @Args 2>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed for $Name" }
    return $Log
}

$LogNormal = Build-And-Run -Name "normal" -ExtraFlags @()
$LogNoUnroll = Build-And-Run -Name "no_unroll" -ExtraFlags @("-fno-unroll-loops")

Write-Host ""
Write-Host "============================================================"
Write-Host " DONE"
Write-Host "============================================================"
Write-Host "Normal log    : $LogNormal"
Write-Host "No-unroll log : $LogNoUnroll"
Write-Host ""
Write-Host "paired_speedup > 1 means x8meta wins; < 1 means llama-style AVX2 wins."
