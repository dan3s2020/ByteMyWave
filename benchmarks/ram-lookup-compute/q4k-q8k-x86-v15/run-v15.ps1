param(
    [Parameter(Mandatory = $true)]
    [string]$ModelBlob,

    [string]$Tensor = "blk.0.ffn_gate.weight",

    [ValidateSet("Native", "Ivy", "Both")]
    [string]$Target = "Both",

    [int]$Iters = 15
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$source = Join-Path $PSScriptRoot "transit_q4k_q8k_exact_v15.cpp"
if (-not (Test-Path -LiteralPath $source)) {
    throw "Missing source: $source"
}
if (-not (Test-Path -LiteralPath $ModelBlob)) {
    throw "Model blob not found: $ModelBlob"
}

$compiler = $null
foreach ($name in @("g++", "clang++")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) {
        $compiler = $cmd.Source
        break
    }
}
if (-not $compiler) {
    throw "No g++ or clang++ found in PATH. On this machine g++ should be under C:\Strawberry\c\bin."
}

Write-Host "Compiler : $compiler"
Write-Host "Source   : $source"
Write-Host "Model    : $ModelBlob"
Write-Host "Tensor   : $Tensor"
Write-Host "Target   : $Target"
Write-Host ""

function Build-And-Run {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$ArchFlags
    )

    $exe = Join-Path $PSScriptRoot ("transit-v15-{0}.exe" -f $Name.ToLowerInvariant())
    $flags = @(
        "-O3",
        "-std=c++17",
        "-fno-unroll-loops",
        "-static-libgcc",
        "-static-libstdc++"
    ) + $ArchFlags

    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " BUILD $Name" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ("{0} {1} `"{2}`" -o `"{3}`"" -f $compiler, ($flags -join " "), $source, $exe)

    & $compiler @flags $source -o $exe
    if ($LASTEXITCODE -ne 0) {
        throw "$Name compilation failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host " RUN $Name" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green

    & $exe --model $ModelBlob --tensor $Tensor --iters $Iters
    if ($LASTEXITCODE -ne 0) {
        throw "$Name benchmark failed with exit code $LASTEXITCODE"
    }
    Write-Host ""
}

switch ($Target) {
    "Native" {
        Build-And-Run -Name "Native" -ArchFlags @("-march=native")
    }
    "Ivy" {
        Build-And-Run -Name "Ivy" -ArchFlags @("-march=ivybridge", "-mno-avx2")
    }
    "Both" {
        Build-And-Run -Name "Native" -ArchFlags @("-march=native")
        Build-And-Run -Name "Ivy" -ArchFlags @("-march=ivybridge", "-mno-avx2")
    }
}
