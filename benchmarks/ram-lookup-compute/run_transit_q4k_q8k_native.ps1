param(
    [string]$ModelBlob = "C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490",
    [string]$Work = "$env:TEMP\q4k_exact_v10",
    [int]$Iterations = 10,
    [int]$Warmup = 2,
    [int]$Cpu = -1,
    [switch]$ForcePrepare,
    [switch]$Avx2Only
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $Here "transit_q4k_q8k_native.cpp"
$Prepare = Join-Path $Here "prepare_q4k_exact.py"

if (-not (Test-Path $Source)) {
    throw "Missing native source: $Source"
}

if (-not (Test-Path $Prepare)) {
    throw "Missing prepare script: $Prepare"
}

New-Item -ItemType Directory -Force -Path $Work | Out-Null

$Q4Path = Join-Path $Work "tensor-q4k.bin"
$XPath = Join-Path $Work "activation-f32.bin"
$RefPath = Join-Path $Work "reference-f32.bin"
$MetaPath = Join-Path $Work "meta.json"

$NeedPrepare = $ForcePrepare -or
    -not (Test-Path $Q4Path) -or
    -not (Test-Path $XPath) -or
    -not (Test-Path $RefPath) -or
    -not (Test-Path $MetaPath)

if ($NeedPrepare) {
    if (-not (Test-Path $ModelBlob)) {
        throw "Model blob does not exist: $ModelBlob"
    }

    $Python = Get-Command python.exe, python -ErrorAction SilentlyContinue |
        Select-Object -First 1

    if ($null -eq $Python) {
        throw "python was not found in PATH"
    }

    Write-Host ""
    Write-Host "Preparing original Q4_K bytes + activation + independent FP32 reference..."
    & $Python.Source $Prepare $ModelBlob $Work
    if ($LASTEXITCODE -ne 0) {
        throw "prepare_q4k_exact.py failed with exit code $LASTEXITCODE"
    }
}

$Meta = Get-Content $MetaPath -Raw | ConvertFrom-Json

$Gpp = Get-Command g++.exe, g++ -ErrorAction SilentlyContinue |
    Select-Object -First 1

if ($null -eq $Gpp) {
    throw "g++ was not found in PATH"
}

Write-Host ""
Write-Host "============================================================"
Write-Host " COMPILER"
Write-Host "============================================================"
Write-Host "g++: $($Gpp.Source)"
& $Gpp.Source --version | Select-Object -First 1
$DumpMachine = & $Gpp.Source -dumpmachine
Write-Host "target: $DumpMachine"

$ExeAvx2 = Join-Path $Work "transit_q4k_q8k_avx2.exe"
$ExeNative = Join-Path $Work "transit_q4k_q8k_native.exe"

$CommonFlags = @(
    $Source,
    "-O3",
    "-std=c++17",
    "-DNDEBUG",
    "-Wall",
    "-Wextra",
    "-Wpedantic"
)

Write-Host ""
Write-Host "============================================================"
Write-Host " BUILD AVX2 BASELINE"
Write-Host "============================================================"

$Avx2Flags = @(
    "-mavx2",
    "-mfma",
    "-mssse3",
    "-msse4.1",
    "-o",
    $ExeAvx2
)

Write-Host (($Gpp.Source, $CommonFlags, $Avx2Flags) -join " ")
& $Gpp.Source @CommonFlags @Avx2Flags
if ($LASTEXITCODE -ne 0) {
    throw "AVX2 build failed with exit code $LASTEXITCODE"
}

$RunArgs = @(
    "--q4", $Q4Path,
    "--x", $XPath,
    "--ref", $RefPath,
    "--rows", ([string][int]$Meta.rows),
    "--cols", ([string][int]$Meta.cols),
    "--iterations", ([string]$Iterations),
    "--warmup", ([string]$Warmup)
)

if ($Cpu -ge 0) {
    $RunArgs += @("--cpu", ([string]$Cpu))
}

$Avx2Log = Join-Path $Work "transit_q4k_q8k_avx2-results.txt"

Write-Host ""
Write-Host "============================================================"
Write-Host " RUN AVX2 BASELINE BUILD"
Write-Host "============================================================"
& $ExeAvx2 @RunArgs 2>&1 | Tee-Object -FilePath $Avx2Log
if ($LASTEXITCODE -ne 0) {
    throw "AVX2 benchmark failed with exit code $LASTEXITCODE"
}

if (-not $Avx2Only) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host " BUILD NATIVE CPU TARGET"
    Write-Host "============================================================"

    $NativeFlags = @(
        "-march=native",
        "-mtune=native",
        "-o",
        $ExeNative
    )

    Write-Host (($Gpp.Source, $CommonFlags, $NativeFlags) -join " ")
    & $Gpp.Source @CommonFlags @NativeFlags
    if ($LASTEXITCODE -ne 0) {
        throw "native build failed with exit code $LASTEXITCODE"
    }

    $NativeLog = Join-Path $Work "transit_q4k_q8k_native-results.txt"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " RUN NATIVE CPU TARGET"
    Write-Host "============================================================"
    & $ExeNative @RunArgs 2>&1 | Tee-Object -FilePath $NativeLog
    if ($LASTEXITCODE -ne 0) {
        throw "native benchmark failed with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " DONE"
Write-Host "============================================================"
Write-Host "Work directory : $Work"
Write-Host "AVX2 log       : $Avx2Log"
if (-not $Avx2Only) {
    Write-Host "Native log     : $(Join-Path $Work 'transit_q4k_q8k_native-results.txt')"
}
Write-Host ""
Write-Host "Send back the complete AVX2 and native outputs."
