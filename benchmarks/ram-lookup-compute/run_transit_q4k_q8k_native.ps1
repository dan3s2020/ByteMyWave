param(
    [string]$ModelBlob = "C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490",
    [string]$Work = "$env:TEMP\q4k_exact_v11",
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

# V10 and earlier wrote the tensor dimensions in the wrong orientation because
# GGUFReader.shape is GGML/ne order, not NumPy rows/cols.  Never silently reuse
# one of those stale fixtures.
if (-not $NeedPrepare -and (Test-Path $MetaPath)) {
    try {
        $ExistingMeta = Get-Content $MetaPath -Raw | ConvertFrom-Json
        $HasGeometryVersion = $null -ne $ExistingMeta.PSObject.Properties["geometry_version"]
        if (-not $HasGeometryVersion -or [int]$ExistingMeta.geometry_version -lt 2) {
            Write-Host "Legacy/wrong-geometry fixture detected; regenerating." -ForegroundColor Yellow
            $NeedPrepare = $true
        }
    }
    catch {
        Write-Host "Unreadable meta.json; regenerating fixture." -ForegroundColor Yellow
        $NeedPrepare = $true
    }
}

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
    Write-Host "Preparing original Q4_K bytes + real-geometry activation + independent FP32 reference..."
    & $Python.Source $Prepare $ModelBlob $Work
    if ($LASTEXITCODE -ne 0) {
        throw "prepare_q4k_exact.py failed with exit code $LASTEXITCODE"
    }
}

$Meta = Get-Content $MetaPath -Raw | ConvertFrom-Json

if ($null -eq $Meta.PSObject.Properties["geometry_version"] -or [int]$Meta.geometry_version -lt 2) {
    throw "Refusing legacy Q4_K fixture: geometry_version >= 2 is required. Re-run with -ForcePrepare."
}

$Rows = [int]$Meta.rows
$Cols = [int]$Meta.cols

if (($Cols % 256) -ne 0) {
    throw "Invalid Q4_K geometry: cols=$Cols is not divisible by 256"
}

$ExpectedBlocks = [int64]$Rows * [int64]($Cols / 256)
$ExpectedQ4Bytes = $ExpectedBlocks * 144L
$ExpectedXBytes = [int64]$Cols * 4L
$ExpectedRefBytes = [int64]$Rows * 4L

$ActualQ4Bytes = (Get-Item $Q4Path).Length
$ActualXBytes = (Get-Item $XPath).Length
$ActualRefBytes = (Get-Item $RefPath).Length

if ($ActualQ4Bytes -ne $ExpectedQ4Bytes) {
    throw "Q4_K fixture byte mismatch: got $ActualQ4Bytes expected $ExpectedQ4Bytes"
}
if ($ActualXBytes -ne $ExpectedXBytes) {
    throw "Activation fixture byte mismatch: got $ActualXBytes expected $ExpectedXBytes"
}
if ($ActualRefBytes -ne $ExpectedRefBytes) {
    throw "Reference fixture byte mismatch: got $ActualRefBytes expected $ExpectedRefBytes"
}

Write-Host ""
Write-Host "============================================================"
Write-Host " REAL GGUF GEOMETRY"
Write-Host "============================================================"
Write-Host "geometry_version : $($Meta.geometry_version)"
Write-Host "GEMV matrix      : $Rows rows x $Cols cols"
Write-Host "GGUF ne[0]/ne[1] : $($Meta.gguf_ne0) / $($Meta.gguf_ne1)"
Write-Host "blocks/row       : $($Meta.blocks_per_row)"
Write-Host "Q4_K bytes       : $ActualQ4Bytes"
Write-Host "activation bytes : $ActualXBytes"
Write-Host "reference bytes  : $ActualRefBytes"

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
    "--rows", ([string]$Rows),
    "--cols", ([string]$Cols),
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
