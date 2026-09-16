param(
    [Parameter(Mandatory = $true)]
    [string]$ModelBlob,
    [string]$Tensor = "blk.0.ffn_gate.weight",
    [int]$Iters = 15,
    [int]$Threads = 12,
    [int]$V11Cpu = 4,
    [string]$CpuList = "",
    [double]$V11WorkingSetMiB = 101.25,
    [double]$V20WorkingSetMiB = 520.31,
    [switch]$SkipOriginalV11,
    [switch]$RunIvyProxyFull
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$source = Join-Path $PSScriptRoot "transit_lut123_v21.cpp"
if (-not (Test-Path -LiteralPath $source)) {
    # The repository stores the audited source as gzip+base64 text parts so the
    # exact verified bytes survive text-only GitHub connectors. Restore it
    # transparently before compiling.
    $parts = @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter "transit_lut123_v21.cpp.gz.b64.part*" | Sort-Object Name)
    if ($parts.Count -eq 0) {
        $singleArchive = Join-Path $PSScriptRoot "transit_lut123_v21.cpp.gz.b64"
        if (Test-Path -LiteralPath $singleArchive) {
            $encodedText = [IO.File]::ReadAllText($singleArchive).Trim()
        } else {
            throw "Missing source and source archive: $source"
        }
    } else {
        $sb = New-Object Text.StringBuilder
        foreach ($part in $parts) {
            [void]$sb.Append([IO.File]::ReadAllText($part).Trim())
        }
        $encodedText = $sb.ToString()
    }

    $compressedSource = [Convert]::FromBase64String($encodedText)
    $sourceInput = New-Object IO.MemoryStream(,$compressedSource)
    $sourceGzip = New-Object IO.Compression.GZipStream($sourceInput, [IO.Compression.CompressionMode]::Decompress)
    $sourceOutput = [IO.File]::Create($source)
    try { $sourceGzip.CopyTo($sourceOutput) } finally { $sourceOutput.Dispose(); $sourceGzip.Dispose(); $sourceInput.Dispose() }

    $expectedSourceSha = "8A1DDF084BE0F63194E246CB12F756E243D35C42C2005EC569D706CB171FDC1A"
    $actualSourceSha = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actualSourceSha -ne $expectedSourceSha) {
        Remove-Item -LiteralPath $source -Force -ErrorAction SilentlyContinue
        throw "V21 source SHA mismatch. Expected $expectedSourceSha, got $actualSourceSha"
    }
    Write-Host "Restored V21 source SHA: PASS ($actualSourceSha)" -ForegroundColor Green
}
if (-not (Test-Path -LiteralPath $ModelBlob)) { throw "Model blob not found: $ModelBlob" }

$compiler = $null
foreach ($name in @("g++", "clang++")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { $compiler = $cmd.Source; break }
}
if (-not $compiler) {
    $candidate = "C:\Strawberry\c\bin\g++.exe"
    if (Test-Path -LiteralPath $candidate) { $compiler = $candidate }
}
if (-not $compiler) { throw "No g++/clang++ found. Expected Strawberry g++ at C:\Strawberry\c\bin\g++.exe" }

$cpuName = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
if ([string]::IsNullOrWhiteSpace($CpuList)) {
    if ($cpuName -match "i5-12500H" -and $Threads -eq 12) {
        # Exact physical-first order recorded by V20 on the benchmark laptop.
        $CpuList = "0,2,4,6,8,9,10,11,12,13,14,15"
    } else {
        $CpuList = ((0..($Threads - 1)) -join ",")
    }
}

$nativeExe = Join-Path $PSScriptRoot "transit-lut123-v21-native.exe"
$ivyExe    = Join-Path $PSScriptRoot "transit-lut123-v21-ivy.exe"
$csv       = Join-Path $PSScriptRoot "RESULTS-LUT123-V21.csv"
$log       = Join-Path $PSScriptRoot "RESULTS-LUT123-V21.txt"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " TRANSIT LUT1/LUT2/LUT3 V21 - AUDITED RUN" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CPU        : $cpuName"
Write-Host "Compiler   : $compiler"
Write-Host "Model      : $ModelBlob"
Write-Host "Tensor     : $Tensor"
Write-Host "Iters      : $Iters"
Write-Host "Threads    : $Threads"
Write-Host "V11 CPU    : $V11Cpu"
Write-Host "CPU list   : $CpuList"
Write-Host "V11 gate   : $V11WorkingSetMiB MiB"
Write-Host "V20 gate   : $V20WorkingSetMiB MiB"
Write-Host ""

$common = @(
    "-O3", "-std=c++17", "-DNDEBUG", "-fomit-frame-pointer", "-pthread",
    "-Wall", "-Wextra", "-Wshadow", "-Wconversion", "-Wpedantic",
    "-static-libgcc", "-static-libstdc++"
)

Write-Host "[1/5] Building native audited harness..." -ForegroundColor Yellow
& $compiler @common "-march=native" "-mtune=native" $source -o $nativeExe
if ($LASTEXITCODE -ne 0) { throw "Native compile failed: $LASTEXITCODE" }

Write-Host "[2/5] Native self-test (must be bit-exact)..." -ForegroundColor Yellow
& $nativeExe --selftest
if ($LASTEXITCODE -ne 0) { throw "Native self-test failed: $LASTEXITCODE" }

Write-Host "[3/5] Building Ivy Bridge / E5-2680 v2 ISA path..." -ForegroundColor Yellow
& $compiler @common "-march=ivybridge" "-mtune=ivybridge" "-mno-avx2" $source -o $ivyExe
if ($LASTEXITCODE -ne 0) { throw "Ivy compile failed: $LASTEXITCODE" }

Write-Host "[4/5] Ivy build self-test (compile + exactness gate)..." -ForegroundColor Yellow
& $ivyExe --selftest
if ($LASTEXITCODE -ne 0) { throw "Ivy self-test failed: $LASTEXITCODE" }

if (-not $SkipOriginalV11) {
    Write-Host "[audit] Restoring the exact archived Native V11 source and checking its SHA-256..." -ForegroundColor Yellow
    $encoded = Join-Path $PSScriptRoot "..\native-q4k-q8k\sources\kernel_v11.cpp.gz.b64"
    if (-not (Test-Path -LiteralPath $encoded)) { throw "Original Native V11 archive not found: $encoded" }
    $restored = Join-Path $PSScriptRoot "kernel_v11.original.cpp"
    $v11exe   = Join-Path $PSScriptRoot "kernel_v11.original.exe"

    $b64 = [IO.File]::ReadAllText($encoded).Trim()
    $compressed = [Convert]::FromBase64String($b64)
    $input = New-Object IO.MemoryStream(,$compressed)
    $gzip = New-Object IO.Compression.GZipStream($input, [IO.Compression.CompressionMode]::Decompress)
    $output = [IO.File]::Create($restored)
    try { $gzip.CopyTo($output) } finally { $output.Dispose(); $gzip.Dispose(); $input.Dispose() }

    $expected = "A4F25E15F6D4667894B9F726B4FFBBD2F1198BCC925F2F4DBEFB2C006630A938"
    $actual = (Get-FileHash -LiteralPath $restored -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actual -ne $expected) { throw "Native V11 SHA mismatch. Expected $expected, got $actual" }
    Write-Host "Native V11 source SHA: PASS ($actual)" -ForegroundColor Green

    & $compiler @common "-march=native" "-mtune=native" $restored -o $v11exe
    if ($LASTEXITCODE -ne 0) { throw "Original V11 compile failed: $LASTEXITCODE" }
    Write-Host "Running original V11 checkpoint on logical CPU $V11Cpu..." -ForegroundColor Green
    & $v11exe $V11Cpu
    if ($LASTEXITCODE -ne 0) { throw "Original V11 run failed: $LASTEXITCODE" }
}

Write-Host "[5/5] Running real-Qwen dual-gate LUT1/LUT2/LUT3 benchmark..." -ForegroundColor Green
$arguments = @(
    "--model", $ModelBlob,
    "--tensor", $Tensor,
    "--iters", "$Iters",
    "--threads", "$Threads",
    "--v11-cpu", "$V11Cpu",
    "--v11-mib", "$V11WorkingSetMiB",
    "--v20-mib", "$V20WorkingSetMiB",
    "--cpu-list", $CpuList,
    "--csv", $csv
)
& $nativeExe @arguments 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "Native benchmark failed: $LASTEXITCODE" }

if ($RunIvyProxyFull) {
    Write-Host "Running the same full benchmark through the Ivy-compatible binary (proxy only; still on this laptop)..." -ForegroundColor Green
    $ivyCsv = Join-Path $PSScriptRoot "RESULTS-LUT123-V21-IVY-PROXY.csv"
    $ivyLog = Join-Path $PSScriptRoot "RESULTS-LUT123-V21-IVY-PROXY.txt"
    $ivyArgs = @(
        "--model", $ModelBlob,
        "--tensor", $Tensor,
        "--iters", "$Iters",
        "--threads", "$Threads",
        "--v11-cpu", "$V11Cpu",
        "--v11-mib", "$V11WorkingSetMiB",
        "--v20-mib", "$V20WorkingSetMiB",
        "--cpu-list", $CpuList,
        "--csv", $ivyCsv
    )
    & $ivyExe @ivyArgs 2>&1 | Tee-Object -FilePath $ivyLog
    if ($LASTEXITCODE -ne 0) { throw "Ivy proxy benchmark failed: $LASTEXITCODE" }
}

Write-Host ""
Write-Host "DONE" -ForegroundColor Green
Write-Host "CSV : $csv"
Write-Host "LOG : $log"
