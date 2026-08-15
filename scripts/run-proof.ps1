param(
    [int[]]$M = @(64, 128, 256, 512, 1024, 2048),
    [int]$K = 4096,
    [int]$N = 2048,
    [int]$Tiles = 32,
    [int]$Warmup = 3,
    [int]$Device = 0
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Build = Join-Path $Root 'build'
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$RunDir = Join-Path $Root (Join-Path 'runs' $Stamp)

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw 'cmake was not found in PATH.'
}

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw 'nvcc was not found in PATH. Install/activate the NVIDIA CUDA Toolkit first.'
}

New-Item -ItemType Directory -Force -Path $Build | Out-Null
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

Write-Host "=== TensorWave Phase-1 build ===" -ForegroundColor Cyan
cmake -S $Root -B $Build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE" }

cmake --build $Build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "CMake build failed with exit code $LASTEXITCODE" }

$Candidates = @(
    (Join-Path $Build 'Release\tensorwave_stream_proof.exe'),
    (Join-Path $Build 'tensorwave_stream_proof.exe')
)
$Exe = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Exe) {
    throw "Built executable not found under $Build"
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi | Out-File -Encoding utf8 (Join-Path $RunDir 'nvidia-smi.txt')
}

$Metadata = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    computer_name = $env:COMPUTERNAME
    powershell = $PSVersionTable.PSVersion.ToString()
    k = $K
    n = $N
    tiles = $Tiles
    warmup = $Warmup
    device = $Device
    m_sweep = $M
}
$Metadata | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 (Join-Path $RunDir 'run-config.json')

Write-Host "`n=== TensorWave Phase-1 sweep ===" -ForegroundColor Cyan
Write-Host "Results directory: $RunDir"

foreach ($ThisM in $M) {
    $Json = Join-Path $RunDir ("m-{0}.json" -f $ThisM)
    $Log = Join-Path $RunDir ("m-{0}.log.txt" -f $ThisM)

    Write-Host "`n--- M=$ThisM K=$K N=$N tiles=$Tiles ---" -ForegroundColor Yellow

    & $Exe `
        --device $Device `
        --m $ThisM `
        --k $K `
        --n $N `
        --tiles $Tiles `
        --warmup $Warmup `
        --json $Json 2>&1 | Tee-Object -FilePath $Log

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed for M=$ThisM with exit code $LASTEXITCODE. See $Log"
    }
}

Write-Host "`nDONE. Raw results are in: $RunDir" -ForegroundColor Green
Write-Host 'The important curve is steady_starvation_pct versus M.'
