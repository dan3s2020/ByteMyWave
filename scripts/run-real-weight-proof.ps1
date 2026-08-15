param(
    [Parameter(Mandatory = $true)]
    [string]$ModelDir,

    [int[]]$M = @(64, 128, 256, 512, 1024),
    [int]$TileN = 256,
    [int]$MaxTiles = 64,
    [ValidateSet('auto', 'F16', 'BF16')]
    [string]$DType = 'auto',
    [int]$K = 0,
    [int]$Warmup = 3,
    [int]$Device = 0
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ModelDir = (Resolve-Path $ModelDir).Path
$Build = Join-Path $Root 'build'
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$RunDir = Join-Path $Root (Join-Path 'runs' ("phase2-real-$Stamp"))
$Prepared = Join-Path $RunDir 'prepared'

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw 'cmake was not found in PATH.'
}
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw 'nvcc was not found in PATH. Install/activate the NVIDIA CUDA Toolkit first.'
}

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    $Python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $Python) {
    throw 'Python 3 was not found in PATH.'
}

New-Item -ItemType Directory -Force -Path $Build | Out-Null
New-Item -ItemType Directory -Force -Path $Prepared | Out-Null

Write-Host '=== TensorWave Phase-2: build real Weight Atlas + tile pack ===' -ForegroundColor Cyan
$PackArgs = @(
    (Join-Path $Root 'tools\pack_stream_tiles.py'),
    '--model-dir', $ModelDir,
    '--output-dir', $Prepared,
    '--dtype', $DType,
    '--tile-n', $TileN,
    '--max-tiles', $MaxTiles
)
if ($K -gt 0) {
    $PackArgs += @('--k', $K)
}

& $Python.Source @PackArgs
if ($LASTEXITCODE -ne 0) {
    throw "Weight pack preparation failed with exit code $LASTEXITCODE"
}

$PlanPath = Join-Path $Prepared 'execution-plan.json'
$PackPath = Join-Path $Prepared 'weights.pack'
$Plan = Get-Content -Raw $PlanPath | ConvertFrom-Json

$PlanDType = [string]$Plan.geometry.dtype
$PlanK = [int]$Plan.geometry.k
$PlanN = [int]$Plan.geometry.n
$PlanTiles = [int]$Plan.geometry.tile_count

switch ($PlanDType) {
    'F16'  { $CliDType = 'fp16' }
    'BF16' { $CliDType = 'bf16' }
    default { throw "Unsupported plan dtype: $PlanDType" }
}

Write-Host "`n=== TensorWave Phase-2 build ===" -ForegroundColor Cyan
cmake -S $Root -B $Build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE" }
cmake --build $Build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "CMake build failed with exit code $LASTEXITCODE" }

$Candidates = @(
    (Join-Path $Build 'Release\tensorwave_real_weight_proof.exe'),
    (Join-Path $Build 'tensorwave_real_weight_proof.exe')
)
$Exe = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Exe) {
    throw "Built tensorwave_real_weight_proof executable not found under $Build"
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi | Out-File -Encoding utf8 (Join-Path $RunDir 'nvidia-smi.txt')
}

$Metadata = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    computer_name = $env:COMPUTERNAME
    powershell = $PSVersionTable.PSVersion.ToString()
    model_dir = $ModelDir
    plan = $PlanPath
    pack = $PackPath
    dtype = $PlanDType
    k = $PlanK
    n = $PlanN
    tiles = $PlanTiles
    m_sweep = $M
    device = $Device
}
$Metadata | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 (Join-Path $RunDir 'run-config.json')

Write-Host "`n=== Real-checkpoint streaming sweep ===" -ForegroundColor Cyan
Write-Host "Model dir:  $ModelDir"
Write-Host "DType:      $PlanDType"
Write-Host "K/N:        $PlanK/$PlanN"
Write-Host "Tiles:      $PlanTiles"
Write-Host "Pack MiB:   $([math]::Round(([double]$Plan.geometry.pack_bytes / 1MB), 2))"
Write-Host "Results:    $RunDir"

foreach ($ThisM in $M) {
    $Json = Join-Path $RunDir ("m-{0}.json" -f $ThisM)
    $Log = Join-Path $RunDir ("m-{0}.log.txt" -f $ThisM)

    Write-Host "`n--- REAL WEIGHTS: M=$ThisM K=$PlanK N=$PlanN tiles=$PlanTiles dtype=$PlanDType ---" -ForegroundColor Yellow

    & $Exe `
        --device $Device `
        --weights-file $PackPath `
        --dtype $CliDType `
        --m $ThisM `
        --k $PlanK `
        --n $PlanN `
        --tiles $PlanTiles `
        --warmup $Warmup `
        --json $Json 2>&1 | Tee-Object -FilePath $Log

    if ($LASTEXITCODE -ne 0) {
        throw "Real-weight experiment failed for M=$ThisM with exit code $LASTEXITCODE. See $Log"
    }
}

Write-Host "`n=== Summarizing sweep ===" -ForegroundColor Cyan
& $Python.Source (Join-Path $Root 'tools\summarize_runs.py') --run-dir $RunDir
if ($LASTEXITCODE -ne 0) {
    throw "Result summarization failed with exit code $LASTEXITCODE"
}

Write-Host "`nDONE. Phase-2 results: $RunDir" -ForegroundColor Green
Write-Host "Summary: $(Join-Path $RunDir 'SUMMARY.md')"
Write-Host 'Inspect steady_starvation_pct, hidden transfer, speedup, and correctness for each M.'
