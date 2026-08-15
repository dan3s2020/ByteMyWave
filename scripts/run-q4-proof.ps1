param(
    [Parameter(Mandatory = $true)]
    [string]$ModelDir,

    [int[]]$M = @(64, 128, 256, 512, 1024),
    [int]$TileN = 256,
    [int]$MaxTiles = 64,
    [ValidateSet('auto', 'F16', 'BF16')]
    [string]$SourceDType = 'auto',
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
$RunDir = Join-Path $Root (Join-Path 'runs' ("phase3-q4-$Stamp"))
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

& $Python.Source -c 'import numpy; print(numpy.__version__)' *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'Phase 3 requires NumPy for offline vectorized Q4 conversion. Run: python -m pip install numpy'
}

New-Item -ItemType Directory -Force -Path $Build | Out-Null
New-Item -ItemType Directory -Force -Path $Prepared | Out-Null

Write-Host '=== Phase 3A: exact real-weight atlas + 16-bit source pack ===' -ForegroundColor Cyan
$PackArgs = @(
    (Join-Path $Root 'tools\pack_stream_tiles.py'),
    '--model-dir', $ModelDir,
    '--output-dir', $Prepared,
    '--dtype', $SourceDType,
    '--tile-n', $TileN,
    '--max-tiles', $MaxTiles
)
if ($K -gt 0) {
    $PackArgs += @('--k', $K)
}
& $Python.Source @PackArgs
if ($LASTEXITCODE -ne 0) {
    throw "Real-weight source pack failed with exit code $LASTEXITCODE"
}

$PlanPath = Join-Path $Prepared 'execution-plan.json'
$SourcePackPath = Join-Path $Prepared 'weights.pack'
$SchedulePath = Join-Path $Prepared 'runtime-schedule.json'

Write-Host "`n=== Phase 3B: compile static two-slot schedule ===" -ForegroundColor Cyan
& $Python.Source `
    (Join-Path $Root 'tools\build_runtime_schedule.py') `
    --plan $PlanPath `
    --output $SchedulePath `
    --slots 2
if ($LASTEXITCODE -ne 0) {
    throw "Runtime schedule compilation failed with exit code $LASTEXITCODE"
}

Write-Host "`n=== Phase 3C: offline Q4 quantization ===" -ForegroundColor Cyan
& $Python.Source `
    (Join-Path $Root 'tools\quantize_q4_pack.py') `
    --plan $PlanPath `
    --input-pack $SourcePackPath `
    --output-dir $Prepared
if ($LASTEXITCODE -ne 0) {
    throw "Q4 quantization failed with exit code $LASTEXITCODE"
}

$SourcePlan = Get-Content -Raw $PlanPath | ConvertFrom-Json
$Q4PlanPath = Join-Path $Prepared 'q4-plan.json'
$Q4PackPath = Join-Path $Prepared 'weights-q4.pack'
$Q4Plan = Get-Content -Raw $Q4PlanPath | ConvertFrom-Json
$Schedule = Get-Content -Raw $SchedulePath | ConvertFrom-Json

$PlanK = [int]$Q4Plan.geometry.k
$PlanN = [int]$Q4Plan.geometry.n
$PlanTiles = [int]$Q4Plan.geometry.tile_count

if ([int]$Schedule.tile_count -ne $PlanTiles) {
    throw 'Q4 tile count differs from the compiled runtime schedule.'
}
if ([int]$Q4Plan.quantization.group_size -ne 32 -or [int]$Q4Plan.quantization.group_bytes -ne 20) {
    throw 'CUDA Q4 proof currently implements only Q4_SYM_G32_F32S.'
}

Write-Host "`n=== Phase 3D: build CUDA Q4 dequant+GEMM proof ===" -ForegroundColor Cyan
cmake -S $Root -B $Build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE" }
cmake --build $Build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "CMake build failed with exit code $LASTEXITCODE" }

$Candidates = @(
    (Join-Path $Build 'Release\tensorwave_q4_stream_proof.exe'),
    (Join-Path $Build 'tensorwave_q4_stream_proof.exe')
)
$Exe = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Exe) {
    throw "Built tensorwave_q4_stream_proof executable not found under $Build"
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi | Out-File -Encoding utf8 (Join-Path $RunDir 'nvidia-smi.txt')
}

$Metadata = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    computer_name = $env:COMPUTERNAME
    powershell = $PSVersionTable.PSVersion.ToString()
    model_dir = $ModelDir
    source_plan = $PlanPath
    runtime_schedule = $SchedulePath
    q4_plan = $Q4PlanPath
    q4_pack = $Q4PackPath
    source_dtype = [string]$Q4Plan.geometry.source_dtype
    q4_format = [string]$Q4Plan.quantization.name
    compression_x = [double]$Q4Plan.geometry.source_to_q4_compression_x
    weight_rms_error = [double]$Q4Plan.quality.rms_error
    weight_snr_db = [double]$Q4Plan.quality.snr_db
    k = $PlanK
    n = $PlanN
    tiles = $PlanTiles
    slots = 2
    m_sweep = $M
    device = $Device
}
$Metadata | ConvertTo-Json -Depth 8 | Out-File -Encoding utf8 (Join-Path $RunDir 'run-config.json')

Write-Host "`n=== Q4 compressed RAM -> VRAM sweep ===" -ForegroundColor Cyan
Write-Host "Model dir:       $ModelDir"
Write-Host "Source dtype:    $($Q4Plan.geometry.source_dtype)"
Write-Host "Q4 format:       $($Q4Plan.quantization.name)"
Write-Host "K/N:             $PlanK/$PlanN"
Write-Host "Tiles:           $PlanTiles"
Write-Host "Compression:     $([math]::Round([double]$Q4Plan.geometry.source_to_q4_compression_x, 3))x"
Write-Host "Source MiB:      $([math]::Round(([double]$Q4Plan.geometry.source_pack_bytes / 1MB), 2))"
Write-Host "Q4 MiB:          $([math]::Round(([double]$Q4Plan.geometry.q4_pack_bytes / 1MB), 2))"
Write-Host "Weight RMS err:  $($Q4Plan.quality.rms_error)"
Write-Host "Weight SNR dB:   $($Q4Plan.quality.snr_db)"
Write-Host 'VRAM Q4 slots:   2 fixed compressed addresses'
Write-Host 'Dequant buffer:  1 fixed FP16 tile'
Write-Host "Results:         $RunDir"

foreach ($ThisM in $M) {
    $Json = Join-Path $RunDir ("m-{0}.json" -f $ThisM)
    $Log = Join-Path $RunDir ("m-{0}.log.txt" -f $ThisM)

    Write-Host "`n--- Q4 REAL WEIGHTS: M=$ThisM K=$PlanK N=$PlanN tiles=$PlanTiles ---" -ForegroundColor Yellow

    & $Exe `
        --device $Device `
        --weights-file $Q4PackPath `
        --m $ThisM `
        --k $PlanK `
        --n $PlanN `
        --tiles $PlanTiles `
        --warmup $Warmup `
        --json $Json 2>&1 | Tee-Object -FilePath $Log

    if ($LASTEXITCODE -ne 0) {
        throw "Q4 streaming experiment failed for M=$ThisM with exit code $LASTEXITCODE. See $Log"
    }
}

Write-Host "`n=== Summarizing Q4 sweep ===" -ForegroundColor Cyan
& $Python.Source (Join-Path $Root 'tools\summarize_runs.py') --run-dir $RunDir
if ($LASTEXITCODE -ne 0) {
    throw "Result summarization failed with exit code $LASTEXITCODE"
}

Write-Host "`nDONE. Phase-3 results: $RunDir" -ForegroundColor Green
Write-Host "Q4 plan:  $Q4PlanPath"
Write-Host "Summary:  $(Join-Path $RunDir 'SUMMARY.md')"
Write-Host 'The key comparison is Phase-2 16-bit H2D starvation versus Phase-3 Q4 H2D+GPU-dequant starvation.'
