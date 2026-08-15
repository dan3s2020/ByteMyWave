param(
    [Parameter(Mandatory = $true)]
    [string]$ModelDir,

    [int[]]$TileN = @(128, 256),
    [int[]]$M = @(1, 4, 16, 64, 128, 256, 512, 1024),
    [int]$MaxTiles = 32,
    [ValidateSet('auto', 'F16', 'BF16')]
    [string]$SourceDType = 'auto',
    [int]$Warmup = 3,
    [int]$Device = 0,
    [switch]$Skip16BitComparison
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ModelDir = (Resolve-Path $ModelDir).Path
$RunsRoot = Join-Path $Root 'runs'
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$Phase4Dir = Join-Path $RunsRoot ("phase4-feasibility-$Stamp")
New-Item -ItemType Directory -Force -Path $Phase4Dir | Out-Null

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) { $Python = Get-Command py -ErrorAction SilentlyContinue }
if (-not $Python) { throw 'Python 3 was not found in PATH.' }

function Get-NewestRun([string]$Pattern) {
    if (-not (Test-Path $RunsRoot)) { return $null }
    return Get-ChildItem $RunsRoot -Directory -Filter $Pattern -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
}

function Assert-NewRun($Before, $After, [string]$Label) {
    if (-not $After) { throw "$Label did not create a run directory." }
    if ($Before -and $Before.FullName -eq $After.FullName) {
        throw "$Label did not create a new run directory; newest path is unchanged: $($After.FullName)"
    }
}

$ManifestRuns = @()

Write-Host '=== TensorWave Phase 4: measured feasibility-map program ===' -ForegroundColor Cyan
Write-Host "ModelDir:  $ModelDir"
Write-Host "TileN:     $($TileN -join ', ')"
Write-Host "M sweep:   $($M -join ', ')"
Write-Host "MaxTiles:  $MaxTiles"
Write-Host "Output:    $Phase4Dir"

foreach ($ThisTileN in $TileN) {
    Write-Host "`n============================================================" -ForegroundColor DarkCyan
    Write-Host "Q4 crossover run: TileN=$ThisTileN" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor DarkCyan

    $BeforeQ4 = Get-NewestRun 'phase3-q4-*'

    & (Join-Path $Root 'scripts\run-q4-proof.ps1') `
        -ModelDir $ModelDir `
        -M $M `
        -TileN $ThisTileN `
        -MaxTiles $MaxTiles `
        -SourceDType $SourceDType `
        -Warmup $Warmup `
        -Device $Device

    if ($LASTEXITCODE -ne 0) {
        throw "Phase-3 Q4 run failed for TileN=$ThisTileN with exit code $LASTEXITCODE"
    }

    $Q4Run = Get-NewestRun 'phase3-q4-*'
    Assert-NewRun $BeforeQ4 $Q4Run "Phase-3 Q4 TileN=$ThisTileN"

    $Q4Config = Get-Content -Raw (Join-Path $Q4Run.FullName 'run-config.json') | ConvertFrom-Json
    $SelectedK = [int]$Q4Config.k
    $SelectedSourceDType = [string]$Q4Config.source_dtype

    $TileOut = Join-Path $Phase4Dir ("tile-n-$ThisTileN")
    New-Item -ItemType Directory -Force -Path $TileOut | Out-Null

    $CalibrationPath = Join-Path $TileOut 'q4-calibration.json'
    & $Python.Source `
        (Join-Path $Root 'tools\calibrate_feasibility_map.py') `
        --run-dir $Q4Run.FullName `
        --output $CalibrationPath
    if ($LASTEXITCODE -ne 0) { throw "Calibration failed for $($Q4Run.FullName)" }

    $Calibration = Get-Content -Raw $CalibrationPath | ConvertFrom-Json
    $MeasuredH2D = [double]$Calibration.effective_h2d_gbps_median
    $MeasuredTF = [double]$Calibration.effective_tflops_median

    $MapOut = Join-Path $TileOut 'map-q4-measured'
    & $Python.Source `
        (Join-Path $Root 'tools\build_feasibility_map.py') `
        --output-dir $MapOut `
        --pcie-gbps $MeasuredH2D `
        --effective-tflops $MeasuredTF `
        --bytes-per-param 0.625 `
        --models-b '7,13,33,70,120' `
        --m-values '1,4,16,64,128,256,512,1024,2048'
    if ($LASTEXITCODE -ne 0) { throw "Map build failed for TileN=$ThisTileN" }

    $ManifestRuns += [ordered]@{
        phase = 'Q4'
        tile_n = $ThisTileN
        k = $SelectedK
        source_dtype = $SelectedSourceDType
        run_dir = $Q4Run.FullName
        calibration = $CalibrationPath
        map_dir = $MapOut
    }

    if (-not $Skip16BitComparison) {
        Write-Host "`n16-bit comparison: TileN=$ThisTileN K=$SelectedK dtype=$SelectedSourceDType" -ForegroundColor Yellow
        $Before16 = Get-NewestRun 'phase2-real-*'

        & (Join-Path $Root 'scripts\run-real-weight-proof.ps1') `
            -ModelDir $ModelDir `
            -M $M `
            -TileN $ThisTileN `
            -MaxTiles $MaxTiles `
            -DType $SelectedSourceDType `
            -K $SelectedK `
            -Warmup $Warmup `
            -Device $Device

        if ($LASTEXITCODE -ne 0) {
            throw "Phase-2 16-bit comparison failed for TileN=$ThisTileN with exit code $LASTEXITCODE"
        }

        $Run16 = Get-NewestRun 'phase2-real-*'
        Assert-NewRun $Before16 $Run16 "Phase-2 16-bit TileN=$ThisTileN"

        $ManifestRuns += [ordered]@{
            phase = '16bit'
            tile_n = $ThisTileN
            k = $SelectedK
            source_dtype = $SelectedSourceDType
            run_dir = $Run16.FullName
        }
    }
}

$AggregateArgs = @(
    (Join-Path $Root 'tools\aggregate_feasibility_runs.py'),
    '--output-dir', $Phase4Dir
)
foreach ($Entry in $ManifestRuns) {
    $AggregateArgs += @('--run-dir', [string]$Entry.run_dir)
}
& $Python.Source @AggregateArgs
if ($LASTEXITCODE -ne 0) { throw 'Feasibility aggregation failed.' }

$Manifest = [ordered]@{
    schema = 'tensorwave.phase4-feasibility-run.v1'
    timestamp = (Get-Date).ToString('o')
    model_dir = $ModelDir
    tile_n_sweep = $TileN
    m_sweep = $M
    max_tiles = $MaxTiles
    device = $Device
    compare_16bit = (-not $Skip16BitComparison)
    runs = $ManifestRuns
}
$Manifest | ConvertTo-Json -Depth 8 | Out-File -Encoding utf8 (Join-Path $Phase4Dir 'run-manifest.json')

Write-Host "`n=== PHASE 4 COMPLETE ===" -ForegroundColor Green
Write-Host "Measured aggregation: $(Join-Path $Phase4Dir 'MEASURED-FEASIBILITY.md')"
Write-Host "CSV:                  $(Join-Path $Phase4Dir 'feasibility-runs.csv')"
Write-Host "Manifest:             $(Join-Path $Phase4Dir 'run-manifest.json')"
Write-Host ''
Write-Host 'The decisive output is starvation versus M for each TileN, plus Q4-vs-16bit at identical K/N/M.'
