param(
    [string]$Model = "C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490",
    [string]$Tensor = "blk.0.ffn_gate.weight",
    [int]$Iterations = 7,
    [int]$Warmup = 2,
    [int]$RotateCopies = 0,
    [switch]$IncludeHybridMixed
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Root = $PSScriptRoot
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ResultDir = Join-Path $Root "RESULTS\$Stamp"
$CacheDir = Join-Path $Root "CACHE"
New-Item -ItemType Directory -Force -Path $ResultDir,$CacheDir | Out-Null

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " TRANSIT V33 PAIRFUSE - EXACT Q4_K, V27 PRESERVED" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Pinned prior model only. No autodiscovery. No synthetic fallback." -ForegroundColor Yellow
Write-Host "V33 attacks redundant Q4 nibble loads + scalar min/reduction work; promotion uses paired AB/BA stability." -ForegroundColor Yellow

$cpus = @(Get-CimInstance Win32_Processor)
$mem = @(Get-CimInstance Win32_PhysicalMemory)
$os = Get-CimInstance Win32_OperatingSystem
$totalGB = [math]::Round(($mem | Measure-Object Capacity -Sum).Sum/1GB,2)
$freeGB = [math]::Round($os.FreePhysicalMemory/1MB,2)
if ($RotateCopies -le 0) {
    if ($totalGB -ge 512) { $RotateCopies = 64 }
    elseif ($totalGB -ge 64) { $RotateCopies = 32 }
    else { $RotateCopies = 16 }
}
$hw = [ordered]@{
    Timestamp=(Get-Date).ToString("o")
    OS="$($os.Caption) $($os.Version) build $($os.BuildNumber)"
    CPUs=@($cpus | ForEach-Object { [ordered]@{Name=$_.Name.Trim();Cores=$_.NumberOfCores;Logical=$_.NumberOfLogicalProcessors;MaxMHz=$_.MaxClockSpeed} })
    DIMMs=@($mem | ForEach-Object { [ordered]@{Manufacturer=$_.Manufacturer;PartNumber=("$($_.PartNumber)").Trim();CapacityGB=[math]::Round($_.Capacity/1GB,2);Speed=$_.Speed;Configured=$_.ConfiguredClockSpeed} })
    TotalPhysicalGB=$totalGB; FreePhysicalGB=$freeGB; RotateCopies=$RotateCopies
}
$hw | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $ResultDir "HARDWARE.json")
$cpus | Format-Table Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed -AutoSize
Write-Host ("RAM: {0} GB total, ~{1} GB free; rotating-address copies = {2}" -f $totalGB,$freeGB,$RotateCopies) -ForegroundColor Green

$Src = Join-Path $Root "v33_pairfuse.cpp"
$Exe = Join-Path $Root "v33_pairfuse.exe"
$Gpp = Get-Command g++.exe, g++ -ErrorAction SilentlyContinue | Select-Object -First 1
$Clang = Get-Command clang++.exe, clang++ -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -ne $Gpp) { $compiler=$Gpp.Source }
elseif ($null -ne $Clang) { $compiler=$Clang.Source }
else { throw "Need g++ or clang++ in PATH." }
Write-Host "Compiler: $compiler" -ForegroundColor Green
$cxxArgs = @(
    "-O3","-std=c++17","-march=ivybridge","-mssse3","-mavx","-mno-avx2","-mno-fma","-pthread",
    $Src,"-o",$Exe
)
& $compiler @cxxArgs
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Exe)) { throw "Compilation failed." }

Write-Host "`nPinned prior-model input:" -ForegroundColor DarkCyan
Write-Host "Model : $Model"
Write-Host "Tensor: $Tensor"
if (-not (Test-Path $Model -PathType Leaf)) { throw "Exact prior ByteMyWave model blob missing: $Model" }
& $Exe --probe --model $Model --tensor $Tensor *> $null
if ($LASTEXITCODE -ne 0) { throw "Pinned blob exists but '$Tensor' is not the expected compatible Q4_K tensor." }
Write-Host "USING EXACT PRIOR MODEL/TENSOR" -ForegroundColor Green
"REAL GGUF BLOB (PINNED PRIOR MODEL)`r`n$Model`r`nTensor=$Tensor" | Set-Content (Join-Path $ResultDir "INPUT.txt")

$perSocket = [int](($cpus | Measure-Object -Property NumberOfCores -Minimum).Minimum)
if ($perSocket -lt 1) { $perSocket=[Math]::Max(1,[Environment]::ProcessorCount) }
$cpuName = (($cpus | Select-Object -First 1).Name).Trim()
$hybrid12500 = $cpuName -match "i5-12500H"
if ($hybrid12500) {
    $threadSweep = @(1,2,4)
    if ($IncludeHybridMixed) { $threadSweep += @(8,12) }
    Write-Host "`nHybrid 12500H detected: promotion sweep is P-core-only 1/2/4. 8/12 mix P+E and are diagnostic only." -ForegroundColor Cyan
} elseif ($perSocket -ge 10) {
    $threadSweep = @(1,2,4,6,8,10,$perSocket) | Where-Object { $_ -le $perSocket } | Sort-Object -Unique
} elseif ($perSocket -ge 8) {
    $threadSweep = @(1,2,4,6,8,$perSocket) | Where-Object { $_ -le $perSocket } | Sort-Object -Unique
} else {
    $threadSweep = @(1,2,4,$perSocket) | Where-Object { $_ -le $perSocket } | Sort-Object -Unique
}
Write-Host "Physical-core-count sweep target: $($threadSweep -join ', ')" -ForegroundColor Cyan
Write-Host "The native harness selects one logical processor per physical core. CPUID 0x1A orders Intel P-cores before E-cores." -ForegroundColor DarkCyan

$all = New-Object System.Collections.Generic.List[object]
foreach ($th in $threadSweep) {
    Write-Host "`n-------------------- THREADS=$th --------------------" -ForegroundColor Magenta
    $csv = Join-Path $ResultDir ("V33_T{0}.csv" -f $th)
    $json = Join-Path $ResultDir ("ATLAS_T{0}.json" -f $th)
    $log = Join-Path $ResultDir ("V33_T{0}.log" -f $th)
    $args = @("--threads","$th","--warmup","$Warmup","--iters","$Iterations","--node","0","--rotate-copies","$RotateCopies","--csv",$csv,"--json",$json,"--model",$Model,"--tensor",$Tensor)
    if ($th -eq ($threadSweep | Select-Object -Last 1)) { $args += @("--cache",$CacheDir) }
    if ($cpus.Count -ge 2 -and $th -eq ($threadSweep | Select-Object -Last 1)) { $args += "--numa" }
    & $Exe @args 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "V33 benchmark failed at threads=$th. See $log" }
    Import-Csv $csv | ForEach-Object { $_ | Add-Member -NotePropertyName Threads -NotePropertyValue $th; $all.Add($_) }
}

$master=Join-Path $ResultDir "MASTER_RESULTS.csv"
$all | Export-Csv -NoTypeInformation -Encoding UTF8 $master

function Show-Board([string]$mode,[string]$title) {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host " $title" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    $all | Where-Object { $_.mode -eq $mode -and $_.batch -eq "1" -and [double]$_.max_abs_diff -le 0.0001 } |
        Sort-Object {[double]$_.speedup_vs_baseline} -Descending |
        Select-Object Threads,name,rowblock,median_ms,speedup_vs_baseline,p10,p90,win_rate,max_abs_diff | Format-Table -AutoSize
}
Show-Board "HOT" "HOT PAIRED-STABLE SINGLE-TOKEN"
Show-Board "ROTATE" "ROTATING / BEYOND-LLC PAIRED-STABLE SINGLE-TOKEN"

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " MULTI-TOKEN / SPECULATIVE REUSE (directional; not V33 promotion gate)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
$all | Where-Object { [int]$_.batch -gt 1 } |
    Sort-Object Threads,{[int]$_.batch},{[double]$_.speedup_vs_baseline} -Descending |
    Select-Object Threads,batch,name,median_ms,speedup_vs_baseline | Format-Table -AutoSize

$stable = $all | Where-Object {
    $_.batch -eq "1" -and
    $_.name -ne "V27_META" -and
    [double]$_.max_abs_diff -le 0.0001 -and
    [double]$_.speedup_vs_baseline -gt 1.0 -and
    [double]$_.p10 -gt 1.0 -and
    [double]$_.win_rate -ge 0.80
}
$bestHot=$stable | Where-Object {$_.mode -eq "HOT"} | Sort-Object {[double]$_.speedup_vs_baseline} -Descending | Select-Object -First 1
$bestRot=$stable | Where-Object {$_.mode -eq "ROTATE"} | Sort-Object {[double]$_.speedup_vs_baseline} -Descending | Select-Object -First 1
$bestVec=$all | Where-Object {[int]$_.batch -gt 1 -and $_.name -like "V31_VEC*"} | Sort-Object {[double]$_.speedup_vs_baseline} -Descending | Select-Object -First 1

$summary=@"
TRANSIT V33 PAIRFUSE
Timestamp: $(Get-Date -Format o)
Input: PINNED PRIOR REAL GGUF: $Model / $Tensor
Rotating beyond-LLC control: $RotateCopies address-distinct META/X8F copies
Per-socket physical cores reported by WMI: $perSocket

STABLE PROMOTION GATE:
median speedup > 1.0 AND p10 > 1.0 AND win_rate >= 0.80 AND diff=0

BEST STABLE HOT candidate: $(if($null -ne $bestHot){"$($bestHot.name) @ $($bestHot.Threads) threads = $($bestHot.speedup_vs_baseline)x, p10=$($bestHot.p10), win=$($bestHot.win_rate)"}else{"NONE - keep V27"})
BEST STABLE ROTATE candidate: $(if($null -ne $bestRot){"$($bestRot.name) @ $($bestRot.Threads) threads = $($bestRot.speedup_vs_baseline)x, p10=$($bestRot.p10), win=$($bestRot.win_rate)"}else{"NONE - keep V27"})
BEST directional multi-token reuse: $(if($null -ne $bestVec){"$($bestVec.name) batch=$($bestVec.batch) @ $($bestVec.Threads) threads = $($bestVec.speedup_vs_baseline)x"}else{"n/a"})

RULES:
- V27_META is never removed.
- V31 FSA_RB4/RB16 remain controls.
- V33_PAIR4_VR / V33_PAIR8_VR are exact pair-fused nibble-load experiments.
- HOT and ROTATE are separate claims.
- On the 12500H only 1/2/4 P-core runs are promotion-grade; mixed P+E runs are diagnostics.
- E5-2680 v2 physical-server results decide production scaling.

Artifacts:
$master
$CacheDir
"@
$summary | Set-Content -Encoding UTF8 (Join-Path $ResultDir "SUMMARY.txt")
Write-Host "`n$summary" -ForegroundColor Yellow
Write-Host "RESULTS DIRECTORY: $ResultDir" -ForegroundColor Green
