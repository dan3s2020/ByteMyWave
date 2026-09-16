param(
    [string]$Model = "C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490",
    [string]$Tensor = "blk.0.ffn_gate.weight",
    [int]$Iterations = 7,
    [int]$Warmup = 2,
    [int]$EvictMB = 0
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
Write-Host " TRANSIT V31 FUSIONFORGE - EXACT Q4_K, V27 PRESERVED" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Pinned prior model only. No autodiscovery. No synthetic fallback." -ForegroundColor Yellow
Write-Host "V27 and V30-X8F remain baselines; V31 is promoted only after exactness + same-run win." -ForegroundColor Yellow

$cpus = @(Get-CimInstance Win32_Processor)
$mem = @(Get-CimInstance Win32_PhysicalMemory)
$os = Get-CimInstance Win32_OperatingSystem
$totalGB = [math]::Round(($mem | Measure-Object Capacity -Sum).Sum/1GB,2)
$freeGB = [math]::Round($os.FreePhysicalMemory/1MB,2)
if ($EvictMB -le 0) {
    if ($totalGB -ge 512) { $EvictMB = 2048 }
    elseif ($totalGB -ge 64) { $EvictMB = 1024 }
    else { $EvictMB = 256 }
}
$hw = [ordered]@{
    Timestamp=(Get-Date).ToString("o")
    OS="$($os.Caption) $($os.Version) build $($os.BuildNumber)"
    CPUs=@($cpus | ForEach-Object { [ordered]@{Name=$_.Name.Trim();Cores=$_.NumberOfCores;Logical=$_.NumberOfLogicalProcessors;MaxMHz=$_.MaxClockSpeed} })
    DIMMs=@($mem | ForEach-Object { [ordered]@{Manufacturer=$_.Manufacturer;PartNumber=("$($_.PartNumber)").Trim();CapacityGB=[math]::Round($_.Capacity/1GB,2);Speed=$_.Speed;Configured=$_.ConfiguredClockSpeed} })
    TotalPhysicalGB=$totalGB; FreePhysicalGB=$freeGB; EvictMB=$EvictMB
}
$hw | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $ResultDir "HARDWARE.json")
$cpus | Format-Table Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed -AutoSize
Write-Host ("RAM: {0} GB total, ~{1} GB free; cold-control eviction surface = {2} MiB" -f $totalGB,$freeGB,$EvictMB) -ForegroundColor Green

$Src = Join-Path $Root "v31_ramforge.cpp"
$Exe = Join-Path $Root "v31_ramforge.exe"
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
$threadSweep = @(1,4,8,$perSocket) | Where-Object { $_ -ge 1 -and $_ -le $perSocket } | Sort-Object -Unique
Write-Host "`nPhysical-core-count sweep target: $($threadSweep -join ', ')" -ForegroundColor Cyan
Write-Host "Inside the native harness V31 chooses ONE logical CPU per physical core; on hybrid CPUs higher EfficiencyClass is preferred first." -ForegroundColor DarkCyan

$all = New-Object System.Collections.Generic.List[object]
foreach ($th in $threadSweep) {
    Write-Host "`n-------------------- THREADS=$th --------------------" -ForegroundColor Magenta
    $csv = Join-Path $ResultDir ("V31_T{0}.csv" -f $th)
    $json = Join-Path $ResultDir ("ATLAS_T{0}.json" -f $th)
    $log = Join-Path $ResultDir ("V31_T{0}.log" -f $th)
    $args = @("--threads","$th","--warmup","$Warmup","--iters","$Iterations","--node","0","--evict-mb","$EvictMB","--csv",$csv,"--json",$json,"--model",$Model,"--tensor",$Tensor)
    if ($th -eq $perSocket) { $args += @("--numa","--cache",$CacheDir) }
    & $Exe @args 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "V31 benchmark failed at threads=$th. See $log" }
    Import-Csv $csv | ForEach-Object { $_ | Add-Member -NotePropertyName Threads -NotePropertyValue $th; $all.Add($_) }
}

$master=Join-Path $ResultDir "MASTER_RESULTS.csv"
$all | Export-Csv -NoTypeInformation -Encoding UTF8 $master

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " HOT SINGLE-TOKEN LEADERBOARD" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
$all | Where-Object { $_.mode -eq "HOT" -and $_.batch -eq "1" -and [double]$_.max_abs_diff -le 0.0001 } |
    Sort-Object {[double]$_.speedup_vs_baseline} -Descending |
    Select-Object Threads,name,rowblock,prefetch,median_ms,speedup_vs_baseline,max_abs_diff | Format-Table -AutoSize

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " COLD / BEYOND-LLC SINGLE-TOKEN LEADERBOARD" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
$all | Where-Object { $_.mode -eq "COLD_EVICT" -and $_.batch -eq "1" -and [double]$_.max_abs_diff -le 0.0001 } |
    Sort-Object {[double]$_.speedup_vs_baseline} -Descending |
    Select-Object Threads,name,rowblock,prefetch,median_ms,speedup_vs_baseline,max_abs_diff | Format-Table -AutoSize

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " MULTI-TOKEN / SPECULATIVE REUSE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
$all | Where-Object { [int]$_.batch -gt 1 } |
    Sort-Object Threads,{[int]$_.batch},{[double]$_.speedup_vs_baseline} -Descending |
    Select-Object Threads,batch,name,median_ms,speedup_vs_baseline | Format-Table -AutoSize

$bestHot=$all | Where-Object {$_.mode -eq "HOT" -and $_.batch -eq "1" -and [double]$_.max_abs_diff -le 0.0001} | Sort-Object {[double]$_.speedup_vs_baseline} -Descending | Select-Object -First 1
$bestCold=$all | Where-Object {$_.mode -eq "COLD_EVICT" -and $_.batch -eq "1" -and [double]$_.max_abs_diff -le 0.0001} | Sort-Object {[double]$_.speedup_vs_baseline} -Descending | Select-Object -First 1
$bestVec=$all | Where-Object {[int]$_.batch -gt 1 -and $_.name -like "V31_VEC*"} | Sort-Object {[double]$_.speedup_vs_baseline} -Descending | Select-Object -First 1

$lastAtlas=Join-Path $ResultDir ("ATLAS_T{0}.json" -f $perSocket)
$capacityPlan=Join-Path $ResultDir "RAMFORGE_CAPACITY_PLAN.txt"
if (Test-Path $lastAtlas) {
    $aj=Get-Content $lastAtlas -Raw | ConvertFrom-Json
    $orig=[double]$aj.original_q4_bytes
    $budgets=@(64GB,128GB,256GB,512GB)
    $lines=New-Object System.Collections.Generic.List[string]
    $lines.Add("V31 FUSIONFORGE compiled-representation capacity arithmetic (NOT a performance claim)")
    foreach($p in $aj.representations.PSObject.Properties){
        $ratio=[double]$p.Value/$orig
        $lines.Add("")
        $lines.Add(("{0}: {1:N3}x original Q4 bytes" -f $p.Name,$ratio))
        foreach($b in $budgets){$eq=($b/$ratio)/1GB;$lines.Add(("  budget {0,4} GB -> ~{1:N1} GB original-Q4-equivalent coverage" -f [int]($b/1GB),$eq))}
    }
    $lines | Set-Content -Encoding UTF8 $capacityPlan
}

$summary=@"
TRANSIT V31 FUSIONFORGE
Timestamp: $(Get-Date -Format o)
Input: PINNED PRIOR REAL GGUF: $Model / $Tensor
Eviction control: $EvictMB MiB, touched outside timed region
Per-socket physical cores reported by WMI: $perSocket

BEST HOT exact candidate: $($bestHot.name) @ $($bestHot.Threads) threads = $($bestHot.speedup_vs_baseline)x vs same-run HOT V27
BEST COLD exact candidate: $($bestCold.name) @ $($bestCold.Threads) threads = $($bestCold.speedup_vs_baseline)x vs same-run COLD V27
BEST multi-token reuse: $($bestVec.name) batch=$($bestVec.batch) @ $($bestVec.Threads) threads = $($bestVec.speedup_vs_baseline)x vs serial V27 for same batch

PROMOTION RULE:
- V27_META is never removed.
- V30_META_X8F remains the previous measured winner/control.
- A V31 path is accepted only when diff=0 (or stated exact tolerance), same input, same run, and faster than the matching V27 mode.
- HOT and COLD_EVICT are separate claims. Do not use HOT speedup as a DRAM-streaming claim.
- Laptop results select mechanisms. E5-2680 v2 physical-server results decide the Transit production path.

Artifacts:
$master
$capacityPlan
$CacheDir
"@
$summary | Set-Content -Encoding UTF8 (Join-Path $ResultDir "SUMMARY.txt")
Write-Host "`n$summary" -ForegroundColor Yellow
Write-Host "RESULTS DIRECTORY: $ResultDir" -ForegroundColor Green
