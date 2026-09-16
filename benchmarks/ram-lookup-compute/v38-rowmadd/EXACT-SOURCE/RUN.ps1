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
Write-Host " TRANSIT V38 ROWMADD - EXACT Q4_K, V27 PRESERVED" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Pinned prior model only. No autodiscovery. No synthetic fallback." -ForegroundColor Yellow
Write-Host "V38 keeps V37 ROW4LANE as the champion control and tests direct PMADDWD scale fusion in row lanes. ROW4MADD uses the same bytes as V37; ROW4MADD_PRE spends modest RAM on pre-expanded scales. Promotion uses paired AB/BA stability." -ForegroundColor Yellow

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

$Src = Join-Path $Root "v38_rowmadd.cpp"
$Exe = Join-Path $Root "v38_rowmadd.exe"
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
    $csv = Join-Path $ResultDir ("V38_T{0}.csv" -f $th)
    $json = Join-Path $ResultDir ("ATLAS_T{0}.json" -f $th)
    $log = Join-Path $ResultDir ("V38_T{0}.log" -f $th)
    $args = @("--threads","$th","--warmup","$Warmup","--iters","$Iterations","--node","0","--rotate-copies","$RotateCopies","--csv",$csv,"--json",$json,"--model",$Model,"--tensor",$Tensor)
    if ($th -eq ($threadSweep | Select-Object -Last 1)) { $args += @("--cache",$CacheDir) }
    if ($cpus.Count -ge 2 -and $th -eq ($threadSweep | Select-Object -Last 1)) { $args += "--numa" }
    & $Exe @args 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "V38 benchmark failed at threads=$th. See $log" }
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
Write-Host " MULTI-TOKEN / SPECULATIVE REUSE (directional; not V38 promotion gate)" -ForegroundColor Cyan
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
TRANSIT V38 ROWMADD
Timestamp: $(Get-Date -Format o)
Input: PINNED PRIOR REAL GGUF: $Model / $Tensor
Rotating beyond-LLC control: $RotateCopies address-distinct META/X8F/ROW4LANE/ROW4MADD copies
Per-socket physical cores reported by WMI: $perSocket

STABLE PROMOTION GATE:
median speedup > 1.0 AND p10 > 1.0 AND win_rate >= 0.80 AND diff=0

BEST STABLE HOT candidate: $(if($null -ne $bestHot){"$($bestHot.name) @ $($bestHot.Threads) threads = $($bestHot.speedup_vs_baseline)x, p10=$($bestHot.p10), win=$($bestHot.win_rate)"}else{"NONE - keep V27"})
BEST STABLE ROTATE candidate: $(if($null -ne $bestRot){"$($bestRot.name) @ $($bestRot.Threads) threads = $($bestRot.speedup_vs_baseline)x, p10=$($bestRot.p10), win=$($bestRot.win_rate)"}else{"NONE - keep V27"})
BEST directional multi-token reuse: $(if($null -ne $bestVec){"$($bestVec.name) batch=$($bestVec.batch) @ $($bestVec.Threads) threads = $($bestVec.speedup_vs_baseline)x"}else{"n/a"})

RULES:
- V27_META is never removed.
- V31 FSA_RB4/RB16 remain controls.
- V33_PAIR4_VR / V33_PAIR8_VR remain exact pair-fused controls.
- V34_PAIR4_APM moves activation-invariant qsum/min coefficients out of the row loop.
- V34_PAIR4_APM is the previous activation-precomputed control.
- V35_PAIR4_MF isolates exact PMADDWD folding without APM.
- V35_PAIR4_APM_MF combines APM + exact PMADDWD folding and is the primary V35 candidate.
- V36_PAIR4_HF isolates horizontal-word/scale fusion without APM.
- V36_PAIR4_APM_HF combines APM + horizontal-word/scale fusion and is the primary V36 candidate.
- V34_PAIR4_APM_SB remains a RAM-for-broadcast negative/control unless it independently passes the stable gate.
- V36 HF remains a control.
- V37_ROW4LANE is the previous stable champion and remains mandatory control.
- V38_ROW4MADD replaces PMADDWD(...,1)+PMULLD(scale) with direct PMADDWD(sum16,scale16) using runtime PSHUFB scale expansion.
- V38_ROW4MADD_I2 adds two independent accumulation chains to expose ILP.
- V38_ROW4MADD_PRE spends +192 B/tile on pre-expanded scale vectors as a RAM-for-instruction control.
- V38 does not remove V27 or V37; the runtime Atlas may select different exact paths by core count and HOT/ROTATE regime.
- HOT and ROTATE are separate claims.
- On the 12500H only 1/2/4 P-core runs are promotion-grade; mixed P+E runs are diagnostics.
- E5-2680 v2 physical-server results decide production scaling.

Artifacts:
$master
$CacheDir
"@

function Get-BestStable([string]$mode,[bool]$newOnly) {
    $q = $stable | Where-Object { $_.mode -eq $mode }
    if ($newOnly) { $q = $q | Where-Object { $_.name -like "V38_*" } }
    else { $q = $q | Where-Object { $_.name -like "V37_*" } }
    return $q | Sort-Object {[double]$_.speedup_vs_baseline} -Descending | Select-Object -First 1
}
function Verdict-Line([string]$mode) {
    $n = Get-BestStable $mode $true
    $p = Get-BestStable $mode $false
    if ($null -eq $n) {
        return [pscustomobject]@{ Mode=$mode; Status="FAIL"; Text="FAIL - V38 did not pass stability gate"; Advance=$false }
    }
    $v27pct = (([double]$n.speedup_vs_baseline - 1.0) * 100.0)
    if ($null -eq $p) {
        return [pscustomobject]@{ Mode=$mode; Status="SUCCESS"; Text=("SUCCESS +{0:N2}% vs V27; no prior stable candidate in mode" -f $v27pct); Advance=$true }
    }
    $prevpct = (([double]$n.speedup_vs_baseline / [double]$p.speedup_vs_baseline - 1.0) * 100.0)
    if ($prevpct -gt 0) {
        return [pscustomobject]@{ Mode=$mode; Status="SUCCESS"; Text=("SUCCESS +{0:N2}% vs V27; +{1:N2}% vs previous champion {2}" -f $v27pct,$prevpct,$p.name); Advance=$true }
    }
    return [pscustomobject]@{ Mode=$mode; Status="FAIL"; Text=("FAIL TO ADVANCE: +{0:N2}% vs V27 but {1:N2}% vs previous champion {2}" -f $v27pct,$prevpct,$p.name); Advance=$false }
}
$vh = Verdict-Line "HOT"
$vr = Verdict-Line "ROTATE"
$advanced = @($vh,$vr) | Where-Object { $_.Advance }
$finalStatus = if ($vh.Advance -and $vr.Advance) { "SUCCESS TO ADVANCE (HOT + ROTATE)" }
               elseif ($advanced.Count -gt 0) { "PARTIAL SUCCESS TO ADVANCE (" + (($advanced | ForEach-Object {$_.Mode}) -join " + ") + ")" }
               else { "FAIL TO ADVANCE" }
$verdict=@"
============================================================
 V38 FINAL VERDICT
============================================================
HOT:    $($vh.Text)
ROTATE: $($vr.Text)
EXACTNESS: PASS only for candidates admitted by the stable gate (diff=0)
STABILITY GATE: median>1, p10>1, win_rate>=80%
FINAL: $finalStatus
============================================================
"@
$fullSummary = $summary + "`r`n`r`n" + $verdict
$fullSummary | Set-Content -Encoding UTF8 (Join-Path $ResultDir "SUMMARY.txt")
Write-Host "`n$summary" -ForegroundColor Yellow
Write-Host "`n$verdict" -ForegroundColor $(if($advanced.Count -gt 0){"Green"}else{"Red"})
Write-Host "RESULTS DIRECTORY: $ResultDir" -ForegroundColor Green
