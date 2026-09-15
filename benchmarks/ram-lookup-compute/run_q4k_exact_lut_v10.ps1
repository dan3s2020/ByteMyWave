param(
    [Parameter(Mandatory = $true)]
    [string]$ModelBlob,

    [string]$Work = "$env:TEMP\q4k_exact_v10",

    [int]$Iterations = 10
)

$ErrorActionPreference = "Stop"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $Work | Out-Null

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$prepare = Join-Path $scriptDir "prepare_q4k_exact.py"
$kernel = Join-Path $scriptDir "q4k_exact_lut_v10.cs"

python $prepare $ModelBlob $Work

$meta = Get-Content (Join-Path $Work "meta.json") -Raw |
    ConvertFrom-Json

$source = Get-Content $kernel -Raw

$cp = New-Object System.CodeDom.Compiler.CompilerParameters
$cp.CompilerOptions = "/unsafe /optimize+"
$cp.GenerateInMemory = $true
$cp.GenerateExecutable = $false

$stopwatchAssembly = [System.Diagnostics.Stopwatch].Assembly.Location
if ($stopwatchAssembly)
{
    $cp.ReferencedAssemblies.Add($stopwatchAssembly) | Out-Null
}

if (-not ("Q4KExactLutV10" -as [type]))
{
    Add-Type `
        -TypeDefinition $source `
        -Language CSharp `
        -CompilerParameters $cp
}

[Q4KExactLutV10]::Run(
    (Join-Path $Work "tensor-q4k.bin"),
    (Join-Path $Work "activation-f32.bin"),
    (Join-Path $Work "reference-f32.bin"),
    [int]$meta.rows,
    [int]$meta.cols,
    $Iterations
)
