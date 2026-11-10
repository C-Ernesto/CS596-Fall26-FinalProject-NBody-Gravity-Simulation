param(
    [int[]]$BodyCounts = @(1024, 4096, 16384),
    [int[]]$BlockSizes = @(64, 128, 256, 512),
    [int]$Steps = 10,
    [int]$Repetitions = 5,
    [string]$OutputPath = "",
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot "build-cuda-explicit\Release\nbody.exe"
$resultsDirectory = Join-Path $projectRoot "results"
$implementations = "cuda_naive", "cuda_tiled"

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $resultsDirectory "cuda_block_sweep.csv"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $projectRoot $OutputPath
}

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Release executable not found: $executable"
}
if ($BodyCounts.Count -eq 0 -or ($BodyCounts | Where-Object { $_ -le 0 })) {
    throw "Every body count must be positive"
}
if ($BlockSizes.Count -eq 0 -or ($BlockSizes | Where-Object { $_ -le 0 })) {
    throw "Every block size must be positive"
}
if ($Steps -le 0) {
    throw "Steps must be positive"
}
if ($Repetitions -le 0) {
    throw "Repetitions must be positive"
}

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

if (Test-Path -LiteralPath $OutputPath) {
    if (-not $Overwrite) {
        throw "Output already exists. Use -Overwrite or choose another path: $OutputPath"
    }
    Remove-Item -LiteralPath $OutputPath
}

$configurationCount = (
    $implementations.Count * $BodyCounts.Count * $BlockSizes.Count
)
$configurationIndex = 0

foreach ($type in $implementations) {
    foreach ($count in $BodyCounts) {
        foreach ($blockSize in $BlockSizes) {
            $configurationIndex++
            Write-Host (
                "[{0}/{1}] {2}: bodies={3}, block-size={4}" -f `
                $configurationIndex,
                $configurationCount,
                $type,
                $count,
                $blockSize
            )

            & $executable `
                --implementation $type `
                --bodies $count `
                --steps $Steps `
                --repetitions $Repetitions `
                --block-size $blockSize `
                --output $OutputPath

            if ($LASTEXITCODE -ne 0) {
                throw (
                    "Benchmark failed for {0}, bodies={1}, block-size={2}" -f `
                    $type,
                    $count,
                    $blockSize
                )
            }
        }
    }
}

Write-Host "Block-size sweep completed: $OutputPath"
