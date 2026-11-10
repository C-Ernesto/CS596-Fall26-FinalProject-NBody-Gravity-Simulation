Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot "build-cuda-explicit\Release\nbody.exe"
$resultsDirectory = Join-Path $projectRoot "results"
$outputPath = Join-Path $resultsDirectory "cuda_optimized_scaling.csv"

$implementations = "cuda_naive", "cuda_tiled"
$bodyCounts = 256, 512, 1024, 2048, 4096, 8192, 16384

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Release executable not found: $executable"
}

New-Item -ItemType Directory -Path $resultsDirectory -Force | Out-Null

foreach ($type in $implementations) {
    foreach ($count in $bodyCounts) {
        Write-Host "Running $type with $count bodies..."

        & $executable `
            --implementation $type `
            --bodies $count `
            --steps 10 `
            --repetitions 5 `
            --block-size 64 `
            --output $outputPath

        if ($LASTEXITCODE -ne 0) {
            throw "Benchmark failed for $type with $count bodies"
        }
    }
}

Write-Host "Benchmark results appended to: $outputPath"
