param(
    [string]$BuildDirectory = "build-cuda-explicit",
    [int[]]$BodyCounts = @(256, 512, 1024, 2048, 4096, 8192),
    [int[]]$ThreadCounts = @(1, 2, 4, 8, 16),
    [string[]]$Families = @(
        "uniform_cube",
        "plummer_sphere",
        "two_cluster_collision",
        "rotating_disk"
    ),
    [int]$TrajectoriesPerCount = 3,
    [int]$Repetitions = 5,
    [string]$Output = "results\cpu_force_sweep.csv",
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot "$BuildDirectory\Release\nbody_benchmark_force.exe"
$outputPath = Join-Path $projectRoot $Output
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Force benchmark not found: $executable"
}
if (Test-Path -LiteralPath $outputPath) {
    if (-not $Overwrite) {
        throw "Output already exists; pass -Overwrite to replace it: $outputPath"
    }
    Remove-Item -LiteralPath $outputPath
}
New-Item -ItemType Directory -Path (Split-Path -Parent $outputPath) -Force | Out-Null

$counts = $BodyCounts -join ","
$append = $false
foreach ($familyIndex in 0..($Families.Count - 1)) {
    $family = $Families[$familyIndex]
    $baseSeed = 700000 + 100000 * $familyIndex

    $common = @(
        "--body-counts", $counts,
        "--base-seed", $baseSeed,
        "--trajectories-per-count", $TrajectoriesPerCount,
        "--repetitions", $Repetitions,
        "--initial-condition", $family,
        "--output", $outputPath
    )
    $arguments = @("--implementations", "serial") + $common
    if ($append) { $arguments += "--append" }
    & $executable @arguments
    if ($LASTEXITCODE -ne 0) { throw "Serial benchmark failed for $family" }
    $append = $true

    foreach ($threads in $ThreadCounts) {
        & $executable `
            --implementations openmp `
            --threads $threads `
            @common `
            --append
        if ($LASTEXITCODE -ne 0) {
            throw "OpenMP benchmark failed for $family with $threads threads"
        }
    }
}
Write-Host "CPU force sweep written to: $outputPath"
