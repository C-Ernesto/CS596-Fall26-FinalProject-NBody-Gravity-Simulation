param(
    [string]$BuildDirectory = "build-cuda-explicit",
    [string]$DatasetDirectory = "data\raw\candidate_v1",
    [string]$Output = "results\cpu_label_validation.csv",
    [int]$TrajectoriesPerCount = 1,
    [string]$SnapshotIds = "0,7",
    [double]$Tolerance = 1.0e-10,
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot "$BuildDirectory\Release\nbody_validate_dataset.exe"
$datasetPath = Join-Path $projectRoot $DatasetDirectory
$outputPath = Join-Path $projectRoot $Output

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Dataset validator not found: $executable"
}
if (-not (Test-Path -LiteralPath $datasetPath -PathType Container)) {
    throw "Dataset directory not found: $datasetPath"
}
if (Test-Path -LiteralPath $outputPath) {
    if (-not $Overwrite) {
        throw "Output already exists; pass -Overwrite to replace it: $outputPath"
    }
    Remove-Item -LiteralPath $outputPath
}

New-Item -ItemType Directory -Path (Split-Path -Parent $outputPath) -Force | Out-Null
$files = Get-ChildItem -LiteralPath $datasetPath -Filter "*.csv" -File |
    Where-Object { $_.Name -ne "manifest.csv" } |
    Sort-Object Name

foreach ($file in $files) {
    Write-Host "Validating $($file.Name)..."
    & $executable `
        --input $file.FullName `
        --output $outputPath `
        --trajectories-per-count $TrajectoriesPerCount `
        --snapshot-ids $SnapshotIds `
        --tolerance $Tolerance
    if ($LASTEXITCODE -ne 0) {
        throw "Serial validation failed for $($file.FullName)"
    }
}

$rows = Import-Csv -LiteralPath $outputPath
$failed = @($rows | Where-Object { $_.passed -ne "true" })
if ($failed.Count -ne 0) {
    throw "$($failed.Count) selected snapshots failed serial validation"
}
Write-Host "Validated $($rows.Count) stratified snapshots. Results: $outputPath"
