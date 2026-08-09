$ErrorActionPreference = "Stop"

Write-Host "[AUDIT] Repository: $PWD"
$env:PYTHONPATH = "$PWD\src"

Write-Host "[AUDIT] Forbidden generated artifacts..."
$forbidden = @(
    Get-ChildItem -Path ".\src" -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
    Get-ChildItem -Path ".\src" -Recurse -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue
)
if ($forbidden.Count -gt 0) {
    $forbidden | ForEach-Object { Write-Warning $_.FullName }
    throw "Generated Python metadata/cache remains in source tree."
}

if (Test-Path ".\runs") {
    throw "runs\ must not be part of the public source release. Publish the approved model as a GitHub release asset."
}

Write-Host "[AUDIT] Compiling package..."
python -m compileall -q ".\src\points2sbl"
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }

Write-Host "[AUDIT] Running tests..."
pytest -q
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

Write-Host "[AUDIT] Validating configs..."
foreach ($cfg in @(
    ".\configs\point_transformer.yaml",
    ".\configs\pointnet2.yaml",
    ".\configs\pointnext.yaml"
)) {
    points2sbl validate-config --config $cfg
    if ($LASTEXITCODE -ne 0) { throw "Config validation failed: $cfg" }
}

Write-Host "[AUDIT] Searching shipped source/config/docs for removed V2 integration..."
$v2 = Get-ChildItem -Path ".\src", ".\configs", ".\docs", ".\examples" -Recurse -File -ErrorAction SilentlyContinue |
    Select-String -Pattern "point_transformer_v2|model_v2|features_v2|train_v2|prep_v2|predict_v2|dataset_v2|run_pipeline_v2|cli_v2|io_las_v2"

if ($v2) {
    $v2
    throw "Residual V2 integration references found."
}

Write-Host "[AUDIT] Searching public files for private machine paths..."
$privatePaths = Get-ChildItem -Path ".\src", ".\configs", ".\docs", ".\examples", ".\tools" -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "audit_release\.ps1$" } |
    Select-String -Pattern 'C:\\Users\\fareed|D:\\USA\\|E:\\|F:\\Labelled'

if ($privatePaths) {
    $privatePaths
    throw "Machine-specific paths remain in public files."
}

Write-Host "[AUDIT] Model manager..."
points2sbl model path
if ($LASTEXITCODE -ne 0) { throw "model manager failed" }

Write-Host "[AUDIT] Building package..."
python -m build
if ($LASTEXITCODE -ne 0) { throw "package build failed" }

Write-Host "[AUDIT] Twine metadata..."
python -m twine check ".\dist\*"
if ($LASTEXITCODE -ne 0) { throw "twine check failed" }

Write-Host "[AUDIT] COMPLETE"
