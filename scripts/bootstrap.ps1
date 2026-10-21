# Bootstrap the workspace on Windows PowerShell.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1

$ErrorActionPreference = "Stop"
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Bootstrapping the TinyCode workspace in $WorkspaceRoot"

foreach ($repo in @("tinycode-model-builder", "tinycode-inference-engine")) {
    $path = Join-Path $WorkspaceRoot $repo
    if (-not (Test-Path $path)) {
        Write-Error "missing repository: $path"
    }
    Write-Host ""
    Write-Host "== $repo =="
    Push-Location $path
    try {
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
        python -m pip install pytest
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "== golden fixture =="
python (Join-Path $WorkspaceRoot "scripts/make_golden_fixture.py")

Write-Host ""
Write-Host "== contract check =="
python (Join-Path $WorkspaceRoot "scripts/verify_contracts.py")

Write-Host ""
Write-Host "Done. Start the builder with: cd tinycode-model-builder; python main.py"
