# Build TensorScope Windows executable.
# Run from the repo root: .\build\build_windows.ps1
#
# Requires: Python venv with requirements_bundle.txt + pyinstaller installed.
# Output: dist\TensorScope\TensorScope.exe

param(
    [switch]$Gpu  # pass -Gpu to build with CUDA torch (requires CUDA installed)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot

# Activate venv if present
$venv = Join-Path $root ".venv"
if (Test-Path "$venv\Scripts\Activate.ps1") {
    . "$venv\Scripts\Activate.ps1"
}

if ($Gpu) {
    Write-Host "Installing CUDA torch (cu124)..."
    pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
}

pip install --upgrade pyinstaller
pyinstaller "$root\build\tensorscope.spec" --distpath "$root\dist" --workpath "$root\build\work"

Write-Host "Done. Executable: $root\dist\TensorScope\TensorScope.exe"
