# OBSERVE - one-line installer (Windows)
#
#   irm https://raw.githubusercontent.com/tritsystem/012-trit-search/main/install.ps1 | iex
#
# Installs into an isolated venv (%USERPROFILE%\.observe) and registers the
# `observe`, `observe-search`, and `observe-mcp` commands on your PATH.
$ErrorActionPreference = "Stop"

$Repo       = "git+https://github.com/tritsystem/012-trit-search.git"
$InstallDir = if ($env:OBSERVE_HOME) { $env:OBSERVE_HOME } else { Join-Path $HOME ".observe" }
$Venv       = Join-Path $InstallDir "venv"
$Scripts    = Join-Path $Venv "Scripts"

Write-Host ""
Write-Host "  OBSERVE - local semantic code search"
Write-Host "  ===================================="

# 1. Find Python >= 3.10 -----------------------------------------------------
$py = $null
foreach ($c in @("python", "py")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) {
        try {
            $v = & $c -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))" 2>$null
            if ($v -and ([version]$v -ge [version]"3.10")) { $py = $c; break }
        } catch {}
    }
}
if (-not $py) {
    Write-Host "  ERROR: Python 3.10+ is required. Install from https://python.org and re-run."
    return
}
Write-Host "  Python : $(& $py --version)"
Write-Host "  Install: $InstallDir"

# 2. Isolated virtual environment --------------------------------------------
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
if (-not (Test-Path $Venv)) { & $py -m venv $Venv }
$Pip = Join-Path $Scripts "pip.exe"

# 3. CPU-only torch first (keeps install ~180MB, not ~2GB), then OBSERVE ------
Write-Host "  Upgrading pip..."
& $Pip install --upgrade --quiet pip
Write-Host "  Installing CPU-only PyTorch (~180MB, one time)..."
& $Pip install --quiet torch --extra-index-url https://download.pytorch.org/whl/cpu
Write-Host "  Installing OBSERVE + dependencies..."
& $Pip install --quiet $Repo

# 4. Add the venv Scripts dir to the user PATH -------------------------------
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$Scripts*") {
    [Environment]::SetEnvironmentVariable("PATH", "$Scripts;$userPath", "User")
    Write-Host "  Added $Scripts to your PATH (restart your terminal to pick it up)."
}

Write-Host ""
Write-Host "  Done. Commands: observe (GUI), observe-search (CLI), observe-mcp (MCP server)"
Write-Host "  Try:  observe-search --help"
Write-Host ""
