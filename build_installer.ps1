# Package the Nuitka standalone build into a single Windows installer (setup .exe)
# using Inno Setup.
#
# Usage:
#   .\build_installer.ps1            # package an existing build\nuitka\main.dist
#   .\build_installer.ps1 -Build     # run build_nuitka.ps1 first, then package
#
# Output: build\installer\HolOrama-Setup-<version>.exe

param(
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# -- Locate the Inno Setup compiler (winget installs it per-user by default) ---
$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $iscc) {
    throw "Inno Setup (ISCC.exe) not found. Install it with: winget install JRSoftware.InnoSetup"
}

# -- Read the app version from src\version.py --------------------------------
$versionLine = Select-String -Path 'src\version.py' -Pattern "__version__\s*=\s*'([^']+)'" | Select-Object -First 1
if (-not $versionLine) { throw "Could not read __version__ from src\version.py" }
$version = $versionLine.Matches[0].Groups[1].Value

# -- Optionally (re)build the Nuitka standalone first ------------------------
if ($Build) {
    Write-Host "Running Nuitka build first..." -ForegroundColor Cyan
    & "$PSScriptRoot\build_nuitka.ps1"
    if ($LASTEXITCODE -ne 0) { throw "Nuitka build failed" }
}

$dist = Join-Path $PSScriptRoot 'build\nuitka\main.dist'
if (-not (Test-Path (Join-Path $dist 'HolOrama.exe'))) {
    throw "Standalone build not found at $dist. Run .\build_nuitka.ps1 first (or pass -Build)."
}

# -- Compile the installer ---------------------------------------------------
Write-Host "Building installer for HolOrama $version ..." -ForegroundColor Cyan
& $iscc "/DMyAppVersion=$version" 'installer\HolOrama.iss'
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed with exit code $LASTEXITCODE" }

Write-Host ""
Write-Host "Done. Installer at: build\installer\HolOrama-Setup-$version.exe" -ForegroundColor Green
