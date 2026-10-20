<#
.SYNOPSIS
    Builds the portable «Нууус» app folder with PyInstaller.

.DESCRIPTION
    Runs packaging\Nuuus.spec with the project's venv, then replaces
    app_built\ at the repo root with the fresh build\dist\Nuuus folder
    (it contains Нууус.exe with the app icon and the embedded Ollama from
    vendor\ollama in ollama\). Ship app_built\ as <folder>\app\ next to
    <folder>\data\, or wrap it into setup.exe with
    packaging\build_installer.ps1 -SkipExe.

    Works from any current directory: paths are resolved from the script
    location.

.PARAMETER Clean
    Drop PyInstaller's cache and build\work before building (use after
    upgrading packages or when a build behaves strangely).

.EXAMPLE
    .\packaging\build.ps1

.EXAMPLE
    .\packaging\build.ps1 -Clean
#>
# Saved as UTF-8 WITH BOM on purpose: Windows PowerShell 5.1 reads BOM-less
# scripts in the ANSI code page, which would garble the Cyrillic exe name.
[CmdletBinding()]
param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$AppName = "Нууус"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SpecFile = Join-Path $PSScriptRoot "Nuuus.spec"
$DistDir = Join-Path $RepoRoot "build\dist"
$WorkDir = Join-Path $RepoRoot "build\work"
$BuiltDir = Join-Path $DistDir "Nuuus"
$TargetDir = Join-Path $RepoRoot "app_built"
$FfmpegExe = Join-Path $RepoRoot "app\ffmpeg\bin\ffmpeg.exe"
$IconFile = Join-Path $RepoRoot "app\assets\icon.ico"
$VendorOllama = Join-Path $RepoRoot "vendor\ollama\ollama.exe"

function Invoke-Native {
    # Runs a native command, streams its output (stderr included) as plain
    # text and throws on a non-zero exit code. pip and PyInstaller log to
    # stderr; under $ErrorActionPreference = "Stop" Windows PowerShell 5.1
    # would turn the first stderr line into a terminating error whenever the
    # output is redirected, so the preference is relaxed inside this function.
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @()
    )
    $ErrorActionPreference = "Continue"
    & $FilePath @Arguments 2>&1 | ForEach-Object { Write-Host "$_" }
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "Command failed with exit code ${code}: $FilePath $($Arguments -join ' ')"
    }
}

function Test-PythonModule {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][string]$Module
    )
    $ErrorActionPreference = "Continue"
    & $Python -c "import $Module" *> $null
    return ($LASTEXITCODE -eq 0)
}

Write-Host "== $AppName build =="
Write-Host "Repo root: $RepoRoot"

# --- Python: the project's venv first, PATH as a fallback ------------------
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvPython) {
    $Python = $VenvPython
}
else {
    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $PythonCmd) {
        throw "Python not found. Create .venv in the repo root (see README.md) or put python on PATH."
    }
    $Python = $PythonCmd.Source
    Write-Warning "No .venv found in the repo root; building with $Python"
}
Write-Host "Python:    $Python"

# --- Sanity checks before a long build ---------------------------------------
if (-not (Test-PythonModule -Python $Python -Module "PyInstaller")) {
    Write-Host "PyInstaller is not installed; installing it..."
    Invoke-Native -FilePath $Python -Arguments @("-m", "pip", "install", "pyinstaller")
}
foreach ($module in @("PyQt6", "torch", "gigaam")) {
    if (-not (Test-PythonModule -Python $Python -Module $module)) {
        throw "Python package '$module' is not importable with $Python. Install the dependencies first (README.md)."
    }
}
if (-not (Test-Path -LiteralPath $IconFile)) {
    throw "$IconFile not found: the exe and the windows would have no app icon."
}
if (-not (Test-Path -LiteralPath $VendorOllama)) {
    Write-Warning "$VendorOllama not found: the build will have no embedded Ollama (users would need an installed one)."
}
if (-not (Test-Path -LiteralPath $FfmpegExe)) {
    Write-Warning "$FfmpegExe not found: the build will not bundle ffmpeg and will rely on ffmpeg from PATH."
}
$running = Get-Process -Name $AppName -ErrorAction SilentlyContinue
if ($null -ne $running) {
    throw "$AppName.exe is running. Close it first, otherwise app_built\ cannot be replaced."
}

# --- Build ---------------------------------------------------------------------
if ($Clean -and (Test-Path -LiteralPath $WorkDir)) {
    Write-Host "Removing $WorkDir"
    Remove-Item -LiteralPath $WorkDir -Recurse -Force
}

$pyiArgs = @(
    "-m", "PyInstaller", $SpecFile,
    "--distpath", $DistDir,
    "--workpath", $WorkDir,
    "--noconfirm"
)
if ($Clean) {
    $pyiArgs += "--clean"
}

$started = Get-Date
Push-Location $RepoRoot
try {
    Invoke-Native -FilePath $Python -Arguments $pyiArgs
}
finally {
    Pop-Location
}

$ExePath = Join-Path $BuiltDir "$AppName.exe"
if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "PyInstaller finished but $ExePath is missing; see the log above."
}

# --- Replace app_built\ at the repo root ----------------------------------------
if (Test-Path -LiteralPath $TargetDir) {
    Write-Host "Replacing $TargetDir"
    Remove-Item -LiteralPath $TargetDir -Recurse -Force
}
Move-Item -LiteralPath $BuiltDir -Destination $TargetDir

$sizeBytes = (Get-ChildItem -LiteralPath $TargetDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
$sizeGb = [math]::Round($sizeBytes / 1GB, 2)
$minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)

Write-Host ""
Write-Host "Done in $minutes min: $TargetDir\$AppName.exe ($sizeGb GB)"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Try it in place: .\app_built\$AppName.exe"
Write-Host "     (it uses .\data next to app_built\, the same data folder as the source checkout)."
Write-Host "  2. To ship: copy app_built\ to the target machine as <folder>\app\;"
Write-Host "     <folder>\data\ is created next to it on first start."
Write-Host "  3. Ollama is embedded (app_built\ollama); the models download on first start"
Write-Host "     (Settings > Models) into data\models. GPU mode needs an NVIDIA driver."
Write-Host "  4. setup.exe: .\packaging\build_installer.ps1 -SkipExe"
Write-Host "  5. If the exe does not start: read data\logs\app.log; for import errors set"
Write-Host "     CONSOLE = True in packaging\Nuuus.spec, rebuild and run the exe from a terminal."
