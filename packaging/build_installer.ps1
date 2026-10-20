<#
.SYNOPSIS
    Builds the «Нууус» installer (setup.exe) with Inno Setup 6.

.DESCRIPTION
    1. Unless -SkipExe: runs packaging\build.ps1 (PyInstaller, ~15 min),
       which leaves app_built\ at the repo root with Нууус.exe, _internal\
       and the bundled Ollama in ollama\.
    2. Packs the CUDA libraries (_internal\torch\lib and Ollama's
       lib\ollama\cuda_*) into build\installer\Nuuus-<ver>-libs-<n>.7z
       (parallel LZMA2, each archive well under GitHub's 2 GiB asset limit).
    3. Compiles packaging\installer.iss with ISCC.exe. The setup holds the
       rest of the app and, while installing, takes each libs archive from
       its own folder if it lies there, otherwise downloads it from
       -ReleaseUrl and checks its SHA-256. Upload the setup AND the archives
       to the release; users only download the setup.

    -Monolithic builds the old single setup (~2.6 GB, too big for a GitHub
    release asset) and skips step 2.

    The installer needs no admin rights: it installs per user into
    %LOCALAPPDATA%\Programs\Nuuus (two clicks: Установить -> Завершить),
    creates Start menu / desktop shortcuts and asks on uninstall whether to
    delete the data folder. Models are downloaded by the app on first start.

    Inno Setup 6 is looked up in %LOCALAPPDATA%\Programs\Inno Setup 6 (the
    per-user install), Program Files (x86)\Inno Setup 6, Program Files and
    PATH. Get it from https://jrsoftware.org/isdl.php.

.PARAMETER SkipExe
    Reuse the existing app_built\ instead of rebuilding it.

.PARAMETER Clean
    Passed to build.ps1 (drop PyInstaller's cache first).

.PARAMETER AppVer
    Version shown in the installer and in its file name (Nuuus-Setup-<ver>.exe).

.PARAMETER Monolithic
    One self-contained setup.exe with the CUDA libraries inside.

.PARAMETER ReleaseUrl
    Where the setup downloads the libs archives from
    (default: the GitHub release v<AppVer> of omagith/Nuuus).

.PARAMETER PartLimitMB
    Uncompressed size of one libs archive (the largest DLL, torch_cuda.dll,
    is ~1 GB; LZMA2 shrinks these libraries to ~40 %).

.EXAMPLE
    .\packaging\build_installer.ps1

.EXAMPLE
    .\packaging\build_installer.ps1 -SkipExe -AppVer 1.0.1
#>
# Saved as UTF-8 WITH BOM on purpose: Windows PowerShell 5.1 reads BOM-less
# scripts in the ANSI code page, which would garble the Cyrillic exe name.
[CmdletBinding()]
param(
    [switch]$SkipExe,
    [switch]$Clean,
    [string]$AppVer = "1.0.0",
    [switch]$Monolithic,
    [string]$ReleaseUrl = "",
    [int]$PartLimitMB = 1400
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$AppName = "Нууус"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildScript = Join-Path $PSScriptRoot "build.ps1"
$IssFile = Join-Path $PSScriptRoot "installer.iss"
$BuiltDir = Join-Path $RepoRoot "app_built"
$OutputDir = Join-Path $RepoRoot "build\installer"
if (-not $ReleaseUrl) {
    $ReleaseUrl = "https://github.com/omagith/Nuuus/releases/download/v$AppVer"
}
# Relative to app_built\; whole folders go into the libs archives.
$LibsDirs = @("_internal\torch\lib", "ollama\lib\ollama\cuda_*")
# GitHub refuses release assets of 2 GiB and more.
$AssetLimit = 2GB - 64MB

function Find-Iscc {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
    )
    if (${env:ProgramFiles(x86)}) {
        $candidates += (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
    }
    if ($env:ProgramFiles) {
        $candidates += (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    }
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) {
            return $path
        }
    }
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    return $null
}

Write-Host "== $AppName installer =="

$Iscc = Find-Iscc
if ($null -eq $Iscc) {
    throw "Inno Setup 6 (ISCC.exe) not found. Install it from https://jrsoftware.org/isdl.php (a per-user install is enough)."
}
Write-Host "ISCC:      $Iscc"

# --- 1. The app folder -------------------------------------------------------
if (-not $SkipExe) {
    $buildArgs = @{}
    if ($Clean) {
        $buildArgs["Clean"] = $true
    }
    & $BuildScript @buildArgs
}

$ExePath = Join-Path $BuiltDir "$AppName.exe"
if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "$ExePath is missing. Run without -SkipExe (or .\packaging\build.ps1) first."
}
$OllamaExe = Join-Path $BuiltDir "ollama\ollama.exe"
if (-not (Test-Path -LiteralPath $OllamaExe)) {
    throw "$OllamaExe is missing: put Ollama into vendor\ollama (README.md) and rebuild, otherwise the installed app has no embedded Ollama."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$started = Get-Date
$isccArgs = @("/Qp", "/DAppVer=$AppVer", "/DSourceDir=$BuiltDir", "/DOutputDir=$OutputDir")

# --- 2. libs archives ---------------------------------------------------------
$Parts = @()
if (-not $Monolithic) {
    $Tar = Join-Path $env:SystemRoot "System32\tar.exe"
    if (-not (Test-Path -LiteralPath $Tar)) {
        throw "$Tar not found (bsdtar ships with Windows 10 1803+); it writes the .7z archives."
    }
    $dirs = @()
    foreach ($pattern in $LibsDirs) {
        $dirs += @(Get-ChildItem -Path (Join-Path $BuiltDir $pattern) -Directory -ErrorAction SilentlyContinue)
    }
    if ($dirs.Count -eq 0) {
        throw "None of $($LibsDirs -join ', ') found in $BuiltDir."
    }
    $files = @($dirs | ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -File -Recurse } |
        Sort-Object Length -Descending)

    # First fit, largest files first.
    $limit = [long]$PartLimitMB * 1MB
    $bins = New-Object System.Collections.ArrayList
    foreach ($f in $files) {
        $bin = $null
        foreach ($b in $bins) {
            if ($b.Size + $f.Length -le $limit) { $bin = $b; break }
        }
        if ($null -eq $bin) {
            $bin = [pscustomobject]@{ Size = [long]0; Files = New-Object System.Collections.ArrayList }
            [void]$bins.Add($bin)
        }
        $bin.Size += $f.Length
        [void]$bin.Files.Add($f.FullName.Substring($BuiltDir.Length + 1).Replace("\", "/"))
    }

    Get-ChildItem -LiteralPath $OutputDir -Filter "Nuuus-$AppVer-libs-*.7z" | Remove-Item -Force
    Write-Host "Packing $($files.Count) files into $($bins.Count) libs archives (LZMA2, in parallel)..."
    $procs = @()
    for ($i = 0; $i -lt $bins.Count; $i++) {
        $name = "Nuuus-$AppVer-libs-$($i + 1).7z"
        $list = Join-Path $OutputDir "$name.txt"
        [System.IO.File]::WriteAllLines($list, [string[]]$bins[$i].Files)
        $archive = Join-Path $OutputDir $name
        $tarArgs = "--format 7zip --options 7zip:compression=lzma2,7zip:compression-level=9 " +
            "-cf `"$archive`" -C `"$BuiltDir`" -T `"$list`""
        $p = Start-Process -FilePath $Tar -ArgumentList $tarArgs -NoNewWindow -PassThru
        $null = $p.Handle  # keeps ExitCode readable after the process ends
        $procs += [pscustomobject]@{ Proc = $p; Name = $name; Path = $archive; List = $list; Size = $bins[$i].Size }
    }
    foreach ($item in $procs) {
        $item.Proc.WaitForExit()
        if ($item.Proc.ExitCode -ne 0) {
            throw "tar failed on $($item.Name) with exit code $($item.Proc.ExitCode)."
        }
        Remove-Item -LiteralPath $item.List
        $length = (Get-Item -LiteralPath $item.Path).Length
        if ($length -ge $AssetLimit) {
            throw "$($item.Name) is $([math]::Round($length / 1GB, 2)) GB, too big for a GitHub release; lower -PartLimitMB."
        }
        $hash = (Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash.ToLowerInvariant()
        $Parts += [pscustomobject]@{ Name = $item.Name; Size = $item.Size; Hash = $hash; Length = $length }
        Write-Host ("  {0}: {1:N2} GB" -f $item.Name, ($length / 1GB))
    }

    # One pair of [Files] entries per archive: from the setup's folder, else downloaded.
    $flags = "external extractarchive ignoreversion recursesubdirs createallsubdirs"
    $lines = @()
    foreach ($part in $Parts) {
        $common = "DestDir: `"{app}\app`"; ExternalSize: $($part.Size); Hash: `"$($part.Hash)`""
        $lines += "Source: `"{src}\$($part.Name)`"; $common; Flags: $flags; Check: LibsPartIsLocal('$($part.Name)')"
        $lines += "Source: `"$ReleaseUrl/$($part.Name)`"; DestName: `"$($part.Name)`"; $common; Flags: $flags download; Check: not LibsPartIsLocal('$($part.Name)')"
    }
    $libsIss = Join-Path $OutputDir "libs.iss"
    [System.IO.File]::WriteAllLines($libsIss, [string[]]$lines, (New-Object System.Text.UTF8Encoding $true))
    $excludes = ($dirs | ForEach-Object { "," + $_.FullName.Substring($BuiltDir.Length) }) -join ""
    $isccArgs += @("/DLibsInclude=$libsIss", "/DLibsExcludes=$excludes")
}

# --- 3. setup.exe -------------------------------------------------------------
$isccArgs += $IssFile
$ErrorActionPreference = "Continue"
& $Iscc @isccArgs 2>&1 | ForEach-Object { Write-Host "$_" }
$code = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($code -ne 0) {
    throw "ISCC failed with exit code $code; see the log above."
}

$Setup = Join-Path $OutputDir "Nuuus-Setup-$AppVer.exe"
if (-not (Test-Path -LiteralPath $Setup)) {
    throw "ISCC finished but $Setup is missing."
}
$sizeGb = [math]::Round((Get-Item -LiteralPath $Setup).Length / 1GB, 2)
$minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)

Write-Host ""
Write-Host "Done in $minutes min: $Setup ($sizeGb GB)"
if ($Parts.Count -gt 0) {
    Write-Host "Upload to the release v$AppVer together with the setup:"
    foreach ($part in $Parts) {
        Write-Host "  $($part.Name)  sha256 $($part.Hash)"
    }
}
Write-Host ""
Write-Host "Notes:"
Write-Host "  - The setup is not code-signed: Windows SmartScreen shows 'Windows protected your PC';"
Write-Host "    users click 'More info' -> 'Run anyway' (only a code-signing certificate removes this)."
Write-Host "  - Try a clean install in Windows Sandbox (no Ollama, no VC++ runtime, no CUDA)."
