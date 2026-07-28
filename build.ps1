<#
    build_bundle.ps1 - Quarry F4 dependency-free release builder.

    Produces  dist/Quarry/  (the end user unzips and runs it; nothing Python-side
    needs installing - no system Python, pip, numpy, Pillow or lz4) and zips it to
    dist/Quarry-v<ver>.zip.

    ONE prerequisite: the .NET Desktop Runtime 8. Carrying it inside the exe cost
    155 MB against a 0.26 MB app, and WinForms is not trim-compatible, so the bundle
    asks for it instead and README.txt says where to get it.

    Layout produced:
      dist/Quarry/
        Quarry.exe                      framework-dependent .NET (win-x64)
        python/                         embedded CPython 3.14.3
          python.exe, python314.dll, python314.zip
          python314._pth                stdlib + . + Lib\site-packages + ..\bakers
          Lib/site-packages/            numpy 2.4.4, Pillow 12.2.0, lz4 4.4.5
        bakers/
          *.py                          every tools/*.py (bakers + local modules)
          map_export/                   sa_source.py + siblings
          gvcslib/                      vendored .py-only (NO 5.1G work/ assets)
          formats/  core/               vendored SAW subset

    The recipe (embed + _pth-sans-site isolation, dead-dev-path skip, .py-only
    vendor) was validated end-to-end before this script was written; see
    docs/superpowers/specs/2026-07-25-quarry-nodeps-bundle-design.md.

    Usage:  pwsh -File tools/quarry/build_bundle.ps1 [-Ver 750] [-Clean]
#>
[CmdletBinding()]
param(
    [string]$Ver = "dev",
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# ---- pinned versions (match the dev host system-Python the bakers validated on) ----
$PyVer   = "3.14.3"
$NumPy   = "2.4.4"
$Pillow  = "12.2.0"
$Lz4     = "4.4.5"

# ---- paths ----
$QuarryDir = $PSScriptRoot                                   # tools/quarry
$Repo      = (Resolve-Path (Join-Path $QuarryDir "..\..")).Path
$Tools     = Join-Path $Repo "tools"
$Csproj    = Join-Path $QuarryDir "Quarry\Quarry.csproj"
$Dist      = Join-Path $QuarryDir "dist"
$Out       = Join-Path $Dist "Quarry"
$Cache     = Join-Path $QuarryDir ".cache"                   # embed zip + get-pip, downloaded once

# vendored source roots (dev tree). Override via env if the layout differs.
$GvcsInner = if ($env:QUARRY_GVCS) { $env:QUARRY_GVCS } else { throw "set QUARRY_GVCS to the vendored library root" }
$Saw       = if ($env:QUARRY_SAW)  { $env:QUARRY_SAW  } else { throw "set QUARRY_SAW to the workbench root" }

function Say($m) { Write-Host "[bundle] $m" -ForegroundColor Cyan }

if ($Clean -and (Test-Path $Out)) { Say "clean $Out"; Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force -Path $Out, $Cache | Out-Null

# ---------------------------------------------------------------- 1. .NET publish
# Refuse to build while a Quarry from the target folder is running. The build wipes
# python/ before re-extracting it, and doing that under a live convert pulls numpy out
# from under the bakers mid-run - it fails with a circular-import error that looks like
# a code fault and is not one.
$running = Get-Process -Name Quarry -ErrorAction SilentlyContinue |
           Where-Object { $_.Path -and $_.Path.StartsWith($Dist, [StringComparison]::OrdinalIgnoreCase) }
if ($running) {
    throw "Quarry is running from $Dist (pid $($running.Id -join ', ')). Close it first: this build replaces the embedded Python underneath it."
}

# The zip name comes from -Ver while the number the app SHOWS comes from the source
# constant. When they disagree the bundle looks new and reports something else, which
# is how a v771 build shipped inside a zip called v773. Refuse rather than mislead.
if ($Ver -ne "dev") {
    $src = Get-Content (Join-Path $QuarryDir "Quarry\GameVersion.cs") -Raw
    if ($src -match 'Version = "v(\d+)"') {
        if ($Matches[1] -ne $Ver) {
            throw "version mismatch: -Ver $Ver but GameVersion.cs says v$($Matches[1]). Update the constant first."
        }
        Say "version check: source and -Ver agree on v$Ver"
    } else { throw "could not read the version constant from GameVersion.cs" }
}

Say "dotnet publish (framework-dependent win-x64)"
# Framework-dependent, so the .NET runtime is NOT carried inside the exe: that alone
# was 155 MB of the bundle against 0.26 MB for the app itself, and WinForms cannot be
# trimmed to close the gap. The user installs the .NET Desktop Runtime once; README.txt
# says where. Everything Python-side still ships, so the only prerequisite is that one.
# -p:DebugType=none: without it the PE keeps a debug directory holding the FULL build
# path of the .pdb, which ships the developer's directory layout inside a public binary.
$exePath = Join-Path $Out "Quarry.exe"
$before  = if (Test-Path $exePath) { (Get-Item $exePath).LastWriteTimeUtc } else { [datetime]::MinValue }
dotnet publish $Csproj -c Release -r win-x64 --self-contained false -p:DebugType=none -p:DebugSymbols=false -o $Out | Out-Null
# Existence is not proof: a failed publish leaves the PREVIOUS exe sitting there and
# the bundle ships stale. A broken csproj comment did exactly that once. Check the
# exit code and that the file actually moved.
if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed (exit $LASTEXITCODE) - run it without -o redirection to see why" }
if (-not (Test-Path $exePath)) { throw "publish did not produce Quarry.exe" }
if ((Get-Item $exePath).LastWriteTimeUtc -le $before) { throw "Quarry.exe was not rewritten - the publish silently kept the old binary" }
# drop debug symbols - not needed by end users
Get-ChildItem $Out -Filter *.pdb -ErrorAction SilentlyContinue | Remove-Item -Force

# ---------------------------------------------------------------- 2. embed CPython
$PyDir  = Join-Path $Out "python"
$PyZip  = Join-Path $Cache "python-$PyVer-embed-amd64.zip"
if (-not (Test-Path $PyZip)) {
    Say "download python-$PyVer-embed-amd64.zip"
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$PyVer/python-$PyVer-embed-amd64.zip" -OutFile $PyZip
}
Say "extract embed -> python/"
if (Test-Path $PyDir) { Remove-Item -Recurse -Force $PyDir }
Expand-Archive -Path $PyZip -DestinationPath $PyDir -Force

# ---- _pth: stdlib zip + own dir + site-packages + sibling bakers.
# NO `import site` - an embeddable with a ._pth and no site() builds sys.path from
# these lines ALONE, which isolates the bundle from the user's %APPDATA% site-packages
# (a real leak otherwise) and makes PYTHONPATH irrelevant.
$PthMajorMinor = ($PyVer -split '\.')[0..1] -join ''       # 3.14.3 -> 314
$Pth = Join-Path $PyDir "python$PthMajorMinor._pth"
@"
python$PthMajorMinor.zip
.
Lib\site-packages
..\bakers
..\bakers\gvcslib
"@ | Set-Content -Path $Pth -Encoding ascii
Say "wrote $($(Split-Path $Pth -Leaf))"

# ---------------------------------------------------------------- 3. pip + wheels
$PyExe   = Join-Path $PyDir "python.exe"
$GetPip  = Join-Path $Cache "get-pip.py"
if (-not (Test-Path $GetPip)) {
    Say "download get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPip
}
Say "bootstrap pip"
& $PyExe $GetPip --no-warn-script-location | Out-Null
$SitePk = Join-Path $PyDir "Lib\site-packages"
Say "pip install numpy==$NumPy Pillow==$Pillow lz4==$Lz4 -> embed"
# --ignore-installed + --target: force the wheels INTO the embed regardless of any
# global/user install on the build host.
& $PyExe -m pip install --no-warn-script-location --ignore-installed --target $SitePk `
    "numpy==$NumPy" "Pillow==$Pillow" "lz4==$Lz4" | Out-Null

# runtime needs only numpy/Pillow/lz4 - drop pip/setuptools (build-time only) and
# the wheels' bundled test suites, so nothing extra ships.
Get-ChildItem $SitePk -Directory | Where-Object { $_.Name -match '^(pip|setuptools|pkg_resources|_distutils_hack)($|-)' } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $SitePk -Recurse -Directory -Filter tests -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- 4. vendor scripts + libs
$Bakers = Join-Path $Out "bakers"
if (Test-Path $Bakers) { Remove-Item -Recurse -Force $Bakers }
New-Item -ItemType Directory -Force -Path $Bakers | Out-Null

Say "vendor tools/*.py + map_export/"
Copy-Item (Join-Path $Tools "*.py") $Bakers
robocopy (Join-Path $Tools "map_export") (Join-Path $Bakers "map_export") /E /NJH /NJS /NFL /NDL /XD __pycache__ | Out-Null

# content/: the two inputs that are the DEMAKE's own, not the disc's - the script
# listing the assembler turns into script/scripts.scm, and the string table it prints
# through. Every other baker input comes off the user's disc; these ship with the tool.
$Content = Join-Path $Out "content"
if (Test-Path $Content) { Remove-Item -Recurse -Force $Content }
Say "vendor content/ (script listing + string table)"
foreach ($rel in @("script\scripts.scm.txt", "text\strings.txt", "effects\marker.bin")) {
    $src = Join-Path $Repo "data\$rel"
    if (-not (Test-Path $src)) { throw "content input missing: data\$rel" }
    $dst = Join-Path $Content $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
    Copy-Item $src $dst
}

# Drop dev/build-only tools that the CONVERTER never invokes (verified: none are
# referenced by ConvertPipeline or imported by any converter baker). They are not
# part of the shipped toolchain and some need unbundled deps (imageio_ffmpeg,
# soundfile, qrcode, websocket) or do dev-time file I/O -> excluding them keeps the
# release clean and every remaining baker importable under the embedded interpreter.
$DevOnly = @(
    "gen_argcount.py","gen_condstub.py","gen_ctrlvar.py",   # SCM .inc code-gen for the ENGINE build
    "logo_bake.py","theme_bake.py","qrsplash_gen.py",       # not wired into ConvertPipeline; unbundled deps
    "font_ttf_bake.py",                                     # font4 dropped: SA has no disc Pricedown; engine aliases font4->font2
    "ppsspp_red_watch.py","ppsspp_vram_shot.py","shot2png.py",  # emulator dev tooling
    "sonic_asset_dump.py","psp_dashboard_server.py","pt_bin2chrome.py"  # mod dev / telemetry / profiler
)
foreach ($f in $DevOnly) {
    $fp = Join-Path $Bakers $f
    if (Test-Path $fp) { Remove-Item -Force $fp }
}

# gvcslib: .py-only (recursive) so gvcslib.work.sa_export_pmap comes along WITHOUT the
# 5.1 GB of asset data under work/. Plus the small data/ lookup dir. Exclude the web
# viewer up front.
Say "vendor gvcslib (.py only, excl web/ + work assets)"
robocopy $GvcsInner (Join-Path $Bakers "gvcslib") *.py /S /NJH /NJS /NFL /NDL /XD __pycache__ web | Out-Null
if (Test-Path (Join-Path $GvcsInner "data")) {
    robocopy (Join-Path $GvcsInner "data") (Join-Path $Bakers "gvcslib\data") /E /NJH /NJS /NFL /NDL | Out-Null
}
# Trim gvcslib to the converter's format-parser closure. Drop (1) all of work/ except
# sa_export_pmap (the only work module the world bake imports) and (2) the VCS-modding
# half of the library (CLI/project/SA->VCS injection/zone tooling) - a separate project
# not used by, and not shipped with, the GTA-SA converter. Verified: the 109 shipped
# bakers all import cleanly after this trim.
$GvWork = Join-Path $Bakers "gvcslib\work"
if (Test-Path $GvWork) {
    Get-ChildItem $GvWork -Filter *.py |
        Where-Object { $_.Name -notin @("sa_export_pmap.py", "__init__.py") } |
        Remove-Item -Force
}
$GvDrop = @(
    "cli.py","project.py","sa_inject.py","sa_map_inject.py","sa_to_vcs.py","sa_gltf.py",
    "dtz_instances.py","dtz_layout.py","mocap.py","model_map.py","sa_zones.py"
)
foreach ($f in $GvDrop) {
    $fp = Join-Path $Bakers (Join-Path "gvcslib" $f)
    if (Test-Path $fp) { Remove-Item -Force $fp }
}

Say "vendor SAW formats/ + core/"
robocopy (Join-Path $Saw "formats") (Join-Path $Bakers "formats") /E /NJH /NJS /NFL /NDL /XD __pycache__ | Out-Null
robocopy (Join-Path $Saw "core")    (Join-Path $Bakers "core")    /E /NJH /NJS /NFL /NDL /XD __pycache__ | Out-Null

# robocopy exit codes 0-7 are success; normalize so the script does not abort.
if ($LASTEXITCODE -lt 8) { $global:LASTEXITCODE = 0 }

# ---------------------------------------------------------------- 5. smoke + zip
Say "smoke: embed resolves numpy/PIL/lz4 + vendored gvcslib/formats/core"
& $PyExe -c "import numpy,PIL,lz4,gvcslib; from formats import dff; from core import imgarchive; print('bundle imports OK')"
if ($LASTEXITCODE -ne 0) { throw "bundle smoke import failed" }

# the smoke import wrote .pyc caches into bakers/, strip them so the zip ships
# only source (no __pycache__).
Get-ChildItem $Out -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

$Zip = Join-Path $Dist "Quarry-v$Ver.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Say "zip -> $Zip"
Compress-Archive -Path $Out -DestinationPath $Zip

$SizeMb = [math]::Round((Get-ChildItem $Out -Recurse | Measure-Object Length -Sum).Sum / 1MB, 1)
Say "DONE. folder=$SizeMb MB  zip=$Zip"
