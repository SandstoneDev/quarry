<#
    build.ps1 - builds the Quarry release bundle from this repository.

    Produces dist/Quarry/ (unzip and run: no Python, pip or libraries to install)
    and zips it to dist/Quarry-v<ver>.zip.

    Needs the .NET 8 SDK. The first run also needs internet access: the embedded
    Python and three wheels are downloaded once into .cache/ and reused after that.
    Everything else comes from this repository.

    Layout produced:
      dist/Quarry/
        Quarry.exe      framework-dependent .NET 8 (win-x64)
        python/         embedded CPython with numpy, Pillow and lz4
        bakers/         the Python converters from bakers/
        content/        inputs that ship with the tool instead of coming from the
                        disc: the script listing, the string table, and the door
                        marker mesh (generated here by bakers/marker_gen.py)

    Usage:
      powershell -ExecutionPolicy Bypass -File build.ps1 [-Ver 1009] [-Clean]
      pwsh -File build.ps1 [-Ver 1009] [-Clean]
#>
[CmdletBinding()]
param(
    [string]$Ver = "dev",
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# ---- pinned versions (the ones the bakers are validated against) ----
$PyVer   = "3.14.3"
$NumPy   = "2.4.4"
$Pillow  = "12.2.0"
$Lz4     = "4.4.5"

# ---- paths: everything is relative to this script ----
$Repo       = $PSScriptRoot
$Csproj     = Join-Path $Repo "src\Quarry.csproj"
$VersionSrc = Join-Path $Repo "src\GameVersion.cs"
$BakersSrc  = Join-Path $Repo "bakers"
$ContentSrc = Join-Path $Repo "content"
$Dist       = Join-Path $Repo "dist"
$Out        = Join-Path $Dist "Quarry"
$Cache      = Join-Path $Repo ".cache"

# Windows PowerShell 5.1 may still default to TLS 1.0/1.1, which python.org refuses.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Say($m) { Write-Host "[build] $m" -ForegroundColor Cyan }

# robocopy reports success with exit codes 0-7. Anything from 8 up is a real failure.
function Invoke-Robocopy([string[]]$RoboArgs) {
    & robocopy @RoboArgs | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit $LASTEXITCODE): $($RoboArgs -join ' ')" }
    $global:LASTEXITCODE = 0
}

foreach ($p in @($Csproj, $VersionSrc, $BakersSrc, $ContentSrc)) {
    if (-not (Test-Path $p)) { throw "missing from the repository: $p" }
}

if ($Clean -and (Test-Path $Out)) { Say "clean $Out"; Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force -Path $Out, $Cache | Out-Null

# ---------------------------------------------------------------- 1. .NET publish
# Refuse to build while a Quarry from the output folder is running: the build
# replaces python/ underneath it, which breaks a convert that is in progress.
$running = Get-Process -Name Quarry -ErrorAction SilentlyContinue |
           Where-Object { $_.Path -and $_.Path.StartsWith($Dist, [StringComparison]::OrdinalIgnoreCase) }
if ($running) {
    throw "Quarry is running from $Dist (pid $($running.Id -join ', ')). Close it first."
}

# The zip name comes from -Ver and the number the app shows comes from the source
# constant. Refuse to build when they disagree.
if ($Ver -ne "dev") {
    $src = Get-Content $VersionSrc -Raw
    if ($src -match 'Version = "v(\d+)"') {
        if ($Matches[1] -ne $Ver) {
            throw "version mismatch: -Ver $Ver but src\GameVersion.cs says v$($Matches[1]). Update the constant first."
        }
        Say "version check: source and -Ver agree on v$Ver"
    } else { throw "could not read the version constant from src\GameVersion.cs" }
}

Say "dotnet publish (framework-dependent win-x64)"
# DebugType=none keeps the build machine's paths out of the published binary.
$exePath = Join-Path $Out "Quarry.exe"
$before  = if (Test-Path $exePath) { (Get-Item $exePath).LastWriteTimeUtc } else { [datetime]::MinValue }
dotnet publish $Csproj -c Release -r win-x64 --self-contained false -p:DebugType=none -p:DebugSymbols=false -o $Out | Out-Null
if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed (exit $LASTEXITCODE) - run it without -o redirection to see why" }
if (-not (Test-Path $exePath)) { throw "publish did not produce Quarry.exe" }
if ((Get-Item $exePath).LastWriteTimeUtc -le $before) { throw "Quarry.exe was not rewritten - the publish kept an old binary" }
Get-ChildItem $Out -Filter *.pdb -ErrorAction SilentlyContinue | Remove-Item -Force

# ---------------------------------------------------------------- 2. embedded CPython
$PyDir = Join-Path $Out "python"
$PyZip = Join-Path $Cache "python-$PyVer-embed-amd64.zip"
if (-not (Test-Path $PyZip)) {
    Say "download python-$PyVer-embed-amd64.zip"
    Invoke-WebRequest -UseBasicParsing -Uri "https://www.python.org/ftp/python/$PyVer/python-$PyVer-embed-amd64.zip" -OutFile $PyZip
}
Say "extract embedded Python -> python/"
if (Test-Path $PyDir) { Remove-Item -Recurse -Force $PyDir }
Expand-Archive -Path $PyZip -DestinationPath $PyDir -Force

# The ._pth file alone defines sys.path, with no `import site`. That isolates the
# bundle from any Python installed on the user's machine and makes PYTHONPATH irrelevant.
$PthMajorMinor = ($PyVer -split '\.')[0..1] -join ''
$Pth = Join-Path $PyDir "python$PthMajorMinor._pth"
@"
python$PthMajorMinor.zip
.
Lib\site-packages
..\bakers
..\bakers\gvcslib
"@ | Set-Content -Path $Pth -Encoding ascii
Say "wrote $(Split-Path $Pth -Leaf)"

# ---------------------------------------------------------------- 3. pip + wheels
$PyExe  = Join-Path $PyDir "python.exe"
$GetPip = Join-Path $Cache "get-pip.py"
if (-not (Test-Path $GetPip)) {
    Say "download get-pip.py"
    Invoke-WebRequest -UseBasicParsing -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPip
}
Say "bootstrap pip"
& $PyExe $GetPip --no-warn-script-location | Out-Null
if ($LASTEXITCODE -ne 0) { throw "get-pip failed (exit $LASTEXITCODE)" }
$SitePk = Join-Path $PyDir "Lib\site-packages"
Say "pip install numpy==$NumPy Pillow==$Pillow lz4==$Lz4"
& $PyExe -m pip install --no-warn-script-location --ignore-installed --target $SitePk `
    "numpy==$NumPy" "Pillow==$Pillow" "lz4==$Lz4" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }

# The runtime only needs the three wheels: drop the build-time packaging tools and
# the test suites the wheels carry.
Get-ChildItem $SitePk -Directory | Where-Object { $_.Name -match '^(pip|setuptools|pkg_resources|_distutils_hack)($|-)' } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $SitePk -Recurse -Directory -Filter tests -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- 4. bakers + content
$Bakers = Join-Path $Out "bakers"
if (Test-Path $Bakers) { Remove-Item -Recurse -Force $Bakers }
Say "copy bakers/"
Invoke-Robocopy @($BakersSrc, $Bakers, "/E", "/NJH", "/NJS", "/NFL", "/NDL", "/XD", "__pycache__")

# Unit tests and development utilities live in bakers/ next to the converters but
# are never run by a convert. None of them are imported by anything that is.
$DevOnly = @(
    "test_cull_air_bake.py", "test_ps2_uv_tess.py", "test_scm_asm.py",
    "logo_bake.py", "theme_bake.py", "qrsplash_gen.py",
    "mipsref.py", "uv_span_stat.py"
)
foreach ($f in $DevOnly) {
    $fp = Join-Path $Bakers $f
    if (Test-Path $fp) { Remove-Item -Force $fp }
}

$Content = Join-Path $Out "content"
if (Test-Path $Content) { Remove-Item -Recurse -Force $Content }
Say "copy content/"
Invoke-Robocopy @($ContentSrc, $Content, "/E", "/NJH", "/NJS", "/NFL", "/NDL")

# The door marker is a plain generated cone, so it is built here from its source
# rather than kept in the repository as a binary.
$Marker = Join-Path $Content "effects\marker.bin"
New-Item -ItemType Directory -Force -Path (Split-Path $Marker) | Out-Null
Say "generate content/effects/marker.bin"
& $PyExe (Join-Path $Bakers "marker_gen.py") $Marker | Out-Null
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Marker) -or (Get-Item $Marker).Length -eq 0) {
    throw "marker_gen.py did not produce $Marker"
}

# ---------------------------------------------------------------- 5. smoke + zip
Say "smoke: the embedded Python resolves numpy/PIL/lz4 and the bundled libraries"
& $PyExe -c "import numpy,PIL,lz4,gvcslib; from formats import dff; from core import imgarchive; print('bundle imports OK')"
if ($LASTEXITCODE -ne 0) { throw "bundle smoke import failed" }

# The smoke import leaves .pyc caches in bakers/. Strip them so only source ships.
Get-ChildItem $Out -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

$Zip = Join-Path $Dist "Quarry-v$Ver.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Say "zip -> $Zip"
Compress-Archive -Path $Out -DestinationPath $Zip

$SizeMb = [math]::Round((Get-ChildItem $Out -Recurse | Measure-Object Length -Sum).Sum / 1MB, 1)
Say "DONE. folder=$SizeMb MB  zip=$Zip"
