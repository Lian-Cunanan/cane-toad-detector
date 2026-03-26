param(
    [ValidateSet('debug', 'release')]
    [string]$BuildType = 'debug',

    [switch]$Install
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "[INFO] Mobile app directory: $scriptDir"

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Error "WSL is not installed. Install WSL first, then rerun this script."
}

$wslAppDir = (wsl wslpath -a "$scriptDir").Trim()
if (-not $wslAppDir) {
    Write-Error "Failed to convert Windows path to WSL path."
}

$wslBuildDir = "/tmp/cane_toad_mobile_build"

$buildCmd = 'set -e; export PIP_BREAK_SYSTEM_PACKAGES=1; rm -rf ''{1}''; mkdir -p ''{1}''; cp -a ''{0}''/. ''{1}''/; cd ''{1}''; yes | buildozer android {2}; mkdir -p ''{0}''/bin; cp -f ''{1}''/bin/*.apk ''{0}''/bin/' -f $wslAppDir, $wslBuildDir, $BuildType

Write-Host "[INFO] Running Buildozer in WSL temp dir: $wslBuildDir"

wsl --cd / bash -lc "$buildCmd"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Buildozer build failed. Review the WSL output above."
}

$apkDir = Join-Path $scriptDir 'bin'
if (-not (Test-Path $apkDir)) {
    Write-Warning "Build completed but APK folder not found at: $apkDir"
    exit 0
}

$apk = Get-ChildItem -Path $apkDir -Filter *.apk | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $apk) {
    Write-Warning "Build completed but no APK was found in: $apkDir"
    exit 0
}

Write-Host "[SUCCESS] APK created: $($apk.FullName)"

if ($Install) {
    if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
        Write-Warning "adb not found in PATH. Install Android platform-tools to use -Install."
        exit 0
    }

    Write-Host "[INFO] Installing APK to connected Android device..."
    & adb install -r "$($apk.FullName)"

    if ($LASTEXITCODE -ne 0) {
        Write-Error "APK install failed. Make sure USB debugging is enabled and a device is connected."
    }

    Write-Host "[SUCCESS] APK installed on device."
}
