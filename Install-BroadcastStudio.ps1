# Full Windows install for Broadcast Studio.
# Creates the venv, installs the CUDA PyTorch wheel, installs app dependencies,
# and adds Desktop and Start Menu shortcuts.
param(
    [string]$InstallDir = (Split-Path -Parent $MyInvocation.MyCommand.Path)
)

$ErrorActionPreference = "Stop"
Set-Location $InstallDir
Write-Host "Broadcast Studio full install"
Write-Host "Folder: $InstallDir"

function Find-Python {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { return "py -3" }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) { return "python" }
    return $null
}

$pythonCmd = Find-Python
if (-not $pythonCmd) {
    Write-Host "Python was not found. Trying winget..."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Install Python 3.10 or newer from python.org, then run Setup.exe again."
    }
    winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
    $pythonCmd = Find-Python
    if (-not $pythonCmd) { throw "Python installed but is not on PATH yet. Open a new terminal and run Setup.exe again." }
}

if (-not (Test-Path "$InstallDir\.venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    cmd /c "$pythonCmd -m venv .venv"
}
$pyexe = Join-Path $InstallDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pyexe)) { throw "venv was not created." }

Write-Host "Installing CUDA PyTorch. This is the large download."
& $pyexe -m pip install --upgrade pip
& $pyexe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Host "CUDA wheel failed. Installing the default PyTorch build instead."
    & $pyexe -m pip install torch torchvision
}
Write-Host "Installing studio dependencies..."
& $pyexe -m pip install -r (Join-Path $InstallDir "requirements.txt")

$desktop = [Environment]::GetFolderPath("Desktop")
$start = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$launcher = Join-Path $InstallDir "BroadcastStudio.exe"
$ws = New-Object -ComObject WScript.Shell
foreach ($pair in @(
    @{ Path = (Join-Path $desktop "Broadcast Studio.lnk"); Target = $launcher },
    @{ Path = (Join-Path $start "Broadcast Studio.lnk"); Target = $launcher }
)) {
    $sc = $ws.CreateShortcut($pair.Path)
    $sc.TargetPath = $pair.Target
    $sc.WorkingDirectory = $InstallDir
    $sc.Description = "Broadcast Studio"
    $sc.Save()
    Write-Host "Shortcut: $($pair.Path)"
}

Write-Host ""
Write-Host "Installed. Launch Broadcast Studio from the desktop shortcut."
Write-Host "OBS Virtual Camera must be installed once so Zoom or Discord can select the output."
Write-Host "First studio launch downloads the matting weights."
