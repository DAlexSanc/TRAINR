param (
    [string]$InstallRoot = $env:TRAINR_ROOT,
    [string]$VenvName = "venv",
    [string]$AppFolderName = "app"
)
# ------------------------------
# Ensure script is run as Admin
# ------------------------------
$IsAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $IsAdmin) {
    Write-Host "Re-launching as administrator..."
    Start-Process powershell `
        -Verb RunAs `
        -ArgumentList "-NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

# ------------------------------
# Get install paths
# ------------------------------
# ------------------------------
# Get install paths
# ------------------------------
if (-not $InstallRoot -or $InstallRoot.Trim() -eq "") {
    Write-Host ""
    Write-Host "Enter installation directory (default: C:\TRAINR)"
    $inputRoot = Read-Host "Install path"

    if ($inputRoot.Trim()) {
        $InstallRoot = $inputRoot
    } else {
        $InstallRoot = "C:\TRAINR"
    }
}

# ------------------------------
# Resolve install paths
# ------------------------------


$resolved = Resolve-Path -Path $InstallRoot -ErrorAction SilentlyContinue
if ($resolved) {
    $ROOT = $resolved.Path
} else {
    $ROOT = $InstallRoot
}

$VENV = Join-Path $ROOT $VenvName
$APP  = Join-Path $ROOT $AppFolderName

Write-Host "Install root: $ROOT"
Write-Host "Venv path:    $VENV"
Write-Host "App path:     $APP"

New-Item -ItemType Directory -Force -Path $ROOT | Out-Null

# ================================
# TRAINR Environment Preparation
# ================================

$ErrorActionPreference = "Stop"

# ------------------------------
# Logging header
# ------------------------------
Write-Host "================================="
Write-Host " TRAINR ENVIRONMENT PREPARATION"
Write-Host "================================="
Write-Host "Running as admin: $IsAdmin"
Write-Host "PID: $PID"
Write-Host "Script: $PSCommandPath"
Write-Host ""


New-Item -ItemType Directory -Force -Path $ROOT | Out-Null

# ------------------------------
# Python check
# ------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10+ is required and not found in PATH"
}
# ------------------------------
# CUDA / NVIDIA driver detection
# ------------------------------

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "NVIDIA GPU not detected (nvidia-smi missing)"
}

$cudaLine = nvidia-smi | Select-String "CUDA Version"
if (-not $cudaLine) {
    throw "Unable to detect CUDA compatibility from NVIDIA driver"
}

$driverCuda = ($cudaLine -split "CUDA Version:")[1].Trim()
Write-Host "NVIDIA driver supports CUDA up to: $driverCuda"

# ------------------------------
# PyTorch wheel selection
# ------------------------------

# Default to safest modern option
$torchIndex = "cu121"

if ($driverCuda -match "^12\.(8|7|6|5|4|3|2)") {
    $torchIndex = "cu128"
}
elseif ($driverCuda -match "^13\.") {
    $torchIndex = "cu130"
}
else {
    Write-Warning "Unknown CUDA compatibility ($driverCuda), falling back to cu121"
    $torchIndex = "cu121"
}

Write-Host "Selected PyTorch wheel: $torchIndex"

# ------------------------------
# Create venv
# ------------------------------
if (-not (Test-Path $VENV)) {
    Write-Host "Creating virtual environment..."
    python -m venv $VENV
} else {
    Write-Host "Virtual environment already exists"
}

$PY  = "$VENV\Scripts\python.exe"

# ------------------------------
# Upgrade pip
# ------------------------------
Write-Host "Upgrading pip tooling..."
& $PY -m pip install --upgrade pip setuptools wheel

# ------------------------------
# Install PyTorch (official)
# ------------------------------
Write-Host "Installing PyTorch + TorchVision..."
& $PY -m pip install `
    torch torchvision `
    --index-url https://download.pytorch.org/whl/$torchIndex

# ------------------------------
# Install remaining deps
# ------------------------------
Write-Host "Installing application dependencies..."
& $PY -m pip install `
    ultralytics `
    labelme `
    labelme2yolo `
    pyside6 `
    pillow `
    onnx `
    onnxruntime-gpu `
    onnxruntime `
    pyyaml

# ------------------------------
# Verification
# ------------------------------
Write-Host ""
Write-Host "Verifying installation..."
& $PY -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

Write-Host ""
Write-Host "========================================="
Write-Host " TRAINR ENVIRONMENT READY"
Write-Host "========================================="
Read-Host "Press ENTER to close this window"
