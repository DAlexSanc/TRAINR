# setup_env.ps1  —  TRAINR environment bootstrapper
# Run by the installer after files are copied.
# Safe to re-run: skips steps already completed.

param (
    [string]$InstallRoot = ""
)

# ── Admin elevation ───────────────────────────────────────────────────────────
$IsAdmin = ([Security.Principal.WindowsPrincipal]
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $IsAdmin) {
    Start-Process powershell `
        -Verb RunAs `
        -ArgumentList "-NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`" -InstallRoot `"$InstallRoot`""
    exit
}

# ── Resolve install root ──────────────────────────────────────────────────────
if (-not $InstallRoot -or $InstallRoot.Trim() -eq "") {
    # When launched by the installer it passes the path via -InstallRoot.
    # When run manually, derive it from the script's own location.
    $InstallRoot = Split-Path -Parent $PSCommandPath
}

$ROOT = $InstallRoot.TrimEnd('\')
$VENV = "$ROOT\venv"
$APP  = "$ROOT\app"
$PY   = "$VENV\Scripts\python.exe"

Clear-Host
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  TRAINR  —  Environment Setup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Install root : $ROOT"
Write-Host "Virtual env  : $VENV"
Write-Host ""

$ErrorActionPreference = "Stop"

# ── Python check ──────────────────────────────────────────────────────────────
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "ERROR: Python 3.10+ not found in PATH." -ForegroundColor Red
    Write-Host "Download from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "Make sure 'Add Python to PATH' is checked during install."
    Read-Host "`nPress ENTER to exit"
    exit 1
}
$pyVer = python --version 2>&1
Write-Host "Found: $pyVer" -ForegroundColor Green

# ── GPU check ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/6] Detecting GPU..." -ForegroundColor Yellow
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "WARNING: nvidia-smi not found. CPU-only mode." -ForegroundColor Yellow
    $torchIndex = "cpu"
} else {
    $cudaLine = nvidia-smi | Select-String "CUDA Version"
    if (-not $cudaLine) {
        Write-Host "WARNING: Cannot read CUDA version. Defaulting to cu121." -ForegroundColor Yellow
        $torchIndex = "cu121"
    } else {
        $driverCuda = ($cudaLine -split "CUDA Version:")[1].Trim()
        Write-Host "NVIDIA driver supports CUDA up to: $driverCuda" -ForegroundColor Green

        if     ($driverCuda -match "^1[3-9]\.")          { $torchIndex = "cu128" }
        elseif ($driverCuda -match "^12\.(8|7|6|5|4|3|2)") { $torchIndex = "cu128" }
        elseif ($driverCuda -match "^12\.(1|0)")          { $torchIndex = "cu121" }
        else                                              { $torchIndex = "cu121" }

        Write-Host "Selected PyTorch wheel: $torchIndex" -ForegroundColor Green
    }
}

# ── Create venv ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/6] Setting up virtual environment..." -ForegroundColor Yellow
if (Test-Path "$VENV\Scripts\python.exe") {
    Write-Host "Virtual environment already exists — skipping creation." -ForegroundColor Green
} else {
    python -m venv $VENV
    Write-Host "Virtual environment created." -ForegroundColor Green
}

# ── Upgrade pip ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[4/6] Upgrading pip..." -ForegroundColor Yellow
& $PY -m pip install --upgrade pip setuptools wheel --quiet

# ── PyTorch ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[5/6] Installing PyTorch (this may take several minutes)..." -ForegroundColor Yellow

if ($torchIndex -eq "cpu") {
    & $PY -m pip install torch torchvision --quiet
} else {
    & $PY -m pip install torch torchvision `
        --index-url "https://download.pytorch.org/whl/$torchIndex"
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyTorch installation failed." -ForegroundColor Red
    Read-Host "Press ENTER to exit"
    exit 1
}
Write-Host "PyTorch installed." -ForegroundColor Green

# ── App dependencies ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[6/6] Installing application dependencies..." -ForegroundColor Yellow

$deps = @(
    "ultralytics",
    "labelme",
    "pyside6",
    "pillow",
    "matplotlib",
    "albumentations",
    "opencv-python",
    "onnx",
    "onnxruntime-gpu",
    "pyyaml"
)

foreach ($dep in $deps) {
    Write-Host "  Installing $dep..." -NoNewline
    & $PY -m pip install $dep --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "WARNING: $dep could not be installed. Some features may not work." -ForegroundColor Yellow
    }
}

# ── Verification ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Verifying installation..." -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

& $PY -c @"
import sys
results = []

checks = [
    ('torch',        lambda: __import__('torch').__version__),
    ('CUDA',         lambda: str(__import__('torch').cuda.is_available())),
    ('PySide6',      lambda: __import__('PySide6').__version__),
    ('ultralytics',  lambda: __import__('ultralytics').__version__),
    ('matplotlib',   lambda: __import__('matplotlib').__version__),
    ('albumentations', lambda: __import__('albumentations').__version__),
    ('cv2',          lambda: __import__('cv2').__version__),
    ('PIL',          lambda: __import__('PIL').__version__),
]

for name, fn in checks:
    try:
        val = fn()
        print(f'  {name:<20} {val}')
    except Exception as e:
        print(f'  {name:<20} MISSING  ({e})')
"@

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  TRAINR is ready to use!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now launch TRAINR from your Desktop shortcut."
Write-Host ""
Read-Host "Press ENTER to close this window"
