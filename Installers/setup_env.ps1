# setup_env.ps1  -  TRAINR environment bootstrapper
# Uses $PSScriptRoot so install path never needs to be passed as an argument.
# Safe to re-run: skips steps already completed.

$ROOT = $PSScriptRoot
$VENV = Join-Path $ROOT "venv"
$PY   = Join-Path $VENV "Scripts\python.exe"

# Admin elevation - single-line form, no multi-line cast
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal   = New-Object Security.Principal.WindowsPrincipal($currentUser)
$IsAdmin     = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $IsAdmin) {
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

Clear-Host
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  TRAINR  -  Environment Setup"              -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Install root : $ROOT"
Write-Host "Virtual env  : $VENV"
Write-Host ""


# Step 1: Python
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python 3.10+ not found in PATH." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "Tick 'Add Python to PATH' during install." -ForegroundColor Red
    Read-Host "`nPress ENTER to exit"
    exit 1
}
$pyVer = python --version 2>&1
Write-Host "Found: $pyVer" -ForegroundColor Green

# Step 2: GPU / CUDA detection
Write-Host ""
Write-Host "[2/6] Detecting GPU..." -ForegroundColor Yellow
$torchIndex = "cpu"

# Search for nvidia-smi in all known locations including SysNative
# (SysNative = real 64-bit System32 as seen from a 32-bit process)
$smiCandidates = @(
    "C:\Windows\SysNative\nvidia-smi.exe",
    "C:\Windows\System32\nvidia-smi.exe",
    "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
)

$nvidiaSmi = $null
$found = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($found) {
    $nvidiaSmi = $found.Source
} else {
    foreach ($c in $smiCandidates) {
        if (Test-Path $c) { $nvidiaSmi = $c; break }
    }
}

if ($nvidiaSmi) {
    $smiOut   = & $nvidiaSmi 2>&1
    $cudaLine = $smiOut | Select-String "CUDA Version"
    if ($cudaLine) {
        $driverCuda = (($cudaLine -split "CUDA Version:")[1].Trim() -split "\s+")[0]
        Write-Host "NVIDIA driver - max CUDA: $driverCuda" -ForegroundColor Green
        if ($driverCuda -match "^(\d+)\.(\d+)") {
            $key = [int]$Matches[1] * 10 + [int]$Matches[2]
            if ($key -ge 132) { $torchIndex = "cu132" }
            elseif ($key -ge 131) { $torchIndex = "cu130" }
            elseif ($key -ge 130) { $torchIndex = "cu130" }
            elseif ($key -ge 128) { $torchIndex = "cu128" }
            elseif ($key -ge 126) { $torchIndex = "cu126" }
            elseif ($key -ge 124) { $torchIndex = "cu124" }
            elseif ($key -ge 121) { $torchIndex = "cu121" }
            elseif ($key -ge 118) { $torchIndex = "cu118" }
            else                  { $torchIndex = "cu118" }
        }
    } else {
        Write-Host "nvidia-smi found but CUDA version unreadable." -ForegroundColor Yellow
    }
} else {
    Write-Host "nvidia-smi not found - CPU-only PyTorch." -ForegroundColor Yellow
    Write-Host "If you have an NVIDIA GPU, ensure drivers are installed." -ForegroundColor Yellow
}
Write-Host "PyTorch wheel : $torchIndex" -ForegroundColor Green

# Step 3: Virtual environment
Write-Host ""
Write-Host "[3/6] Setting up virtual environment..." -ForegroundColor Yellow
if (Test-Path $PY) {
    Write-Host "Already exists - skipping." -ForegroundColor Green
} else {
    Write-Host "Creating venv at: $VENV"
    python -m venv $VENV
    if (-not (Test-Path $PY)) {
        Write-Host "ERROR: venv creation failed. Check your Python installation." -ForegroundColor Red
        Read-Host "Press ENTER to exit"
        exit 1
    }
    Write-Host "Created." -ForegroundColor Green
}

# Step 4: Upgrade pip
Write-Host ""
Write-Host "[4/6] Upgrading pip..." -ForegroundColor Yellow
& $PY -m pip install --upgrade pip setuptools wheel --quiet
Write-Host "Done." -ForegroundColor Green

# Step 5: PyTorch
Write-Host ""
Write-Host "[5/6] Installing PyTorch (may take several minutes)..." -ForegroundColor Yellow
$null = & $PY -m pip show torch 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Already installed." -ForegroundColor Green
} else {
    if ($torchIndex -eq "cpu") {
        & $PY -m pip install torch torchvision
    } else {
        & $PY -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$torchIndex"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: PyTorch install failed. Check internet and retry." -ForegroundColor Red
        Read-Host "Press ENTER to exit"
        exit 1
    }
    Write-Host "Installed." -ForegroundColor Green
}

# Step 6: Application dependencies
Write-Host ""
Write-Host "[6/6] Installing application dependencies..." -ForegroundColor Yellow
Write-Host ""

$failed = @()

$depNames    = @("PySide6",   "Ultralytics/YOLO", "LabelMe",  "Pillow",  "PyYAML",  "Matplotlib",  "OpenCV",         "NumPy",  "Albumentations", "ONNX",  "ONNX Runtime GPU")
$depPkgs     = @("pyside6",   "ultralytics",       "labelme",  "pillow",  "pyyaml",  "matplotlib",  "opencv-python",  "numpy",  "albumentations", "onnx",  "onnxruntime-gpu")
$depRequired = @($true,       $true,               $true,      $true,     $true,     $true,         $true,            $true,    $false,           $false,  $false)

for ($i = 0; $i -lt $depNames.Length; $i++) {
    $name     = $depNames[$i]
    $pkg      = $depPkgs[$i]
    $required = $depRequired[$i]

    Write-Host "  $name..." -NoNewline
    & $PY -m pip install $pkg --quiet 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } elseif ($required) {
        Write-Host " FAILED" -ForegroundColor Red
        $failed += $name
    } else {
        Write-Host " FAILED (optional)" -ForegroundColor Yellow
    }
}

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Required packages failed to install:" -ForegroundColor Red
    foreach ($f in $failed) { Write-Host "  - $f" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Run setup_env.ps1 again to retry." -ForegroundColor Yellow
    Read-Host "`nPress ENTER to exit"
    exit 1
}

# Verification
Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Verifying..."                               -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

$verifyPkgs = @("torch", "pyside6", "ultralytics", "labelme", "matplotlib", "opencv-python", "numpy")
foreach ($pkg in $verifyPkgs) {
    $null = & $PY -m pip show $pkg 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  $pkg" -NoNewline
        Write-Host "  OK" -ForegroundColor Green
    } else {
        Write-Host "  $pkg" -NoNewline
        Write-Host "  MISSING" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  TRAINR is ready to use!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Use the Desktop shortcut to launch TRAINR."
Write-Host "To repair: right-click setup_env.ps1 and Run with PowerShell."
Write-Host ""
Read-Host "Press ENTER to close"
