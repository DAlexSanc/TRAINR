@echo off
:: launch_setup.bat
:: Called by Inno Setup at the end of installation.
:: Opens a visible PowerShell console to run setup_env.ps1
:: The /WAIT flag means the installer waits for this to finish before closing.

powershell.exe -NoProfile -ExecutionPolicy Bypass ^
    -File "%~dp0setup_env.ps1" ^
    -InstallRoot "%~dp0"
