#define AppName    "TRAINR"
#define AppVersion "1.1.0"
#define AppIcon    "SWAI.ico"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
PrivilegesRequired=admin
WizardStyle=modern
DisableDirPage=no
OutputBaseFilename=TRAINR_Setup

SetupIconFile={#AppIcon}
UninstallDisplayIcon={app}\{#AppIcon}

Compression=lzma2
SolidCompression=yes

[Files]
; ── Python source ─────────────────────────────────────────────────────────────
Source: "Scripts\main.py";               DestDir: "{app}\Scripts"; Flags: ignoreversion
Source: "Scripts\theme.py";              DestDir: "{app}\Scripts"; Flags: ignoreversion
Source: "Scripts\paths.py";              DestDir: "{app}\Scripts"; Flags: ignoreversion
Source: "Scripts\HailoDetectionYolo.py"; DestDir: "{app}\Scripts"; Flags: ignoreversion
Source: "Scripts\extractor.py";          DestDir: "{app}\Scripts"; Flags: ignoreversion
Source: "Scripts\core\*"; DestDir: "{app}\Scripts\core"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "Scripts\ui\*";   DestDir: "{app}\Scripts\ui";   Flags: recursesubdirs createallsubdirs ignoreversion

; ── Models ────────────────────────────────────────────────────────────────────
Source: "App\Models\*"; DestDir: "{app}\Models"; Flags: recursesubdirs createallsubdirs ignoreversion

; ── Config (only if not already present — preserves user settings on upgrade) ─
Source: "config.json"; DestDir: "{app}"; Flags: ignoreversion onlyifdoesntexist

; ── Icon ──────────────────────────────────────────────────────────────────────
Source: "{#AppIcon}"; DestDir: "{app}"; Flags: ignoreversion

; ── Launcher ─────────────────────────────────────────────────────────────────
Source: "Installers\launcher.vbs"; DestDir: "{app}"; Flags: ignoreversion

; ── Environment bootstrapper ─────────────────────────────────────────────────
Source: "Installers\setup_env.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Desktop shortcut
Name: "{commondesktop}\{#AppName}"; \
  Filename: "{app}\launcher.vbs"; \
  IconFilename: "{app}\{#AppIcon}"

; Start Menu
Name: "{commonprograms}\{#AppName}\{#AppName}"; \
  Filename: "{app}\launcher.vbs"; \
  IconFilename: "{app}\{#AppIcon}"

; Repair shortcut — re-runs setup without reinstalling
Name: "{commonprograms}\{#AppName}\Repair Environment"; \
  Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -NoExit -File ""{app}\setup_env.ps1"""; \
  WorkingDir: "{app}"; \
  IconFilename: "{app}\{#AppIcon}"

[Run]
; ── Environment setup ─────────────────────────────────────────────────────────
; Calls powershell.exe directly — no bat intermediary, no path argument.
; setup_env.ps1 uses $PSScriptRoot to find its own location.
; -NoExit keeps the window open so the user can read the output.
; waituntilterminated blocks the installer finish button until setup completes.
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -NoExit -File ""{app}\setup_env.ps1"""; \
  WorkingDir: "{app}"; \
  Flags: shellexec waituntilterminated; \
  StatusMsg: "Setting up Python environment (this takes a few minutes)...";

; ── Optional: launch after install ───────────────────────────────────────────
Filename: "{app}\launcher.vbs"; \
  Description: "Launch {#AppName}"; \
  Flags: shellexec nowait postinstall skipifsilent unchecked

[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"
