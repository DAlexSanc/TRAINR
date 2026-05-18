#define AppName    "TRAINR"
#define AppVersion "1.1.0"
#define AppExe     "main.exe"
#define AppIcon    "SWAI.ico"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
PrivilegesRequired=admin
WizardStyle=modern
DisableDirPage=no

SetupIconFile={#AppIcon}
UninstallDisplayIcon={app}\{#AppIcon}

Compression=lzma2
SolidCompression=yes

; Output file name
OutputBaseFilename=TRAINR_Setup

[Files]
; ── App executable ────────────────────────────────────────────────────────────
Source: "App\main.exe";         DestDir: "{app}\app";          Flags: ignoreversion

; ── Scripts — new folder structure ───────────────────────────────────────────
; Root-level scripts (theme.py, paths.py, HailoDetectionYolo.py, extractor.py)
Source: "Scripts\theme.py";              DestDir: "{app}\app\Scripts";          Flags: ignoreversion
Source: "Scripts\paths.py";              DestDir: "{app}\app\Scripts";          Flags: ignoreversion
Source: "Scripts\HailoDetectionYolo.py"; DestDir: "{app}\app\Scripts";          Flags: ignoreversion
Source: "Scripts\extractor.py";          DestDir: "{app}\app\Scripts";          Flags: ignoreversion

; core/
Source: "Scripts\core\*"; DestDir: "{app}\app\Scripts\core"; Flags: recursesubdirs createallsubdirs ignoreversion

; ui/
Source: "Scripts\ui\*"; DestDir: "{app}\app\Scripts\ui"; Flags: recursesubdirs createallsubdirs ignoreversion

; ── Models ────────────────────────────────────────────────────────────────────
Source: "App\Models\*"; DestDir: "{app}\app\Models"; Flags: recursesubdirs createallsubdirs ignoreversion

; ── Icon ──────────────────────────────────────────────────────────────────────
Source: "{#AppIcon}"; DestDir: "{app}"; Flags: ignoreversion

; ── Environment bootstrapper ──────────────────────────────────────────────────
; Copied to {app} so setup_env.ps1 can find paths.py via Split-Path
Source: "Installers\setup_env.ps1";  DestDir: "{app}"; Flags: ignoreversion
Source: "Installers\launch_setup.bat"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Desktop shortcut
Name: "{commondesktop}\{#AppName}"; \
  Filename: "{app}\app\{#AppExe}"; \
  IconFilename: "{app}\{#AppIcon}"

; Start Menu shortcut
Name: "{commonprograms}\{#AppName}\{#AppName}"; \
  Filename: "{app}\app\{#AppExe}"; \
  IconFilename: "{app}\{#AppIcon}"

[Run]
; ── Phase 1: Launch the environment setup in a visible terminal ───────────────
; shellexec + waituntilterminated = opens a real console the user can watch,
; installer waits for it to complete before showing the "Finish" button.
Filename: "{app}\launch_setup.bat"; \
  Description: "Set up Python environment (required — takes a few minutes)"; \
  Flags: shellexec waituntilterminated; \
  StatusMsg: "Setting up Python environment...";

; ── Phase 2: Optionally launch the app ───────────────────────────────────────
Filename: "{app}\app\{#AppExe}"; \
  Description: "Launch {#AppName}"; \
  Flags: nowait postinstall skipifsilent unchecked

[UninstallDelete]
; Clean up the venv on uninstall so it doesn't leave gigabytes behind
Type: filesandordirs; Name: "{app}\venv"
