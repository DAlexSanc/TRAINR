#define AppName "TRAINR"
#define AppVersion "1.0.0"
#define AppExe "interface.exe"
#define AppIcon "SWAI.ico"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
PrivilegesRequired=admin
WizardStyle=modern
DisableDirPage=no

; Installer branding
SetupIconFile={#AppIcon}
UninstallDisplayIcon={app}\{#AppIcon}

Compression=lzma2
SolidCompression=yes

[Files]
Source: "App\interface.exe"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "Scripts\*"; DestDir: "{app}\app\Scripts"; Flags: recursesubdirs createallsubdirs
Source: "Models\*"; DestDir: "{app}\app\Models"; Flags: recursesubdirs createallsubdirs
Source: "{#AppIcon}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Desktop shortcut (required)
Name: "{commondesktop}\{#AppName}"; \
Filename: "{app}\app\{#AppExe}"; \
IconFilename: "{app}\{#AppIcon}"

; Start Menu shortcut (optional but good UX)
Name: "{commonprograms}\{#AppName}\{#AppName}"; \
Filename: "{app}\app\{#AppExe}"; \
IconFilename: "{app}\{#AppIcon}"

[Run]
Filename: "{app}\app\{#AppExe}"; \
Description: "Launch {#AppName}"; \
Flags: nowait postinstall skipifsilent



