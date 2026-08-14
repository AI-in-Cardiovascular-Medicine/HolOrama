; Inno Setup script for HolOrama.
;
; Packages the Nuitka standalone build (build\nuitka\main.dist) into a single
; setup executable that non-technical users can run: it asks where to install,
; copies the whole app (exe + media + config.yaml), creates Start-Menu and
; optional Desktop shortcuts, and registers an uninstaller.
;
; Per-user install by default (PrivilegesRequired=lowest): no admin rights / UAC
; prompt required, and {app} lands in %LOCALAPPDATA%\Programs where the app can
; freely write its logs\ and models\ output folders.
;
; Build with build_installer.ps1 (passes MyAppVersion in from src\version.py).

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

#define MyAppName "HolOrama"
#define MyAppPublisher "HolOrama"
#define MyAppExeName "HolOrama.exe"
; Path to the Nuitka dist, relative to this .iss file (installer\ -> repo root).
#define DistDir "..\build\nuitka\main.dist"

[Setup]
; Stable AppId — keep this constant across versions so upgrades replace in place.
AppId={{A1E9C4F2-7B3D-4E6A-9C21-8F0B5D2A7E14}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
; Per-user, no admin rights needed.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes
OutputDir=..\build\installer
OutputBaseFilename=HolOrama-Setup-{#MyAppVersion}
SetupIconFile=..\media\desktop_img.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
; The app is 64-bit only.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Recursively include the entire Nuitka dist folder.
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
