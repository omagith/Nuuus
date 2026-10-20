; Inno Setup 6 script for «Нууус»: setup.exe that installs in a couple of
; clicks (Далее -> Установить -> Завершить), per user, without admin rights.
;
; Build with packaging\build_installer.ps1 (it builds app_built\ first and
; passes the version). By hand:
;   ISCC.exe /DAppVer=1.0.0 packaging\installer.iss
;   ISCC.exe /DSourceDir=C:\some\dummy /DOutputDir=C:\tmp packaging\installer.iss
;
; SourceDir = the PyInstaller folder (app_built\: Нууус.exe, _internal\ and
; ollama\ - packaging\Nuuus.spec copies vendor\ollama next to the exe).
;
; Layout after install (app/core/paths.py: data lives next to app\):
;   %LOCALAPPDATA%\Programs\Nuuus\app\Нууус.exe, app\_internal\, app\ollama\
;   %LOCALAPPDATA%\Programs\Nuuus\data\  (settings, recordings, models; made by the app)
; The folder is ASCII on purpose: torch.jit.load (Silero VAD) and
; sentencepiece (GigaAM tokenizers) cannot open files under a Cyrillic path
; on Windows (checked: both fail with "No such file"). Menu entries and
; shortcuts carry the real name. A profile whose own path is not ASCII
; installs to <system drive>\Nuuus instead (creating a folder there needs no
; admin rights).
;
; Split build (the default of build_installer.ps1): a GitHub release asset may
; not exceed 2 GiB, so the CUDA libraries (_internal\torch\lib and Ollama's
; lib\ollama\cuda_*) go into Nuuus-<ver>-libs-<n>.7z archives next to the
; setup. build_installer.ps1 passes /DLibsExcludes (those folders) and
; /DLibsInclude (a generated file with one pair of [Files] entries per
; archive): an archive lying next to the setup is extracted from there
; (offline install), otherwise it is downloaded from the release, checked by
; SHA-256 and extracted. Without these defines this script builds one
; self-contained setup as before.
;
; Comments: only whole lines starting with ";" ("value ; text" would become
; part of the value); "//" inside [Code].

#define AppName "Нууус"
#define AppExe "Нууус.exe"
#define InstallFolder "Nuuus"
#ifndef AppVer
  #define AppVer "1.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\app_built"
#endif
#ifndef OutputDir
  #define OutputDir "..\build\installer"
#endif
#ifndef LibsExcludes
  #define LibsExcludes ""
#endif

[Setup]
; Never change AppId: a new version then installs over the old one.
AppId={{6E7A2C1B-3F0D-4E8A-9B51-4C2D7F1A9E30}
AppName={#AppName}
AppVersion={#AppVer}
AppVerName={#AppName} {#AppVer}
AppPublisher={#AppName}
DefaultDirName={code:GetInstallDir}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Pages: tasks (shortcut) -> ready -> finish.
DisableDirPage=yes
DisableProgramGroupPage=yes
WizardStyle=modern
ShowLanguageDialog=no
; app/main.py holds this mutex while running: setup and uninstall wait for the
; app to close, and the embedded Ollama goes away with it (job object).
AppMutex=Nuuus.SingleInstance
CloseApplications=yes
SetupIconFile=..\app\assets\icon.ico
UninstallDisplayIcon={app}\app\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
OutputDir={#OutputDir}
OutputBaseFilename=Nuuus-Setup-{#AppVer}
#ifdef LibsInclude
; is7zxr.dll: plain .7z with less memory for the 1 GB DLLs inside.
ArchiveExtraction=enhanced/nopassword
#endif

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"

; Files of the previous version must not linger (data\ is never touched here).
[InstallDelete]
Type: filesandordirs; Name: "{app}\app"

[Files]
; Qt's QML modules are never used by this widgets app: 2 243 files and the
; longest paths of the build (MAX_PATH), so they stay out of the installer.
Source: "{#SourceDir}\*"; DestDir: "{app}\app"; Excludes: "\_internal\PyQt6\Qt6\qml{#LibsExcludes}"; Flags: ignoreversion recursesubdirs createallsubdirs
#ifdef LibsInclude
  #include LibsInclude
#endif

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\app\{#AppExe}"; WorkingDir: "{app}\app"; AppUserModelID: "Nuuus.Desktop"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\app\{#AppExe}"; WorkingDir: "{app}\app"; AppUserModelID: "Nuuus.Desktop"; Tasks: desktopicon

[Run]
Filename: "{app}\app\{#AppExe}"; Description: "Запустить {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\app"

[Code]
function IsAsciiPath(const Path: String): Boolean;
var
  I: Integer;
begin
  Result := True;
  for I := 1 to Length(Path) do
    if Ord(Path[I]) > 127 then begin
      Result := False;
      Exit;
    end;
end;

// Split build: a libs archive next to the setup is used instead of downloading it.
function LibsPartIsLocal(const Name: String): Boolean;
begin
  Result := FileExists(ExpandConstant('{src}\') + Name);
end;

function GetInstallDir(Param: String): String;
begin
  Result := ExpandConstant('{localappdata}\Programs\{#InstallFolder}');
  // torch.jit / sentencepiece break under non-ASCII paths (see the header).
  if not IsAsciiPath(Result) then
    Result := ExpandConstant('{sd}\{#InstallFolder}');
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Data: String;
begin
  if CurUninstallStep = usPostUninstall then begin
    Data := ExpandConstant('{app}\data');
    if DirExists(Data) and not UninstallSilent and
       (MsgBox('Удалить также ваши записи, настройки и скачанные модели (папка data)?' + #13#10 +
               'Нажмите «Нет», чтобы сохранить их для повторной установки.',
               mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES) then
      DelTree(Data, True, True, True);
    // Removed only when empty, i.e. when the data folder is gone too.
    RemoveDir(ExpandConstant('{app}'));
  end;
end;
