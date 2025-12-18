@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

%POWERSHELL% -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install.ps1"

echo.
echo Installation terminee. Fermez cette fenetre ou appuyez sur une touche.
pause
