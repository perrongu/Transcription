@echo off
setlocal

set "ROOT=%~dp0"
set "VENV_PY=%ROOT%tools\venv\Scripts\python.exe"
set "EMBED_PY=%ROOT%tools\python\python.exe"

if exist "%VENV_PY%" (
  set "PYTHON=%VENV_PY%"
) else if exist "%EMBED_PY%" (
  set "PYTHON=%EMBED_PY%"
) else (
  set "PYTHON=python"
)

set "FFMPEG_BIN=%ROOT%tools\ffmpeg\bin"
if exist "%FFMPEG_BIN%\ffmpeg.exe" (
  set "PATH=%FFMPEG_BIN%;%PATH%"
)

echo == Interface web locale ==
echo Disponible sur http://localhost:8765
echo.
"%PYTHON%" -m uvicorn web.app:app --host 127.0.0.1 --port 8765
echo.
echo Serveur arrêté. Fermez la fenêtre ou appuyez sur une touche.
pause
endlocal
