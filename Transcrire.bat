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

pushd "%ROOT%"
if "%~1"=="" (
  "%PYTHON%" "scripts\transcribe.py"
) else (
  "%PYTHON%" "scripts\transcribe.py" --input "%~1"
)
popd

echo.
echo Transcription terminee. Fermez la fenetre ou appuyez sur une touche.
pause
endlocal
