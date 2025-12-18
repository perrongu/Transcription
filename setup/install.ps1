[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Root = Split-Path -Parent $ScriptDir
$ToolsDir = Join-Path $Root "tools"
$VenvDir = Join-Path $ToolsDir "venv"
$EmbeddedDir = Join-Path $ToolsDir "python"
$FfmpegDir = Join-Path $ToolsDir "ffmpeg"
$ModelDir = Join-Path $Root "models"
$Requirements = Join-Path $Root "requirements.txt"

function Write-Info($msg) { Write-Host $msg -ForegroundColor Cyan }
function Ensure-Dir($path) { if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path | Out-Null } }

function Get-Python {
    param([string]$EmbeddedTarget)
    $embeddedPython = Join-Path $EmbeddedTarget "python.exe"
    if (Test-Path $embeddedPython) { return $embeddedPython }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pyExec = & py -3 -c "import sys; import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return (& py -3 -c "import sys; print(sys.executable)")
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return "python"
        }
    }

    Write-Info "Téléchargement de Python (package embeddable)..."
    Ensure-Dir $EmbeddedTarget
    $pythonUrl = "https://www.python.org/ftp/python/3.12.4/python-3.12.4-embed-amd64.zip"
    $archive = Join-Path $ToolsDir "python-embed.zip"
    Invoke-WebRequest $pythonUrl -OutFile $archive
    Expand-Archive -Path $archive -DestinationPath $EmbeddedTarget -Force

    $pthFile = Join-Path $EmbeddedTarget "python312._pth"
    (Get-Content $pthFile) -replace '#import site', 'import site' | Set-Content $pthFile -Encoding ASCII

    $getPip = Join-Path $EmbeddedTarget "get-pip.py"
    Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip
    & $embeddedPython $getPip
    Remove-Item $archive, $getPip -Force
    return $embeddedPython
}

Write-Info "== Déploiement clé-en-main (Windows) =="
Ensure-Dir $ToolsDir
Ensure-Dir $ModelDir

Write-Info "[1/5] Recherche/installation Python..."
$pythonCandidate = Get-Python -EmbeddedTarget $EmbeddedDir
if (-not $pythonCandidate) { throw "Impossible de récupérer Python 3.10+." }

Write-Info "[2/5] Création du venv local (tools/venv)..."
if (-not (Test-Path (Join-Path $VenvDir "Scripts/python.exe"))) {
    & $pythonCandidate -m venv $VenvDir
}
$PythonExe = if (Test-Path (Join-Path $VenvDir "Scripts/python.exe")) { Join-Path $VenvDir "Scripts/python.exe" } else { $pythonCandidate }

Write-Info "[3/5] Installation des dépendances Python..."
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r $Requirements

Write-Info "[4/5] ffmpeg portable..."
Ensure-Dir (Join-Path $FfmpegDir "bin")
if (-not (Test-Path (Join-Path $FfmpegDir "bin/ffmpeg.exe"))) {
    $ffmpegUrl = if ($env:FFMPEG_URL) { $env:FFMPEG_URL } else { "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" }
    $ffArchive = Join-Path $ToolsDir "ffmpeg.zip"
    Invoke-WebRequest $ffmpegUrl -OutFile $ffArchive
    Expand-Archive -Path $ffArchive -DestinationPath $ToolsDir -Force
    $folder = Get-ChildItem $ToolsDir -Directory | Where-Object { $_.Name -like "ffmpeg-*" } | Select-Object -First 1
    if ($null -ne $folder) {
        Move-Item (Join-Path $folder.FullName "bin/ffmpeg.exe") (Join-Path $FfmpegDir "bin/ffmpeg.exe") -Force
        Move-Item (Join-Path $folder.FullName "bin/ffprobe.exe") (Join-Path $FfmpegDir "bin/ffprobe.exe") -Force
        Remove-Item $folder.FullName -Recurse -Force
    }
    Remove-Item $ffArchive -Force
} else {
    Write-Info "  ffmpeg déjà présent, étape sautée."
}

Write-Info "[5/5] Pré-téléchargement du modèle large-v3..."
$downloadScript = @"
from pathlib import Path
from faster_whisper.utils import download_model
target = Path(r"$ModelDir")
target.mkdir(parents=True, exist_ok=True)
print("  Téléchargement/validation du modèle large-v3...")
download_model("large-v3", download_root=target)
print("  Modèle prêt dans", target)
"@
$tmpPy = Join-Path $ToolsDir "download_model.py"
Set-Content -Path $tmpPy -Value $downloadScript -Encoding ASCII
& $PythonExe $tmpPy
Remove-Item $tmpPy -Force

Write-Host "`n✅ Installation terminée. Les employés peuvent utiliser Transcrire.bat." -ForegroundColor Green
