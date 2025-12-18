#!/usr/bin/env bash
set -euo pipefail

# Installation clé-en-main pour macOS/Linux.
# - Crée un venv dans tools/venv
# - Installe les dépendances Python
# - Télécharge ffmpeg portable
# - Pré-télécharge le modèle Whisper large-v3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT/tools"
VENV_DIR="$TOOLS_DIR/venv"
FFMPEG_DIR="$TOOLS_DIR/ffmpeg"
MODEL_DIR="$ROOT/models"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS="$ROOT/requirements.txt"

echo "== Déploiement clé-en-main (macOS/Linux) =="
echo "[1/5] Vérification Python (>=3.10)..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.10+ est requis. Installez-le puis relancez ce script."
  exit 1
fi

python_version="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "$python_version" < "3.10" ]]; then
  echo "Python $python_version détecté, mais 3.10+ est requis."
  exit 1
fi

mkdir -p "$TOOLS_DIR" "$MODEL_DIR"

echo "[2/5] Création du venv local (tools/venv)..."
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "  venv déjà présent, on réutilise."
fi
PYTHON="$VENV_DIR/bin/python"

echo "[3/5] Installation des dépendances Python..."
"$PYTHON" -m pip install --upgrade pip >/dev/null
"$PYTHON" -m pip install -r "$REQUIREMENTS"

download_ffmpeg_linux() {
  local url="${FFMPEG_URL:-https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz}"
  local archive="$TOOLS_DIR/ffmpeg.tar.xz"
  echo "  Téléchargement ffmpeg Linux: $url"
  curl -L "$url" -o "$archive"
  mkdir -p "$FFMPEG_DIR"
  tar -xf "$archive" -C "$TOOLS_DIR"
  local extracted
  extracted="$(find "$TOOLS_DIR" -maxdepth 1 -type d -name 'ffmpeg-*' | head -n 1)"
  mkdir -p "$FFMPEG_DIR/bin"
  mv "$extracted/ffmpeg" "$extracted/ffprobe" "$FFMPEG_DIR/bin/"
  rm -rf "$extracted" "$archive"
}

download_ffmpeg_macos() {
  local ffmpeg_url="${FFMPEG_URL:-https://evermeet.cx/ffmpeg/ffmpeg-6.1.1.zip}"
  local ffprobe_url="${FFPROBE_URL:-https://evermeet.cx/ffmpeg/ffprobe-6.1.1.zip}"
  local tmpdir
  tmpdir="$(mktemp -d)"
  mkdir -p "$FFMPEG_DIR/bin"
  echo "  Téléchargement ffmpeg macOS..."
  curl -L "$ffmpeg_url" -o "$tmpdir/ffmpeg.zip"
  curl -L "$ffprobe_url" -o "$tmpdir/ffprobe.zip"
  unzip -o "$tmpdir/ffmpeg.zip" -d "$tmpdir" >/dev/null
  unzip -o "$tmpdir/ffprobe.zip" -d "$tmpdir" >/dev/null
  mv "$tmpdir/ffmpeg" "$FFMPEG_DIR/bin/"
  mv "$tmpdir/ffprobe" "$FFMPEG_DIR/bin/"
  chmod +x "$FFMPEG_DIR/bin/ffmpeg" "$FFMPEG_DIR/bin/ffprobe"
  rm -rf "$tmpdir"
}

echo "[4/5] ffmpeg portable..."
if [[ -x "$FFMPEG_DIR/bin/ffmpeg" || -x "$FFMPEG_DIR/bin/ffmpeg.exe" ]]; then
  echo "  ffmpeg déjà présent."
else
  mkdir -p "$FFMPEG_DIR/bin"
  case "$OSTYPE" in
    linux*) download_ffmpeg_linux ;;
    darwin*) download_ffmpeg_macos ;;
    *) echo "Système non reconnu pour le téléchargement automatique de ffmpeg ($OSTYPE)." ;;
  esac
fi

echo "[5/5] Pré-téléchargement du modèle large-v3 (dans models/)..."
"$PYTHON" - <<PY
from pathlib import Path
from faster_whisper.utils import download_model

model_dir = Path(r"$MODEL_DIR")
model_dir.mkdir(parents=True, exist_ok=True)
print("  Téléchargement/validation du modèle large-v3...")
download_model("large-v3", download_root=model_dir)
print("  Modèle prêt dans", model_dir)
PY

echo
echo "✅ Installation terminée."
echo "Les employés peuvent utiliser Transcrire.command (macOS) ou Transcrire.sh (Linux) avec un double-clic ou glisser-déposer."
