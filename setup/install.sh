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
echo "[1/5] Vérification Python (>=3.9)..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.9+ est requis. Installez-le puis relancez ce script."
  exit 1
fi

# Vérification de version Python (comparaison numérique)
python_ok="$($PYTHON_BIN - <<'PY'
import sys
major, minor = sys.version_info.major, sys.version_info.minor
print("1" if (major > 3) or (major == 3 and minor >= 9) else "0")
PY
)"
python_version="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$python_ok" != "1" ]]; then
  echo "Python $python_version détecté, mais 3.9+ est requis."
  echo ""
  echo "Sur macOS, installez Python 3.9+ avec Homebrew :"
  echo "  brew install python@3.12"
  echo ""
  echo "Puis relancez le script avec :"
  echo "  PYTHON_BIN=/opt/homebrew/bin/python3.12 ./setup/install.sh"
  exit 1
fi
echo "  Python $python_version détecté ✓"

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
download_model("large-v3", output_dir=str(model_dir))
print("  Modèle prêt dans", model_dir)
PY

echo
echo "[Vérification] Test de l'installation..."
"$PYTHON" - <<PY
import sys
try:
    import faster_whisper
    from faster_whisper import WhisperModel
    print("  ✓ faster-whisper importé avec succès")

    import tqdm
    print("  ✓ tqdm importé avec succès")

    # Vérifier que le modèle existe
    from pathlib import Path
    model_dir = Path(r"$MODEL_DIR")
    model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
    if model_files:
        print(f"  ✓ Modèle trouvé dans {model_dir} ({len(model_files)} fichier(s))")
    else:
        print(f"  ⚠ Modèle non trouvé dans {model_dir}")
        sys.exit(1)

    print("  ✅ Toutes les vérifications passées")
except ImportError as e:
    print(f"  ❌ Erreur d'import: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ Erreur: {e}")
    sys.exit(1)
PY

if [ $? -ne 0 ]; then
    echo
    echo "❌ Erreur lors de la vérification. L'installation peut être incomplète."
    exit 1
fi

echo
echo "✅ Installation terminée et vérifiée."
echo "Les employés peuvent utiliser Transcrire.command (macOS) ou Transcrire.sh (Linux) avec un double-clic ou glisser-déposer."
