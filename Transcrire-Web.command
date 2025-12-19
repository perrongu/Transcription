#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/tools/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  if [[ -x "$ROOT/tools/python/bin/python3" ]]; then
    PYTHON="$ROOT/tools/python/bin/python3"
  else
    PYTHON="python3"
  fi
fi

FFMPEG_BIN="$ROOT/tools/ffmpeg/bin"
if [[ -d "$FFMPEG_BIN" ]]; then
  export PATH="$FFMPEG_BIN:$PATH"
fi

cd "$ROOT"
echo "== Interface web locale =="
echo "Disponible sur http://localhost:8765"
"$PYTHON" -m uvicorn web.app:app --host 127.0.0.1 --port 8765
