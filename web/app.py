#!/usr/bin/env python3
import asyncio
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Rendre le module scripts/ importable même si on lance depuis web/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.transcribe import (  # noqa: E402
    TranscriptionConfig,
    TranscriptionResult,
    transcribe_with_progress,
)

STATIC_DIR = ROOT_DIR / "web" / "static"
JOBS_DIR = ROOT_DIR / "web" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def read_version() -> str:
    version_file = ROOT_DIR / "VERSION"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip()
    return "dev"


app = FastAPI(
    title="Transcription Locale",
    version=read_version(),
    description="Interface web locale pour Transcription (FastAPI + vanilla JS)",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8765", "http://127.0.0.1:8765"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@dataclass
class JobState:
    id: str
    status: str
    input_path: Path
    output_dir: Path
    formats: List[str]
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    created_at: float = field(default_factory=time.time)
    error: Optional[str] = None
    result: Optional[TranscriptionResult] = None
    last_progress: Optional[Dict] = None


jobs: Dict[str, JobState] = {}

# Constantes de sécurité
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB max
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".mp4", ".avi", ".mov", ".mkv", ".m4a", ".flac", ".ogg", ".webm"}
JOB_CLEANUP_AGE = 3600  # 1 heure en secondes


def sanitize_filename(name: str) -> str:
    """Nettoie un nom de fichier pour éviter les paths relatifs dangereux."""
    fallback = "audio"
    if not name:
        return fallback
    # Extraire uniquement le nom de fichier (sans path)
    cleaned = Path(name).name
    # Remplacer les caractères dangereux
    cleaned = cleaned.replace(" ", "_").replace("/", "_").replace("\\", "_")
    # Nettoyer les caractères non-ASCII et caractères spéciaux
    cleaned = re.sub(r'[^\w.\-]', '_', cleaned)
    # Limiter la longueur
    cleaned = cleaned[:200] if len(cleaned) > 200 else cleaned
    return cleaned or fallback


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def sse_event(payload: Dict) -> str:
    event_name = payload.get("type", "message")
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_progress_callback(job: JobState, loop: asyncio.AbstractEventLoop):
    def callback(payload: Dict):
        enriched = dict(payload)
        enriched["job_id"] = job.id
        enriched["timestamp"] = time.time()
        job.last_progress = enriched
        asyncio.run_coroutine_threadsafe(job.queue.put(enriched), loop)

    return callback


async def run_job(job_id: str, config: TranscriptionConfig, sample_minutes: Optional[float]):
    job = jobs[job_id]
    loop = asyncio.get_event_loop()
    callback = build_progress_callback(job, loop)
    job.status = "processing"
    try:
        result = await asyncio.to_thread(
            transcribe_with_progress,
            job.input_path,
            job.output_dir,
            config,
            job.formats,
            sample_minutes,
            callback,
            False,  # append_input_stem: on utilise le dossier du job directement
            False,  # console_progress: silencieux côté serveur
        )
        job.result = result
        job.status = "done"
        job.last_progress = job.last_progress or {
            "type": "complete",
            "job_id": job.id,
            "output_dir": str(result.output_dir),
            "formats": result.formats,
        }
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        await job.queue.put({
            "type": "error",
            "job_id": job.id,
            "message": str(exc),
        })
    finally:
        await job.queue.put(None)  # sentinelle pour fermer le flux SSE


def server_busy() -> bool:
    return any(j.status == "processing" for j in jobs.values())


def cleanup_old_jobs():
    """Supprime les jobs terminés de plus d'une heure."""
    now = time.time()
    to_remove = []
    for job_id, job in jobs.items():
        if job.status in ("done", "error") and (now - job.created_at) > JOB_CLEANUP_AGE:
            to_remove.append(job_id)
    for job_id in to_remove:
        job = jobs.pop(job_id, None)
        if job and job.output_dir.exists():
            import shutil
            try:
                shutil.rmtree(job.output_dir)
            except Exception:
                pass  # Ignore les erreurs de suppression


@app.post("/api/transcribe")
async def start_transcription(
    file: UploadFile = File(...),
    model: str = Form("large-v3"),
    language: str = Form("fr"),
    compute_type: str = Form("auto"),
    device: str = Form("auto"),
    beam_size: int = Form(5),
    vad_filter: bool = Form(True),
    sample_minutes: Optional[float] = Form(None),
    formats: str = Form("txt,srt,vtt,json"),
):
    if server_busy():
        raise HTTPException(status_code=409, detail="Une transcription est déjà en cours. Réessayez dans un instant.")

    # Validation du modèle
    valid_models = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
    if model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Modèle invalide. Choisir parmi: {', '.join(sorted(valid_models))}")

    # Validation de la langue (codes ISO 639-1 de base)
    if language != "auto" and len(language) != 2:
        raise HTTPException(status_code=400, detail="Code langue invalide (2 caractères ou 'auto')")

    # Validation du compute_type
    valid_compute_types = {"auto", "float16", "float32", "int8", "int8_float16"}
    if compute_type not in valid_compute_types:
        raise HTTPException(status_code=400, detail=f"Compute type invalide. Choisir parmi: {', '.join(sorted(valid_compute_types))}")

    # Validation du beam_size
    if not (1 <= beam_size <= 20):
        raise HTTPException(status_code=400, detail="Beam size doit être entre 1 et 20")

    # Validation de l'extension du fichier
    safe_name = sanitize_filename(file.filename or "audio")
    file_ext = Path(safe_name).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format de fichier non supporté. Formats acceptés: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Nettoyage des anciens jobs
    cleanup_old_jobs()

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / safe_name
    total_size = 0
    try:
        with open(input_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    input_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Fichier trop volumineux (max {MAX_FILE_SIZE / (1024**3):.1f} GB)"
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'upload: {str(exc)}")

    formats_list = [f.strip().lower() for f in formats.split(",") if f.strip()]
    job = JobState(
        id=job_id,
        status="pending",
        input_path=input_path,
        output_dir=job_dir,
        formats=formats_list,
    )
    jobs[job_id] = job

    config = TranscriptionConfig(
        model_name=model,
        language=language,
        device=device,
        compute_type=compute_type,
        beam_size=int(beam_size),
        vad_filter=parse_bool(vad_filter),
        update_interval=1.0,
    )

    if os.environ.get("TRANSCRIBE_SYNC") == "1":
        await run_job(job_id, config, sample_minutes)
    else:
        asyncio.create_task(run_job(job_id, config, sample_minutes))
    return {"job_id": job_id}


@app.get("/api/progress/{job_id}")
async def progress(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")

    async def event_generator():
        if job.last_progress:
            yield sse_event(job.last_progress)
        while True:
            payload = await job.queue.get()
            if payload is None:
                break
            yield sse_event(payload)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/status")
async def status():
    return {
        "status": "busy" if server_busy() else "ready",
        "jobs": [
            {"id": j.id, "status": j.status, "created_at": j.created_at, "error": j.error}
            for j in jobs.values()
        ],
    }


@app.get("/api/download/{job_id}/{fmt}")
async def download(job_id: str, fmt: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail="Job en cours ou incomplet")

    fmt_lower = fmt.lower()
    valid_formats = {"txt", "srt", "vtt", "json"}
    if fmt_lower not in valid_formats:
        raise HTTPException(status_code=400, detail=f"Format invalide. Choisir parmi: {', '.join(sorted(valid_formats))}")

    file_path = job.result.output_files.get(fmt_lower)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Format introuvable")

    # Sécurité: vérifier que le fichier est bien dans le répertoire du job
    try:
        file_path.resolve().relative_to(job.output_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Accès refusé")

    media_type = "text/plain"
    if fmt_lower in ("srt", "vtt"):
        media_type = "text/vtt" if fmt_lower == "vtt" else "application/x-subrip"
    elif fmt_lower == "json":
        media_type = "application/json"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name,
    )


@app.get("/favicon.ico")
async def favicon():
    """Retourne 204 No Content pour éviter le 404 du favicon."""
    from fastapi.responses import Response
    return Response(status_code=204)

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web.app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
    )
