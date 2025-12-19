#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# Rendre le module scripts/ importable même si on lance depuis web/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.transcribe import (  # noqa: E402
    TranscriptionConfig,
    TranscriptionResult,
    TranscriptionCancelled,
    transcribe_with_progress,
)

STATIC_DIR = ROOT_DIR / "web" / "static"
JOBS_DIR = ROOT_DIR / "web" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def read_version() -> str:
    version_file = ROOT_DIR / "VERSION"
    if version_file.exists():
        try:
            return version_file.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning(f"Erreur lors de la lecture de VERSION: {exc}")
            return "dev"
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
    cancel_requested: bool = False


jobs: Dict[str, JobState] = {}

# Constantes de sécurité
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB max
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".mp4", ".avi", ".mov", ".mkv", ".m4a", ".flac", ".ogg", ".webm"}
JOB_CLEANUP_AGE = 3600  # 1 heure en secondes
MAX_JOBS_IN_MEMORY = 100  # Limite de jobs en mémoire
VALID_FORMATS = {"txt", "srt", "vtt", "json"}


def is_valid_job_id(job_id: str) -> bool:
    """Valide qu'un job_id est un UUID hex valide."""
    if not job_id or len(job_id) != 32:
        return False
    try:
        int(job_id, 16)
        return True
    except ValueError:
        return False


# Codes ISO 639-1 valides (échantillon des plus courants)
VALID_LANGUAGE_CODES = {
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az",
    "ba", "be", "bg", "bh", "bi", "bm", "bn", "bo", "br", "bs",
    "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy",
    "da", "de", "dv", "dz",
    "ee", "el", "en", "eo", "es", "et", "eu",
    "fa", "ff", "fi", "fj", "fo", "fr", "fy",
    "ga", "gd", "gl", "gn", "gu", "gv",
    "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz",
    "ia", "id", "ie", "ig", "ii", "ik", "io", "is", "it", "iu",
    "ja", "jv",
    "ka", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku", "kv", "kw", "ky",
    "la", "lb", "lg", "li", "ln", "lo", "lt", "lu", "lv",
    "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
    "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny",
    "oc", "oj", "om", "or", "os",
    "pa", "pi", "pl", "ps", "pt",
    "qu",
    "rm", "rn", "ro", "ru", "rw",
    "sa", "sc", "sd", "se", "sg", "si", "sk", "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw",
    "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts", "tt", "tw", "ty",
    "ug", "uk", "ur", "uz",
    "ve", "vi", "vo",
    "wa", "wo",
    "xh",
    "yi", "yo",
    "za", "zh", "zu",
}


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
        if job.cancel_requested:
            return
        # Vérifier que le job existe toujours
        if job.id not in jobs:
            return
        enriched = dict(payload)
        enriched["job_id"] = job.id
        enriched["timestamp"] = time.time()
        job.last_progress = enriched
        try:
            # Limiter la taille de la queue pour éviter les fuites mémoire
            # Si la queue est pleine, on ignore les messages de progression (mais pas les événements critiques)
            queue_size = getattr(job.queue, 'qsize', lambda: 0)()
            if queue_size < 100:  # Limite raisonnable
                asyncio.run_coroutine_threadsafe(job.queue.put(enriched), loop)
            elif payload.get("type") in ("error", "cancelled", "complete"):
                # Toujours envoyer les événements critiques même si la queue est pleine
                asyncio.run_coroutine_threadsafe(job.queue.put(enriched), loop)
        except Exception as exc:
            logger.warning(f"Erreur lors de l'envoi du callback pour job {job.id}: {exc}")

    return callback


async def run_job(job_id: str, config: TranscriptionConfig, sample_minutes: Optional[float]):
    job = jobs.get(job_id)
    if not job:
        logger.warning(f"Job {job_id} introuvable lors du démarrage du traitement")
        return
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
            lambda: job.cancel_requested,
        )
        job.result = result
        job.status = "done"
        job.last_progress = job.last_progress or {
            "type": "complete",
            "job_id": job.id,
            "output_dir": str(result.output_dir),
            "formats": result.formats,
        }
    except TranscriptionCancelled:
        job.status = "cancelled"
        logger.info(f"Job {job_id} annulé par l'utilisateur")
        await job.queue.put({
            "type": "cancelled",
            "job_id": job.id,
            "message": "Cancelled",
        })
    except Exception as exc:
        job.status = "error"
        # Limiter la longueur du message d'erreur pour éviter les problèmes
        error_msg = str(exc)[:500] if exc else "Erreur inconnue"
        job.error = error_msg
        logger.error(f"Erreur lors du traitement du job {job_id}: {exc}", exc_info=True)
        await job.queue.put({
            "type": "error",
            "job_id": job.id,
            "message": error_msg,
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
        if job.status in ("done", "error", "cancelled") and (now - job.created_at) > JOB_CLEANUP_AGE:
            to_remove.append(job_id)
    for job_id in to_remove:
        job = jobs.pop(job_id, None)
        if job:
            # Nettoyer la queue pour libérer la mémoire
            try:
                # Vider la queue de manière asynchrone
                while True:
                    try:
                        job.queue.get_nowait()
                    except Exception:
                        break  # Queue vide ou erreur
            except Exception:
                pass  # Ignore les erreurs de nettoyage de queue
            if job.output_dir.exists():
                try:
                    shutil.rmtree(job.output_dir)
                    logger.debug(f"Job {job_id} nettoyé (âge: {now - job.created_at:.0f}s)")
                except Exception as exc:
                    logger.warning(f"Erreur lors du nettoyage du job {job_id}: {exc}")


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

    # Validation de la langue (codes ISO 639-1)
    language_lower = language.lower() if language else ""
    if language != "auto":
        if len(language_lower) != 2 or language_lower not in VALID_LANGUAGE_CODES:
            raise HTTPException(
                status_code=400,
                detail=f"Code langue invalide. Utiliser 'auto' ou un code ISO 639-1 valide (ex: fr, en, es, de)"
            )

    # Validation du compute_type
    valid_compute_types = {"auto", "float16", "float32", "int8", "int8_float16"}
    if compute_type not in valid_compute_types:
        raise HTTPException(status_code=400, detail=f"Compute type invalide. Choisir parmi: {', '.join(sorted(valid_compute_types))}")

    # Validation du beam_size
    if not (1 <= beam_size <= 20):
        raise HTTPException(status_code=400, detail="Beam size doit être entre 1 et 20")

    # Validation de sample_minutes
    if sample_minutes is not None:
        try:
            sample_minutes_float = float(sample_minutes)
            if not (0 < sample_minutes_float <= 10080):  # Max 7 jours (10080 minutes)
                raise HTTPException(status_code=400, detail="sample_minutes doit être entre 0 et 10080 minutes")
            sample_minutes = sample_minutes_float
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="sample_minutes doit être un nombre valide")

    # Validation de l'extension du fichier
    safe_name = sanitize_filename(file.filename or "audio")
    file_ext = Path(safe_name).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format de fichier non supporté. Formats acceptés: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Nettoyage des anciens jobs (optimisé: seulement si nécessaire)
    if len(jobs) > 10:  # Nettoyer seulement si beaucoup de jobs en mémoire
        cleanup_old_jobs()

    # Vérifier la limite de jobs en mémoire
    if len(jobs) >= MAX_JOBS_IN_MEMORY:
        raise HTTPException(
            status_code=503,
            detail=f"Trop de jobs en cours. Limite: {MAX_JOBS_IN_MEMORY}. Réessayez plus tard."
        )

    # Race condition: vérifier à nouveau après nettoyage
    if server_busy():
        raise HTTPException(status_code=409, detail="Une transcription est déjà en cours. Réessayez dans un instant.")

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
        logger.error(f"Erreur lors de l'upload du fichier: {exc}", exc_info=True)
        # Ne pas exposer les détails de l'erreur système au client
        raise HTTPException(status_code=500, detail="Erreur lors de l'upload du fichier")

    formats_list = [f.strip().lower() for f in formats.split(",") if f.strip()]
    # Validation des formats
    invalid_formats = [f for f in formats_list if f not in VALID_FORMATS]
    if invalid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Formats invalides: {', '.join(invalid_formats)}. Formats valides: {', '.join(sorted(VALID_FORMATS))}"
        )
    if not formats_list:
        raise HTTPException(status_code=400, detail="Au moins un format doit être sélectionné")

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
        language=language_lower if language != "auto" else "auto",
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
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Format de job_id invalide")
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")

    async def event_generator():
        try:
            # Envoyer le dernier progrès connu si disponible
            current_job = jobs.get(job_id)
            if current_job and current_job.last_progress:
                yield sse_event(current_job.last_progress)
            while True:
                # Vérifier que le job existe toujours
                current_job = jobs.get(job_id)
                if not current_job:
                    logger.debug(f"Job {job_id} supprimé pendant le streaming SSE")
                    break
                # Utiliser current_job au lieu de job pour éviter les références obsolètes
                try:
                    # Timeout pour éviter de bloquer indéfiniment si le client se déconnecte
                    # Timeout plus long pour les longues transcriptions (5 minutes)
                    payload = await asyncio.wait_for(current_job.queue.get(), timeout=300.0)
                except asyncio.TimeoutError:
                    # Vérifier à nouveau que le job existe avant d'envoyer le heartbeat
                    if job_id in jobs:
                        yield sse_event({"type": "heartbeat", "job_id": job_id, "timestamp": time.time()})
                    continue
                if payload is None:
                    break
                yield sse_event(payload)
        except asyncio.CancelledError:
            logger.debug(f"Stream SSE annulé pour job {job_id} (client déconnecté)")
        except Exception as exc:
            logger.error(f"Erreur dans le générateur SSE pour job {job_id}: {exc}", exc_info=True)
            yield sse_event({
                "type": "error",
                "job_id": job_id,
                "message": "Erreur de connexion",
            })

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


@app.post("/api/cancel/{job_id}")
async def cancel(job_id: str):
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Format de job_id invalide")
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.status not in {"processing", "pending"}:
        raise HTTPException(status_code=409, detail="Job déjà terminé")
    # Vérifier que le job n'a pas été annulé entre temps (race condition)
    current_job = jobs.get(job_id)
    if not current_job or current_job.status not in {"processing", "pending"}:
        raise HTTPException(status_code=409, detail="Job déjà terminé")
    current_job.cancel_requested = True
    await current_job.queue.put({
        "type": "cancelled",
        "job_id": current_job.id,
        "message": "Cancelled by user",
    })
    return {"status": "cancelled"}


@app.get("/api/download/{job_id}/{fmt}")
async def download(job_id: str, fmt: str):
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Format de job_id invalide")
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail="Job en cours ou incomplet")

    fmt_lower = fmt.lower()
    if fmt_lower not in VALID_FORMATS:
        raise HTTPException(status_code=400, detail=f"Format invalide. Choisir parmi: {', '.join(sorted(VALID_FORMATS))}")

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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Retourne 204 No Content pour éviter le 404 du favicon."""
    return Response(status_code=204)

# Mount static files en dernier pour servir index.html pour toutes les autres routes
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
