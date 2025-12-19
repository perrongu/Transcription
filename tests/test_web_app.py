import os
import json
import time
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

# Évite de charger les dépendances lourdes pendant les tests
os.environ.setdefault("TRANSCRIBE_SKIP_DEPS", "1")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from web import app as web_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIBE_SYNC", "1")
    web_app.jobs.clear()
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(web_app, "JOBS_DIR", jobs_dir)
    from scripts.transcribe import TranscriptionResult

    def fake_transcribe_with_progress(
        input_path: Path,
        output_dir: Path,
        config,
        formats=None,
        sample_minutes=None,
        progress_callback=None,
        append_input_stem=False,
        console_progress=False,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        segments = [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello world"}]
        info = {"language": "fr", "language_probability": 0.99, "duration": 1.0}
        progress_stats = {"elapsed_time": 0.1, "speed": 10.0, "segments_count": 1, "words_count": 2, "audio_duration": 1.0}

        if progress_callback:
            progress_callback({"type": "stage", "stage": "prepare"})
            progress_callback({"type": "progress", "progress": 0.5, "segments": 1, "words": 2, "last_text": "hello world", "total": 1.0})

        output_files = {}
        name_map = {"txt": "transcript.txt", "srt": "transcript.srt", "vtt": "transcript.vtt", "json": "segments.json"}
        for fmt in formats or []:
            fname = name_map.get(fmt, f"file.{fmt}")
            path = output_dir / fname
            if fmt == "json":
                path.write_text(json.dumps({"segments": segments, "info": info}), encoding="utf-8")
            else:
                path.write_text("dummy content", encoding="utf-8")
            output_files[fmt] = path

        if progress_callback:
            progress_callback({"type": "complete", "formats": formats or [], "segments": 1, "words": 2, "last_text": "hello world"})

        return TranscriptionResult(
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            formats=formats or [],
            segments=segments,
            info=info,
            progress_stats=progress_stats,
            output_files=output_files,
            audio_duration=1.0,
        )

    monkeypatch.setattr(web_app, "transcribe_with_progress", fake_transcribe_with_progress)
    return TestClient(web_app.app)


def test_full_flow(client):
    resp = client.post(
        "/api/transcribe",
        files={"file": ("sample.wav", b"fake-audio", "audio/wav")},
        data={
            "model": "tiny",
            "language": "fr",
            "formats": "txt,srt",
            "beam_size": "3",
            "vad_filter": "true",
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # Attendre la fin du job
    status = None
    deadline = time.time() + 3
    while time.time() < deadline:
        status_resp = client.get("/api/status").json()
        job = next((j for j in status_resp["jobs"] if j["id"] == job_id), None)
        status = job["status"] if job else None
        if status == "done":
            break
        time.sleep(0.05)
    assert status == "done"

    # Progress SSE doit renvoyer au moins un event
    with client.stream("GET", f"/api/progress/{job_id}") as stream:
        first_chunk = next(stream.iter_text())
        assert "data:" in first_chunk

    # Téléchargements disponibles
    for fmt, expected_name in [("txt", "transcript.txt"), ("srt", "transcript.srt")]:
        dl = client.get(f"/api/download/{job_id}/{fmt}")
        assert dl.status_code == 200
        assert expected_name in dl.headers.get("content-disposition", "")
        assert dl.text


def test_reject_when_busy(client):
    # Simuler un job en cours
    web_app.jobs["busy"] = web_app.JobState(
        id="busy",
        status="processing",
        input_path=Path("dummy"),
        output_dir=Path("dummy"),
        formats=["txt"],
    )
    resp = client.post(
        "/api/transcribe",
        files={"file": ("sample.wav", b"fake-audio", "audio/wav")},
        data={"formats": "txt"},
    )
    assert resp.status_code == 409
