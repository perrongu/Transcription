#!/usr/bin/env python3
"""
FR (débutant) : transcrire un fichier audio/vidéo en local.
EN (beginner) : transcribe an audio/video file locally.

Usage rapide / Quick start:
  python scripts/transcribe.py --input "mon_fichier.mp4"
Résultats / Outputs:
  out/<nom_fichier>/
    - transcript.txt (texte)
    - transcript.srt / transcript.vtt (sous-titres)
    - segments.json (timestamps + métadonnées)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Constantes
DEFAULT_UPDATE_INTERVAL = 5.0
ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT_DIR / "tools"
DEFAULT_MODEL_DIR = ROOT_DIR / "models"

# Détection automatique du venv local
VENV_PYTHON = TOOLS_DIR / "venv" / "bin" / "python"
if sys.platform == "win32":
    VENV_PYTHON = TOOLS_DIR / "venv" / "Scripts" / "python.exe"

# Si le venv existe et que les dépendances ne sont pas disponibles, suggérer de l'utiliser
def check_and_suggest_venv():
    """Vérifie si un venv local existe et suggère son utilisation si nécessaire."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return None


@dataclass
class TranscriptionConfig:
    """Configuration centralisée pour la transcription."""
    model_name: str = "large-v3"
    language: str = "fr"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    update_interval: float = DEFAULT_UPDATE_INTERVAL


def check_dependencies():
    """Check required Python deps, guide beginners."""
    missing = []
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        missing.append("faster-whisper")

    try:
        import tqdm  # noqa: F401
    except ImportError:
        missing.append("tqdm")

    if missing:
        venv_python = check_and_suggest_venv()
        print(f"Erreur / Error: dépendances manquantes: {', '.join(missing)}")
        
        if venv_python:
            print(f"\n⚠️  Un environnement virtuel local a été détecté dans tools/venv/")
            print(f"   Utilisez-le avec: {venv_python} scripts/transcribe.py --input \"fichier.mp4\"")
            print(f"\n   Ou utilisez les lanceurs (recommandé):")
            if sys.platform == "win32":
                print("   - Transcrire.bat")
            elif sys.platform == "darwin":
                print("   - Transcrire.command")
            else:
                print("   - Transcrire.sh")
            print(f"\n   Ces lanceurs utilisent automatiquement le Python du venv.")
        else:
            print(f"\nInstallez / Install with: pip install {' '.join(missing)}")
            print("\nOu exécutez le script d'installation: ./setup/install.sh (macOS/Linux) ou setup\\install.bat (Windows)")
        
        sys.exit(1)


# Vérification des dépendances au démarrage
check_dependencies()

# Imports après vérification
from faster_whisper import WhisperModel
from tqdm import tqdm


# ============================================================================
# FORMATAGE
# ============================================================================

def format_timestamp_srt(seconds: float) -> str:
    """Formate en HH:MM:SS,mmm (format SRT)."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Formate en HH:MM:SS.mmm (format VTT)."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_duration(seconds: float) -> str:
    """Formate une durée en HH:MM:SS."""
    if not (seconds >= 0) or not (seconds == seconds):  # Gestion des NaN, inf, et négatifs
        return "--:--:--"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_speed(speed: float) -> str:
    """Formate la vitesse (x temps réel)."""
    if speed <= 0:
        return "---"
    if speed >= 1:
        return f"{speed:.1f}x"
    return f"{speed:.2f}x"


# ============================================================================
# AUDIO
# ============================================================================

def find_executable(candidates: List[str]) -> Optional[str]:
    """Retourne le premier exécutable existant parmi une liste."""
    for candidate in candidates:
        if not candidate:
            continue
        path_candidate = Path(candidate).expanduser()
        if path_candidate.exists():
            return str(path_candidate)
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return shutil.which(candidate)
    return None


def detect_ffmpeg_binaries() -> Tuple[str, str]:
    """Détecte ffmpeg/ffprobe locaux dans tools/ffmpeg ou dans le PATH."""
    ffmpeg_candidates = [
        os.environ.get("FFMPEG_BIN"),
        TOOLS_DIR / "ffmpeg" / "bin" / "ffmpeg",
        TOOLS_DIR / "ffmpeg" / "ffmpeg",
        TOOLS_DIR / "ffmpeg" / "bin" / "ffmpeg.exe",
        TOOLS_DIR / "ffmpeg" / "ffmpeg.exe",
        "ffmpeg",
    ]
    ffprobe_candidates = [
        os.environ.get("FFPROBE_BIN"),
        TOOLS_DIR / "ffmpeg" / "bin" / "ffprobe",
        TOOLS_DIR / "ffmpeg" / "ffprobe",
        TOOLS_DIR / "ffmpeg" / "bin" / "ffprobe.exe",
        TOOLS_DIR / "ffmpeg" / "ffprobe.exe",
        "ffprobe",
    ]
    ffmpeg_cmd = find_executable([str(c) for c in ffmpeg_candidates if c is not None]) or "ffmpeg"
    ffprobe_cmd = find_executable([str(c) for c in ffprobe_candidates if c is not None]) or "ffprobe"
    return ffmpeg_cmd, ffprobe_cmd


def probe_audio_duration(audio_path: Path, ffprobe_cmd: str) -> float:
    """Retourne la durée (secondes) d'un fichier audio via ffprobe, 0 si erreur."""
    probe_cmd = [
        ffprobe_cmd, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Avertissement/Warning: échec ffprobe pour {audio_path.name}: {result.stderr.strip()}")
        return 0.0
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        return 0.0


def extract_audio(input_path: Path, output_wav: Path, ffmpeg_cmd: str, ffprobe_cmd: str, sample_minutes: Optional[float] = None) -> float:
    """
    Extrait l'audio en WAV mono 16kHz PCM.
    Retourne la durée en secondes.
    """
    cmd = [
        ffmpeg_cmd, "-y", "-i", str(input_path),
        "-ac", "1",           # mono
        "-ar", "16000",       # 16kHz
        "-c:a", "pcm_s16le",  # PCM 16-bit
    ]
    
    if sample_minutes:
        cmd.extend(["-t", str(sample_minutes * 60)])
    
    cmd.append(str(output_wav))
    
    print(f"[ffmpeg] Extraction audio / Audio extraction → {output_wav.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Erreur ffmpeg / ffmpeg error:\n{result.stderr}")
        sys.exit(1)
    
    return probe_audio_duration(output_wav, ffprobe_cmd)


# ============================================================================
# PROGRESSION
# ============================================================================

class ProgressTracker:
    """Suivi de progression avec stats détaillées et mise à jour périodique."""

    def __init__(self, total_duration: float, update_interval: float = DEFAULT_UPDATE_INTERVAL):
        self.total_duration = total_duration
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.segments_count = 0
        self.words_count = 0
        self.current_position = 0.0
        self.last_text = ""
        self._elapsed_cache = 0.0  # Cache pour optimiser les calculs

        # Barre de progression tqdm
        self.pbar = tqdm(
            total=total_duration,
            unit="s",
            unit_scale=False,
            ncols=95,
            leave=True,
            dynamic_ncols=False,
        )
        self.pbar.set_description("Transcription")

    def update(self, segment_end: float, text: str):
        """Met à jour la progression avec un nouveau segment."""
        self.segments_count += 1
        self.words_count += len(text.split())
        self.current_position = max(self.current_position, segment_end)  # Évite les régressions
        self.last_text = text[:50] + "..." if len(text) > 50 else text

        # Mise à jour de la barre (évite les décréments)
        increment = max(0, self.current_position - self.pbar.n)
        if increment > 0:
            self.pbar.update(increment)

        # Stats périodiques
        now = time.time()
        if now - self.last_update_time >= self.update_interval:
            self._print_stats(now)
            self.last_update_time = now

    def _print_stats(self, current_time: float):
        """Affiche les stats détaillées (version simple et lisible)."""
        elapsed = current_time - self.start_time
        self._elapsed_cache = elapsed

        # Calculs optimisés
        if elapsed > 0:
            speed = self.current_position / elapsed  # x temps réel
            remaining_audio = max(0, self.total_duration - self.current_position)
            eta = remaining_audio / speed if speed > 0 else 0
            segments_per_min = (self.segments_count / elapsed) * 60
            words_per_min = (self.words_count / elapsed) * 60
        else:
            speed = 0
            eta = 0
            segments_per_min = 0
            words_per_min = 0

        # Affichage compact et compréhensible
        progress_pct = (self.current_position / self.total_duration * 100) if self.total_duration else 0
        stats_line = (
            f"⏳ {progress_pct:5.1f}% | "
            f"{format_duration(self.current_position)}/{format_duration(self.total_duration)} | "
            f"Vitesse/Speed {format_speed(speed)} | "
            f"Reste/Remaining {format_duration(eta)} | "
            f"{self.segments_count} segments ({segments_per_min:.0f}/min) | "
            f"{self.words_count} mots/words ({words_per_min:.0f}/min)"
        )
        tqdm.write(stats_line)

        # Dernier segment (aperçu)
        if self.last_text:
            tqdm.write(f"    ↳ Dernier/Last: \"{self.last_text}\"")

    def finish(self) -> Dict:
        """Termine la progression et retourne les stats finales."""
        # S'assurer que la barre n'excède pas 100% (évite le clamping warning)
        remaining = max(0, self.total_duration - self.pbar.n)
        if remaining > 0:
            self.pbar.update(remaining)

        self.pbar.close()

        # Utilise le cache si disponible
        elapsed = self._elapsed_cache if self._elapsed_cache > 0 else (time.time() - self.start_time)
        speed = self.current_position / elapsed if elapsed > 0 else 0

        return {
            "elapsed_time": elapsed,
            "speed": speed,
            "segments_count": self.segments_count,
            "words_count": self.words_count,
            "audio_duration": self.current_position,
        }


# ============================================================================
# TRANSCRIPTION
# ============================================================================

def transcribe_audio(
    wav_path: Path,
    audio_duration: float,
    model_name: str = "large-v3",
    language: str = "fr",
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
    model_dir: Optional[Path] = None,
) -> Tuple[List[Dict], Dict, Dict]:
    """
    Transcrit l'audio avec faster-whisper.
    Retourne (segments, info, progress_stats).
    """
    # Auto-détection device/compute
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    resolved_model_dir = model_dir or DEFAULT_MODEL_DIR
    resolved_model_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = resolved_model_dir / model_name
    model_source = str(local_model_path) if local_model_path.exists() else model_name

    print(f"\n[whisper] Chargement modèle / Loading model {model_source} sur {device} ({compute_type})...")
    model = WhisperModel(
        model_source,
        device=device,
        compute_type=compute_type,
        download_root=str(resolved_model_dir),
    )

    print(f"[whisper] Transcription / Transcribing (langue/lang={language}, beam={beam_size}, vad={vad_filter})\n")

    segments_gen, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
    )

    # Initialiser le tracker de progression
    tracker = ProgressTracker(audio_duration)

    segments = []
    try:
        for seg in segments_gen:
            segments.append({
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
            tracker.update(seg.end, seg.text.strip())
    except KeyboardInterrupt:
        print("\n\nInterruption détectée, arrêt propre de la transcription...")
        tqdm.write("Sauvegarde des segments déjà traités...")

    progress_stats = tracker.finish()

    info_dict = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
    }

    return segments, info_dict, progress_stats


# ============================================================================
# EXPORT
# ============================================================================

def write_txt(segments: List[Dict], output_path: Path):
    """Écrit le transcript brut (texte seul)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg["text"] + "\n")
    print(f"  ✓ {output_path.name}")


def write_srt(segments: List[Dict], output_path: Path):
    """Écrit au format SRT."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")
    print(f"  ✓ {output_path.name}")


def write_vtt(segments: List[Dict], output_path: Path):
    """Écrit au format WebVTT."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp_vtt(seg['start'])} --> {format_timestamp_vtt(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")
    print(f"  ✓ {output_path.name}")


def write_json(segments: List[Dict], info: Dict, output_path: Path):
    """Écrit les segments + métadonnées en JSON."""
    data = {
        "info": info,
        "segments": segments,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {output_path.name}")


# ============================================================================
# STATS FINALES
# ============================================================================

def print_final_stats(segments: List[Dict], audio_duration: float, info: Dict, progress_stats: Dict):
    """Affiche le résumé final détaillé."""
    total_words = sum(len(seg["text"].split()) for seg in segments)
    transcribed_duration = segments[-1]["end"] if segments else 0
    elapsed = progress_stats.get("elapsed_time", 0)
    speed = progress_stats.get("speed", 0)
    
    print("\n" + "═" * 60)
    print("          TRANSCRIPTION TERMINÉE / TRANSCRIPTION DONE")
    print("═" * 60)
    print()
    print(f"  📊 AUDIO / AUDIO")
    print(f"     Durée totale / total       : {format_duration(audio_duration)} ({audio_duration:.1f}s)")
    print(f"     Transcrit / transcribed    : {format_duration(transcribed_duration)} ({transcribed_duration:.1f}s)")
    print(f"     Langue détectée / detected : {info.get('language', '?')} ({info.get('language_probability', 0):.1%})")
    print()
    print(f"  ⚡ PERFORMANCE")
    print(f"     Temps / time      : {format_duration(elapsed)}")
    print(f"     Vitesse / speed   : {format_speed(speed)} temps réel / real-time")
    print()
    print(f"  📝 CONTENU / CONTENT")
    print(f"     Segments          : {len(segments)}")
    print(f"     Mots total / words: {total_words}")
    if audio_duration > 0:
        print(f"     Mots/minute / WPM : {total_words / (audio_duration / 60):.1f}")
    print()
    print("═" * 60)


# ============================================================================
# MAIN
# ============================================================================

def create_config_from_args(args) -> TranscriptionConfig:
    """Crée une configuration à partir des arguments CLI."""
    return TranscriptionConfig(
        model_name=args.model,
        language=args.lang,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        update_interval=DEFAULT_UPDATE_INTERVAL,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transcription audio/vidéo locale / Local transcription with faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python transcribe.py --input video.mp4
  python transcribe.py --input audio.wav --model medium --lang fr
  python transcribe.py --input video.mp4 --sample 5 --outdir test/
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=False,
        help="Fichier audio/vidéo à transcrire / file to transcribe (.mp4, .wav, .mp3, etc.)",
    )
    parser.add_argument(
        "--outdir", "-o",
        default="out",
        help="Dossier de sortie / output folder (défaut/default: out/)",
    )
    parser.add_argument(
        "--lang", "-l",
        default="fr",
        help="Langue de transcription / transcription language (défaut/default: fr)",
    )
    parser.add_argument(
        "--model", "-m",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Modèle Whisper / Whisper model (défaut/default: large-v3)",
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device (auto/cpu/cuda, défaut/default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "float16", "float32", "int8", "int8_float16"],
        help="Type de calcul / compute type (défaut/default auto → float16 GPU, int8 CPU)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size (défaut/default 5; plus = meilleur/slow, moins = rapide/lower quality)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Désactiver VAD / disable Voice Activity Detection",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Transcrire seulement les N premières minutes / only first N minutes (test)",
    )
    parser.add_argument(
        "--formats",
        default="txt,srt,vtt,json",
        help="Formats de sortie séparés par virgule / comma-separated outputs (défaut/default: txt,srt,vtt,json)",
    )

    args = parser.parse_args()
    if not args.input:
        try:
            print("Mode interactif: indique le fichier audio/vidéo à transcrire (glisser-déposer possible).")
            user_value = input("Chemin du fichier / File path: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            user_value = ""
        if not user_value:
            print("Aucun fichier fourni, arrêt.")
            sys.exit(1)
        args.input = user_value

    config = create_config_from_args(args)
    ffmpeg_cmd, ffprobe_cmd = detect_ffmpeg_binaries()
    
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Erreur / Error: fichier introuvable / file not found: {input_path}")
        sys.exit(1)
    
    # Créer dossier de sortie
    basename = input_path.stem
    out_dir = Path(args.outdir).resolve() / basename
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # En-tête
    print()
    print("╔" + "═" * 58 + "╗")
    print(f"║{'TRANSCRIPTION':^58}║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Fichier/File : {input_path.name[:41]:<50}║")
    print(f"║  Modèle/Model : {args.model:<50}║")
    print(f"║  Langue/Lang  : {args.lang:<50}║")
    print(f"║  Sortie/Out   : {str(out_dir)[-45:]:<50}║")
    if args.sample:
        print(f"║  Sample   : {args.sample} min{' ' * 38}║")
    print("╚" + "═" * 58 + "╝")
    
    # Extraction audio (fichier temporaire si vidéo)
    suffix = input_path.suffix.lower()
    use_temp_audio = True
    if suffix == ".wav" and args.sample is None:
        wav_path = input_path
        audio_duration = probe_audio_duration(wav_path, ffprobe_cmd)
        use_temp_audio = False
    else:
        wav_path = out_dir / "audio_temp.wav"
        audio_duration = extract_audio(input_path, wav_path, ffmpeg_cmd=ffmpeg_cmd, ffprobe_cmd=ffprobe_cmd, sample_minutes=args.sample)
    
    if audio_duration <= 0:
        print("Erreur / Error: impossible de déterminer la durée audio / cannot read audio duration")
        sys.exit(1)

    print(f"\n⏱️  Durée audio / Audio length: {format_duration(audio_duration)}. Début / Start...")
    print("Astuce/Tip: laisse tourner; tu peux interrompre avec Ctrl+C (les segments déjà faits sont conservés).")
    
    # Transcription
    segments, info, progress_stats = transcribe_audio(
        wav_path=wav_path,
        audio_duration=audio_duration,
        model_name=config.model_name,
        language=config.language,
        device=config.device,
        compute_type=config.compute_type,
        beam_size=config.beam_size,
        vad_filter=config.vad_filter,
        model_dir=DEFAULT_MODEL_DIR,
    )
    
    # Exports
    formats = [f.strip().lower() for f in args.formats.split(",")]
    print("\n[export] Génération des fichiers...")
    
    if "txt" in formats:
        write_txt(segments, out_dir / "transcript.txt")
    if "srt" in formats:
        write_srt(segments, out_dir / "transcript.srt")
    if "vtt" in formats:
        write_vtt(segments, out_dir / "transcript.vtt")
    if "json" in formats:
        write_json(segments, info, out_dir / "segments.json")
    
    # Nettoyage audio temp
    if use_temp_audio and wav_path.exists():
        wav_path.unlink()
    
    # Stats finales
    print_final_stats(segments, audio_duration, info, progress_stats)
    
    print(f"\n📁 Fichiers dans / Files in: {out_dir}\n")


if __name__ == "__main__":
    main()
