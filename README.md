# Transcription Audio/Video | Audio/Video Transcription (Whisper local)

**FR** : Transcription audio/vidéo 100% locale et sécurisée avec [faster-whisper](https://github.com/SYSTRAN/faster-whisper).  
**EN** : 100% local and secure audio/video transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## 🔒 Confidentialité / Privacy

**FR**  
✅ **Traitement 100% local** — Aucune donnée n'est transmise à l'extérieur.  
✅ **Aucune API requise** — Fonctionne entièrement hors ligne après le téléchargement initial des modèles.  
✅ **Sécurisé pour contenu privé** — Tes fichiers audio/vidéo et transcriptions restent sur ta machine.  
✅ **Pas de télémétrie** — Aucun tracking, aucune collecte de données.

**EN**  
✅ **100% local processing** — No data is transmitted externally.  
✅ **No API required** — Works completely offline after initial model download.  
✅ **Secure for private content** — Your audio/video files and transcriptions stay on your machine.  
✅ **No telemetry** — No tracking, no data collection.

---

## 📋 Fonctionnalités / Features

**FR**  
- Transcription audio/vidéo en local (MP4, MP3, WAV, etc.)  
- Support multi-langues (FR, EN, ES, DE, etc.)  
- Export multiple formats : TXT, SRT, VTT, JSON  
- Support GPU NVIDIA (optionnel) pour accélération  
- Barre de progression avec statistiques détaillées  
- Détection automatique de langue  

**EN**  
- Local audio/video transcription (MP4, MP3, WAV, etc.)  
- Multi-language support (FR, EN, ES, DE, etc.)  
- Multiple export formats: TXT, SRT, VTT, JSON  
- NVIDIA GPU support (optional) for acceleration  
- Progress bar with detailed statistics  
- Automatic language detection  

---

## 🚀 Démarrage rapide / Quick start

### Prérequis / Requirements

**FR**  
- Python 3.10+  
- ffmpeg  

**EN**  
- Python 3.10+  
- ffmpeg  

### Installation / Installation

**FR**  
1. Clone ou télécharge ce dépôt  
2. Installe les dépendances :  
```bash
pip install -r requirements.txt
```

**EN**  
1. Clone or download this repository  
2. Install dependencies:  
```bash
pip install -r requirements.txt
```

### Utilisation / Usage

**FR**  
```bash
python3 scripts/transcribe.py --input "mon_fichier.mp4"    # macOS/Linux
# ou / or
python scripts/transcribe.py --input "mon_fichier.mp4"     # Windows
```

**EN**  
```bash
python3 scripts/transcribe.py --input "my_file.mp4"    # macOS/Linux
# or
python scripts/transcribe.py --input "my_file.mp4"      # Windows
```

Les résultats sont dans `out/<nom_du_fichier>/` / Results are in `out/<file_name>/`

---

## 📦 Installation détaillée / Detailed installation

### 1. Prérequis système / System requirements

#### 1.1 macOS

**FR**  
1. Ouvre Terminal (Spotlight > tape "Terminal")  
2. Vérifie Python (doit afficher 3.10 ou plus) :  
```bash
python3 --version
```
Si Python n'est pas installé ou version < 3.10, installe-le avec Homebrew (étapes 3-4) ou télécharge depuis https://www.python.org/downloads/  
3. Vérifie Homebrew (si une version s'affiche, c'est ok) :  
```bash
brew --version
```
4. Pas de Homebrew ? Installe-le :  
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
5. Installe Python et ffmpeg :  
```bash
brew install python3 ffmpeg
```

**EN**  
1. Open Terminal (Spotlight > type "Terminal")  
2. Check Python (should display 3.10 or higher) :  
```bash
python3 --version
```
If Python is not installed or version < 3.10, install it with Homebrew (steps 3-4) or download from https://www.python.org/downloads/  
3. Check Homebrew (if a version displays, you're good) :  
```bash
brew --version
```
4. No Homebrew? Install it :  
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
5. Install Python and ffmpeg :  
```bash
brew install python3 ffmpeg
```

#### 1.2 Ubuntu/Debian

**FR**  
1. Vérifie Python (doit afficher 3.10 ou plus) :  
```bash
python3 --version
```
2. Si Python n'est pas installé ou version < 3.10, installe-le avec ffmpeg :  
```bash
sudo apt update
sudo apt install python3 python3-pip ffmpeg
```

**EN**  
1. Check Python (should display 3.10 or higher) :  
```bash
python3 --version
```
2. If Python is not installed or version < 3.10, install it with ffmpeg :  
```bash
sudo apt update
sudo apt install python3 python3-pip ffmpeg
```

#### 1.3 Windows (PowerShell)

**FR**  
1. Vérifie Python (doit afficher 3.10 ou plus) :  
```powershell
python --version
```
Si Python n'est pas installé ou version < 3.10 :  
2. Installe Python 3.10+ depuis https://www.python.org/downloads/  
   ⚠️ **Important** : Coche "Add python.exe to PATH" pendant l'installation  
3. Vérifie que pip est installé :  
```powershell
pip --version
```
4. Installe ffmpeg via winget :  
```powershell
winget install Gyan.FFmpeg
```

**EN**  
1. Check Python (should display 3.10 or higher) :  
```powershell
python --version
```
If Python is not installed or version < 3.10 :  
2. Install Python 3.10+ from https://www.python.org/downloads/  
   ⚠️ **Important** : Check "Add python.exe to PATH" during installation  
3. Verify pip is installed :  
```powershell
pip --version
```
4. Install ffmpeg via winget :  
```powershell
winget install Gyan.FFmpeg
```

### 2. Installation du projet / Project setup

**FR**  
1. Clone ou télécharge ce dépôt  
2. Ouvre un terminal dans ce dossier  
3. Vérifie que Python et pip fonctionnent :  
```bash
python3 --version    # macOS/Linux
pip3 --version       # macOS/Linux
# ou
python --version     # Windows
pip --version        # Windows
```
4. Installe les dépendances Python :  
```bash
pip3 install -r requirements.txt    # macOS/Linux
# ou
pip install -r requirements.txt      # Windows
```

**EN**  
1. Clone or download this repository  
2. Open a terminal in this folder  
3. Verify Python and pip work :  
```bash
python3 --version    # macOS/Linux
pip3 --version       # macOS/Linux
# or
python --version     # Windows
pip --version        # Windows
```
4. Install Python dependencies :  
```bash
pip3 install -r requirements.txt    # macOS/Linux
# or
pip install -r requirements.txt      # Windows
```

#### Option GPU (NVIDIA) / Optional GPU

**FR**  
Pour accélérer avec une carte NVIDIA :  
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
Pas de GPU ? Ça marche en CPU (plus lent).

**EN**  
To accelerate with an NVIDIA GPU :  
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
No GPU? CPU works, just slower.

---

## 💡 Comment ça marche / How it works

**FR**  
1. Vérifie que les bibliothèques Python sont installées (faster-whisper, tqdm)  
2. Convertit l'audio de ta vidéo en WAV mono 16 kHz avec ffmpeg  
3. Transcrit avec le modèle Whisper choisi  
4. Écrit les fichiers dans `out/<nom>/`

**EN**  
1. Checks that Python libraries are installed (faster-whisper, tqdm)  
2. Converts your video audio to mono 16 kHz WAV with ffmpeg  
3. Transcribes with the chosen Whisper model  
4. Writes files to `out/<name>/`

---

## ⚙️ Options / Options

| Option | Défaut / Default | Description |
|--------|------------------|-------------|
| `--input`, `-i` | (requis) | Fichier à transcrire / File to transcribe |
| `--outdir`, `-o` | `out` | Dossier de sortie / Output folder |
| `--lang`, `-l` | `fr` | Langue de transcription / Transcription language |
| `--model`, `-m` | `large-v3` | Modèle Whisper / Whisper model |
| `--device`, `-d` | `auto` | `cpu`, `cuda`, ou `auto` |
| `--beam-size` | `5` | Qualité vs vitesse / Quality vs speed |
| `--no-vad` | - | Désactiver VAD / Disable Voice Activity Detection |
| `--sample` | - | Transcrire seulement N minutes / Transcribe first N minutes |
| `--formats` | `txt,srt,vtt,json` | Formats de sortie séparés par virgule / Comma-separated outputs |

### Exemples / Examples

```bash
# FR complet / Full run
python3 scripts/transcribe.py -i "reunion.mp4"    # macOS/Linux
python scripts/transcribe.py -i "reunion.mp4"       # Windows

# Test rapide 3 minutes / Quick 3-minute test
python3 scripts/transcribe.py -i "reunion.mp4" --sample 3 --model medium

# CPU uniquement, SRT seulement / CPU only, SRT only
python3 scripts/transcribe.py -i "audio.wav" -d cpu --formats srt

# Anglais avec beam-size plus haut / English with higher beam-size
python3 scripts/transcribe.py -i "interview.mp4" -l en --beam-size 10
```

---

## 🎯 Choisir un modèle / Pick a model

| Modèle | RAM approx | Pour qui ? / Best for |
|--------|------------|-----------------------|
| `small` | ~2 GB | PC léger, tests rapides / Light PCs, quick tests |
| `medium` | ~5 GB | Bon compromis / Good balance |
| `large-v3` | ~10 GB | Meilleure qualité / Best quality (défaut) |

**FR**  
Conseils : `small` si ton ordi rame ; `large-v3` pour réunions importantes.

**EN**  
Tips : use `small` on low-power PCs ; `large-v3` for important meetings.

---

## 📁 Structure de sortie / Output structure

```
out/
└── mon_fichier/
    ├── transcript.txt   # Texte brut / Raw text
    ├── transcript.srt   # Sous-titres SRT
    ├── transcript.vtt   # Sous-titres WebVTT
    └── segments.json    # Timestamps + métadonnées / Timestamps + metadata
```

---

## 💪 Pour de bons résultats / Tips for better results

**FR**  
- Audio clair => meilleurs sous-titres  
- Spécifie la langue avec `--lang` pour éviter les erreurs  
- Laisse la VAD activée par défaut pour couper les silences  
- Monte `--beam-size` (8-10) pour la précision, baisse (1-2) pour la vitesse

**EN**  
- Clean audio => better subtitles  
- Set language with `--lang` to avoid mistakes  
- Keep VAD on by default for better segmentation  
- Raise `--beam-size` (8-10) for accuracy, lower (1-2) for speed

---

## 🔧 Dépannage / Troubleshooting

**FR**  
- **"python: command not found"** ou **"python3: command not found"** : installe Python (voir section Installation détaillée)  
- **"pip: command not found"** : utilise `pip3` sur macOS/Linux, ou réinstalle Python avec l'option "Add to PATH" sur Windows  
- **"ffmpeg: command not found"** : installe ffmpeg (voir section Installation)  
- **"Python version < 3.10"** : mets à jour Python vers 3.10+  
- **Trop lent** : modèle plus petit (`--model small/medium`), `--device cuda` si GPU, baisse `--beam-size`  
- **Texte incorrect** : vérifie `--lang`, essaie un modèle plus grand (`large-v3`), augmente `--beam-size`  
- **Erreur "dépendances manquantes"** : lance `pip3 install -r requirements.txt` (macOS/Linux) ou `pip install -r requirements.txt` (Windows)

**EN**  
- **"python: command not found"** or **"python3: command not found"** : install Python (see Detailed installation section)  
- **"pip: command not found"** : use `pip3` on macOS/Linux, or reinstall Python with "Add to PATH" option on Windows  
- **"ffmpeg: command not found"** : install ffmpeg (see Installation section)  
- **"Python version < 3.10"** : update Python to 3.10+  
- **Too slow** : smaller model (`--model small/medium`), `--device cuda` if GPU, lower `--beam-size`  
- **Incorrect text** : check `--lang`, try a larger model (`large-v3`), increase `--beam-size`  
- **"Missing dependencies" error** : run `pip3 install -r requirements.txt` (macOS/Linux) or `pip install -r requirements.txt` (Windows)

---

## 📄 Licence / License

**FR**  
Ce projet utilise [faster-whisper](https://github.com/SYSTRAN/faster-whisper) qui est sous licence MIT.

**EN**  
This project uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) which is licensed under MIT.

---

## 🤝 Contribution / Contributing

**FR**  
Les contributions sont les bienvenues ! N'hésite pas à ouvrir une issue ou une pull request.

**EN**  
Contributions are welcome! Feel free to open an issue or pull request.
