# Transcription Audio/Vidéo | Audio/Video Transcription

**FR** : Transcription audio/vidéo 100% locale avec [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Aucune donnée envoyée sur internet.  
**EN** : 100% local audio/video transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No data sent to the internet.

---

## 🎯 Pour les utilisateurs / For Users

### Comment utiliser / How to use

**FR**  
**Méthode recommandée (glisser-déposer)** :  
1. **Glisse-dépose** ton fichier audio/vidéo sur `Transcrire.bat` (Windows), `Transcrire.command` (macOS) ou `Transcrire.sh` (Linux)
2. Attends que la transcription se termine
3. Récupère les fichiers dans le dossier `out/<nom_du_fichier>/`

**Méthode alternative (ligne de commande)** :  
```bash
# macOS/Linux
./Transcrire.command "fichier.mp4"
# ou
./Transcrire.sh "fichier.mp4"

# Windows
Transcrire.bat "fichier.mp4"
```

**EN**  
**Recommended method (drag and drop)** :  
1. **Drag and drop** your audio/video file onto `Transcrire.bat` (Windows), `Transcrire.command` (macOS) or `Transcrire.sh` (Linux)
2. Wait for transcription to complete
3. Find your files in `out/<file_name>/`

**Alternative method (command line)** :  
```bash
# macOS/Linux
./Transcrire.command "file.mp4"
# or
./Transcrire.sh "file.mp4"

# Windows
Transcrire.bat "file.mp4"
```

### Fichiers générés / Generated files

| Fichier / File | Description |
|----------------|-------------|
| `transcript.txt` | Texte brut / Plain text |
| `transcript.srt` | Sous-titres SRT (lecteurs vidéo) / SRT subtitles |
| `transcript.vtt` | Sous-titres WebVTT (web) / WebVTT subtitles |
| `segments.json` | Données structurées avec timestamps / Structured data with timestamps |

### Conseils / Tips

**FR**  
- **Utilise toujours les lanceurs** (`Transcrire.command`, `Transcrire.sh`, `Transcrire.bat`) — ils utilisent automatiquement le bon Python avec les dépendances
- Audio clair = meilleure transcription
- La première transcription peut prendre du temps (chargement du modèle)
- Tu peux interrompre avec `Ctrl+C` : les segments déjà faits sont conservés
- Si tu vois une erreur "dépendances manquantes", le script te dira automatiquement quelle commande utiliser

**EN**  
- **Always use the launchers** (`Transcrire.command`, `Transcrire.sh`, `Transcrire.bat`) — they automatically use the correct Python with dependencies
- Clear audio = better transcription
- First transcription may take time (model loading)
- You can interrupt with `Ctrl+C`: already processed segments are saved
- If you see a "missing dependencies" error, the script will automatically tell you which command to use

---

## 🏢 Pour l'IT / For IT

### Installation (une seule fois / one time only)

**FR**  
1. Télécharge le projet (ZIP ou `git clone`)
2. Exécute le script d'installation :

| Système | Commande |
|---------|----------|
| Windows | Double-clic sur `setup\install.bat` ou : `powershell -ExecutionPolicy Bypass -File setup\install.ps1` |
| macOS/Linux | `chmod +x setup/install.sh && ./setup/install.sh` |

3. C'est terminé. Le dossier est prêt à être distribué aux utilisateurs.

**EN**  
1. Download the project (ZIP or `git clone`)
2. Run the installation script:

| System | Command |
|--------|---------|
| Windows | Double-click `setup\install.bat` or: `powershell -ExecutionPolicy Bypass -File setup\install.ps1` |
| macOS/Linux | `chmod +x setup/install.sh && ./setup/install.sh` |

3. Done. The folder is ready to be distributed to users.

### Ce que fait le script d'installation / What the install script does

1. **Python** : Télécharge Python embeddable (Windows) ou vérifie Python 3.10+ (macOS/Linux)
2. **Environnement virtuel** : Crée `tools/venv/` avec toutes les dépendances Python
3. **ffmpeg** : Télécharge ffmpeg portable dans `tools/ffmpeg/`
4. **Modèle Whisper** : Pré-télécharge le modèle large-v3 (~6 Go) dans `models/`

### Distribution aux utilisateurs / Distribution to users

**FR**  
Copiez le dossier complet (incluant `tools/` et `models/`) sur les postes utilisateurs. Les utilisateurs n'ont besoin que de glisser-déposer leurs fichiers sur les lanceurs.

**EN**  
Copy the entire folder (including `tools/` and `models/`) to user workstations. Users only need to drag and drop files onto the launchers.

### Structure du projet après installation / Project structure after installation

```
Transcription/
├── Transcrire.bat          # Lanceur Windows / Windows launcher
├── Transcrire.command      # Lanceur macOS / macOS launcher
├── Transcrire.sh           # Lanceur Linux / Linux launcher
├── setup/
│   ├── install.bat         # Script IT Windows
│   ├── install.ps1         
│   └── install.sh          # Script IT macOS/Linux
├── tools/                  # Créé par l'installation / Created by installation
│   ├── python/             # Python embeddable (Windows uniquement)
│   ├── ffmpeg/             # ffmpeg portable
│   └── venv/               # Environnement Python avec dépendances
├── models/                 # Modèle Whisper pré-téléchargé / Pre-downloaded model
├── scripts/
│   └── transcribe.py       # Script principal
├── out/                    # Résultats des transcriptions / Transcription results
├── requirements.txt
└── README.md
```

### Configuration réseau / Network configuration

**FR**  
- L'installation nécessite un accès internet pour télécharger Python, ffmpeg et le modèle Whisper
- Après installation, **aucun accès internet n'est requis**
- Les proxies HTTP_PROXY/HTTPS_PROXY sont détectés automatiquement lors de l'installation
- Pour un réseau très restrictif : téléchargez manuellement les fichiers et placez-les dans les dossiers appropriés

**EN**  
- Installation requires internet access to download Python, ffmpeg and Whisper model
- After installation, **no internet access is required**
- HTTP_PROXY/HTTPS_PROXY proxies are automatically detected during installation
- For very restrictive networks: manually download files and place them in appropriate folders

---

## 🔒 Confidentialité / Privacy

**FR**  
✅ Traitement 100% local — Aucune donnée transmise à l'extérieur  
✅ Aucune API requise — Fonctionne hors ligne après installation  
✅ Pas de télémétrie — Aucun tracking, aucune collecte de données

**EN**  
✅ 100% local processing — No data transmitted externally  
✅ No API required — Works offline after installation  
✅ No telemetry — No tracking, no data collection

---

## ⚙️ Options avancées / Advanced options

### Utilisation directe du script Python / Direct Python script usage

**FR**  
Si tu veux utiliser directement le script Python (au lieu des lanceurs), tu dois utiliser le Python du venv local :

```bash
# macOS/Linux
tools/venv/bin/python scripts/transcribe.py --input "fichier.mp4"

# Windows
tools\venv\Scripts\python.exe scripts\transcribe.py --input "fichier.mp4"
```

**⚠️ Important** : N'utilise **pas** `python3 scripts/transcribe.py` directement — cela utilise le Python système qui n'a pas les dépendances installées. Si tu essaies, le script détectera automatiquement le venv local et t'indiquera la bonne commande à utiliser.

**EN**  
If you want to use the Python script directly (instead of the launchers), you must use the Python from the local venv:

```bash
# macOS/Linux
tools/venv/bin/python scripts/transcribe.py --input "file.mp4"

# Windows
tools\venv\Scripts\python.exe scripts\transcribe.py --input "file.mp4"
```

**⚠️ Important** : Do **not** use `python3 scripts/transcribe.py` directly — this uses the system Python which doesn't have the dependencies installed. If you try, the script will automatically detect the local venv and tell you the correct command to use.

### Options disponibles / Available options

| Option | Défaut / Default | Description |
|--------|------------------|-------------|
| `--input`, `-i` | (interactif) | Fichier à transcrire / File to transcribe |
| `--outdir`, `-o` | `out` | Dossier de sortie / Output folder |
| `--lang`, `-l` | `fr` | Langue / Language (fr, en, es, de, etc.) |
| `--model`, `-m` | `large-v3` | Modèle Whisper / Whisper model |
| `--device`, `-d` | `auto` | `cpu`, `cuda` ou `auto` |
| `--beam-size` | `5` | Qualité vs vitesse / Quality vs speed |
| `--no-vad` | - | Désactiver VAD / Disable Voice Activity Detection |
| `--sample` | - | Transcrire N premières minutes / First N minutes only |
| `--formats` | `txt,srt,vtt,json` | Formats de sortie / Output formats |

### Exemples / Examples

**FR**  
**Avec les lanceurs (recommandé)** :
```bash
# Test rapide (3 premières minutes)
./Transcrire.command "reunion.mp4" --sample 3 --model medium

# Anglais, haute qualité
./Transcrire.command "interview.mp4" --lang en --beam-size 10

# SRT uniquement
./Transcrire.command "video.mp4" --formats srt
```

**Avec le script Python directement** :
```bash
# Test rapide (3 premières minutes)
tools/venv/bin/python scripts/transcribe.py -i "reunion.mp4" --sample 3 --model medium

# Anglais, haute qualité
tools/venv/bin/python scripts/transcribe.py -i "interview.mp4" -l en --beam-size 10

# SRT uniquement
tools/venv/bin/python scripts/transcribe.py -i "video.mp4" --formats srt
```

**EN**  
**With launchers (recommended)** :
```bash
# Quick test (first 3 minutes)
./Transcrire.command "meeting.mp4" --sample 3 --model medium

# English, high quality
./Transcrire.command "interview.mp4" --lang en --beam-size 10

# SRT only
./Transcrire.command "video.mp4" --formats srt
```

**With Python script directly** :
```bash
# Quick test (first 3 minutes)
tools/venv/bin/python scripts/transcribe.py -i "meeting.mp4" --sample 3 --model medium

# English, high quality
tools/venv/bin/python scripts/transcribe.py -i "interview.mp4" -l en --beam-size 10

# SRT only
tools/venv/bin/python scripts/transcribe.py -i "video.mp4" --formats srt
```

### Choix du modèle / Model selection

| Modèle / Model | RAM | Usage recommandé / Recommended use |
|----------------|-----|-----------------------------------|
| `small` | ~2 GB | Tests rapides, PC léger / Quick tests, light PC |
| `medium` | ~5 GB | Bon compromis / Good balance |
| `large-v3` | ~10 GB | Meilleure qualité (défaut) / Best quality (default) |

---

## 🔧 Dépannage / Troubleshooting

### Problèmes courants / Common issues

| Problème / Problem | Solution |
|--------------------|----------|
| **"dépendances manquantes: faster-whisper, tqdm"** | Utilise les lanceurs (`Transcrire.command`, `Transcrire.sh`, `Transcrire.bat`) ou le Python du venv : `tools/venv/bin/python scripts/transcribe.py` |
| "Aucun fichier fourni" | Glisse un fichier sur le lanceur ou utilise `--input` |
| Transcription lente | Utilise `--model small` ou `--model medium` |
| Texte incorrect | Vérifie `--lang`, augmente `--beam-size` |
| Erreur ffmpeg | Relance l'installation IT |
| Le script ne trouve pas le modèle | Vérifie que `models/large-v3/` existe (relance l'installation si nécessaire) |

### Détection automatique du venv / Automatic venv detection

**FR**  
Si tu utilises `python3 scripts/transcribe.py` directement et que les dépendances manquent, le script détecte automatiquement le venv local dans `tools/venv/` et affiche la commande exacte à utiliser :

```
⚠️  Un environnement virtuel local a été détecté dans tools/venv/
   Utilisez-le avec: tools/venv/bin/python scripts/transcribe.py --input "fichier.mp4"
```

**EN**  
If you use `python3 scripts/transcribe.py` directly and dependencies are missing, the script automatically detects the local venv in `tools/venv/` and displays the exact command to use:

```
⚠️  A local virtual environment was detected in tools/venv/
   Use it with: tools/venv/bin/python scripts/transcribe.py --input "file.mp4"
```

### Logs et debug

Le script affiche la progression en temps réel. En cas d'erreur, le message indique généralement la cause.

---

## 📄 Licence / License

Ce projet utilise [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (licence MIT).

---

## 🤝 Contribution / Contributing

Les contributions sont les bienvenues ! / Contributions are welcome!
