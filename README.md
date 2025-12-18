# Transcription Audio/Vidéo | Audio/Video Transcription

**FR** : Transcription audio/vidéo 100% locale avec [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Aucune donnée envoyée sur internet.  
**EN** : 100% local audio/video transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No data sent to the internet.

---

## 🎯 Pour les utilisateurs / For Users

### Comment utiliser / How to use

**FR**  
1. **Glisse-dépose** ton fichier audio/vidéo sur `Transcrire.bat` (Windows), `Transcrire.command` (macOS) ou `Transcrire.sh` (Linux)
2. Attends que la transcription se termine
3. Récupère les fichiers dans le dossier `out/<nom_du_fichier>/`

**EN**  
1. **Drag and drop** your audio/video file onto `Transcrire.bat` (Windows), `Transcrire.command` (macOS) or `Transcrire.sh` (Linux)
2. Wait for transcription to complete
3. Find your files in `out/<file_name>/`

### Fichiers générés / Generated files

| Fichier / File | Description |
|----------------|-------------|
| `transcript.txt` | Texte brut / Plain text |
| `transcript.srt` | Sous-titres SRT (lecteurs vidéo) / SRT subtitles |
| `transcript.vtt` | Sous-titres WebVTT (web) / WebVTT subtitles |
| `segments.json` | Données structurées avec timestamps / Structured data with timestamps |

### Conseils / Tips

**FR**  
- Audio clair = meilleure transcription
- La première transcription peut prendre du temps (chargement du modèle)
- Tu peux interrompre avec `Ctrl+C` : les segments déjà faits sont conservés

**EN**  
- Clear audio = better transcription
- First transcription may take time (model loading)
- You can interrupt with `Ctrl+C`: already processed segments are saved

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

### Utilisation en ligne de commande / Command line usage

```bash
# Windows
python scripts\transcribe.py --input "fichier.mp4"

# macOS/Linux
python3 scripts/transcribe.py --input "fichier.mp4"
```

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

```bash
# Test rapide (3 premières minutes) / Quick test (first 3 minutes)
python scripts/transcribe.py -i "reunion.mp4" --sample 3 --model medium

# Anglais, haute qualité / English, high quality
python scripts/transcribe.py -i "interview.mp4" -l en --beam-size 10

# SRT uniquement / SRT only
python scripts/transcribe.py -i "video.mp4" --formats srt
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
| "Aucun fichier fourni" | Glisse un fichier sur le lanceur ou utilise `--input` |
| Transcription lente | Utilise `--model small` ou `--model medium` |
| Texte incorrect | Vérifie `--lang`, augmente `--beam-size` |
| Erreur ffmpeg | Relance l'installation IT |

### Logs et debug

Le script affiche la progression en temps réel. En cas d'erreur, le message indique généralement la cause.

---

## 📄 Licence / License

Ce projet utilise [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (licence MIT).

---

## 🤝 Contribution / Contributing

Les contributions sont les bienvenues ! / Contributions are welcome!
