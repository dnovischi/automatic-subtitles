# Video Subtitle Pipeline

Transcribe video audio, translate subtitles into a target language, and embed the translated subtitles into the final video.
Supports both **single-file processing** and **recursive batch processing** across folder trees.

**Pipeline:**
1. Read the reference transcript (`.txt` or `.pdf`) to guide Whisper's vocabulary
2. Transcribe the video audio with [Whisper](https://github.com/openai/whisper)
3. Translate each subtitle line with Google Translate (no API key required)
4. Burn the translated subtitles into the video with FFmpeg

For each processed video, the pipeline generates:
- an original-language SRT file
- a translated SRT file
- a subtitled output video in the target language

---

## Requirements

- Ubuntu 20.04 or later
- Python 3.8+
- An NVIDIA GPU with CUDA is optional but strongly recommended for speed

---

## Installation

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install ffmpeg python3-pip python3-venv
```

### 2. Clone or copy this project, then enter the folder

```bash
cd automatic-subtitles
```

### 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Python packages

```bash
pip install -r requirements.txt
```

This installs:
- `openai-whisper` - speech-to-text transcription
- `srt` - SRT file parsing and generation
- `deep-translator` - Google Translate wrapper (free, no API key)

PyTorch is pulled in automatically as a dependency of Whisper.

---

## Usage

### Single file

```
python automatic-subtitles.py <video> <language> [options]
```

| Argument | Description |
|----------|-------------|
| `video` | Path to the input video file (e.g. `lecture.mp4`) |
| `language` | Target language code for the subtitles (e.g. `ro`, `es`, `fr`) |

### Batch / folder

```
python automatic-subtitles.py --batch --input-dir <folder> --output-dir <folder> <language> [options]
```

All videos found recursively under `--input-dir` are processed one by one.
The folder structure is mirrored under `--output-dir`, and each output video is
renamed `<original_stem>_<language><ext>` (e.g. `lecture_ro.mp4`).

If a `.txt` or `.pdf` file with the same name as the video exists next to it,
it is used automatically as the Whisper prompt for that video. Otherwise the
global `--transcript` is used, or Whisper runs in auto-mode if neither exists.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--batch` | off | Enable batch/folder mode |
| `--input-dir` | - | *(batch)* Root folder to scan for videos recursively |
| `--output-dir` | - | *(batch)* Root folder for output; mirrors input structure |
| `--transcript` | none | Optional reference file (`.txt` or `.pdf`). Improves Whisper accuracy. Used as fallback in batch mode. |
| `--model` | `medium` | Whisper model to use. Larger models are slower but more accurate. |
| `--device` | auto | `cpu` or `cuda`. Auto-detects CUDA if available. |
| `--source-lang` | `auto` | Source language code for translation. Auto-detected by default. |
| `--output` | `<name>_<lang>.<ext>` | *(single)* Custom output video path. |

### Whisper model sizes

| Model | VRAM needed | Speed | Quality |
|-------|-------------|-------|---------|
| `tiny` | ~1 GB | fastest | basic |
| `base` | ~1 GB | fast | good |
| `small` | ~2 GB | fast | good |
| `medium` | ~5 GB | balanced | very good |
| `large` | ~10 GB | slow | excellent |
| `large-v2` | ~10 GB | slow | excellent |
| `large-v3` | ~10 GB | slow | best |

For a 7-8 GB GPU, `medium` is the largest model that fits reliably.

---

## Examples

### Single file

Translate a lecture to Romanian using the medium model on GPU:

```bash
python automatic-subtitles.py lecture.mp4 ro --model medium --device cuda
```

With a plain text transcript to guide Whisper:

```bash
python automatic-subtitles.py lecture.mp4 ro --transcript lecture.txt --model medium --device cuda
```

With a PDF (e.g. lecture slides or course notes):

```bash
python automatic-subtitles.py lecture.mp4 ro --transcript slides.pdf --model medium --device cuda
```

Translate to Spanish using a large model (requires ~10 GB VRAM):

```bash
python automatic-subtitles.py lecture.mp4 es --model large-v2 --device cuda
```

Translate to French on CPU (no GPU required):

```bash
python automatic-subtitles.py lecture.mp4 fr --model medium --device cpu
```

Save the output to a specific path:

```bash
python automatic-subtitles.py lecture.mp4 de --output lecture_german.mp4
```

### Batch / folder

Process all videos under `class01/`, mirror the folder tree into `class01_ro/`,
and translate everything to Romanian:

```bash
python automatic-subtitles.py --batch --input-dir ./class01 --output-dir ./class01_ro ro
```

Same, using the large model on GPU:

```bash
python automatic-subtitles.py --batch --input-dir ./class01 --output-dir ./class01_ro ro --model large --device cuda
```

With a global fallback transcript for videos that don't have their own:

```bash
python automatic-subtitles.py --batch --input-dir ./class01 --output-dir ./class01_ro ro --transcript ./class01/overview.txt
```

In batch mode each video can have its **own** transcript placed beside it with
the same stem:

```
class01/
  week1/
    lecture.mp4
    lecture.txt   # used automatically for lecture.mp4
  week2:
    demo.mp4      # no .txt/.pdf - falls back to --transcript or Whisper auto-mode
```

---

## Output files

### Single file

Three files are created next to the input video:

| File | Description |
|------|-------------|
| `lecture_<lang>.mp4` | Output video with subtitles burned in |
| `lecture_original.srt` | Subtitles in the original spoken language |
| `lecture_<lang>.srt` | Subtitles translated into the target language |

### Batch

The same three files are produced for every video, placed under the mirrored
folder inside `--output-dir`:

```
class01_ro/
  week1/
    lecture_ro.mp4         # video with subtitles burned in
    lecture_original.srt   # original-language subtitles
    lecture_ro.srt         # translated subtitles
  week2/
    demo_ro.mp4
    demo_original.srt
    demo_ro.srt
```

---

## Supported languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| `ro` | Romanian | `es` | Spanish |
| `fr` | French | `de` | German |
| `it` | Italian | `pt` | Portuguese |
| `nl` | Dutch | `pl` | Polish |
| `ru` | Russian | `uk` | Ukrainian |
| `zh-CN` | Chinese (Simplified) | `ja` | Japanese |
| `ar` | Arabic | `cs` | Czech |
| `tr` | Turkish | `sv` | Swedish |
| `fi` | Finnish | `hu` | Hungarian |
| `ko` | Korean | `hi` | Hindi |

Any language code supported by Google Translate will work.

---

## Troubleshooting

**CUDA out of memory**

The model is too large for your GPU. Switch to a smaller model:

```bash
python automatic-subtitles.py lecture.mp4 ro --model medium --device cuda
```

If VRAM is occupied by another process, the pipeline automatically falls back to CPU and prints a warning.

**A video in batch mode fails**

Errors on individual videos are non-fatal in batch mode. The script logs the
error and moves on to the next video so the rest of the batch still completes.

**FFmpeg not found**

```bash
sudo apt install ffmpeg
```

**Missing Python packages**

Make sure the virtual environment is activated before running the script:

```bash
source .venv/bin/activate
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Support

If you find this project useful, consider buying me a coffee!

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-dan.novischi-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/dan.novischi)
