#!/usr/bin/env python3
"""
Video Subtitle & Dubbing Pipeline

Transcribes a video using a reference transcript, translates the subtitles to a
target language, and burns them into the output video.  With --dub, the pipeline
also generates a fully dubbed video with translated speech using the original
speaker's pace and pauses extracted directly from the audio signal.

Subtitle pipeline (always):
    Read reference transcript -> Transcribe with Whisper -> Translate SRT
    -> Burn subtitles into video

Dubbing pipeline (with --dub, runs after subtitles):
    Extract VAD speech segments from audio -> Align Whisper text to those windows
    -> Translate -> TTS per segment -> Fit TTS to original duration (pace)
    -> Insert original pauses between segments -> Mix with background -> Export

Outputs (--dub mode):
    <stem>_<lang>.<ext>        - subtitled video (standard output)
    <stem>_dubbed_<lang>.<ext> - dubbed video with translated speech
    <stem>_original.srt        - original-language subtitles
    <stem>_<lang>.srt          - translated subtitles

Usage (subtitles - single file):
    python automatic-subtitles.py <video> <language> [options]

Usage (dubbing - single file):
    python automatic-subtitles.py <video> <language> --dub [options]

Usage (batch / folder):
    python automatic-subtitles.py --batch --input-dir <folder> --output-dir <folder> <language> [options]

Examples:
    python automatic-subtitles.py lecture.mp4 ro
    python automatic-subtitles.py lecture.mp4 ro --dub
    python automatic-subtitles.py lecture.mp4 ro --transcript lecture.txt
    python automatic-subtitles.py lecture.mp4 es --model large --device cuda
    python automatic-subtitles.py lecture.mp4 fr --output out.mp4
    python automatic-subtitles.py --batch --input-dir /data/courses --output-dir /data/out ro
    python automatic-subtitles.py --batch --input-dir /data/courses --output-dir /data/out ro --dub
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

# Reduce CUDA memory fragmentation before torch is imported
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# Dependency checks

def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        sys.exit(
            "Error: ffmpeg not found.\n"
            "  Install with: sudo apt install ffmpeg"
        )


def check_python_deps():
    required = [
        ("openai-whisper", "whisper"),
        ("srt",            "srt"),
        ("deep-translator","deep_translator"),
    ]
    missing = []
    for pkg, mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(
            f"Error: Missing Python packages: {', '.join(missing)}\n"
            f"  Install with: pip install {' '.join(missing)}"
        )


def detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def detect_nvenc() -> bool:
    """Return True if ffmpeg was built with h264_nvenc AND a CUDA GPU is present."""
    if detect_device() != "cuda":
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def read_transcript(path) -> str:
    """Read a reference transcript from a .txt or .pdf file.

    Returns the extracted text, or None if path is None.
    For PDFs, all pages are concatenated. Image-only (scanned) PDFs will
    produce an empty string and cause an early exit with a clear message.
    """
    if path is None:
        return None
    if not path.exists():
        sys.exit(f"Error: Transcript file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8").strip()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            sys.exit(
                "Error: pypdf is required to read PDF files.\n"
                "  Install with: pip install pypdf"
            )
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            sys.exit(
                f"Error: Could not extract text from '{path.name}'.\n"
                "  The PDF may be image-only (scanned). Use a .txt transcript instead."
            )
        return text
    sys.exit(f"Error: Unsupported transcript format '{suffix}'. Use .txt or .pdf.")


# Progress helpers

def _progress_bar(current: int, total: int, width: int = 38) -> str:
    """Return an ASCII progress bar: [=====>    ]  50.0%  25/50"""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    if filled >= width:
        bar = "=" * width
    else:
        bar = "=" * filled + ">" + " " * (width - filled - 1)
    return f"[{bar}] {pct * 100:5.1f}%  {current}/{total}"


def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds using ffprobe, or 0.0 on failure."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return 0.0


def detect_speech_start(video_path: Path, silence_thresh: int = -30,
                        min_silence_dur: float = 1.0) -> float:
    """Detect when speech actually begins in the video using ffmpeg silencedetect.

    Returns the timestamp (in seconds) where the first silence ends, i.e.
    where the voice starts.  Falls back to 0.0 if detection fails.
    """
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-af", f"silencedetect=noise={silence_thresh}dB:d={min_silence_dur}",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # silencedetect outputs to stderr
        output = result.stderr
        # Find "silence_end: <time>" - the first one is where speech begins
        match = re.search(r'silence_end:\s*([\d.]+)', output)
        if match:
            speech_start = float(match.group(1))
            return speech_start
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return 0.0


# Transcription

def transcribe(video_path: Path, model_name: str, device: str,
               initial_prompt: str = None):
    """
    Transcribe the audio track of *video_path* with Whisper.

    The *initial_prompt* (first 500 chars of the reference transcript) is fed
    to Whisper so it can disambiguate domain-specific vocabulary and improve
    accuracy without needing to run forced alignment.

    If the requested device is 'cuda' and the model doesn't fit in VRAM, the
    function automatically falls back to CPU.
    """
    import whisper
    import torch

    kwargs = {}
    if initial_prompt:
        # Whisper uses up to ~224 tokens as a prompt; 500 chars is a safe limit.
        kwargs["initial_prompt"] = initial_prompt[:500]

    def _load_and_transcribe(dev: str):
        print(f"  Loading Whisper model '{model_name}' on {dev} ...")
        model = whisper.load_model(model_name, device=dev)
        print(f"  Transcribing '{video_path.name}' ...")
        result = model.transcribe(
            str(video_path),
            # Drop segments where Whisper is not confident speech exists
            no_speech_threshold=0.5,
            # Also drop segments whose average log-probability is suspiciously low
            # (Whisper default is -1.0; -0.5 is more aggressive)
            logprob_threshold=-0.5,
            # Prevent repetition/looping hallucinations between windows
            condition_on_previous_text=False,
            # Show Whisper's built-in tqdm progress bar (one tick per 30-s chunk)
            verbose=False,
            **kwargs,
        )
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        return result["segments"], result.get("language", "en")

    if device == "cuda":
        try:
            return _load_and_transcribe("cuda")
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(
                f"\n  Warning: CUDA out of memory (GPU has {vram_total:.1f} GiB total).\n"
                f"  '{model_name}' is too large for the available VRAM.\n"
                f"  Options:\n"
                f"    - Free GPU memory from other processes, then retry\n"
                f"    - Use a smaller model:  --model medium  (needs ~5 GiB)\n"
                f"                            --model small   (needs ~2 GiB)\n"
                f"  Falling back to CPU (slower but will complete) ...\n"
            )
            return _load_and_transcribe("cpu")

    return _load_and_transcribe(device)


# Regex to detect non-speech tokens Whisper commonly hallucinates during silence or music.
# Covers: bracketed/parenthesised tags, music symbols, common filler phrases, and
# anything that is just punctuation or whitespace.
_HALLUCINATION_RE = re.compile(
    r'^[\[\(].*[\]\)]$'           # [Music], (Applause), [BLANK_AUDIO], etc.
    r'|^[\u266a\u266b\u266c\u266d\u266e\u266f\s\.,!?\-]+'  # music notes or only punctuation/whitespace
    r'|^\s*$'                       # empty / whitespace-only
    # Common Whisper filler phrases generated during silence:
    r'|^(thank you\.?|thanks\.?|thank you for watching\.?|please subscribe\.?'
    r'|like and subscribe\.?|see you (in the )?next (video|lecture|time)\.?'
    r'|subtitles by .+|transcribed by .+|www\..+|https?://.+'
    r'|\.\.\.|[\.,!?]{1,3})$',
    re.IGNORECASE,
)


def segments_to_srt(segments, speech_start: float = 0.0) -> str:
    import srt

    MIN_CHARS_PER_SEC = 2.0
    LEAD_IN_SEC = 3.0

    # First pass: filter and collect (start_sec, end_sec, text) tuples.
    filtered = []
    prev_text = None
    for seg in segments:
        if seg.get("no_speech_prob", 0.0) > 0.5:
            continue
        text = seg["text"].strip()
        if _HALLUCINATION_RE.match(text):
            continue
        if len(text) < 3:
            continue
        start_sec = max(0.0, float(seg["start"]))
        end_sec   = max(0.0, float(seg["end"]))

        # If Whisper assigned a start before the detected speech onset and the
        # segment spans across it, clamp the start to where speech really begins.
        if start_sec < speech_start < end_sec:
            start_sec = speech_start

        duration  = end_sec - start_sec
        if duration > 0 and len(text) / duration < MIN_CHARS_PER_SEC:
            continue
        if text.lower() == (prev_text or "").lower():
            continue
        prev_text = text
        filtered.append((start_sec, end_sec, text))

    # Second pass: assign display timestamps so subtitles never overlap.
    # Each subtitle ends exactly when the next one starts (or at its natural
    # end if it's the last one).  A lead-in is applied when there is a silence
    # gap of more than LEAD_IN_SEC before the next spoken segment.
    entries = []
    for i, (start_sec, end_sec, text) in enumerate(filtered):
        # Determine the earliest point the next subtitle could appear
        next_start = filtered[i + 1][0] if i + 1 < len(filtered) else None

        # Resolve display start: apply lead-in only when there is genuine silence
        prev_end = entries[-1].end.total_seconds() if entries else 0.0
        proposed_start = start_sec - LEAD_IN_SEC
        # Never display before the detected speech onset
        proposed_start = max(proposed_start, speech_start)
        if proposed_start > 0.0 and proposed_start >= prev_end:
            display_start_sec = proposed_start
        else:
            display_start_sec = max(prev_end, start_sec)

        # Cap the end at the next subtitle's display start to avoid overlap.
        # Cap also at the segment's own natural end so we don't overshoot.
        if next_start is not None:
            next_display_start = max(next_start - LEAD_IN_SEC, next_start)
            display_end_sec = min(end_sec, next_display_start)
        else:
            display_end_sec = end_sec

        # Ensure non-zero duration
        if display_end_sec <= display_start_sec:
            display_end_sec = display_start_sec + max(duration, 1.0)

        entries.append(srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=display_start_sec),
            end=timedelta(seconds=display_end_sec),
            content=text,
        ))

    return srt.compose(entries)


# DeepL target language code mapping (differs from Google/ISO in a few places)
_DEEPL_LANG_MAP = {
    "zh-CN": "ZH",   # DeepL uses ZH for Simplified Chinese
    "zh-TW": "ZH-HANT",
    "pt":    "PT-BR",  # Default to Brazilian Portuguese
}

# NLLB-200 BCP-47 language codes (used by facebook/nllb-200-* models)
_NLLB_LANG_MAP = {
    "en":    "eng_Latn",
    "ro":    "ron_Latn",
    "es":    "spa_Latn",
    "fr":    "fra_Latn",
    "de":    "deu_Latn",
    "it":    "ita_Latn",
    "pt":    "por_Latn",
    "nl":    "nld_Latn",
    "pl":    "pol_Latn",
    "ru":    "rus_Cyrl",
    "uk":    "ukr_Cyrl",
    "zh-CN": "zho_Hans",
    "zh-TW": "zho_Hant",
    "ja":    "jpn_Jpan",
    "ar":    "arb_Arab",
    "cs":    "ces_Latn",
    "tr":    "tur_Latn",
    "sv":    "swe_Latn",
    "fi":    "fin_Latn",
    "hu":    "hun_Latn",
    "ko":    "kor_Hang",
    "hi":    "hin_Deva",
}


# Translation

def translate_srt(srt_content: str, source_lang: str, target_lang: str,
                  translator: str = "google", deepl_key: str = None,
                  local_model: str = "facebook/nllb-200-distilled-600M",
                  device: str = "cpu") -> str:
    import srt

    subs  = list(srt.parse(srt_content))
    total = len(subs)
    translated_subs = []

    if translator == "local":
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            sys.exit(
                "Error: 'transformers' and 'sentencepiece' are required for --translator local.\n"
                "  Install with: pip install transformers sentencepiece"
            )
        src_nllb = _NLLB_LANG_MAP.get(source_lang, "eng_Latn")  # default to English
        tgt_nllb = _NLLB_LANG_MAP.get(target_lang)
        if not tgt_nllb:
            sys.exit(f"Error: No NLLB language code for target '{target_lang}'. "
                     "Use --translator google instead.")
        print(f"  Loading local model '{local_model}' on {device} ...")
        _dev = torch.device("cuda" if device == "cuda" else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(local_model)
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(local_model).to(_dev)
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
        for i, sub in enumerate(subs, 1):
            try:
                inputs = tokenizer(sub.content, return_tensors="pt",
                                   padding=True, truncation=True,
                                   src_lang=src_nllb).to(_dev)
                with torch.no_grad():
                    output_ids = nllb_model.generate(
                        **inputs,
                        forced_bos_token_id=tgt_lang_id,
                        max_length=512,
                    )
                sub.content = tokenizer.decode(output_ids[0], skip_special_tokens=True) or sub.content
            except Exception as exc:
                print(f"\n  Warning: subtitle {i} not translated: {exc}")
            translated_subs.append(sub)
            print(f"\r  {_progress_bar(i, total)}", end="", flush=True)
        del nllb_model

    elif translator == "deepl":
        try:
            import deepl as deepl_lib
        except ImportError:
            sys.exit(
                "Error: 'deepl' package is required for --translator deepl.\n"
                "  Install with: pip install deepl"
            )
        if not deepl_key:
            sys.exit(
                "Error: DeepL API key is required.\n"
                "  Pass it with --deepl-key KEY or set the DEEPL_AUTH_KEY environment variable."
            )
        deepl_translator = deepl_lib.Translator(deepl_key)
        deepl_target = _DEEPL_LANG_MAP.get(target_lang, target_lang.upper())
        deepl_source = None if source_lang == "auto" else source_lang.upper()

        for i, sub in enumerate(subs, 1):
            try:
                result      = deepl_translator.translate_text(
                    sub.content, source_lang=deepl_source, target_lang=deepl_target
                )
                sub.content = result.text or sub.content
            except Exception as exc:
                print(f"\n  Warning: subtitle {i} not translated: {exc}")
            translated_subs.append(sub)
            print(f"\r  {_progress_bar(i, total)}", end="", flush=True)

    else:
        from deep_translator import GoogleTranslator
        google_translator = GoogleTranslator(source=source_lang, target=target_lang)

        for i, sub in enumerate(subs, 1):
            try:
                translated      = google_translator.translate(sub.content)
                sub.content     = translated or sub.content
            except Exception as exc:
                print(f"\n  Warning: subtitle {i} not translated: {exc}")
            translated_subs.append(sub)
            print(f"\r  {_progress_bar(i, total)}", end="", flush=True)

    print()  # newline after progress bar
    return srt.compose(translated_subs)


# Subtitle burning

def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path):
    """
    Burn the SRT file into the video via the FFmpeg 'subtitles' filter.

    The SRT is copied to a temp directory with a plain ASCII filename so that
    the FFmpeg filtergraph never sees special characters or spaces in the path.
    When an NVIDIA GPU with NVENC support is detected the video stream is
    encoded with h264_nvenc instead of the default software encoder.
    """
    duration = get_video_duration(video_path)
    use_nvenc = detect_nvenc()
    video_codec = ["h264_nvenc", "-rc", "vbr", "-cq", "23"] if use_nvenc else ["libx264", "-crf", "23", "-preset", "fast"]
    if use_nvenc:
        print("  GPU detected — using h264_nvenc for encoding.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_srt = Path(tmpdir) / "subtitles.srt"
        shutil.copy(srt_path, tmp_srt)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles={tmp_srt}",
            "-c:v", *video_codec,
            "-c:a", "copy",
            "-progress", "pipe:1",
            "-nostats",
            str(output_path),
        ]

        print("  Burning subtitles ...")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    elapsed_s = int(line.split("=", 1)[1]) / 1_000_000
                    if duration > 0:
                        print(
                            f"\r  {_progress_bar(int(elapsed_s), int(duration))}",
                            end="", flush=True,
                        )
                except ValueError:
                    pass
            elif line == "progress=end":
                total_s = int(duration) if duration > 0 else 1
                print(f"\r  {_progress_bar(total_s, total_s)}", flush=True)

        proc.wait()
        stderr_output = proc.stderr.read()
        print()  # newline after progress bar

    if proc.returncode != 0:
        print("  FFmpeg stderr (last 3000 chars):")
        print(stderr_output[-3000:])
        sys.exit(f"Error: FFmpeg failed with return code {proc.returncode}")


# Dubbing pipeline

# Mapping of language codes to edge-tts voice names (one per supported language)
EDGE_TTS_VOICES = {
    "ro": "ro-RO-EmilNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "it": "it-IT-DiegoNeural",
    "pt": "pt-BR-AntonioNeural",
    "nl": "nl-NL-MaartenNeural",
    "pl": "pl-PL-MarekNeural",
    "ru": "ru-RU-DmitryNeural",
    "uk": "uk-UA-OstapNeural",
    "zh-CN": "zh-CN-YunxiNeural",
    "ja": "ja-JP-KeitaNeural",
    "ar": "ar-SA-HamedNeural",
    "cs": "cs-CZ-AntoninNeural",
    "tr": "tr-TR-AhmetNeural",
    "sv": "sv-SE-MattiasNeural",
    "fi": "fi-FI-HarriNeural",
    "hu": "hu-HU-TamasNeural",
    "ko": "ko-KR-InJoonNeural",
    "hi": "hi-IN-MadhurNeural",
    "en": "en-US-GuyNeural",
}


def check_dubbing_deps():
    """Check that dubbing-specific Python packages are installed."""
    missing = []
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        missing.append("edge-tts")
    try:
        from pydub import AudioSegment  # noqa: F401
    except ImportError:
        missing.append("pydub")
    if missing:
        sys.exit(
            f"Error: Missing dubbing dependencies: {', '.join(missing)}\n"
            f"  Install with: pip install {' '.join(missing)}"
        )


async def _tts_one_segment(text: str, voice: str, output_path: Path, retries: int = 4):
    """Generate speech for one segment using edge-tts at natural rate.
    Retries up to *retries* times with exponential backoff on transient errors.
    """
    import asyncio
    import edge_tts
    last_exc = None
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output_path))
            return
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s ...
                await asyncio.sleep(wait)
    raise last_exc


def get_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds using ffprobe, or 0.0 on failure."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return 0.0


def _build_atempo_chain(ratio: float) -> str:
    """Build an ffmpeg atempo filter chain for the given speed ratio.
    A single atempo filter supports 0.5-2.0; chain multiple filters for larger ratios.
    """
    if ratio <= 0:
        return "atempo=1.0"
    filters = []
    remaining = ratio
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    if remaining < 0.5:
        remaining = 0.5
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def _filter_whisper_for_dubbing(segments_raw: list, speech_start: float = 0.0) -> list:
    """Return filtered Whisper segments with their real speech timestamps.

    Applies the same quality filters as segments_to_srt (hallucination removal,
    low-confidence skip, deduplication) but keeps the original Whisper start/end
    times - not the display-timing adjustments used in subtitles.

    Returns list of ``{start, end}`` dicts (text is not needed here; translated
    text comes from the SRT).
    """
    MIN_CHARS_PER_SEC = 2.0
    out = []
    prev_text = None
    for seg in segments_raw:
        if seg.get("no_speech_prob", 0.0) > 0.5:
            continue
        text = seg["text"].strip()
        if _HALLUCINATION_RE.match(text):
            continue
        if len(text) < 3:
            continue
        start_sec = max(0.0, float(seg["start"]))
        end_sec   = max(0.0, float(seg["end"]))
        if start_sec < speech_start < end_sec:
            start_sec = speech_start
        if end_sec <= speech_start:
            continue
        duration = end_sec - start_sec
        if duration > 0 and len(text) / duration < MIN_CHARS_PER_SEC:
            continue
        if text.lower() == (prev_text or "").lower():
            continue
        prev_text = text
        out.append({"start": start_sec, "end": end_sec})
    return out


def _group_speech_entries(whisper_timing: list, translated_texts: list,
                          merge_gap_s: float = 1.5) -> list:
    """Group speech entries using real Whisper gaps, with translated SRT text.

    *whisper_timing* - list of ``{start, end}`` from the original audio (real
                       pauses, not SRT display timing).
    *translated_texts* - parallel list of translated strings from the SRT.

    Consecutive entries whose Whisper gap (prev.end -> next.start) is <=
    *merge_gap_s* are merged into one TTS group.  This eliminates micro-pauses
    while preserving meaningful silence between topic blocks.

    Returns list of groups:
        start_ms  - Whisper start of first entry in group (for timeline placement)
        window_ms - time until next group's Whisper start (or None for last)
        text      - merged translated text for TTS
    """
    n = min(len(whisper_timing), len(translated_texts))
    if n == 0:
        return []

    groups = []
    cur_start = whisper_timing[0]["start"]
    cur_end   = whisper_timing[0]["end"]
    cur_texts = [translated_texts[0]]

    for idx in range(1, n):
        gap = whisper_timing[idx]["start"] - whisper_timing[idx - 1]["end"]
        if gap <= merge_gap_s:
            cur_end = whisper_timing[idx]["end"]
            cur_texts.append(translated_texts[idx])
        else:
            groups.append({
                "start_ms": int(cur_start * 1000),
                "end_ms":   int(cur_end   * 1000),
                "text":     " ".join(cur_texts),
            })
            cur_start = whisper_timing[idx]["start"]
            cur_end   = whisper_timing[idx]["end"]
            cur_texts = [translated_texts[idx]]

    groups.append({
        "start_ms": int(cur_start * 1000),
        "end_ms":   int(cur_end   * 1000),
        "text":     " ".join(cur_texts),
    })

    for i, grp in enumerate(groups):
        grp["window_ms"] = (
            groups[i + 1]["start_ms"] - grp["start_ms"]
            if i + 1 < len(groups) else None
        )

    return groups


def build_dubbed_track_from_srt(trans_srt: str, whisper_timing: list,
                                 target_lang: str, total_duration: float,
                                 work_dir: Path, merge_gap_s: float = 1.5) -> Path:
    """Build dubbed audio track from translated SRT text + Whisper timestamps.

    Strategy:
    1. Use real Whisper speech gaps (not SRT display timing) to group
       consecutive entries.  Gaps <= merge_gap_s are merged into one TTS
       group - eliminating choppy micro-pauses between subtitle lines.
    2. Generate TTS for each group at the voice's natural speaking rate.
    3. Place the clip at the Whisper start time of the group.
    4. If the clip fits within the window to the next group, play at natural
       rate - silence fills the remaining gap.
    5. If the clip overruns, apply the minimum speedup needed, capped at 1.2x
       so speech stays natural; a small overflow into the gap is accepted.

    Returns path to dubbed_voice.wav.
    """
    import srt
    from pydub import AudioSegment

    subs = list(srt.parse(trans_srt))
    if not subs or not whisper_timing:
        return None

    translated_texts = [s.content.strip() for s in subs]

    voice = EDGE_TTS_VOICES.get(target_lang)
    if not voice:
        print(f"  Warning: No default voice for '{target_lang}', falling back to English")
        voice = EDGE_TTS_VOICES["en"]

    tts_dir = work_dir / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    groups = _group_speech_entries(whisper_timing, translated_texts,
                                   merge_gap_s=merge_gap_s)
    print(f"  {len(groups)} speech group(s) from {len(subs)} subtitle entries "
          f"(merge gap <= {merge_gap_s}s)")

    timeline = AudioSegment.silent(duration=int(total_duration * 1000), frame_rate=44100)
    total = len(groups)

    for i, grp in enumerate(groups):
        start_ms  = grp["start_ms"]
        window_ms = grp["window_ms"]  # None for last group

        mp3_path = tts_dir / f"seg_{i:04d}.mp3"
        wav_path = tts_dir / f"seg_{i:04d}.wav"

        asyncio.run(_tts_one_segment(grp["text"], voice, mp3_path))
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path),
             "-ar", "44100", "-ac", "1", str(wav_path)],
            capture_output=True, timeout=30,
        )

        actual_ms = int(get_audio_duration(wav_path) * 1000)
        clip_path = wav_path

        if window_ms is not None and actual_ms > window_ms and window_ms > 0:
            # Speed up just enough to fit - cap at 1.2x to stay natural
            ratio = min(actual_ms / window_ms, 1.2)
            af = _build_atempo_chain(ratio)
            fitted_path = tts_dir / f"seg_{i:04d}_fitted.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path),
                 "-filter:a", af, "-ar", "44100", "-ac", "1", str(fitted_path)],
                capture_output=True, timeout=30,
            )
            clip_path = fitted_path

        try:
            clip = AudioSegment.from_wav(str(clip_path))
            if start_ms + len(clip) > len(timeline):
                clip = clip[: len(timeline) - start_ms]
            timeline = timeline.overlay(clip, position=start_ms)
        except Exception as exc:
            print(f"\n  Warning: skipping group {i + 1}: {exc}")

        print(f"\r  {_progress_bar(i + 1, total)}", end="", flush=True)

    print()

    dubbed_path = work_dir / "audio" / "dubbed_voice.wav"
    dubbed_path.parent.mkdir(parents=True, exist_ok=True)
    timeline.export(str(dubbed_path), format="wav")
    return dubbed_path


def extract_audio_from_video(video_path: Path, work_dir: Path) -> Path:
    """Extract the full audio track from a video as 44.1 kHz mono WAV."""
    audio_path = work_dir / "audio" / "original_audio.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le",
        str(audio_path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=300)
    return audio_path


def extract_background_audio(original_audio_path: Path, work_dir: Path) -> Path:
    """
    Attempt to separate background/ambient audio from vocals.
    Strategy 1: demucs (high quality, if installed)
    Strategy 2: attenuate original audio (simple fallback)
    """
    background_path = work_dir / "audio" / "background.wav"

    # Try demucs for proper source separation
    try:
        demucs_output = work_dir / "audio" / "demucs"
        result = subprocess.run(
            [sys.executable, "-m", "demucs",
             "--two-stems", "vocals",
             "-o", str(demucs_output),
             str(original_audio_path)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            no_vocals = list(demucs_output.rglob("no_vocals.wav"))
            if no_vocals:
                shutil.copy(no_vocals[0], background_path)
                print("  Background extracted with demucs (high quality)")
                return background_path
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: attenuate the original audio to ~15 % volume
    print("  Using attenuated original audio as background (install demucs for better quality)")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(original_audio_path),
         "-filter:a", "volume=0.15",
         "-ar", "44100", "-ac", "1",
         str(background_path)],
        capture_output=True, timeout=120,
    )
    return background_path


def mix_dubbed_and_background(dubbed_path: Path, background_path: Path,
                              work_dir: Path) -> Path:
    """
    Mix dubbed speech with background audio.
    Background is weighted at 30 %, then the mix is loudness-normalized
    to broadcast standard (EBU R128, -16 LUFS).
    """
    final_mix_path = work_dir / "audio" / "final_mix.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(dubbed_path),
        "-i", str(background_path),
        "-filter_complex",
        "[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=mono[speech];"
        "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=mono[bg];"
        "[speech][bg]amix=inputs=2:weights=1 0.3:duration=longest:normalize=0[mixed];"
        "[mixed]loudnorm=I=-16:TP=-1.5:LRA=11[out]",
        "-map", "[out]",
        "-ar", "44100", "-ac", "1",
        str(final_mix_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        # Fallback: simple mix without loudnorm
        cmd = [
            "ffmpeg", "-y",
            "-i", str(dubbed_path),
            "-i", str(background_path),
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:weights=1 0.3:duration=longest[out]",
            "-map", "[out]",
            "-ar", "44100", "-ac", "1",
            str(final_mix_path),
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)

    return final_mix_path


def mux_audio_into_video(video_path: Path, audio_path: Path, output_path: Path):
    """Replace the video's audio track with the final dubbed audio."""
    duration = get_video_duration(video_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-progress", "pipe:1",
        "-nostats",
        str(output_path),
    ]

    print("  Muxing dubbed audio into video ...")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us="):
            try:
                elapsed_s = int(line.split("=", 1)[1]) / 1_000_000
                if duration > 0:
                    print(
                        f"\r  {_progress_bar(int(elapsed_s), int(duration))}",
                        end="", flush=True,
                    )
            except ValueError:
                pass
        elif line == "progress=end":
            total_s = int(duration) if duration > 0 else 1
            print(f"\r  {_progress_bar(total_s, total_s)}", flush=True)

    proc.wait()
    stderr_output = proc.stderr.read()
    print()

    if proc.returncode != 0:
        print("  FFmpeg stderr (last 3000 chars):")
        print(stderr_output[-3000:])
        sys.exit(f"Error: FFmpeg muxing failed with return code {proc.returncode}")


def process_one_video_dub(
    video_path: Path,
    language: str,
    model_name: str,
    device: str,
    source_lang: str,
    reference_text,
    output_path: Path,
    subtitle_output_path: Path,
    merge_gap_s: float = 1.5,
    translator: str = "google",
    deepl_key: str = None,
    local_model: str = "facebook/nllb-200-distilled-600M",
):
    """Run the complete subtitle + dubbing pipeline on a single video.

    Subtitle pipeline (always runs first, identical to normal mode):
        Transcribe -> generate SRT -> translate SRT -> burn subtitles

    Dubbing pipeline (uses the translated SRT timings directly):
        For each translated subtitle segment, generate TTS at natural rate,
        place it at the SRT start time, apply minimal speedup (<=1.3x) only
        if the clip would overlap the next segment's start - otherwise let
        it play at full natural rate with silence filling the gap.

    Outputs:
        *subtitle_output_path*  - video with burned-in translated subtitles
        *output_path*           - dubbed video with translated speech
        ``<stem>_original.srt`` - original-language SRT
        ``<stem>_<lang>.srt``   - translated SRT
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_output_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = output_path.parent / f".dub_work_{video_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    srt_orig_path  = output_path.parent / f"{video_path.stem}_original.srt"
    srt_trans_path = output_path.parent / f"{video_path.stem}_{language}.srt"

    total_duration = get_video_duration(video_path)
    N = 6

    # Step 1: Detect speech onset
    print(f"\n[1/{N}] Detecting speech start ...")
    speech_start = detect_speech_start(video_path)
    print(f"  Speech begins at: {speech_start:.2f}s")

    # Step 2: Transcribe
    print(f"\n[2/{N}] Transcribing video with Whisper ...")
    segments_raw, detected_lang = transcribe(
        video_path, model_name, device, initial_prompt=reference_text,
    )
    print(f"  Detected language: {detected_lang}")

    # Subtitle pipeline

    # Step 3: Build original SRT, translate it, burn it
    print(f"\n[3/{N}] Building and translating subtitles ...")
    orig_srt = segments_to_srt(segments_raw, speech_start=speech_start)
    srt_orig_path.write_text(orig_srt, encoding="utf-8")
    n_subs = orig_srt.count("\n\n")
    print(f"  {n_subs} subtitle segments - saved: {srt_orig_path.name}")

    trans_srt = translate_srt(orig_srt, source_lang, language, translator=translator,
                              deepl_key=deepl_key, local_model=local_model, device=device)
    srt_trans_path.write_text(trans_srt, encoding="utf-8")
    print(f"  Translated SRT saved: {srt_trans_path.name}")

    print(f"\n[4/{N}] Burning subtitles into video ...")
    burn_subtitles(video_path, srt_trans_path, subtitle_output_path)
    print(f"  Saved: {subtitle_output_path}")

    # Dubbing pipeline

    # Step 5: Extract real Whisper speech timing, then generate TTS grouped by
    #         natural speech pauses.  Short gaps between subtitles are merged
    #         so TTS flows as continuous speech rather than choppy clips.
    print(f"\n[5/{N}] Generating dubbed speech from translated SRT ...")
    whisper_timing = _filter_whisper_for_dubbing(segments_raw, speech_start)
    dubbed_track = build_dubbed_track_from_srt(
        trans_srt, whisper_timing, language, total_duration, work_dir,
        merge_gap_s=merge_gap_s,
    )

    # Step 6: Mix with background audio and export dubbed video
    print(f"\n[6/{N}] Mixing with background audio and exporting ...")
    original_audio = extract_audio_from_video(video_path, work_dir)
    background     = extract_background_audio(original_audio, work_dir)
    final_audio    = mix_dubbed_and_background(dubbed_track, background, work_dir)
    mux_audio_into_video(video_path, final_audio, output_path)
    print(f"  Saved: {output_path}")

    # Cleanup work directory
    try:
        shutil.rmtree(work_dir)
    except OSError:
        pass

    print("\n  Artifacts:")
    print(f"    {srt_orig_path}")
    print(f"    {srt_trans_path}")
    print(f"    {subtitle_output_path}")
    print(f"    {output_path}")


# Known video file extensions processed in batch mode
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".flv", ".wmv", ".m4v", ".ts", ".mts",
}


def process_one_video(
    video_path: Path,
    language: str,
    model_name: str,
    device: str,
    source_lang: str,
    reference_text,
    output_path: Path,
    translator: str = "google",
    deepl_key: str = None,
    local_model: str = "facebook/nllb-200-distilled-600M",
):
    """Transcribe, translate, and burn subtitles into a single video.

    SRT files are saved alongside *output_path* and named after the
    original video stem so they are easy to identify.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srt_orig_path  = output_path.parent / f"{video_path.stem}_original.srt"
    srt_trans_path = output_path.parent / f"{video_path.stem}_{language}.srt"

    # Detect when the speaker actually starts talking
    speech_start = detect_speech_start(video_path)
    if speech_start > 0:
        print(f"  Speech detected at: {speech_start:.2f}s")

    # Step 2: Transcribe
    print("\n[2/4] Transcribing video with Whisper ...")
    segments, detected_lang = transcribe(
        video_path, model_name, device, initial_prompt=reference_text
    )

    orig_srt = segments_to_srt(segments, speech_start=speech_start)
    srt_orig_path.write_text(orig_srt, encoding="utf-8")
    n_subs = orig_srt.count("\n\n")
    print(f"  Detected language : {detected_lang}")
    print(f"  Subtitle segments : {n_subs}")
    print(f"  Saved             : {srt_orig_path.name}")

    # Step 3: Translate
    print(f"\n[3/4] Translating subtitles to '{language}' ...")
    trans_srt = translate_srt(orig_srt, source_lang, language, translator=translator,
                              deepl_key=deepl_key, local_model=local_model, device=device)
    srt_trans_path.write_text(trans_srt, encoding="utf-8")
    print(f"  Saved: {srt_trans_path.name}")

    # Step 4: Burn subtitles
    print("\n[4/4] Burning subtitles into video ...")
    burn_subtitles(video_path, srt_trans_path, output_path)
    print(f"  Saved: {output_path}")

    print("\n  SRT files:")
    print(f"    {srt_orig_path}")
    print(f"    {srt_trans_path}")


def batch_process(
    input_dir: Path,
    output_dir: Path,
    language: str,
    model_name: str,
    device: str,
    source_lang: str,
    transcript_path,
    dub: bool = False,
    merge_gap_s: float = 1.5,
    translator: str = "google",
    deepl_key: str = None,
    local_model: str = "facebook/nllb-200-distilled-600M",
):
    """Walk *input_dir* recursively, process every video file, and write
    results to a mirrored structure under *output_dir*.

    Each output video is named ``<original_stem>_<language><ext>``.
    SRT files are placed next to the output video.
    """
    video_files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        sys.exit(f"Error: No video files found under '{input_dir}'")

    print(f"  Found {len(video_files)} video(s) to process.")

    # Read the global transcript once if provided (fallback for all videos)
    global_reference_text = read_transcript(transcript_path)
    if global_reference_text:
        print(f"  Global transcript: {len(global_reference_text)} chars from '{transcript_path.name}'")
    else:
        print("  No global transcript provided - will look for per-video transcripts")

    for idx, video_path in enumerate(video_files, 1):
        # Mirror the relative path inside input_dir into output_dir
        rel = video_path.relative_to(input_dir)
        if dub:
            out_name = f"{video_path.stem}_dubbed_{language}{video_path.suffix}"
        else:
            out_name = f"{video_path.stem}_{language}{video_path.suffix}"
        output_path = output_dir / rel.parent / out_name

        # Auto-detect a co-located transcript: same stem, .txt preferred over .pdf
        per_video_transcript: Path | None = None
        for ext in (".txt", ".pdf"):
            candidate = video_path.with_suffix(ext)
            if candidate.is_file():
                per_video_transcript = candidate
                break

        if per_video_transcript:
            reference_text = read_transcript(per_video_transcript)
            transcript_note = f"per-video transcript '{per_video_transcript.name}'"
        else:
            reference_text = global_reference_text
            transcript_note = (
                f"global transcript '{transcript_path.name}'" if global_reference_text
                else "no transcript (Whisper auto-mode)"
            )

        print("\n" + "=" * 60)
        print(f"  [{idx}/{len(video_files)}] {rel}")
        print("=" * 60)
        print(f"  Output     : {output_path}")
        print(f"  Model      : {model_name}   Device: {device}")
        print(f"  Transcript : {transcript_note}")

        try:
            if dub:
                subtitle_output_path = output_dir / rel.parent / f"{video_path.stem}_{language}{video_path.suffix}"
                process_one_video_dub(
                    video_path=video_path,
                    language=language,
                    model_name=model_name,
                    device=device,
                    source_lang=source_lang,
                    reference_text=reference_text,
                    output_path=output_path,
                    subtitle_output_path=subtitle_output_path,
                    merge_gap_s=merge_gap_s,
                    translator=translator,
                    deepl_key=deepl_key,
                    local_model=local_model,
                )
            else:
                process_one_video(
                    video_path=video_path,
                    language=language,
                    model_name=model_name,
                    device=device,
                    source_lang=source_lang,
                    reference_text=reference_text,
                    output_path=output_path,
                    translator=translator,
                    deepl_key=deepl_key,
                    local_model=local_model,
                )
        except SystemExit as exc:
            # Non-fatal in batch mode: log the error and continue with next video
            print(f"\n  ERROR processing '{video_path}': {exc}")
            print("  Skipping to next video ...")
            continue

    print("\n" + "=" * 60)
    print(f"  Batch complete. {len(video_files)} video(s) processed.")
    print("=" * 60)


# CLI

EPILOG = """
Language codes:
  ro  Romanian    es  Spanish     fr  French      de  German      it  Italian
  pt  Portuguese  nl  Dutch       pl  Polish       ru  Russian     uk  Ukrainian
  zh-CN Chinese   ja  Japanese    ar  Arabic       cs  Czech       tr  Turkish
  sv  Swedish     fi  Finnish     hu  Hungarian   ko  Korean      hi  Hindi

Examples (single file):
  python automatic-subtitles.py lecture.mp4 ro
  python automatic-subtitles.py lecture.mp4 ro --transcript lecture.txt
  python automatic-subtitles.py lecture.mp4 ro --transcript lecture.pdf
  python automatic-subtitles.py lecture.mp4 es --model large --device cuda
  python automatic-subtitles.py lecture.mp4 fr

Examples (dubbing):
  python automatic-subtitles.py lecture.mp4 ro --dub
  python automatic-subtitles.py lecture.mp4 es --dub --model large --device cuda
  python automatic-subtitles.py --batch --input-dir /data/courses --output-dir /data/out ro --dub

Examples (batch / folder):
  python automatic-subtitles.py --batch --input-dir /data/courses --output-dir /data/out ro
  python automatic-subtitles.py --batch --input-dir /data/courses --output-dir /data/out ro --model large --device cuda
"""


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe, translate, and embed subtitles into a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )

    # batch vs single-file mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--batch", action="store_true",
        help="Batch mode: process all videos under --input-dir recursively."
    )

    # positional argument (single-file mode)
    parser.add_argument("video", nargs="?", default=None,
                        help="Input video file path  (single-file mode)")

    # batch mode arguments
    parser.add_argument("--input-dir", default=None, metavar="DIR",
                        help="[batch] Root folder to scan for video files recursively")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="[batch] Root folder for output; mirrors input folder structure")

    # shared options
    parser.add_argument("language",
                        help="Target subtitle language code  (e.g. ro, es, fr)")
    parser.add_argument("--transcript", default=None, metavar="FILE",
                        help="Optional reference transcript (.txt or .pdf). "
                             "Improves Whisper accuracy for domain-specific vocabulary.")
    parser.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium",
                                 "large", "large-v2", "large-v3", "turbo"],
                        help="Whisper model to use  (default: medium)")
    parser.add_argument("--device", default=None,
                        help="Device: cpu or cuda  (default: auto-detect)")
    parser.add_argument("--source-lang", default="auto",
                        help="Source language for translation  (default: auto)")
    parser.add_argument("--output", default=None,
                        help="[single] Output video path  (default: <input>_<lang>.<ext>)")
    parser.add_argument("--dub", action="store_true",
                        help="Dubbing mode: generate a dubbed video with translated "
                             "speech IN ADDITION to the normal subtitled video. "
                             "Requires: pip install edge-tts pydub")
    parser.add_argument("--merge-gap", type=float, default=1.5, metavar="SEC",
                        help="[dub] Max pause (seconds) between Whisper segments that "
                             "still get merged into one TTS block. Lower = more breaks "
                             "preserved; higher = smoother but less sync.  (default: 1.5)")
    parser.add_argument("--translator", default="google", choices=["google", "deepl", "local"],
                        help="Translation engine to use  (default: google). "
                             "'local' runs facebook/nllb-200 on your GPU, no API key needed.")
    parser.add_argument("--deepl-key", default=None, metavar="KEY",
                        help="DeepL API auth key. Required when --translator deepl. "
                             "Can also be set via the DEEPL_AUTH_KEY environment variable.")
    parser.add_argument("--local-model", default="facebook/nllb-200-distilled-600M",
                        metavar="MODEL",
                        help="HuggingFace model for --translator local  "
                             "(default: facebook/nllb-200-distilled-600M). "
                             "Larger: facebook/nllb-200-distilled-1.3B")

    args = parser.parse_args()

    check_ffmpeg()
    check_python_deps()
    if args.dub:
        check_dubbing_deps()

    device = args.device or detect_device()
    transcript_path = Path(args.transcript).resolve() if args.transcript else None
    merge_gap = args.merge_gap
    translator = args.translator
    deepl_key  = args.deepl_key or os.environ.get("DEEPL_AUTH_KEY")
    local_model = args.local_model
    if translator == "deepl" and not deepl_key:
        parser.error(
            "--translator deepl requires an API key. "
            "Pass --deepl-key KEY or set the DEEPL_AUTH_KEY environment variable."
        )

    # batch mode
    if args.batch:
        if not args.input_dir:
            parser.error("--batch requires --input-dir")
        if not args.output_dir:
            parser.error("--batch requires --output-dir")

        input_dir  = Path(args.input_dir).resolve()
        output_dir = Path(args.output_dir).resolve()

        if not input_dir.is_dir():
            sys.exit(f"Error: --input-dir does not exist or is not a directory: {input_dir}")

        print("=" * 60)
        mode_label = "BATCH + DUB" if args.dub else "BATCH MODE"
        print(f"  Video Subtitle Pipeline  [{mode_label}]")
        print("=" * 60)
        print(f"  Input dir  : {input_dir}")
        print(f"  Output dir : {output_dir}")
        print(f"  Target lang: {args.language}")
        print(f"  Mode       : {'dubbing' if args.dub else 'subtitles'}")
        print(f"  Model      : {args.model}")
        print(f"  Device     : {device}")
        print(f"  Transcript : {args.transcript or 'none (Whisper auto-mode)'}")
        print(f"  Translator : {translator}")
        print("=" * 60)

        batch_process(
            input_dir=input_dir,
            output_dir=output_dir,
            language=args.language,
            model_name=args.model,
            device=device,
            source_lang=args.source_lang,
            transcript_path=transcript_path,
            dub=args.dub,
            merge_gap_s=merge_gap,
            translator=translator,
            deepl_key=deepl_key,
            local_model=local_model,
        )
        return

    # single-file mode
    if not args.video:
        parser.error("Provide a video file path, or use --batch mode.")

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Error: Video file not found: {video_path}")

    if args.output:
        output_path = Path(args.output).resolve()
        subtitle_output_path = output_path
    elif args.dub:
        output_path          = video_path.parent / f"{video_path.stem}_dubbed_{args.language}{video_path.suffix}"
        subtitle_output_path = video_path.parent / f"{video_path.stem}_{args.language}{video_path.suffix}"
    else:
        output_path          = video_path.parent / f"{video_path.stem}_{args.language}{video_path.suffix}"
        subtitle_output_path = output_path

    mode_label = "Video Subtitle + Dubbing Pipeline" if args.dub else "Video Subtitle Pipeline"
    print("=" * 60)
    print(f"  {mode_label}")
    print("=" * 60)
    print(f"  Video      : {video_path.name}")
    print(f"  Transcript : {args.transcript or 'none (Whisper auto-mode)'}")
    print(f"  Target lang: {args.language}")
    print(f"  Mode       : {'subtitles + dubbing' if args.dub else 'subtitles'}")
    print(f"  Model      : {args.model}")
    print(f"  Device     : {device}")
    print(f"  Output     : {output_path.name}")
    print(f"  Translator : {translator}")
    if args.dub:
        print(f"  Sub output : {subtitle_output_path.name}")
    print("=" * 60)

    # Step 1: Read reference transcript (optional)
    print("\n[1/4] Reading reference transcript ...")
    reference_text = read_transcript(transcript_path)
    if reference_text:
        print(f"  {len(reference_text)} characters read from '{transcript_path.name}'")
    else:
        print("  No transcript provided - Whisper will transcribe without a prompt")

    if args.dub:
        process_one_video_dub(
            video_path=video_path,
            language=args.language,
            model_name=args.model,
            device=device,
            source_lang=args.source_lang,
            reference_text=reference_text,
            output_path=output_path,
            subtitle_output_path=subtitle_output_path,
            merge_gap_s=merge_gap,
            translator=translator,
            deepl_key=deepl_key,
            local_model=local_model,
        )
    else:
        process_one_video(
            video_path=video_path,
            language=args.language,
            model_name=args.model,
            device=device,
            source_lang=args.source_lang,
            reference_text=reference_text,
            output_path=output_path,
            translator=translator,
            deepl_key=deepl_key,
            local_model=local_model,
        )

    print("\n" + "=" * 60)
    if args.dub:
        print(f"  Done!  Subtitled : {subtitle_output_path}")
        print(f"         Dubbed    : {output_path}")
    else:
        print(f"  Done!  Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
