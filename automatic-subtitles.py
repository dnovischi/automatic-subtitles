#!/usr/bin/env python3
"""
Video Subtitle Pipeline
=======================
Transcribes a video using a reference transcript, translates the subtitles to a
target language, and burns them into the output video.

Usage (single file):
    python automatic-subtitles.py <video> <language> [options]

Usage (batch / folder):
    python automatic-subtitles.py --batch --input-dir <folder> --output-dir <folder> <language> [options]

Examples:
    python automatic-subtitles.py lecture.mp4 ro
    python automatic-subtitles.py lecture.mp4 ro --transcript lecture.txt
    python automatic-subtitles.py lecture.mp4 ro --transcript lecture.pdf
    python automatic-subtitles.py lecture.mp4 es --model large --device cuda
    python automatic-subtitles.py lecture.mp4 fr --output out.mp4
    python automatic-subtitles.py --batch --input-dir /data/courses --output-dir /data/out ro
    python automatic-subtitles.py --batch --input-dir /data/courses --output-dir /data/out ro --model large
"""

import argparse
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


def segments_to_srt(segments) -> str:
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


# Translation

def translate_srt(srt_content: str, source_lang: str, target_lang: str) -> str:
    import srt
    from deep_translator import GoogleTranslator

    subs  = list(srt.parse(srt_content))
    total = len(subs)
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translated_subs = []

    for i, sub in enumerate(subs, 1):
        try:
            translated      = translator.translate(sub.content)
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
    """
    duration = get_video_duration(video_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_srt = Path(tmpdir) / "subtitles.srt"
        shutil.copy(srt_path, tmp_srt)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles={tmp_srt}",
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
):
    """Transcribe, translate, and burn subtitles into a single video.

    SRT files are saved alongside *output_path* and named after the
    original video stem so they are easy to identify.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srt_orig_path  = output_path.parent / f"{video_path.stem}_original.srt"
    srt_trans_path = output_path.parent / f"{video_path.stem}_{language}.srt"

    # Step 2: Transcribe
    print("\n[2/4] Transcribing video with Whisper ...")
    segments, detected_lang = transcribe(
        video_path, model_name, device, initial_prompt=reference_text
    )

    orig_srt = segments_to_srt(segments)
    srt_orig_path.write_text(orig_srt, encoding="utf-8")
    n_subs = orig_srt.count("\n\n")
    print(f"  Detected language : {detected_lang}")
    print(f"  Subtitle segments : {n_subs}")
    print(f"  Saved             : {srt_orig_path.name}")

    # Step 3: Translate
    print(f"\n[3/4] Translating subtitles to '{language}' ...")
    trans_srt = translate_srt(orig_srt, source_lang, language)
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
            process_one_video(
                video_path=video_path,
                language=language,
                model_name=model_name,
                device=device,
                source_lang=source_lang,
                reference_text=reference_text,
                output_path=output_path,
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

    # ------------------------------------------------------------------ mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--batch", action="store_true",
        help="Batch mode: process all videos under --input-dir recursively."
    )

    # ---------------------------------------------------- positional (single)
    parser.add_argument("video", nargs="?", default=None,
                        help="Input video file path  (single-file mode)")

    # ------------------------------------------------------- batch arguments
    parser.add_argument("--input-dir", default=None, metavar="DIR",
                        help="[batch] Root folder to scan for video files recursively")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="[batch] Root folder for output; mirrors input folder structure")

    # -------------------------------------------------------- shared options
    parser.add_argument("language",
                        help="Target subtitle language code  (e.g. ro, es, fr)")
    parser.add_argument("--transcript", default=None, metavar="FILE",
                        help="Optional reference transcript (.txt or .pdf). "
                             "Improves Whisper accuracy for domain-specific vocabulary.")
    parser.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium",
                                 "large", "large-v2", "large-v3"],
                        help="Whisper model to use  (default: medium)")
    parser.add_argument("--device", default=None,
                        help="Device: cpu or cuda  (default: auto-detect)")
    parser.add_argument("--source-lang", default="auto",
                        help="Source language for translation  (default: auto)")
    parser.add_argument("--output", default=None,
                        help="[single] Output video path  (default: <input>_<lang>.<ext>)")

    args = parser.parse_args()

    check_ffmpeg()
    check_python_deps()

    device = args.device or detect_device()
    transcript_path = Path(args.transcript).resolve() if args.transcript else None

    # ----------------------------------------------------------------- BATCH
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
        print("  Video Subtitle Pipeline  [BATCH MODE]")
        print("=" * 60)
        print(f"  Input dir  : {input_dir}")
        print(f"  Output dir : {output_dir}")
        print(f"  Target lang: {args.language}")
        print(f"  Model      : {args.model}")
        print(f"  Device     : {device}")
        print(f"  Transcript : {args.transcript or 'none (Whisper auto-mode)'}")
        print("=" * 60)

        batch_process(
            input_dir=input_dir,
            output_dir=output_dir,
            language=args.language,
            model_name=args.model,
            device=device,
            source_lang=args.source_lang,
            transcript_path=transcript_path,
        )
        return

    # --------------------------------------------------------------- SINGLE
    if not args.video:
        parser.error("Provide a video file path, or use --batch mode.")

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Error: Video file not found: {video_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else video_path.parent / f"{video_path.stem}_{args.language}{video_path.suffix}"
    )

    print("=" * 60)
    print("  Video Subtitle Pipeline")
    print("=" * 60)
    print(f"  Video      : {video_path.name}")
    print(f"  Transcript : {args.transcript or 'none (Whisper auto-mode)'}")
    print(f"  Target lang: {args.language}")
    print(f"  Model      : {args.model}")
    print(f"  Device     : {device}")
    print(f"  Output     : {output_path.name}")
    print("=" * 60)

    # Step 1: Read reference transcript (optional)
    print("\n[1/4] Reading reference transcript ...")
    reference_text = read_transcript(transcript_path)
    if reference_text:
        print(f"  {len(reference_text)} characters read from '{transcript_path.name}'")
    else:
        print("  No transcript provided - Whisper will transcribe without a prompt")

    process_one_video(
        video_path=video_path,
        language=args.language,
        model_name=args.model,
        device=device,
        source_lang=args.source_lang,
        reference_text=reference_text,
        output_path=output_path,
    )

    print("\n" + "=" * 60)
    print(f"  Done!  Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
