#!/usr/bin/env python3
"""
Subtitle Embed Tool
===================
Burn existing SRT subtitles into one video or a recursively scanned folder tree.

Usage (single file):
    python embed-subtitles.py <video> <subtitle> <suffix>

Usage (batch / folder):
    python embed-subtitles.py --batch --input-dir <folder> <suffix>
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        sys.exit(
            "Error: ffmpeg not found.\n"
            "  Install with: sudo apt install ffmpeg"
        )


def _progress_bar(current: int, total: int, width: int = 38) -> str:
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    if filled >= width:
        bar = "=" * width
    else:
        bar = "=" * filled + ">" + " " * (width - filled - 1)
    return f"[{bar}] {pct * 100:5.1f}%  {current}/{total}"


def get_video_duration(video_path: Path) -> float:
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


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path):
    """
    Burn the SRT file into the video via the FFmpeg 'subtitles' filter.

    The SRT is copied to a temp directory with a plain ASCII filename so that
    the FFmpeg filtergraph never sees special characters or spaces in the path.
    """
    duration = get_video_duration(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        print()

    if proc.returncode != 0:
        print("  FFmpeg stderr (last 3000 chars):")
        print(stderr_output[-3000:])
        sys.exit(f"Error: FFmpeg failed with return code {proc.returncode}")


VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".flv", ".wmv", ".m4v", ".ts", ".mts",
}


def output_path_for_video(video_path: Path, suffix: str, output_root: Path | None = None, input_root: Path | None = None) -> Path:
    output_name = f"{video_path.stem}_{suffix}{video_path.suffix}"
    if output_root is None:
        return video_path.with_name(output_name)
    rel_parent = video_path.relative_to(input_root).parent
    return output_root / rel_parent / output_name


def find_matching_subtitle(video_path: Path) -> Path | None:
    subtitle_path = video_path.with_suffix(".srt")
    if subtitle_path.is_file():
        return subtitle_path
    return None


def validate_suffix(value: str) -> str:
    suffix = value.strip()
    if not suffix:
        raise argparse.ArgumentTypeError("suffix must not be empty")
    if any(char in suffix for char in r'\\/:*?"<>|'):
        raise argparse.ArgumentTypeError(
            "suffix contains filesystem-reserved characters"
        )
    return suffix


def embed_single(video_path: Path, subtitle_path: Path, suffix: str):
    if not video_path.is_file():
        sys.exit(f"Error: Video file not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        sys.exit(f"Error: Unsupported video extension '{video_path.suffix}'")
    if not subtitle_path.is_file():
        sys.exit(f"Error: Subtitle file not found: {subtitle_path}")
    if subtitle_path.suffix.lower() != ".srt":
        sys.exit("Error: Subtitle file must be an .srt file")

    output_path = output_path_for_video(video_path, suffix)

    print("=" * 60)
    print("  Subtitle Embed Tool")
    print("=" * 60)
    print(f"  Video      : {video_path}")
    print(f"  Subtitle   : {subtitle_path}")
    print(f"  Suffix     : {suffix}")
    print(f"  Output     : {output_path}")
    print("=" * 60)

    burn_subtitles(video_path, subtitle_path, output_path)

    print("\n" + "=" * 60)
    print(f"  Done! Output: {output_path}")
    print("=" * 60)


def embed_batch(input_dir: Path, suffix: str):
    if not input_dir.is_dir():
        sys.exit(f"Error: Input directory does not exist or is not a directory: {input_dir}")

    output_dir = input_dir.with_name(f"{input_dir.name}-{suffix}")
    video_files = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        sys.exit(f"Error: No video files found under '{input_dir}'")

    print("=" * 60)
    print("  Subtitle Embed Tool  [BATCH MODE]")
    print("=" * 60)
    print(f"  Input dir  : {input_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Suffix     : {suffix}")
    print(f"  Videos     : {len(video_files)}")
    print("=" * 60)

    processed = 0
    skipped = 0

    for idx, video_path in enumerate(video_files, 1):
        rel = video_path.relative_to(input_dir)
        subtitle_path = find_matching_subtitle(video_path)

        print("\n" + "=" * 60)
        print(f"  [{idx}/{len(video_files)}] {rel}")
        print("=" * 60)

        if subtitle_path is None:
            skipped += 1
            print("  Missing matching subtitle file; expected a .srt with the same stem.")
            print("  Skipping to next video ...")
            continue

        output_path = output_path_for_video(
            video_path,
            suffix,
            output_root=output_dir,
            input_root=input_dir,
        )
        print(f"  Subtitle   : {subtitle_path.relative_to(input_dir)}")
        print(f"  Output     : {output_path}")

        try:
            burn_subtitles(video_path, subtitle_path, output_path)
            processed += 1
            print(f"  Saved      : {output_path}")
        except SystemExit as exc:
            skipped += 1
            print(f"  ERROR      : {exc}")
            print("  Skipping to next video ...")

    print("\n" + "=" * 60)
    print(f"  Batch complete. Processed: {processed}  Skipped: {skipped}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Burn existing SRT subtitles into videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python embed-subtitles.py lecture.mp4 lecture.srt hardcoded\n"
            "  python embed-subtitles.py --batch --input-dir ./course hardcoded\n"
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: process all videos under --input-dir recursively.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        metavar="DIR",
        help="[batch] Root folder to scan for videos recursively.",
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Input video path (single-file mode).",
    )
    parser.add_argument(
        "subtitle",
        nargs="?",
        default=None,
        help="Input subtitle path (.srt) (single-file mode).",
    )
    parser.add_argument(
        "suffix",
        type=validate_suffix,
        help="Suffix added to output video names and batch output folder name.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    check_ffmpeg()

    if args.batch:
        if args.video or args.subtitle:
            parser.error("--batch does not accept positional video/subtitle inputs")
        if not args.input_dir:
            parser.error("--batch requires --input-dir")
        embed_batch(Path(args.input_dir).resolve(), args.suffix)
        return

    if args.input_dir:
        parser.error("--input-dir can only be used together with --batch")
    if not args.video or not args.subtitle:
        parser.error("single-file mode requires: <video> <subtitle> <suffix>")

    embed_single(
        video_path=Path(args.video).resolve(),
        subtitle_path=Path(args.subtitle).resolve(),
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()