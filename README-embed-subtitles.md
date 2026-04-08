# Subtitle Embed Tool

Burn an existing `.srt` subtitle file directly into a video with FFmpeg.
This tool is separate from the full transcription/translation pipeline and is
meant for the simpler case where the subtitle file already exists.

## Requirements

- `ffmpeg` must be installed and available on `PATH`
- Python 3.8+

Install FFmpeg on Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg
```

## Script

```bash
python embed-subtitles.py
```

## Single-file mode

### Single-file command

```bash
python embed-subtitles.py <video> <subtitle.srt> <suffix>
```

### Single-file behavior

- Burns the provided subtitle file into the provided video.
- Writes the output next to the input video.
- Names the output video `<original_name>_<suffix>.<ext>`.

### Single-file example

```bash
python embed-subtitles.py ./lecture.mp4 ./lecture.srt hardcoded
```

Output:

```text
./lecture_hardcoded.mp4
```

## Batch mode

### Batch command

```bash
python embed-subtitles.py --batch --input-dir <folder> <suffix>
```

### Batch behavior

- Scans the input folder recursively for video files.
- For each video, expects a matching `.srt` file in the same folder with the same stem.
- Creates a sibling output folder named `<input-folder-name>-<suffix>`.
- Mirrors the full directory structure into that output folder.
- Writes each output video as `<original_name>_<suffix>.<ext>`.
- Skips videos that do not have a matching `.srt` file and continues.

### Batch example

Input tree:

```text
course/
  week1/
    lecture.mp4
    lecture.srt
  week2/
    demo.mkv
    demo.srt
```

Command:

```bash
python embed-subtitles.py --batch --input-dir ./course hardcoded
```

Output tree:

```text
course-hardcoded/
  week1/
    lecture_hardcoded.mp4
  week2/
    demo_hardcoded.mkv
```

## Supported video extensions

The batch scan processes these extensions:

- `.mp4`
- `.mkv`
- `.avi`
- `.mov`
- `.webm`
- `.flv`
- `.wmv`
- `.m4v`
- `.ts`
- `.mts`

## Notes

- The original videos are left unchanged.
- The tool copies audio streams with `-c:a copy` and re-encodes only the video stream needed for subtitle burning.
- Subtitle files must be `.srt`.
