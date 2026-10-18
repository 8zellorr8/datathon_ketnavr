"""Decode local video with FFmpeg; no network or AI service is used here."""
from __future__ import annotations

import base64
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


class VideoError(RuntimeError):
    pass


def ffmpeg_executable() -> str:
    configured = os.environ.get('FORTE_FFMPEG')
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not candidate.is_file():
            raise VideoError('FORTE_FFMPEG does not point to an executable file')
        return str(candidate)
    installed = shutil.which('ffmpeg')
    if installed:
        return installed
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise VideoError('Install requirements.txt or set FORTE_FFMPEG') from exc


def _timeout(deadline: float | None, maximum: float = 30) -> float:
    if deadline is None:
        return maximum
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise VideoError('Video analysis deadline exceeded')
    return min(maximum, remaining)


def probe_video(video_path: str | Path, *, deadline: float | None = None) -> dict:
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise VideoError(f'Video not found: {video}')
    if video.suffix.lower() != '.mov':
        raise VideoError('The case requires a .mov video')
    try:
        result = subprocess.run(
            [ffmpeg_executable(), '-hide_banner', '-nostdin', '-protocol_whitelist', 'file,pipe',
             '-format_whitelist', 'mov', '-i', str(video)],
            capture_output=True, timeout=_timeout(deadline),
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoError('Video metadata timeout') from exc
    info = result.stderr.decode('utf-8', errors='replace')
    match = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', info)
    if not match or 'Video:' not in info:
        raise VideoError('No decodable video stream or finite duration found')
    hours, minutes, seconds = (float(v) for v in match.groups())
    duration = hours * 3600 + minutes * 60 + seconds
    if not math.isfinite(duration) or duration <= 0:
        raise VideoError('Invalid video duration')
    return {'path': str(video), 'duration_sec': duration, 'size_bytes': video.stat().st_size}


def sample_times(duration_sec: float, step_sec: float) -> list[float]:
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError('duration_sec must be finite and positive')
    if not math.isfinite(step_sec) or step_sec <= 0:
        raise ValueError('step_sec must be finite and positive')
    end = max(0.0, duration_sec - 0.15)
    times = [round(i * step_sec, 3) for i in range(int(end // step_sec) + 1)]
    if end - times[-1] > min(step_sec / 2, 1):
        times.append(round(end, 3))
    return times


def extract_frame(video_path: str | Path, timestamp_sec: float, *,
                  max_side: int = 560, deadline: float | None = None) -> dict:
    """Return the decoded JPEG and its original stream PTS, not a guessed index."""
    if not math.isfinite(timestamp_sec) or timestamp_sec < 0:
        raise ValueError('Invalid frame timestamp')
    if not 128 <= max_side <= 1280:
        raise ValueError('max_side must be between 128 and 1280')
    command = [
        ffmpeg_executable(), '-hide_banner', '-nostdin', '-threads', '1',
        '-ss', f'{timestamp_sec:.6f}', '-copyts', '-protocol_whitelist', 'file,pipe',
        '-format_whitelist', 'mov', '-i', str(Path(video_path).resolve()),
        '-an', '-sn', '-dn', '-vf',
        f'scale={max_side}:{max_side}:force_original_aspect_ratio=decrease,showinfo',
        '-frames:v', '1', '-q:v', '3', '-f', 'image2pipe', '-vcodec', 'mjpeg', 'pipe:1',
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=_timeout(deadline))
    except subprocess.TimeoutExpired as exc:
        raise VideoError('Frame extraction timeout') from exc
    stderr = result.stderr.decode('utf-8', errors='replace')
    if result.returncode or not result.stdout.startswith(b'\xff\xd8'):
        raise VideoError(f'Could not decode frame at {timestamp_sec:g} seconds')
    pts = re.search(r'\bn:\s*0\s+pts:\s*-?\d+\s+pts_time:\s*(-?[\d.]+)', stderr)
    if pts is None:
        raise VideoError('FFmpeg did not return the decoded frame timestamp')
    actual_time = float(pts.group(1))
    if not math.isfinite(actual_time) or actual_time < 0 or abs(actual_time - timestamp_sec) > 1:
        raise VideoError('Unexpected stream timestamps; unsupported video time base')
    return {
        'timestamp_sec': actual_time,
        'image_base64': base64.b64encode(result.stdout).decode('ascii'),
    }
