from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    duration_sec: float
    num_frames: int
    width: int
    height: int


def probe_video(path: Path) -> VideoMetadata:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for video probing. Install with: pip install opencv-python") from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if fps <= 0:
        fps = 30.0
        LOGGER.warning("Could not read FPS for %s; falling back to %.1f", path, fps)
    duration = frame_count / fps if frame_count > 0 else 0.0
    return VideoMetadata(path=path, fps=fps, duration_sec=duration, num_frames=frame_count, width=width, height=height)


def make_low_fps_video(input_path: Path, output_path: Path, *, fps: float, width: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            f"fps={fps},scale={width}:-2",
            "-an",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return output_path

    LOGGER.warning("ffmpeg not found; using OpenCV for low-FPS video generation")
    _opencv_transcode(input_path, output_path, fps=fps, width=width)
    return output_path


def extract_video_clip(
    input_path: Path,
    output_path: Path,
    *,
    start_sec: float,
    end_sec: float,
    fps: float,
    width: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec, float(end_sec))
    duration = max(0.001, end_sec - start_sec)
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(input_path),
            "-vf",
            f"fps={fps},scale={width}:-2",
            "-an",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return output_path

    LOGGER.warning("ffmpeg not found; using OpenCV for clip extraction")
    _opencv_transcode(input_path, output_path, fps=fps, width=width, start_sec=start_sec, end_sec=end_sec)
    return output_path


def save_contact_sheet(
    video_path: Path,
    output_path: Path,
    segments: list[dict[str, object]],
    *,
    max_thumb_width: int = 240,
) -> Path:
    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python and pillow are required for contact sheets. Install with: pip install opencv-python pillow"
        ) from exc

    metadata = probe_video(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video for contact sheet: {video_path}")

    thumbs: list[Image.Image] = []
    labels: list[str] = []
    try:
        for segment in segments:
            start = float(segment["start_time"])
            end = float(segment["end_time"])
            times = [start, (start + end) / 2.0, max(start, end - 1e-3)]
            for label, timestamp in zip(("start", "mid", "end"), times, strict=True):
                frame = _read_frame_at(cap, timestamp, metadata)
                if frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                scale = max_thumb_width / max(1, image.width)
                image = image.resize((max_thumb_width, max(1, int(image.height * scale))))
                thumbs.append(image)
                labels.append(f"{segment['subtask']} {label} {timestamp:.1f}s")
    finally:
        cap.release()

    if not thumbs:
        raise ValueError(f"No frames could be read from {video_path}")

    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    label_h = 28
    cell_w = max(img.width for img in thumbs)
    cell_h = max(img.height for img in thumbs) + label_h
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, image in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = col * cell_w
        y = row * cell_h
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 4, y + 6), labels[idx][:60], fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    return output_path


def save_timeline_plot(output_path: Path, segments: list[dict[str, object]], *, duration_sec: float) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for timeline plots. Install with: pip install matplotlib") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(2.4, 0.36 * len(segments) + 1.2)))
    palette = plt.get_cmap("tab20")
    for idx, segment in enumerate(segments):
        start = float(segment["start_time"])
        end = float(segment["end_time"])
        ax.barh(0, end - start, left=start, height=0.55, color=palette(idx % 20), edgecolor="black")
        label = str(segment["subtask"])
        x = start + (end - start) / 2.0
        ax.text(x, 0, label, ha="center", va="center", fontsize=8, clip_on=True)
    ax.set_xlim(0, max(duration_sec, 0.1))
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title("Subtask timeline")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _opencv_transcode(
    input_path: Path,
    output_path: Path,
    *,
    fps: float,
    width: int,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required when ffmpeg is unavailable") from exc

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or width)
    height = max(2, int(round(src_h * (width / max(src_w, 1)))))
    if height % 2:
        height += 1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {output_path}")

    next_time = start_sec
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    try:
        while True:
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if end_sec is not None and timestamp > end_sec:
                break
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if timestamp + (0.5 / max(src_fps, 1e-6)) < next_time:
                continue
            resized = cv2.resize(frame, (width, height))
            writer.write(resized)
            next_time += 1.0 / fps
    finally:
        writer.release()
        cap.release()


def _read_frame_at(cap: object, timestamp: float, metadata: VideoMetadata) -> object | None:
    import cv2

    frame_index = max(0, int(timestamp * metadata.fps))
    if metadata.num_frames > 0:
        frame_index = min(frame_index, metadata.num_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None
