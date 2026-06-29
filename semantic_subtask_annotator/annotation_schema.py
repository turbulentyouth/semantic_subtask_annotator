from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Segment:
    subtask: str
    start_time: float
    end_time: float
    description: str = ""
    start_signal: str = ""
    end_signal: str = ""
    boundary_reason: str = ""
    confidence: float = 0.0
    subtask_index: int | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    start_timestamp: str | None = None
    end_timestamp: str | None = None

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "Segment":
        return cls(
            subtask=str(item.get("subtask", "")),
            start_time=float(item.get("start_time")),
            end_time=float(item.get("end_time")),
            description=str(item.get("description", "")),
            start_signal=str(item.get("start_signal", "")),
            end_signal=str(item.get("end_signal", "")),
            boundary_reason=str(item.get("boundary_reason", "")),
            confidence=float(item.get("confidence", 0.0)),
            start_timestamp=str(item["start_timestamp"]) if item.get("start_timestamp") is not None else None,
            end_timestamp=str(item["end_timestamp"]) if item.get("end_timestamp") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        optional_text_fields = {"description", "start_signal", "end_signal"}
        return {k: v for k, v in data.items() if v is not None and (k not in optional_text_fields or v)}


@dataclass
class ValidationReport:
    valid: bool
    fixed: bool = False
    messages: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.messages.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "fixed": self.fixed, "messages": self.messages}


@dataclass
class EpisodeAnnotation:
    episode_index: int
    video_key: str
    video_path: str
    dataset_fps: float
    duration_sec: float
    num_frames: int
    main_task: str
    segments: list[Segment]
    validation: ValidationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "video_key": self.video_key,
            "video_path": self.video_path,
            "dataset_fps": self.dataset_fps,
            "duration_sec": self.duration_sec,
            "num_frames": self.num_frames,
            "main_task": self.main_task,
            "segments": [s.to_dict() for s in self.segments],
            "validation": self.validation.to_dict(),
            "frame_interval_convention": "left-closed-right-open [start_frame, end_frame); final end_frame == num_frames",
        }


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_timestamp(timestamp: str) -> float:
    parts = timestamp.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"invalid timestamp: {timestamp!r}")


def attach_frame_indices(
    segments: list[Segment],
    *,
    dataset_fps: float,
    num_frames: int,
    subtask_names: list[str],
) -> None:
    name_to_index: dict[str, int] = {}
    for idx, name in enumerate(subtask_names):
        name_to_index.setdefault(name, idx)
    for idx, segment in enumerate(segments):
        start_frame = int(round(segment.start_time * dataset_fps))
        end_frame = int(round(segment.end_time * dataset_fps))
        start_frame = max(0, min(start_frame, num_frames))
        end_frame = max(start_frame, min(end_frame, num_frames))
        if idx == len(segments) - 1:
            end_frame = num_frames
        segment.start_frame = start_frame
        segment.end_frame = end_frame
        segment.subtask_index = name_to_index.get(segment.subtask)
        segment.start_timestamp = format_timestamp(segment.start_time)
        segment.end_timestamp = format_timestamp(segment.end_time)
