from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .annotation_schema import EpisodeAnnotation, Segment
from .config import SubtaskConfig


TASK_SPEC_VERSION = 1


@dataclass(frozen=True)
class TaskSpec:
    main_task: str
    subtasks: list[SubtaskConfig]
    source_episode_index: int = 0
    source_video_key: str = ""
    source_video_path: str = ""
    source_duration_sec: float = 0.0
    reference_segments: list[Segment] = field(default_factory=list)
    version: int = TASK_SPEC_VERSION

    @classmethod
    def from_annotation(cls, annotation: EpisodeAnnotation) -> "TaskSpec":
        subtasks = [
            SubtaskConfig(
                name=segment.subtask,
                description=segment.description,
                visual_start=segment.start_signal,
                visual_end=segment.end_signal,
            )
            for segment in annotation.segments
        ]
        return cls(
            main_task=annotation.main_task,
            subtasks=subtasks,
            source_episode_index=annotation.episode_index,
            source_video_key=annotation.video_key,
            source_video_path=annotation.video_path,
            source_duration_sec=annotation.duration_sec,
            reference_segments=annotation.segments,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TaskSpec":
        subtasks = [
            SubtaskConfig(
                name=str(item.get("name", "")),
                description=str(item.get("description", "")),
                visual_start=str(item.get("visual_start", "")),
                visual_end=str(item.get("visual_end", "")),
            )
            for item in data.get("subtasks", [])
            if isinstance(item, dict)
        ]
        reference_segments = [
            Segment.from_mapping(item)
            for item in data.get("reference_segments", [])
            if isinstance(item, dict)
        ]
        return cls(
            version=int(data.get("version", TASK_SPEC_VERSION)),
            main_task=str(data.get("main_task", "")),
            subtasks=subtasks,
            source_episode_index=int(data.get("source_episode_index", 0)),
            source_video_key=str(data.get("source_video_key", "")),
            source_video_path=str(data.get("source_video_path", "")),
            source_duration_sec=float(data.get("source_duration_sec", 0.0)),
            reference_segments=reference_segments,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["subtasks"] = [
            {
                "name": item.name,
                "description": item.description,
                "visual_start": item.visual_start,
                "visual_end": item.visual_end,
            }
            for item in self.subtasks
        ]
        data["reference_segments"] = [segment.to_dict() for segment in self.reference_segments]
        return data

    def validate(self) -> None:
        if self.version != TASK_SPEC_VERSION:
            raise ValueError(f"Unsupported task_spec version: {self.version}")
        if not self.main_task.strip():
            raise ValueError("task_spec.main_task must be non-empty")
        if not self.subtasks:
            raise ValueError("task_spec.subtasks must be non-empty")
        seen: set[str] = set()
        for idx, subtask in enumerate(self.subtasks):
            if not subtask.name.strip():
                raise ValueError(f"task_spec.subtasks[{idx}].name must be non-empty")
            if subtask.name in seen:
                raise ValueError(f"duplicate task_spec subtask name: {subtask.name}")
            seen.add(subtask.name)


def write_task_spec(path: Path, spec: TaskSpec) -> Path:
    spec.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_task_spec(path: Path) -> TaskSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("task_spec JSON must be an object")
    spec = TaskSpec.from_mapping(data)
    spec.validate()
    return spec
