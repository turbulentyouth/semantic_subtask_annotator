from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .video_utils import probe_video

LOGGER = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DISALLOWED_AUTO_KEY_PARTS = ("wrist", "side", "left", "right")


@dataclass(frozen=True)
class EpisodeVideo:
    episode_index: int
    video_key: str
    video_path: Path
    dataset_fps: float
    original_fps: float
    duration_sec: float
    num_frames: int


@dataclass(frozen=True)
class DatasetInfo:
    root: Path
    video_keys: list[str]
    selected_video_key: str
    fps: float | None
    episode_indices: list[int]
    video_files: list[Path]


class DatasetLoader:
    """Find main-camera videos without using robot state/action semantics."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.root = config.root.expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")
        self.info = self._load_dataset_info()

    def episodes(self) -> list[int]:
        return self.info.episode_indices

    def get_episode_video(self, episode_index: int) -> EpisodeVideo:
        video_path = self.find_episode_video_path(episode_index)
        if video_path is None:
            raise FileNotFoundError(
                f"Could not find video for episode {episode_index} and key {self.info.selected_video_key!r} under {self.root}"
            )
        metadata = probe_video(video_path)
        fps = self.config.fps or self.info.fps or metadata.fps
        num_frames = metadata.num_frames
        if self.config.fps is not None and metadata.duration_sec > 0:
            num_frames = int(round(metadata.duration_sec * self.config.fps))
        return EpisodeVideo(
            episode_index=episode_index,
            video_key=self.info.selected_video_key,
            video_path=video_path,
            dataset_fps=fps,
            original_fps=metadata.fps,
            duration_sec=metadata.duration_sec,
            num_frames=num_frames,
        )

    def find_episode_video_path(self, episode_index: int) -> Path | None:
        return self._find_video_for_episode(self.info.selected_video_key, episode_index)

    def _load_dataset_info(self) -> DatasetInfo:
        info_json = self.root / "meta" / "info.json"
        if info_json.exists():
            raw = _read_json(info_json)
            video_keys = _extract_video_keys(raw)
            if not video_keys:
                video_keys = self._discover_video_keys_from_files()
            fps = self.config.fps or _extract_fps(raw)
        else:
            raw = {}
            video_keys = self._discover_video_keys_from_files()
            fps = self.config.fps

        if not video_keys:
            raise ValueError(f"No video/image keys found under dataset root: {self.root}")

        selected_key = self._select_main_video_key(video_keys)
        episode_indices = self._discover_episode_indices(selected_key, raw)
        if not episode_indices:
            raise ValueError(f"No episodes found for video key {selected_key!r} under {self.root}")

        return DatasetInfo(
            root=self.root,
            video_keys=video_keys,
            selected_video_key=selected_key,
            fps=fps,
            episode_indices=episode_indices,
            video_files=_list_video_files(self.root),
        )

    def _select_main_video_key(self, available_keys: list[str]) -> str:
        available = set(available_keys)
        explicit_key = self.config.main_video_key
        if explicit_key in available:
            return explicit_key

        for key in self.config.fallback_main_video_keys:
            if key in available:
                if _is_disallowed_auto_key(key):
                    LOGGER.warning("Skipping fallback video key that looks non-main: %s", key)
                    continue
                return key

        available_list = "\n  - ".join(sorted(available_keys))
        raise ValueError(
            "Could not find configured main camera key. "
            f"Requested {explicit_key!r}; fallback keys={self.config.fallback_main_video_keys!r}. "
            "Available image/video keys:\n  - "
            f"{available_list}"
        )

    def _discover_video_keys_from_files(self) -> list[str]:
        video_files = _list_video_files(self.root)
        keys: set[str] = set()
        for path in video_files:
            rel = path.relative_to(self.root)
            parts = rel.parts
            if "videos" in parts:
                idx = parts.index("videos")
                key_parts = parts[idx + 1 : -1]
                if key_parts and re.fullmatch(r"chunk[-_]\d+", key_parts[0], flags=re.IGNORECASE):
                    key_parts = key_parts[1:]
                if key_parts:
                    keys.add(".".join(key_parts))
                    keys.add("/".join(key_parts))
                    continue
            if path.parent == self.root:
                keys.add(self.config.main_video_key)
                continue
            parent_name = path.parent.name
            if parent_name:
                keys.add(parent_name)
            if len(video_files) == 1:
                keys.add(self.config.main_video_key)
        return sorted(keys)

    def _discover_episode_indices(self, selected_key: str, raw_info: dict[str, Any]) -> list[int]:
        total_episodes = _extract_total_episodes(raw_info)
        metadata_indices: set[int] = set(range(total_episodes)) if total_episodes is not None else set()
        video_indices: set[int] = set()

        video_files = _list_video_files(self.root)
        for path in video_files:
            if _file_matches_key(path, self.root, selected_key, total_video_files=len(video_files)):
                episode = _extract_episode_index(path)
                if episode is not None:
                    video_indices.add(episode)

        if video_indices:
            if metadata_indices:
                missing = sorted(metadata_indices - video_indices)
                if missing:
                    LOGGER.warning(
                        "Dataset metadata lists %d episode(s), but %d selected-camera video(s) were found. "
                        "Missing video-backed episodes include: %s",
                        len(metadata_indices),
                        len(video_indices),
                        missing[:20],
                    )
            return sorted(video_indices)

        if len(video_files) == 1:
            return [0]
        return sorted(metadata_indices)

    def _find_video_for_episode(self, selected_key: str, episode_index: int) -> Path | None:
        video_files = self.info.video_files
        candidates = [
            path
            for path in video_files
            if _file_matches_key(path, self.root, selected_key, total_video_files=len(video_files))
            and _extract_episode_index(path) == episode_index
        ]
        if not candidates and episode_index == 0:
            all_videos = video_files
            if len(all_videos) == 1:
                candidates = all_videos
        if not candidates:
            return None
        return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _extract_video_keys(raw: dict[str, Any]) -> list[str]:
    keys: set[str] = set()
    features = raw.get("features")
    if isinstance(features, dict):
        for key, spec in features.items():
            if _feature_is_video_or_image(spec):
                keys.add(str(key))

    for field_name in ("video_keys", "image_keys", "camera_keys"):
        value = raw.get(field_name)
        if isinstance(value, list):
            keys.update(str(item) for item in value if isinstance(item, str))

    return sorted(keys)


def _feature_is_video_or_image(spec: Any) -> bool:
    if not isinstance(spec, dict):
        return False
    dtype = str(spec.get("dtype", "")).lower()
    feature_type = str(spec.get("type", "")).lower()
    shape = spec.get("shape")
    names = [str(value).lower() for value in spec.values() if isinstance(value, str)]
    if dtype in {"video", "image"} or feature_type in {"video", "image"}:
        return True
    if any("video" in item or "image" in item for item in names):
        return True
    return isinstance(shape, list) and len(shape) >= 2 and any(token in dtype for token in ("uint8", "image"))


def _extract_fps(raw: dict[str, Any]) -> float | None:
    for key in ("fps", "video_fps", "dataset_fps"):
        value = raw.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    info = raw.get("info")
    if isinstance(info, dict):
        return _extract_fps(info)
    return None


def _extract_total_episodes(raw: dict[str, Any]) -> int | None:
    for key in ("total_episodes", "num_episodes"):
        value = raw.get(key)
        if isinstance(value, int):
            return value
    splits = raw.get("splits")
    if isinstance(splits, dict):
        train = splits.get("train")
        if isinstance(train, str):
            match = re.search(r":\s*(\d+)\s*\]", train)
            if match:
                return int(match.group(1))
    return None


def _is_disallowed_auto_key(key: str) -> bool:
    lower = key.lower()
    return any(part in lower for part in DISALLOWED_AUTO_KEY_PARTS)


def _list_video_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def _file_matches_key(path: Path, root: Path, key: str, *, total_video_files: int | None = None) -> bool:
    rel = path.relative_to(root)
    if path.parent == root:
        return True
    if total_video_files is None:
        total_video_files = len(_list_video_files(root))
    if total_video_files == 1:
        return True
    normalized_key = key.replace(".", "/")
    rel_no_suffix = str(rel.with_suffix(""))
    rel_posix = rel.as_posix()
    if normalized_key in rel_posix or key in rel_posix:
        return True
    return normalized_key in rel_no_suffix


def _extract_episode_index(path: Path) -> int | None:
    text = path.as_posix()
    patterns = [
        r"episode[_-]?(\d+)",
        r"episodes?[/_-](\d+)",
        r"chunk-\d+/episode_(\d+)",
        r"(\d{6,})(?=\.[^.]+$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    stem_match = re.search(r"(\d+)$", path.stem)
    if stem_match:
        return int(stem_match.group(1))
    return None
