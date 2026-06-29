from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    debug_dir: Path
    temp_dir: Path
    errors_dir: Path


def prepare_output_dirs(output_dir: Path) -> OutputPaths:
    root = output_dir.expanduser().resolve()
    debug = root / "debug"
    temp = root / "tmp"
    errors = root / "errors"
    for path in (root, debug, temp, errors):
        path.mkdir(parents=True, exist_ok=True)
    return OutputPaths(root=root, debug_dir=debug, temp_dir=temp, errors_dir=errors)


def episode_stem(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def write_episode_json(output_dir: Path, annotation: dict[str, Any]) -> Path:
    episode_index = int(annotation["episode_index"])
    path = output_dir / f"{episode_stem(episode_index)}.json"
    path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_error_json(output_dir: Path, *, episode_index: int, error: str, context: dict[str, Any] | None = None) -> Path:
    payload = {"episode_index": episode_index, "error": error, "context": context or {}}
    path = output_dir / f"{episode_stem(episode_index)}_error.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_all_annotations(output_dir: Path, annotations: list[dict[str, Any]]) -> None:
    jsonl_path = output_dir / "all_annotations.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in annotations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    parquet_path = output_dir / "all_annotations.parquet"
    if not annotations:
        LOGGER.info("No successful annotations; skipping %s", parquet_path)
        return
    rows = []
    for item in annotations:
        rows.append(
            {
                "episode_index": item["episode_index"],
                "video_key": item["video_key"],
                "video_path": item["video_path"],
                "dataset_fps": item["dataset_fps"],
                "duration_sec": item["duration_sec"],
                "num_frames": item["num_frames"],
                "main_task": item["main_task"],
                "segments_json": json.dumps(item["segments"], ensure_ascii=False),
                "validation_valid": item["validation"]["valid"],
                "validation_fixed": item["validation"]["fixed"],
                "validation_messages": json.dumps(item["validation"]["messages"], ensure_ascii=False),
            }
        )
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    except Exception as exc:  # noqa: BLE001 - optional dependency/runtime codec.
        LOGGER.warning("Could not write parquet %s: %s", parquet_path, exc)
        fallback = output_dir / "all_annotations.parquet_error.txt"
        fallback.write_text(
            "Failed to write parquet. Install pandas and pyarrow or fastparquet.\n"
            f"Original error: {exc}\n",
            encoding="utf-8",
        )


def write_validation_summary(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_dir / "validation_summary.csv"
    fieldnames = [
        "episode_index",
        "valid",
        "fixed",
        "num_segments",
        "messages",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path
