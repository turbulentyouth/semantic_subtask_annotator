#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_subtask_annotator.config import load_config  # noqa: E402
from semantic_subtask_annotator.dataset_loader import DatasetLoader  # noqa: E402
from semantic_subtask_annotator.video_utils import probe_video  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether a dataset matches the annotator config.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "annotate.yaml")
    parser.add_argument("--limit", type=int, default=5, help="Number of episode video paths to print.")
    parser.add_argument("--probe", action="store_true", help="Probe sample videos for fps/frame/duration metadata.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    loader = DatasetLoader(config.dataset)

    print("Dataset root:", loader.info.root)
    print("Configured main_video_key:", config.dataset.main_video_key)
    print("Selected video_key:", loader.info.selected_video_key)
    print("Dataset fps:", loader.info.fps)
    print("Available video/image keys:")
    for key in loader.info.video_keys:
        print("  -", key)

    print("\nVideo file counts by key:")
    for key, count in _count_video_files_by_key(loader.info.root).most_common():
        print(f"  {key}: {count}")

    metadata_total = _read_total_episodes(loader.info.root)
    if metadata_total is not None:
        print("\nmeta/info.json total_episodes:", metadata_total)
    print("Video-backed selected episodes:", len(loader.episodes()))
    if loader.episodes():
        print("Episode index range:", loader.episodes()[0], "to", loader.episodes()[-1])

    if metadata_total is not None:
        missing = sorted(set(range(metadata_total)) - set(loader.episodes()))
        if missing:
            print("Missing selected-camera videos for metadata episodes:", missing[:30])
            if len(missing) > 30:
                print(f"... and {len(missing) - 30} more")

    print("\nSample selected-camera episode videos:")
    for episode_index in loader.episodes()[: args.limit]:
        path = loader.find_episode_video_path(episode_index)
        print(f"  episode {episode_index:06d}: {path}")
        if args.probe and path is not None:
            try:
                metadata = probe_video(path)
                print(
                    "    "
                    f"fps={metadata.fps:.3f}, frames={metadata.num_frames}, "
                    f"duration={metadata.duration_sec:.3f}s, size={metadata.width}x{metadata.height}"
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic script should keep reporting.
                print(f"    probe failed: {exc}")

    print("\nSampling policy used by annotation:")
    print(
        f"  Global pass: creates a temporary low-FPS video at {config.qwen.fps_for_global:g} FPS, "
        f"width {config.qwen.video_width}px, and sends that to Qwen."
    )
    if config.annotation.enable_boundary_refine:
        print(
            f"  Boundary refine: for each adjacent boundary, clips +/- {config.annotation.refine_window_sec:g}s "
            f"and samples at {config.qwen.fps_for_refine:g} FPS."
        )
    else:
        print("  Boundary refine: disabled.")
    print("  The Qwen API does not receive state/action/wrist/tactile streams.")
    print("  Local ffmpeg/OpenCV still decodes the source video to create sampled clips.")
    return 0


def _count_video_files_by_key(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    videos_root = root / "videos"
    if not videos_root.exists():
        return counts
    for path in videos_root.rglob("*.mp4"):
        rel = path.relative_to(videos_root)
        if len(rel.parts) >= 3:
            counts[rel.parts[0]] += 1
    return counts


def _read_total_episodes(root: Path) -> int | None:
    path = root / "meta" / "info.json"
    if not path.exists():
        return None
    info = json.loads(path.read_text(encoding="utf-8"))
    value = info.get("total_episodes")
    return int(value) if isinstance(value, int) else None


if __name__ == "__main__":
    raise SystemExit(main())
