from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .annotation_schema import EpisodeAnnotation, attach_frame_indices
from .config import AppConfig, QwenConfig, SubtaskConfig, load_config
from .dataset_loader import DatasetLoader, EpisodeVideo
from .prompt_builder import build_discovery_prompt, build_global_prompt, build_refine_prompt
from .qwen_client import QwenClient
from .validator import apply_refined_boundaries, validate_and_repair_annotation
from .video_utils import extract_video_clip, make_low_fps_video, save_contact_sheet, save_timeline_plot
from .writer import (
    OutputPaths,
    episode_stem,
    prepare_output_dirs,
    write_all_annotations,
    write_episode_json,
    write_error_json,
    write_validation_summary,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate robot demonstration subtasks from main-camera videos.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/annotate.yaml")
    parser.add_argument("--episode-index", type=int, default=None, help="Only process one episode index")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N episodes")
    parser.add_argument(
        "--discover-subtasks",
        action="store_true",
        help="Ask Qwen to name and segment subtasks from the video instead of using task.subtasks as a closed vocabulary.",
    )
    parser.add_argument("--no-refine", action="store_true", help="Disable Qwen boundary refinement for this run")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, allow_task_discovery=args.discover_subtasks)
    outputs = prepare_output_dirs(config.dataset.output_dir)

    loader = DatasetLoader(config.dataset)
    episodes = _select_episodes(loader.episodes(), episode_index=args.episode_index, limit=args.limit)
    LOGGER.info(
        "Selected video key %s; processing %d episode(s): %s",
        loader.info.selected_video_key,
        len(episodes),
        episodes,
    )

    if config.annotation.write_lerobot_subtask_index:
        LOGGER.warning("write_lerobot_subtask_index=true is not implemented in this minimal version; source dataset is unchanged")

    client = QwenClient(config.qwen, error_dir=outputs.debug_dir)
    successful: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for episode_index in episodes:
        try:
            annotation = process_episode(
                config,
                loader,
                client,
                outputs,
                episode_index,
                enable_refine=not args.no_refine,
                discover_subtasks=args.discover_subtasks,
            )
            annotation_dict = annotation.to_dict()
            write_episode_json(outputs.root, annotation_dict)
            successful.append(annotation_dict)
            validation_rows.append(
                {
                    "episode_index": episode_index,
                    "valid": annotation.validation.valid,
                    "fixed": annotation.validation.fixed,
                    "num_segments": len(annotation.segments),
                    "messages": " | ".join(annotation.validation.messages),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep batch processing alive.
            LOGGER.exception("Episode %s failed", episode_index)
            write_error_json(outputs.errors_dir, episode_index=episode_index, error=repr(exc))
            validation_rows.append(
                {
                    "episode_index": episode_index,
                    "valid": False,
                    "fixed": False,
                    "num_segments": 0,
                    "messages": "",
                    "error": repr(exc),
                }
            )

    write_all_annotations(outputs.root, successful)
    write_validation_summary(outputs.root, validation_rows)
    LOGGER.info("Done. Successful episodes: %d/%d. Output: %s", len(successful), len(episodes), outputs.root)
    return 0 if successful or not episodes else 1


def process_episode(
    config: AppConfig,
    loader: DatasetLoader,
    client: QwenClient,
    outputs: OutputPaths,
    episode_index: int,
    *,
    enable_refine: bool,
    discover_subtasks: bool = False,
) -> EpisodeAnnotation:
    episode = loader.get_episode_video(episode_index)
    stem = episode_stem(episode_index)
    LOGGER.info("Processing %s from %s", stem, episode.video_path)

    global_video = outputs.temp_dir / f"{stem}_global_{config.qwen.fps_for_global:g}fps.mp4"
    make_low_fps_video(episode.video_path, global_video, fps=config.qwen.fps_for_global, width=config.qwen.video_width)

    if discover_subtasks:
        prompt = build_discovery_prompt(config.task.main_task, episode.duration_sec)
    else:
        prompt = build_global_prompt(config.task.main_task, config.task.subtasks, episode.duration_sec)
    (outputs.debug_dir / f"{stem}_global_prompt.txt").write_text(prompt, encoding="utf-8")

    payload = client.annotate_video(global_video, prompt, fps=config.qwen.fps_for_global, request_name=stem)
    annotation_main_task = config.task.main_task
    if discover_subtasks and not annotation_main_task:
        annotation_main_task = str(payload.get("main_task", "")).strip()
    configured_names = [item.name for item in config.task.subtasks]
    segments, validation = validate_and_repair_annotation(
        payload,
        subtask_names=[] if discover_subtasks else configured_names,
        duration_sec=episode.duration_sec,
        force_subtask_order=False if discover_subtasks else config.annotation.force_subtask_order,
        require_all_subtasks_once=False if discover_subtasks else config.annotation.require_all_subtasks_once,
        min_segment_sec=config.annotation.min_segment_sec,
        require_signals=discover_subtasks,
    )
    (outputs.debug_dir / f"{stem}_validation_report_global.json").write_text(
        json.dumps(validation.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not validation.valid:
        LOGGER.warning("Global annotation for %s is invalid after repair: %s", stem, validation.messages)
        (outputs.errors_dir / f"{stem}_validation_error.json").write_text(
            json.dumps(validation.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if (
        enable_refine
        and config.annotation.enable_boundary_refine
        and validation.valid
        and len(segments) > 1
    ):
        refined_boundaries = _refine_boundaries(
            config,
            client,
            outputs,
            episode,
            segments,
            main_task=annotation_main_task,
            discover_subtasks=discover_subtasks,
        )
        segments = apply_refined_boundaries(
            segments,
            refined_boundaries,
            duration_sec=episode.duration_sec,
            min_segment_sec=config.annotation.min_segment_sec,
        )
        refined_payload = {"segments": [segment.to_dict() for segment in segments]}
        segments, refined_validation = validate_and_repair_annotation(
            refined_payload,
            subtask_names=[] if discover_subtasks else configured_names,
            duration_sec=episode.duration_sec,
            force_subtask_order=False if discover_subtasks else config.annotation.force_subtask_order,
            require_all_subtasks_once=False if discover_subtasks else config.annotation.require_all_subtasks_once,
            min_segment_sec=config.annotation.min_segment_sec,
            require_signals=discover_subtasks,
        )
        if refined_validation.fixed or refined_validation.messages:
            validation.fixed = validation.fixed or refined_validation.fixed
            validation.messages.extend([f"refine: {message}" for message in refined_validation.messages])
        validation.valid = validation.valid and refined_validation.valid
        (outputs.debug_dir / f"{stem}_validation_report_refined.json").write_text(
            json.dumps(refined_validation.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    attach_frame_indices(
        segments,
        dataset_fps=episode.dataset_fps,
        num_frames=episode.num_frames,
        subtask_names=[segment.subtask for segment in segments] if discover_subtasks else configured_names,
    )

    annotation = EpisodeAnnotation(
        episode_index=episode.episode_index,
        video_key=episode.video_key,
        video_path=str(episode.video_path),
        dataset_fps=episode.dataset_fps,
        duration_sec=episode.duration_sec,
        num_frames=episode.num_frames,
        main_task=annotation_main_task,
        segments=segments,
        validation=validation,
    )

    _write_debug_visuals(config, outputs, episode, annotation.to_dict())
    return annotation


def _refine_boundaries(
    config: AppConfig,
    client: QwenClient,
    outputs: OutputPaths,
    episode: EpisodeVideo,
    segments: list[Any],
    *,
    main_task: str | None = None,
    qwen_config: QwenConfig | None = None,
    subtask_configs: list[SubtaskConfig] | None = None,
    discover_subtasks: bool = False,
) -> list[float]:
    refined: list[float] = []
    model_config = qwen_config or config.qwen
    name_to_subtask = {item.name: item for item in (subtask_configs or config.task.subtasks)}
    stem = episode_stem(episode.episode_index)
    for idx in range(len(segments) - 1):
        coarse = float(segments[idx].end_time)
        clip_start = max(0.0, coarse - config.annotation.refine_window_sec)
        clip_end = min(episode.duration_sec, coarse + config.annotation.refine_window_sec)
        clip_path = outputs.temp_dir / f"{stem}_boundary_{idx:02d}_{model_config.fps_for_refine:g}fps.mp4"
        extract_video_clip(
            episode.video_path,
            clip_path,
            start_sec=clip_start,
            end_sec=clip_end,
            fps=model_config.fps_for_refine,
            width=model_config.video_width,
        )
        prompt = build_refine_prompt(
            main_task=main_task if main_task is not None else config.task.main_task,
            previous_subtask=_subtask_for_refine(segments[idx], name_to_subtask, discover_subtasks=discover_subtasks),
            next_subtask=_subtask_for_refine(segments[idx + 1], name_to_subtask, discover_subtasks=discover_subtasks),
            clip_start_sec=clip_start,
            clip_end_sec=clip_end,
            coarse_boundary_sec=coarse,
        )
        (outputs.debug_dir / f"{stem}_boundary_{idx:02d}_prompt.txt").write_text(prompt, encoding="utf-8")
        try:
            payload = client.annotate_video(
                clip_path,
                prompt,
                fps=model_config.fps_for_refine,
                request_name=f"{stem}_boundary_{idx:02d}",
            )
            boundary = float(payload.get("boundary_time", coarse))
            if boundary < clip_start or boundary > clip_end:
                LOGGER.warning(
                    "Refined boundary %.3f is outside [%.3f, %.3f]; keeping coarse %.3f",
                    boundary,
                    clip_start,
                    clip_end,
                    coarse,
                )
                boundary = coarse
            refined.append(boundary)
        except Exception as exc:  # noqa: BLE001 - keep episode usable with coarse boundary.
            LOGGER.warning("Boundary refine failed for %s boundary %d: %s", stem, idx, exc)
            (outputs.errors_dir / f"{stem}_boundary_{idx:02d}_error.txt").write_text(repr(exc), encoding="utf-8")
            refined.append(coarse)
    return refined


def _subtask_for_refine(
    segment: Any,
    configured: dict[str, SubtaskConfig],
    *,
    discover_subtasks: bool,
) -> SubtaskConfig:
    if not discover_subtasks:
        return configured[segment.subtask]
    return SubtaskConfig(
        name=str(segment.subtask),
        description=str(getattr(segment, "description", "")),
        visual_start=str(getattr(segment, "start_signal", "")),
        visual_end=str(getattr(segment, "end_signal", "")),
    )


def _write_debug_visuals(config: AppConfig, outputs: OutputPaths, episode: EpisodeVideo, annotation: dict[str, Any]) -> None:
    stem = episode_stem(episode.episode_index)
    segments = list(annotation["segments"])
    if config.annotation.save_timeline_plot:
        try:
            save_timeline_plot(outputs.debug_dir / f"{stem}_timeline.png", segments, duration_sec=episode.duration_sec)
        except Exception as exc:  # noqa: BLE001 - debug artifact should not fail annotation.
            LOGGER.warning("Could not save timeline for %s: %s", stem, exc)
    if config.annotation.save_debug_frames:
        try:
            save_contact_sheet(episode.video_path, outputs.debug_dir / f"{stem}_contact_sheet.jpg", segments)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not save contact sheet for %s: %s", stem, exc)


def _select_episodes(episodes: list[int], *, episode_index: int | None, limit: int | None) -> list[int]:
    selected = list(episodes)
    if episode_index is not None:
        if episode_index not in selected:
            raise ValueError(f"Requested episode_index={episode_index}, but available episodes are {selected}")
        selected = [episode_index]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        selected = selected[:limit]
    return selected
