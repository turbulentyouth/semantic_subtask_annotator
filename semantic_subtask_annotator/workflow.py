from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .annotation_schema import EpisodeAnnotation, attach_frame_indices
from .cli import _refine_boundaries, _select_episodes, _write_debug_visuals
from .config import AppConfig, load_config
from .dataset_loader import DatasetLoader
from .prompt_builder import build_student_segmentation_prompt, build_teacher_discovery_prompt
from .qwen_client import QwenClient
from .task_spec import TaskSpec, load_task_spec, write_task_spec
from .validator import apply_refined_boundaries, validate_and_repair_annotation
from .video_utils import make_low_fps_video
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
    parser = argparse.ArgumentParser(description="Run staged teacher/student subtask annotation workflow.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["all", "bootstrap", "segment"],
        default=None,
        help="Shortcut stage command. Defaults to all.",
    )
    parser.add_argument("--config", default=Path("configs/annotate.yaml"), type=Path, help="Path to configs/annotate.yaml")
    parser.add_argument(
        "--stage",
        choices=["bootstrap", "segment", "all"],
        default=None,
        help="bootstrap writes task_spec from ep0; segment annotates target episodes with that spec; all runs both.",
    )
    parser.add_argument("--bootstrap-episode-index", type=int, default=0, help="Episode used by the teacher model.")
    parser.add_argument("--episode-index", type=int, default=None, help="Only segment one target episode")
    parser.add_argument("--ep", dest="episode_index", type=int, help="Alias for --episode-index")
    parser.add_argument("--limit", type=int, default=None, help="Segment at most N target episodes")
    parser.add_argument("--task-spec", type=Path, default=None, help="Path to task_spec.json")
    parser.add_argument("--no-refine", action="store_true", help="Disable boundary refinement for this workflow run")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stage = args.stage or args.command or "all"
    config = load_config(args.config, allow_task_discovery=True)
    outputs = prepare_output_dirs(config.dataset.output_dir)
    spec_path = args.task_spec or (outputs.root / "task_spec.json")

    loader = DatasetLoader(config.dataset)
    if config.annotation.write_lerobot_subtask_index:
        LOGGER.warning("write_lerobot_subtask_index=true is not implemented in this minimal version; source dataset is unchanged")

    if stage in {"bootstrap", "all"}:
        teacher = QwenClient(config.teacher_qwen, error_dir=outputs.debug_dir)
        spec = run_bootstrap_stage(
            config,
            loader,
            teacher,
            outputs,
            episode_index=args.bootstrap_episode_index,
            spec_path=spec_path,
            enable_refine=not args.no_refine,
        )
        LOGGER.info("Wrote task spec with %d subtasks: %s", len(spec.subtasks), spec_path)

    if stage in {"segment", "all"}:
        spec = load_task_spec(spec_path)
        student = QwenClient(config.student_qwen, error_dir=outputs.debug_dir)
        target_episodes = _select_target_episodes(
            loader.episodes(),
            bootstrap_episode_index=args.bootstrap_episode_index,
            episode_index=args.episode_index,
            limit=args.limit,
        )
        return run_segment_stage(
            config,
            loader,
            student,
            outputs,
            spec,
            target_episodes,
            enable_refine=not args.no_refine,
        )
    return 0


def run_bootstrap_stage(
    config: AppConfig,
    loader: DatasetLoader,
    client: QwenClient,
    outputs: OutputPaths,
    *,
    episode_index: int,
    spec_path: Path,
    enable_refine: bool,
) -> TaskSpec:
    annotation = process_teacher_episode(config, loader, client, outputs, episode_index, enable_refine=enable_refine)
    annotation_dict = annotation.to_dict()
    write_episode_json(outputs.root, annotation_dict)
    spec = TaskSpec.from_annotation(annotation)
    write_task_spec(spec_path, spec)
    return spec


def process_teacher_episode(
    config: AppConfig,
    loader: DatasetLoader,
    client: QwenClient,
    outputs: OutputPaths,
    episode_index: int,
    *,
    enable_refine: bool,
) -> EpisodeAnnotation:
    episode = loader.get_episode_video(episode_index)
    stem = episode_stem(episode_index)
    LOGGER.info("Teacher bootstrap on %s from %s", stem, episode.video_path)

    global_video = outputs.temp_dir / f"{stem}_teacher_{config.teacher_qwen.fps_for_global:g}fps.mp4"
    make_low_fps_video(
        episode.video_path,
        global_video,
        fps=config.teacher_qwen.fps_for_global,
        width=config.teacher_qwen.video_width,
    )

    prompt = build_teacher_discovery_prompt(config.task.main_task, episode.duration_sec)
    (outputs.debug_dir / f"{stem}_teacher_prompt.txt").write_text(prompt, encoding="utf-8")
    payload = client.annotate_video(global_video, prompt, fps=config.teacher_qwen.fps_for_global, request_name=f"{stem}_teacher")

    main_task = config.task.main_task or str(payload.get("main_task", "")).strip()
    segments, validation = validate_and_repair_annotation(
        payload,
        subtask_names=[],
        duration_sec=episode.duration_sec,
        force_subtask_order=False,
        require_all_subtasks_once=False,
        min_segment_sec=config.annotation.min_segment_sec,
        require_signals=True,
    )
    (outputs.debug_dir / f"{stem}_teacher_validation_report_global.json").write_text(
        json.dumps(validation.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if enable_refine and config.annotation.enable_boundary_refine and validation.valid and len(segments) > 1:
        refined = _refine_boundaries(
            config,
            client,
            outputs,
            episode,
            segments,
            main_task=main_task,
            qwen_config=config.teacher_qwen,
            discover_subtasks=True,
        )
        segments = apply_refined_boundaries(
            segments,
            refined,
            duration_sec=episode.duration_sec,
            min_segment_sec=config.annotation.min_segment_sec,
        )
        refined_payload = {"segments": [segment.to_dict() for segment in segments]}
        segments, refined_validation = validate_and_repair_annotation(
            refined_payload,
            subtask_names=[],
            duration_sec=episode.duration_sec,
            force_subtask_order=False,
            require_all_subtasks_once=False,
            min_segment_sec=config.annotation.min_segment_sec,
            require_signals=True,
        )
        if refined_validation.fixed or refined_validation.messages:
            validation.fixed = validation.fixed or refined_validation.fixed
            validation.messages.extend([f"refine: {message}" for message in refined_validation.messages])
        validation.valid = validation.valid and refined_validation.valid
        (outputs.debug_dir / f"{stem}_teacher_validation_report_refined.json").write_text(
            json.dumps(refined_validation.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if not validation.valid:
        (outputs.errors_dir / f"{stem}_teacher_validation_error.json").write_text(
            json.dumps(validation.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    attach_frame_indices(
        segments,
        dataset_fps=episode.dataset_fps,
        num_frames=episode.num_frames,
        subtask_names=[segment.subtask for segment in segments],
    )
    annotation = EpisodeAnnotation(
        episode_index=episode.episode_index,
        video_key=episode.video_key,
        video_path=str(episode.video_path),
        dataset_fps=episode.dataset_fps,
        duration_sec=episode.duration_sec,
        num_frames=episode.num_frames,
        main_task=main_task,
        segments=segments,
        validation=validation,
    )
    _write_debug_visuals(config, outputs, episode, annotation.to_dict())
    return annotation


def run_segment_stage(
    config: AppConfig,
    loader: DatasetLoader,
    client: QwenClient,
    outputs: OutputPaths,
    spec: TaskSpec,
    episodes: list[int],
    *,
    enable_refine: bool,
) -> int:
    LOGGER.info("Student segment stage processing %d episode(s): %s", len(episodes), episodes)
    successful: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for episode_index in episodes:
        try:
            annotation = process_student_episode(
                config,
                loader,
                client,
                outputs,
                spec,
                episode_index,
                enable_refine=enable_refine,
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
            LOGGER.exception("Student segmentation failed for episode %s", episode_index)
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
    LOGGER.info("Student stage done. Successful episodes: %d/%d. Output: %s", len(successful), len(episodes), outputs.root)
    return 0 if successful or not episodes else 1


def process_student_episode(
    config: AppConfig,
    loader: DatasetLoader,
    client: QwenClient,
    outputs: OutputPaths,
    spec: TaskSpec,
    episode_index: int,
    *,
    enable_refine: bool,
) -> EpisodeAnnotation:
    episode = loader.get_episode_video(episode_index)
    stem = episode_stem(episode_index)
    LOGGER.info("Student segmentation on %s from %s", stem, episode.video_path)

    global_video = outputs.temp_dir / f"{stem}_student_{config.student_qwen.fps_for_global:g}fps.mp4"
    make_low_fps_video(
        episode.video_path,
        global_video,
        fps=config.student_qwen.fps_for_global,
        width=config.student_qwen.video_width,
    )

    prompt = build_student_segmentation_prompt(spec, episode.duration_sec)
    (outputs.debug_dir / f"{stem}_student_prompt.txt").write_text(prompt, encoding="utf-8")
    payload = client.annotate_video(global_video, prompt, fps=config.student_qwen.fps_for_global, request_name=f"{stem}_student")

    subtask_names = [item.name for item in spec.subtasks]
    segments, validation = validate_and_repair_annotation(
        payload,
        subtask_names=subtask_names,
        duration_sec=episode.duration_sec,
        force_subtask_order=True,
        require_all_subtasks_once=True,
        min_segment_sec=config.annotation.min_segment_sec,
    )
    (outputs.debug_dir / f"{stem}_student_validation_report_global.json").write_text(
        json.dumps(validation.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if enable_refine and config.annotation.enable_boundary_refine and validation.valid and len(segments) > 1:
        refined = _refine_boundaries(
            config,
            client,
            outputs,
            episode,
            segments,
            main_task=spec.main_task,
            qwen_config=config.student_qwen,
            subtask_configs=spec.subtasks,
            discover_subtasks=False,
        )
        segments = apply_refined_boundaries(
            segments,
            refined,
            duration_sec=episode.duration_sec,
            min_segment_sec=config.annotation.min_segment_sec,
        )
        refined_payload = {"segments": [segment.to_dict() for segment in segments]}
        segments, refined_validation = validate_and_repair_annotation(
            refined_payload,
            subtask_names=subtask_names,
            duration_sec=episode.duration_sec,
            force_subtask_order=True,
            require_all_subtasks_once=True,
            min_segment_sec=config.annotation.min_segment_sec,
        )
        if refined_validation.fixed or refined_validation.messages:
            validation.fixed = validation.fixed or refined_validation.fixed
            validation.messages.extend([f"refine: {message}" for message in refined_validation.messages])
        validation.valid = validation.valid and refined_validation.valid
        (outputs.debug_dir / f"{stem}_student_validation_report_refined.json").write_text(
            json.dumps(refined_validation.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if not validation.valid:
        (outputs.errors_dir / f"{stem}_student_validation_error.json").write_text(
            json.dumps(validation.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    attach_frame_indices(
        segments,
        dataset_fps=episode.dataset_fps,
        num_frames=episode.num_frames,
        subtask_names=subtask_names,
    )
    annotation = EpisodeAnnotation(
        episode_index=episode.episode_index,
        video_key=episode.video_key,
        video_path=str(episode.video_path),
        dataset_fps=episode.dataset_fps,
        duration_sec=episode.duration_sec,
        num_frames=episode.num_frames,
        main_task=spec.main_task,
        segments=segments,
        validation=validation,
    )
    _write_debug_visuals(config, outputs, episode, annotation.to_dict())
    return annotation


def _select_target_episodes(
    episodes: list[int],
    *,
    bootstrap_episode_index: int,
    episode_index: int | None,
    limit: int | None,
) -> list[int]:
    selected = _select_episodes(episodes, episode_index=episode_index, limit=None)
    if episode_index is None:
        selected = [idx for idx in selected if idx != bootstrap_episode_index]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        selected = selected[:limit]
    return selected
