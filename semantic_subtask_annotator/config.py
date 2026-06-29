from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    fps: float | None = None
    main_video_key: str = "observation.images.main"
    fallback_main_video_keys: list[str] = field(default_factory=list)
    output_dir: Path = Path("./outputs/subtask_annotations")


@dataclass(frozen=True)
class SubtaskConfig:
    name: str
    description: str
    visual_start: str
    visual_end: str


@dataclass(frozen=True)
class TaskConfig:
    main_task: str
    subtasks: list[SubtaskConfig]


@dataclass(frozen=True)
class QwenConfig:
    provider: str = "dashscope"
    model: str = "qwen3-vl-plus"
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url: str = ""
    fps_for_global: float = 1.0
    fps_for_refine: float = 3.0
    video_width: int = 640
    temperature: float = 0.0
    max_retries: int = 3
    timeout_sec: int = 180


@dataclass(frozen=True)
class AnnotationConfig:
    enable_boundary_refine: bool = True
    refine_window_sec: float = 5.0
    min_segment_sec: float = 1.0
    force_subtask_order: bool = True
    require_all_subtasks_once: bool = True
    save_debug_frames: bool = True
    save_timeline_plot: bool = True
    write_lerobot_subtask_index: bool = False


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    task: TaskConfig
    qwen: QwenConfig
    teacher_qwen: QwenConfig
    student_qwen: QwenConfig
    annotation: AnnotationConfig
    config_path: Path | None = None


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _as_path(value: Any, name: str, *, base_dir: Path | None = None) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def _as_optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be null or a number") from exc


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _load_qwen_config(raw: dict[str, Any], prefix: str, defaults: QwenConfig | None = None) -> QwenConfig:
    default = defaults or QwenConfig()
    return QwenConfig(
        provider=str(raw.get("provider", default.provider)),
        model=str(raw.get("model", default.model)),
        api_key_env=str(raw.get("api_key_env", default.api_key_env)),
        base_url=str(raw.get("base_url", default.base_url)),
        fps_for_global=_as_float(raw.get("fps_for_global", default.fps_for_global), f"{prefix}.fps_for_global"),
        fps_for_refine=_as_float(raw.get("fps_for_refine", default.fps_for_refine), f"{prefix}.fps_for_refine"),
        video_width=_as_int(raw.get("video_width", default.video_width), f"{prefix}.video_width"),
        temperature=_as_float(raw.get("temperature", default.temperature), f"{prefix}.temperature"),
        max_retries=_as_int(raw.get("max_retries", default.max_retries), f"{prefix}.max_retries"),
        timeout_sec=_as_int(raw.get("timeout_sec", default.timeout_sec), f"{prefix}.timeout_sec"),
    )


def load_config(path: str | Path, *, allow_task_discovery: bool = False) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load annotate.yaml. Install with: pip install pyyaml") from exc

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _require_mapping(raw, "config")
    dataset_raw = _require_mapping(raw.get("dataset"), "dataset")
    task_raw_value = raw.get("task", {}) if allow_task_discovery else raw.get("task")
    task_raw = _require_mapping(task_raw_value, "task")
    qwen_value = raw.get("qwen", {})
    if qwen_value is None:
        qwen_value = {}
    qwen_raw = _require_mapping(qwen_value, "qwen")
    workflow_raw = raw.get("workflow", {})
    if workflow_raw is None:
        workflow_raw = {}
    workflow_raw = _require_mapping(workflow_raw, "workflow")
    annotation_value = raw.get("annotation", {})
    if annotation_value is None:
        annotation_value = {}
    annotation_raw = _require_mapping(annotation_value, "annotation")

    fallback_keys = dataset_raw.get("fallback_main_video_keys", [])
    if fallback_keys is None:
        fallback_keys = []
    if not isinstance(fallback_keys, list) or not all(isinstance(k, str) for k in fallback_keys):
        raise ValueError("dataset.fallback_main_video_keys must be a list of strings")

    dataset = DatasetConfig(
        root=_as_path(dataset_raw.get("root"), "dataset.root", base_dir=base_dir),
        fps=_as_optional_float(dataset_raw.get("fps"), "dataset.fps"),
        main_video_key=str(dataset_raw.get("main_video_key", "observation.images.main")),
        fallback_main_video_keys=list(fallback_keys),
        output_dir=_as_path(
            dataset_raw.get("output_dir", "./outputs/subtask_annotations"),
            "dataset.output_dir",
            base_dir=base_dir,
        ),
    )

    subtasks_raw = task_raw.get("subtasks", [])
    if subtasks_raw is None:
        subtasks_raw = []
    if not isinstance(subtasks_raw, list) or (not subtasks_raw and not allow_task_discovery):
        raise ValueError("task.subtasks must be a non-empty list")
    subtasks: list[SubtaskConfig] = []
    seen_names: set[str] = set()
    for idx, item in enumerate(subtasks_raw):
        item = _require_mapping(item, f"task.subtasks[{idx}]")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"task.subtasks[{idx}].name must be non-empty")
        if name in seen_names:
            raise ValueError(f"duplicate subtask name: {name}")
        seen_names.add(name)
        subtasks.append(
            SubtaskConfig(
                name=name,
                description=str(item.get("description", "")),
                visual_start=str(item.get("visual_start", "")),
                visual_end=str(item.get("visual_end", "")),
            )
        )
    main_task = str(task_raw.get("main_task", "")).strip()
    if not main_task and not allow_task_discovery:
        raise ValueError("task.main_task must be non-empty")
    task = TaskConfig(main_task=main_task, subtasks=subtasks)

    qwen = _load_qwen_config(qwen_raw, "qwen")
    teacher_qwen = _load_qwen_config(
        _require_mapping(workflow_raw.get("teacher_qwen", {}), "workflow.teacher_qwen"),
        "workflow.teacher_qwen",
        qwen,
    )
    student_defaults = QwenConfig(
        provider=qwen.provider,
        model=qwen.model,
        api_key_env=qwen.api_key_env,
        base_url=qwen.base_url,
        fps_for_global=qwen.fps_for_global,
        fps_for_refine=qwen.fps_for_refine,
        video_width=qwen.video_width,
        temperature=qwen.temperature,
        max_retries=qwen.max_retries,
        timeout_sec=qwen.timeout_sec,
    )
    student_qwen = _load_qwen_config(
        _require_mapping(workflow_raw.get("student_qwen", {}), "workflow.student_qwen"),
        "workflow.student_qwen",
        student_defaults,
    )

    annotation = AnnotationConfig(
        enable_boundary_refine=bool(annotation_raw.get("enable_boundary_refine", True)),
        refine_window_sec=_as_float(annotation_raw.get("refine_window_sec", 5.0), "annotation.refine_window_sec"),
        min_segment_sec=_as_float(annotation_raw.get("min_segment_sec", 1.0), "annotation.min_segment_sec"),
        force_subtask_order=bool(annotation_raw.get("force_subtask_order", True)),
        require_all_subtasks_once=bool(annotation_raw.get("require_all_subtasks_once", True)),
        save_debug_frames=bool(annotation_raw.get("save_debug_frames", True)),
        save_timeline_plot=bool(annotation_raw.get("save_timeline_plot", True)),
        write_lerobot_subtask_index=bool(annotation_raw.get("write_lerobot_subtask_index", False)),
    )

    for name, model_config in (("qwen", qwen), ("workflow.teacher_qwen", teacher_qwen), ("workflow.student_qwen", student_qwen)):
        if model_config.provider.lower() not in {"dashscope", "openai_compatible"}:
            raise ValueError(f"{name}.provider must be 'dashscope' or 'openai_compatible'")
        if model_config.fps_for_global <= 0 or model_config.fps_for_refine <= 0:
            raise ValueError(f"{name} fps values must be positive")
        if model_config.video_width <= 0:
            raise ValueError(f"{name}.video_width must be positive")
        if model_config.provider.lower() == "openai_compatible" and not model_config.base_url.strip():
            raise ValueError(f"{name}.base_url must be set for provider='openai_compatible'")
    if annotation.refine_window_sec <= 0:
        raise ValueError("annotation.refine_window_sec must be positive")
    if annotation.min_segment_sec < 0:
        raise ValueError("annotation.min_segment_sec must be non-negative")

    return AppConfig(
        dataset=dataset,
        task=task,
        qwen=qwen,
        teacher_qwen=teacher_qwen,
        student_qwen=student_qwen,
        annotation=annotation,
        config_path=config_path,
    )
