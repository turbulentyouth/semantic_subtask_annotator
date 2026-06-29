from __future__ import annotations

from typing import Any

from .annotation_schema import Segment, ValidationReport, format_timestamp


def validate_and_repair_annotation(
    payload: dict[str, Any],
    *,
    subtask_names: list[str],
    duration_sec: float,
    force_subtask_order: bool,
    require_all_subtasks_once: bool,
    min_segment_sec: float,
    require_signals: bool = False,
    tolerance_sec: float = 0.5,
) -> tuple[list[Segment], ValidationReport]:
    report = ValidationReport(valid=True)
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        return [], ValidationReport(valid=False, messages=["JSON must contain a non-empty segments list"])

    segments: list[Segment] = []
    for idx, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            report.add(f"segments[{idx}] is not an object")
            report.valid = False
            continue
        subtask = str(item.get("subtask", "")).strip()
        if not subtask:
            report.add(f"segments[{idx}].subtask must be non-empty")
            report.valid = False
        elif subtask_names and subtask not in subtask_names:
            report.add(f"segments[{idx}].subtask {subtask!r} is not in the closed vocabulary")
            report.valid = False
        description = str(item.get("description", "")).strip()
        start_signal = str(item.get("start_signal", "")).strip()
        end_signal = str(item.get("end_signal", "")).strip()
        if require_signals:
            if not description:
                report.add(f"segments[{idx}].description must be non-empty")
                report.valid = False
            if not start_signal:
                report.add(f"segments[{idx}].start_signal must be non-empty")
                report.valid = False
            if not end_signal:
                report.add(f"segments[{idx}].end_signal must be non-empty")
                report.valid = False
        try:
            start_time = float(item.get("start_time"))
            end_time = float(item.get("end_time"))
        except (TypeError, ValueError):
            report.add(f"segments[{idx}] start_time/end_time must be numeric")
            report.valid = False
            continue
        confidence = _coerce_confidence(item.get("confidence", 0.0), report, idx)
        segments.append(
            Segment(
                subtask=subtask,
                start_time=start_time,
                end_time=end_time,
                description=description,
                start_signal=start_signal,
                end_signal=end_signal,
                boundary_reason=str(item.get("boundary_reason", "")),
                confidence=confidence,
                start_timestamp=str(item["start_timestamp"]) if item.get("start_timestamp") is not None else None,
                end_timestamp=str(item["end_timestamp"]) if item.get("end_timestamp") is not None else None,
            )
        )

    if not segments:
        report.valid = False
        return [], report

    segments.sort(key=lambda segment: (segment.start_time, segment.end_time))
    _repair_boundary_endpoints(segments, duration_sec, report)
    _repair_small_gaps(segments, report, tolerance_sec=tolerance_sec)
    _clamp_segments(segments, duration_sec, report)
    _refresh_timestamps(segments)

    for idx, segment in enumerate(segments):
        if segment.start_time < -1e-9:
            report.add(f"segments[{idx}].start_time is negative")
            report.valid = False
        if segment.end_time > duration_sec + 1e-9:
            report.add(f"segments[{idx}].end_time exceeds duration_sec")
            report.valid = False
        if segment.start_time >= segment.end_time:
            report.add(f"segments[{idx}] has start_time >= end_time")
            report.valid = False
        if min_segment_sec > 0 and (segment.end_time - segment.start_time) < min_segment_sec:
            report.add(
                f"segments[{idx}] duration {segment.end_time - segment.start_time:.3f}s is shorter than min_segment_sec={min_segment_sec:.3f}"
            )
            report.valid = False

    if abs(segments[0].start_time - 0.0) > 1e-6:
        report.add("first segment does not start at 0.0")
        report.valid = False
    if abs(segments[-1].end_time - duration_sec) > 1e-6:
        report.add("last segment does not end at duration_sec")
        report.valid = False
    for idx in range(len(segments) - 1):
        if abs(segments[idx].end_time - segments[idx + 1].start_time) > 1e-6:
            report.add(f"segments[{idx}] and segments[{idx + 1}] are not continuous")
            report.valid = False

    if force_subtask_order and subtask_names:
        _validate_order(segments, subtask_names, report)
    if require_all_subtasks_once and subtask_names:
        _validate_all_once(segments, subtask_names, report)

    return segments, report


def apply_refined_boundaries(
    segments: list[Segment],
    boundaries: list[float],
    *,
    duration_sec: float,
    min_segment_sec: float,
) -> list[Segment]:
    if len(boundaries) != max(0, len(segments) - 1):
        raise ValueError("boundary count must be len(segments) - 1")
    fixed = [
        Segment(
            subtask=s.subtask,
            start_time=s.start_time,
            end_time=s.end_time,
            description=s.description,
            start_signal=s.start_signal,
            end_signal=s.end_signal,
            boundary_reason=s.boundary_reason,
            confidence=s.confidence,
        )
        for s in segments
    ]
    previous = 0.0
    for idx, raw_boundary in enumerate(boundaries):
        lower = previous + min_segment_sec
        remaining = len(fixed) - idx - 1
        upper = duration_sec - remaining * min_segment_sec
        boundary = min(max(float(raw_boundary), lower), upper)
        fixed[idx].start_time = previous
        fixed[idx].end_time = boundary
        fixed[idx + 1].start_time = boundary
        previous = boundary
    fixed[-1].end_time = duration_sec
    _refresh_timestamps(fixed)
    return fixed


def _coerce_confidence(value: Any, report: ValidationReport, idx: int) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        report.add(f"segments[{idx}].confidence is not numeric; set to 0.0")
        report.fixed = True
        return 0.0
    if confidence < 0.0 or confidence > 1.0:
        report.add(f"segments[{idx}].confidence out of [0, 1]; clamped")
        report.fixed = True
        confidence = min(max(confidence, 0.0), 1.0)
    return confidence


def _repair_boundary_endpoints(segments: list[Segment], duration_sec: float, report: ValidationReport) -> None:
    if abs(segments[0].start_time - 0.0) <= 0.5:
        if segments[0].start_time != 0.0:
            segments[0].start_time = 0.0
            report.fixed = True
            report.add("fixed first segment start_time to 0.0")
    if abs(segments[-1].end_time - duration_sec) <= 0.5:
        if segments[-1].end_time != duration_sec:
            segments[-1].end_time = duration_sec
            report.fixed = True
            report.add("fixed last segment end_time to duration_sec")


def _repair_small_gaps(segments: list[Segment], report: ValidationReport, *, tolerance_sec: float) -> None:
    for idx in range(len(segments) - 1):
        left = segments[idx]
        right = segments[idx + 1]
        delta = right.start_time - left.end_time
        if abs(delta) <= tolerance_sec:
            boundary = (left.end_time + right.start_time) / 2.0
            if abs(delta) > 1e-9:
                report.fixed = True
                report.add(f"fixed small gap/overlap between segments[{idx}] and segments[{idx + 1}]")
            left.end_time = boundary
            right.start_time = boundary


def _clamp_segments(segments: list[Segment], duration_sec: float, report: ValidationReport) -> None:
    for idx, segment in enumerate(segments):
        old_start, old_end = segment.start_time, segment.end_time
        segment.start_time = min(max(segment.start_time, 0.0), duration_sec)
        segment.end_time = min(max(segment.end_time, 0.0), duration_sec)
        if segment.start_time != old_start or segment.end_time != old_end:
            report.fixed = True
            report.add(f"clamped segment[{idx}] to episode duration")


def _refresh_timestamps(segments: list[Segment]) -> None:
    for segment in segments:
        segment.start_timestamp = format_timestamp(segment.start_time)
        segment.end_timestamp = format_timestamp(segment.end_time)


def _validate_order(segments: list[Segment], subtask_names: list[str], report: ValidationReport) -> None:
    indices = [subtask_names.index(s.subtask) for s in segments if s.subtask in subtask_names]
    if indices != sorted(indices):
        report.add("segments are not in configured subtask order")
        report.valid = False


def _validate_all_once(segments: list[Segment], subtask_names: list[str], report: ValidationReport) -> None:
    observed = [s.subtask for s in segments]
    expected = list(subtask_names)
    if observed != expected:
        report.add(f"segments must contain each configured subtask exactly once in order; observed={observed!r}")
        report.valid = False
