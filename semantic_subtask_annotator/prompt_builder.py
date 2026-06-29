from __future__ import annotations

import json

from .config import SubtaskConfig
from .task_spec import TaskSpec


def build_global_prompt(main_task: str, subtasks: list[SubtaskConfig], duration_sec: float) -> str:
    subtask_lines = "\n".join(
        [
            (
                f"{idx}. name: {item.name}\n"
                f"   description: {item.description}\n"
                f"   visual_start: {item.visual_start}\n"
                f"   visual_end: {item.visual_end}"
            )
            for idx, item in enumerate(subtasks)
        ]
    )
    schema = {
        "main_task": main_task,
        "duration_sec": round(float(duration_sec), 3),
        "segments": [
            {
                "subtask": subtasks[0].name if subtasks else "subtask_name",
                "start_time": 0.0,
                "end_time": 31.0,
                "start_timestamp": "00:00",
                "end_timestamp": "00:31",
                "boundary_reason": "夹爪已经移动到数据线附近，后续开始对准微调",
                "confidence": 0.82,
            }
        ],
    }
    allowed_names = ", ".join(item.name for item in subtasks)
    return f"""你需要只根据主摄像头视频，把机器人示教 episode 划分为连续子任务片段。

重要限制：
- 只能使用视频画面做语义判断，不要假设你能看到状态、动作、腕部摄像头、夹爪传感器或机器人内部变量。
- 子任务标签是封闭词表，只能从候选子任务名中选择，不能创造新标签。
- 候选子任务名：{allowed_names}
- 所有片段必须按时间顺序排列，并连续覆盖整个视频。
- 第一个片段 start_time 必须是 0.0。
- 最后一个片段 end_time 必须是 {float(duration_sec):.3f}。
- 前一个片段 end_time 必须等于后一个片段 start_time。
- 不要平均切分；边界必须对应视觉上的真实阶段变化。
- 边界应对应视觉事件，例如靠近、对准、夹爪闭合、物体离桌、移动到目标上方、夹爪张开、机械臂撤离。
- 如果某个动作不明显，以最接近的视觉变化时刻为边界。
- 如果存在停顿或犹豫，把停顿归入前一个明确子任务。
- confidence 是 0 到 1 的数字，表示你对该片段边界和标签的信心。

主任务：
{main_task}

候选子任务，必须保持这个顺序：
{subtask_lines}

视频总时长：{float(duration_sec):.3f} 秒。

只输出 JSON，不要输出 Markdown，不要解释，不要包裹代码块。
JSON schema 固定如下，字段名必须一致：
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def build_discovery_prompt(main_task: str, duration_sec: float) -> str:
    schema = {
        "main_task": main_task,
        "duration_sec": round(float(duration_sec), 3),
        "segments": [
            {
                "subtask": "concise_snake_case_subtask_name",
                "description": "这个子任务在视频中的语义动作目标",
                "start_time": 0.0,
                "end_time": 31.0,
                "start_timestamp": "00:00",
                "end_timestamp": "00:31",
                "start_signal": "画面中表示该子任务开始的视觉信号",
                "end_signal": "画面中表示该子任务结束的视觉信号",
                "boundary_reason": "为什么在这个时间点切换到下一个子任务",
                "confidence": 0.82,
            }
        ],
    }
    task_block = main_task.strip() if main_task.strip() else "未提供。请只根据视频内容推断整体任务。"
    return f"""你需要只根据主摄像头视频，主动发现并命名机器人示教 episode 中的连续语义子任务。

重要限制：
- 只能使用视频画面做语义判断，不要假设你能看到状态、动作、腕部摄像头、夹爪传感器或机器人内部变量。
- 不会预先给你候选子任务列表；你必须根据视频中真实发生的操作阶段自行命名和划分子任务。
- 子任务名必须简洁、稳定、可复用，使用英文 snake_case，例如 grasp_object、move_to_box、release_object。
- 不要使用 step_1、phase_2、unknown、misc 这类没有语义的信息。
- 每个子任务必须包含 description、start_signal 和 end_signal。
- start_signal 必须描述画面中可观察到的开始信号，例如机械臂开始接近目标、夹爪开始闭合、物体刚离开桌面。
- end_signal 必须描述画面中可观察到的结束信号，例如目标物已被放入容器、夹爪张开、机械臂撤离、下一目标开始被接近。
- 所有片段必须按时间顺序排列，并连续覆盖整个视频。
- 第一个片段 start_time 必须是 0.0。
- 最后一个片段 end_time 必须是 {float(duration_sec):.3f}。
- 前一个片段 end_time 必须等于后一个片段 start_time。
- 不要平均切分；边界必须对应视觉上的真实阶段变化。
- 边界应对应视觉事件，例如靠近、对准、夹爪闭合、物体离桌、移动到目标上方、夹爪张开、机械臂撤离。
- 如果一个动作阶段重复发生，需要按发生顺序拆成多个片段，并用语义清晰的名字区分，例如 right_to_left_shoe_1_take_out。
- 如果存在停顿或犹豫，把停顿归入前一个明确子任务。
- confidence 是 0 到 1 的数字，表示你对该片段边界和命名的信心。

主任务提示：
{task_block}

视频总时长：{float(duration_sec):.3f} 秒。

只输出 JSON，不要输出 Markdown，不要解释，不要包裹代码块。
JSON schema 固定如下，字段名必须一致：
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def build_teacher_discovery_prompt(main_task: str, duration_sec: float) -> str:
    schema = {
        "main_task": main_task or "模型根据视频总结出的完整主任务",
        "duration_sec": round(float(duration_sec), 3),
        "segments": [
            {
                "subtask": "stable_snake_case_subtask_name",
                "description": "这个子任务的可复用语义定义，后续 episode 也应按此理解",
                "start_time": 0.0,
                "end_time": 31.0,
                "start_timestamp": "00:00",
                "end_timestamp": "00:31",
                "start_signal": "画面中表示该子任务开始的可观察信号",
                "end_signal": "画面中表示该子任务结束的可观察信号",
                "boundary_reason": "ep0 中为什么在这个时间点切换到下一个子任务",
                "confidence": 0.92,
            }
        ],
    }
    task_block = main_task.strip() if main_task.strip() else "未提供。请根据 ep0 视频总结一个完整、可复用的主任务描述。"
    return f"""你是 teacher 模型。你需要只根据第一个主摄像头 episode 视频，建立后续所有 episode 都要复用的任务规范，并同时标注 ep0。

工作目标：
- 主动总结 main_task。
- 主动发现、命名并排序一组可复用子任务。
- 为每个子任务写出 description、start_signal、end_signal，供后续小模型按闭集标签划分其它 episode。
- 同时给出 ep0 的连续时间片段。

重要限制：
- 只能使用视频画面做语义判断，不要假设你能看到状态、动作、腕部摄像头、夹爪传感器或机器人内部变量。
- 不会预先给你候选子任务列表；你必须根据 ep0 中真实发生的操作阶段自行命名。
- 子任务名必须简洁、稳定、可复用，使用英文 snake_case。
- 不要使用 step_1、phase_2、unknown、misc 这类没有语义的信息。
- 如果同一语义动作在一次完整任务中重复出现，但对象/方向/轮次不同，应拆成不同的有语义名字，例如 right_to_left_shoe_1_take_out。
- 所有片段必须按时间顺序排列，并连续覆盖整个视频。
- 第一个片段 start_time 必须是 0.0。
- 最后一个片段 end_time 必须是 {float(duration_sec):.3f}。
- 前一个片段 end_time 必须等于后一个片段 start_time。
- 边界必须对应视觉上的真实阶段变化，不要平均切分。
- start_signal 和 end_signal 必须是后续 episode 中也可观察的视觉标准。
- confidence 是 0 到 1 的数字，表示你对该片段命名和边界的信心。

主任务提示：
{task_block}

视频总时长：{float(duration_sec):.3f} 秒。

只输出 JSON，不要输出 Markdown，不要解释，不要包裹代码块。
JSON schema 固定如下，字段名必须一致：
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def build_student_segmentation_prompt(task_spec: TaskSpec, duration_sec: float) -> str:
    subtask_lines = "\n".join(
        [
            (
                f"{idx}. name: {item.name}\n"
                f"   description: {item.description}\n"
                f"   start_signal: {item.visual_start}\n"
                f"   end_signal: {item.visual_end}"
            )
            for idx, item in enumerate(task_spec.subtasks)
        ]
    )
    reference_lines = "\n".join(
        [
            (
                f"- {segment.subtask}: {float(segment.start_time):.3f}s -> {float(segment.end_time):.3f}s, "
                f"start_signal={segment.start_signal}, end_signal={segment.end_signal}"
            )
            for segment in task_spec.reference_segments
        ]
    )
    schema = {
        "main_task": task_spec.main_task,
        "duration_sec": round(float(duration_sec), 3),
        "segments": [
            {
                "subtask": task_spec.subtasks[0].name if task_spec.subtasks else "subtask_name",
                "start_time": 0.0,
                "end_time": 31.0,
                "start_timestamp": "00:00",
                "end_timestamp": "00:31",
                "boundary_reason": "当前 episode 中观察到子任务切换的视觉依据",
                "confidence": 0.82,
            }
        ],
    }
    allowed_names = ", ".join(item.name for item in task_spec.subtasks)
    return f"""你是 student 小模型。你需要只根据当前主摄像头 episode 视频和 teacher 生成的任务规范，划分子任务边界。

重要限制：
- 只能使用视频画面做语义判断，不要假设你能看到状态、动作、腕部摄像头、夹爪传感器或机器人内部变量。
- 不允许创造新子任务名。
- 子任务标签是闭集，只能从候选子任务名中选择：{allowed_names}
- 必须按 teacher 给出的子任务顺序输出。
- 必须覆盖完整视频，片段连续无缝。
- 第一个片段 start_time 必须是 0.0。
- 最后一个片段 end_time 必须是 {float(duration_sec):.3f}。
- 前一个片段 end_time 必须等于后一个片段 start_time。
- 不要重新命名子任务，不要输出 description/start_signal/end_signal，只输出当前 episode 的边界。
- 如果当前 episode 的动作速度和 ep0 不同，要根据当前视频的视觉信号调整边界，不要照抄 ep0 时间。
- 如果存在停顿或犹豫，把停顿归入前一个明确子任务。

主任务：
{task_spec.main_task}

teacher 子任务规范，必须保持顺序：
{subtask_lines}

ep0 参考边界，仅用于理解流程，不能直接照抄：
{reference_lines}

当前视频总时长：{float(duration_sec):.3f} 秒。

只输出 JSON，不要输出 Markdown，不要解释，不要包裹代码块。
JSON schema 固定如下，字段名必须一致：
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def build_refine_prompt(
    *,
    main_task: str,
    previous_subtask: SubtaskConfig,
    next_subtask: SubtaskConfig,
    clip_start_sec: float,
    clip_end_sec: float,
    coarse_boundary_sec: float,
) -> str:
    schema = {
        "boundary_time": round(float(coarse_boundary_sec), 3),
        "boundary_timestamp": "00:31",
        "boundary_reason": "前一子任务的视觉状态结束，后一子任务的视觉动作开始",
        "confidence": 0.82,
    }
    return f"""你需要只根据这个主摄像头局部视频，精修相邻两个子任务之间的边界时间。

重要限制：
- 只能使用视频画面做判断，不要假设你能看到状态、动作、腕部摄像头、夹爪传感器或机器人内部变量。
- 只允许判断这一个边界：前一子任务结束、后一子任务开始。
- 边界时间必须是原始完整 episode 的绝对时间，不能是局部视频内的相对时间。
- 边界时间必须在 [{float(clip_start_sec):.3f}, {float(clip_end_sec):.3f}] 秒范围内。
- 粗边界是 {float(coarse_boundary_sec):.3f} 秒。只有当局部视频显示更准确的视觉阶段变化时才移动它。
- 如果存在停顿或犹豫，把停顿归入前一个明确子任务。

主任务：
{main_task}

前一子任务：
name: {previous_subtask.name}
description: {previous_subtask.description}
visual_start: {previous_subtask.visual_start}
visual_end: {previous_subtask.visual_end}

后一子任务：
name: {next_subtask.name}
description: {next_subtask.description}
visual_start: {next_subtask.visual_start}
visual_end: {next_subtask.visual_end}

只输出 JSON，不要输出 Markdown，不要解释，不要包裹代码块。
JSON schema 固定如下，字段名必须一致：
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
