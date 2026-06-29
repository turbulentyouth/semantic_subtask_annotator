#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_subtask_annotator.config import QwenConfig  # noqa: E402
from semantic_subtask_annotator.qwen_client import QwenClient  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test Qwen VL with one local video and one text prompt.")
    parser.add_argument("--video", required=True, type=Path, help="Local video path, for example /path/to/episode_000000.mp4")
    parser.add_argument("--text", required=True, help="Text prompt to ask about the video")
    parser.add_argument("--model", default="qwen3-vl-plus", help="DashScope model name")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="Environment variable containing the API key")
    parser.add_argument("--fps", type=float, default=1.0, help="Video sampling FPS sent to Qwen")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the model response text")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"video file does not exist: {video_path}")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    client = QwenClient(
        QwenConfig(
            model=args.model,
            api_key_env=args.api_key_env,
            fps_for_global=args.fps,
            fps_for_refine=args.fps,
            temperature=args.temperature,
            max_retries=args.max_retries,
            timeout_sec=args.timeout_sec,
        ),
        error_dir=PROJECT_ROOT / "outputs" / "qwen_smoke_test",
    )
    response = client.generate_text(video_path, args.text, fps=args.fps)

    print("\n===== Qwen Response =====")
    print(response)
    print("=========================\n")

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response, encoding="utf-8")
        print(f"Saved response to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
