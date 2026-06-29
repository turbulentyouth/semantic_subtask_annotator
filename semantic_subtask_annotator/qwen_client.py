from __future__ import annotations

import json
import logging
import os
import re
import time
import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from .config import QwenConfig

LOGGER = logging.getLogger(__name__)


class QwenClientError(RuntimeError):
    pass


class QwenClient:
    def __init__(self, config: QwenConfig, *, error_dir: Path | None = None) -> None:
        self.config = config
        self.error_dir = error_dir
        self.provider = config.provider.lower()
        self.api_key = os.environ.get(config.api_key_env, "")
        if self.provider == "dashscope" and not self.api_key:
            raise QwenClientError(
                f"Environment variable {config.api_key_env} is not set. "
                f"Set it before calling the Qwen API."
            )
        if self.provider == "openai_compatible" and not config.base_url.strip():
            raise QwenClientError("base_url is required for provider='openai_compatible'")

    def annotate_video(self, video_path: Path, prompt: str, *, fps: float, request_name: str) -> dict[str, Any]:
        raw = self._call_with_retries(video_path, prompt, fps=fps, request_name=request_name)
        try:
            return extract_json_object(raw)
        except ValueError as exc:
            self._write_error(f"{request_name}_json_parse_error.txt", raw)
            raise QwenClientError(f"Qwen response was not valid JSON for {request_name}: {exc}") from exc

    def _call_with_retries(self, video_path: Path, prompt: str, *, fps: float, request_name: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                LOGGER.info("Calling Qwen for %s (attempt %d/%d)", request_name, attempt, self.config.max_retries)
                raw = self._call_once(video_path, prompt, fps=fps)
                self._write_debug(f"{request_name}_qwen_raw_response.txt", raw)
                return raw
            except Exception as exc:  # noqa: BLE001 - preserve SDK errors in logs and retry.
                last_error = exc
                LOGGER.warning("Qwen call failed for %s on attempt %d: %s", request_name, attempt, exc)
                self._write_error(f"{request_name}_attempt_{attempt}_error.txt", repr(exc))
                if attempt < self.config.max_retries:
                    time.sleep(min(2**attempt, 10))
        raise QwenClientError(f"Qwen call failed after {self.config.max_retries} attempts: {last_error}") from last_error

    def _call_once(self, video_path: Path, prompt: str, *, fps: float) -> str:
        if self.provider == "openai_compatible":
            return self._call_openai_compatible(video_path, prompt, fps=fps)
        if self.provider != "dashscope":
            raise QwenClientError(f"Unsupported Qwen provider: {self.config.provider}")
        return self._call_dashscope(video_path, prompt, fps=fps)

    def _call_dashscope(self, video_path: Path, prompt: str, *, fps: float) -> str:
        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError as exc:
            raise QwenClientError("dashscope is required. Install with: pip install dashscope") from exc

        dashscope.api_key = self.api_key
        file_uri = video_path.resolve().as_uri()
        messages = [
            {
                "role": "user",
                "content": [
                    {"video": file_uri, "fps": fps},
                    {"text": prompt},
                ],
            }
        ]

        def _invoke() -> Any:
            return MultiModalConversation.call(
                api_key=self.api_key,
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
            )

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_invoke)
        try:
            response = future.result(timeout=self.config.timeout_sec)
        except FutureTimeoutError as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise QwenClientError(f"Qwen call timed out after {self.config.timeout_sec} seconds") from exc
        finally:
            if future.done():
                executor.shutdown(wait=False, cancel_futures=True)

        status_code = getattr(response, "status_code", None)
        if status_code is not None and int(status_code) >= 400:
            code = getattr(response, "code", "")
            message = getattr(response, "message", "")
            request_id = getattr(response, "request_id", "")
            raise QwenClientError(f"DashScope error status={status_code} code={code} message={message} request_id={request_id}")

        text = _extract_response_text(response)
        if not text.strip():
            raise QwenClientError(f"DashScope response did not contain text: {response!r}")
        return text

    def _call_openai_compatible(self, video_path: Path, prompt: str, *, fps: float) -> str:
        try:
            import requests
        except ImportError as exc:
            raise QwenClientError("requests is required for openai_compatible provider. Install with: pip install requests") from exc

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        mime = _guess_video_mime(video_path)
        encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{encoded}"}, "fps": fps},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout_sec)
        if response.status_code >= 400:
            raise QwenClientError(f"OpenAI-compatible API error status={response.status_code}: {response.text[:1000]}")
        data = response.json()
        text = _extract_openai_text(data)
        if not text.strip():
            raise QwenClientError(f"OpenAI-compatible response did not contain text: {data!r}")
        return text

    def _write_error(self, name: str, text: str) -> None:
        if not self.error_dir:
            return
        self.error_dir.mkdir(parents=True, exist_ok=True)
        (self.error_dir / name).write_text(text, encoding="utf-8")

    def _write_debug(self, name: str, text: str) -> None:
        self._write_error(name, text)


def _extract_response_text(response: Any) -> str:
    output = _get_item(response, "output")
    choices = _get_item(output, "choices")
    if isinstance(choices, list) and choices:
        message = _get_item(choices[0], "message")
        content = _get_item(message, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    value = item.get("text") or item.get("content")
                    if isinstance(value, str):
                        chunks.append(value)
            return "\n".join(chunks)
    if isinstance(response, dict):
        for key in ("text", "content"):
            value = response.get(key)
            if isinstance(value, str):
                return value
    return str(response)


def _get_item(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _extract_openai_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return ""


def _guess_video_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".webm":
        return "video/webm"
    if suffix == ".mov":
        return "video/quicktime"
    if suffix == ".avi":
        return "video/x-msvideo"
    return "video/mp4"


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = json.loads(_extract_balanced_json(cleaned))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _extract_balanced_json(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object start found")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("no balanced JSON object found")
