#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "lerobot_dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a local dataset for semantic subtask annotation.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-repo-id", help="Hugging Face dataset repo id, for example lerobot/pusht")
    source.add_argument(
        "--hf-url",
        help=(
            "Hugging Face dataset page URL, for example "
            "https://huggingface.co/datasets/owner/repo/tree/main/videos"
        ),
    )
    source.add_argument("--url", help="Direct dataset file URL. .zip and .tar.* archives are extracted automatically.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Destination directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument("--hf-revision", default=None, help="Optional Hugging Face revision, branch, or commit.")
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Optional Hugging Face endpoint, for example https://hf-mirror.com. Defaults to HF_ENDPOINT when set.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. If omitted, HUGGINGFACE_HUB_TOKEN/HF_TOKEN is used when set.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Hugging Face allow pattern. Can be repeated, for example --include 'videos/**' --include 'meta/**'.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Hugging Face ignore pattern. Can be repeated.",
    )
    parser.add_argument(
        "--videos-only",
        action="store_true",
        help="Only download files under videos/** from a Hugging Face dataset.",
    )
    parser.add_argument(
        "--prefer-stdlib",
        action="store_true",
        help="Deprecated alias for --method stdlib.",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "hub", "cli", "stdlib"],
        default="auto",
        help="Hugging Face download method. auto tries hub, then hf/huggingface-cli, then stdlib.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Ignore HTTP_PROXY/HTTPS_PROXY/ALL_PROXY for this download process.",
    )
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow downloading into a non-empty output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_proxy:
        _disable_proxy_environment()
    method = "stdlib" if args.prefer_stdlib else args.method
    output_dir = args.output_dir.expanduser().resolve()
    _prepare_output_dir(output_dir, allow_existing=args.allow_existing)

    if args.hf_repo_id or args.hf_url:
        repo_id = args.hf_repo_id
        revision = args.hf_revision
        include_patterns = list(args.include or [])
        if args.hf_url:
            parsed = _parse_huggingface_dataset_url(args.hf_url)
            repo_id = parsed["repo_id"]
            revision = revision or parsed.get("revision")
            if parsed.get("subdir"):
                include_patterns.append(f"{parsed['subdir'].rstrip('/')}/**")
        if args.videos_only:
            include_patterns.append("videos/**")
        _download_huggingface_dataset(
            repo_id=repo_id,
            output_dir=output_dir,
            revision=revision,
            endpoint=args.hf_endpoint,
            token=args.hf_token,
            include_patterns=include_patterns or None,
            exclude_patterns=args.exclude,
            method=method,
            no_proxy=args.no_proxy,
            timeout_sec=args.timeout_sec,
        )
    else:
        _download_url(args.url, output_dir, no_proxy=args.no_proxy, timeout_sec=args.timeout_sec)

    print(f"Dataset is ready at: {output_dir}")
    print("The default config already points to: ./data/lerobot_dataset")
    print("Run from the project root:")
    print("  python scripts/run_workflow.py")
    return 0


def _prepare_output_dir(output_dir: Path, *, allow_existing: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}\n"
            "Use --allow-existing to reuse it, or choose a different --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _download_huggingface_dataset(
    *,
    repo_id: str,
    output_dir: Path,
    revision: str | None,
    endpoint: str | None,
    token: str | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    method: str,
    no_proxy: bool,
    timeout_sec: float,
) -> None:
    resolved_token = token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")

    print(f"Downloading Hugging Face dataset {repo_id!r} to {output_dir}")
    if endpoint:
        print(f"Using Hugging Face endpoint: {endpoint.rstrip('/')}")
    if revision:
        print(f"Revision: {revision}")
    if include_patterns:
        print(f"Include patterns: {include_patterns}")
    if exclude_patterns:
        print(f"Exclude patterns: {exclude_patterns}")

    if method == "stdlib":
        print("Using standard-library HTTP download.")
        _download_huggingface_dataset_without_hub(
            repo_id=repo_id,
            output_dir=output_dir,
            revision=revision or "main",
            endpoint=endpoint,
            token=resolved_token,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            no_proxy=no_proxy,
            timeout_sec=timeout_sec,
        )
        return

    if method in {"auto", "hub"}:
        try:
            _download_with_huggingface_hub(
                repo_id=repo_id,
                output_dir=output_dir,
                revision=revision,
                endpoint=endpoint,
                token=resolved_token,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            return
        except Exception as exc:  # noqa: BLE001 - auto mode should keep trying practical fallbacks.
            if method == "hub":
                raise
            print(f"huggingface-hub download failed: {exc!r}")
            print("Trying huggingface-cli fallback.")

    if method in {"auto", "cli"}:
        try:
            _download_with_huggingface_cli(
                repo_id=repo_id,
                output_dir=output_dir,
                revision=revision,
                endpoint=endpoint,
                token=resolved_token,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                no_proxy=no_proxy,
            )
            return
        except Exception as exc:  # noqa: BLE001
            if method == "cli":
                print(f"huggingface CLI path failed: {exc!r}")
                print("Falling back to huggingface_hub Python API.")
                _download_with_huggingface_hub(
                    repo_id=repo_id,
                    output_dir=output_dir,
                    revision=revision,
                    endpoint=endpoint,
                    token=resolved_token,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                )
                return
            print(f"huggingface-cli download failed: {exc!r}")
            print("Trying standard-library HTTP fallback.")

    _download_huggingface_dataset_without_hub(
        repo_id=repo_id,
        output_dir=output_dir,
        revision=revision or "main",
        endpoint=endpoint,
        token=resolved_token,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        no_proxy=no_proxy,
        timeout_sec=timeout_sec,
    )


def _download_with_huggingface_hub(
    *,
    repo_id: str,
    output_dir: Path,
    revision: str | None,
    endpoint: str | None,
    token: str | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is not installed") from exc

    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "token": token,
        "local_dir": output_dir,
        "allow_patterns": include_patterns,
        "ignore_patterns": exclude_patterns,
    }
    if endpoint:
        # Newer huggingface-hub versions accept endpoint directly. Older ones
        # read HF_ENDPOINT from the environment, already set by caller.
        import inspect

        if "endpoint" in inspect.signature(snapshot_download).parameters:
            kwargs["endpoint"] = endpoint.rstrip("/")

    snapshot_download(**kwargs)


def _download_with_huggingface_cli(
    *,
    repo_id: str,
    output_dir: Path,
    revision: str | None,
    endpoint: str | None,
    token: str | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    no_proxy: bool,
) -> None:
    executable = shutil.which("hf") or shutil.which("huggingface-cli")
    if executable is None:
        raise RuntimeError("neither 'hf' nor 'huggingface-cli' was found in PATH")

    executable_name = Path(executable).name
    if executable_name == "hf":
        cmd = [
            executable,
            "download",
            repo_id,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(output_dir),
        ]
    else:
        cmd = [
            executable,
            "download",
            repo_id,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(output_dir),
        ]
    if revision:
        cmd.extend(["--revision", revision])
    if token:
        cmd.extend(["--token", token])
    for pattern in include_patterns or []:
        cmd.extend(["--include", pattern])
    for pattern in exclude_patterns or []:
        cmd.extend(["--exclude", pattern])

    env = os.environ.copy()
    if endpoint:
        env["HF_ENDPOINT"] = endpoint.rstrip("/")
    if no_proxy:
        _remove_proxy_from_env(env)

    subprocess.run(cmd, check=True, env=env)


def _download_huggingface_dataset_without_hub(
    *,
    repo_id: str,
    output_dir: Path,
    revision: str,
    endpoint: str | None,
    token: str | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    no_proxy: bool,
    timeout_sec: float,
) -> None:
    base = (endpoint or "https://huggingface.co").rstrip("/")
    api_url = f"{base}/api/datasets/{repo_id}/tree/{quote(revision, safe='')}"
    files = _list_huggingface_files(api_url, token=token, no_proxy=no_proxy, timeout_sec=timeout_sec)
    selected = [
        path
        for path in files
        if _matches_patterns(path, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
    ]
    if not selected:
        raise RuntimeError(
            f"No files matched the requested patterns. repo_id={repo_id!r}, revision={revision!r}, "
            f"include={include_patterns!r}, exclude={exclude_patterns!r}"
        )

    print(f"Matched {len(selected)} file(s).")
    for idx, rel_path in enumerate(selected, start=1):
        target_path = output_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        file_url = (
            f"{base}/datasets/{repo_id}/resolve/{quote(revision, safe='')}/"
            f"{quote(rel_path, safe='/')}"
        )
        print(f"[{idx}/{len(selected)}] {rel_path}")
        _download_file(file_url, target_path, token=token, no_proxy=no_proxy, timeout_sec=timeout_sec)


def _list_huggingface_files(api_url: str, *, token: str | None, no_proxy: bool, timeout_sec: float) -> list[str]:
    files: list[str] = []
    stack = [api_url]
    while stack:
        current = stack.pop()
        with _urlopen(_make_request(current, token=token), no_proxy=no_proxy, timeout_sec=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Hugging Face API response from {current}: {payload!r}")
        for item in payload:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            path = item.get("path")
            if not isinstance(path, str):
                continue
            if item_type == "file":
                files.append(path)
            elif item_type == "directory":
                stack.append(current.rstrip("/") + "/" + quote(path.rsplit("/", 1)[-1], safe=""))
    return sorted(files)


def _matches_patterns(
    path: str,
    *,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> bool:
    if include_patterns and not any(fnmatch.fnmatch(path, pattern) for pattern in include_patterns):
        return False
    if exclude_patterns and any(fnmatch.fnmatch(path, pattern) for pattern in exclude_patterns):
        return False
    return True


def _download_file(url: str, target_path: Path, *, token: str | None, no_proxy: bool, timeout_sec: float) -> None:
    with _urlopen(_make_request(url, token=token), no_proxy=no_proxy, timeout_sec=timeout_sec) as response:
        with target_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def _make_request(url: str, *, token: str | None) -> urllib.request.Request:
    headers = {
        "Accept": "application/json, application/octet-stream, */*",
        "User-Agent": "huggingface_hub/0.25 semantic-subtask-annotator/0.1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _urlopen(request: urllib.request.Request, *, no_proxy: bool, timeout_sec: float):
    if no_proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=timeout_sec)
    return urllib.request.urlopen(request, timeout=timeout_sec)


def _disable_proxy_environment() -> None:
    _remove_proxy_from_env(os.environ)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def _remove_proxy_from_env(env: os._Environ[str] | dict[str, str]) -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"


def _parse_huggingface_dataset_url(url: str) -> dict[str, str | None]:
    parsed = urlparse(url)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if not parts or parts[0] != "datasets":
        raise ValueError(f"not a Hugging Face dataset URL: {url}")
    if len(parts) < 3:
        raise ValueError(f"dataset URL must include owner and repo name: {url}")

    repo_id = f"{parts[1]}/{parts[2]}"
    revision: str | None = None
    subdir: str | None = None
    if len(parts) >= 5 and parts[3] in {"tree", "blob", "resolve"}:
        revision = parts[4]
        if len(parts) > 5:
            subdir = "/".join(parts[5:])
    return {"repo_id": repo_id, "revision": revision, "subdir": subdir}


def _download_url(url: str, output_dir: Path, *, no_proxy: bool, timeout_sec: float) -> None:
    filename = Path(url.split("?", 1)[0]).name or "dataset_download"
    download_path = output_dir / filename
    print(f"Downloading {url} to {download_path}")
    with _urlopen(_make_request(url, token=None), no_proxy=no_proxy, timeout_sec=timeout_sec) as response:
        with download_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    if zipfile.is_zipfile(download_path):
        print(f"Extracting zip archive to {output_dir}")
        with zipfile.ZipFile(download_path) as archive:
            archive.extractall(output_dir)
        return

    if tarfile.is_tarfile(download_path):
        print(f"Extracting tar archive to {output_dir}")
        with tarfile.open(download_path) as archive:
            _safe_extract_tar(archive, output_dir)
        return

    print("Downloaded file is not a zip/tar archive; leaving it in place.")


def _safe_extract_tar(archive: tarfile.TarFile, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    for member in archive.getmembers():
        target = (output_dir / member.name).resolve()
        if not str(target).startswith(str(output_root) + os.sep) and target != output_root:
            raise RuntimeError(f"unsafe path in tar archive: {member.name}")
    archive.extractall(output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
