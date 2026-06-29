# Semantic Subtask Annotator

This tool annotates robot demonstration episodes into ordered semantic subtasks using only the main-camera video and a Qwen multimodal API call.

It is designed for LeRobot-style datasets, but the minimal version also supports a local folder containing one or more video files.

## Install

For a Conda-based machine, use the setup script:

```bash
cd semantic_subtask_annotator
bash scripts/setup_conda_env.sh
conda activate semantic-subtask-annotator
```

The default and recommended Python version is 3.10, matching `environment.yml`.
You can choose another environment name, and only override Python if you have a reason to:

```bash
bash scripts/setup_conda_env.sh --name subtask-annotator --python 3.10
conda activate subtask-annotator
```

Python 3.10 is the tested/default setup. Python 3.11 should also work. The script installs `ffmpeg`, then installs Python modules from `requirements.txt`. The project is run directly from the source tree through the files in `scripts/`; no editable package install is required.

Alternatively, create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate semantic-subtask-annotator
```

Manual install:

```bash
cd semantic_subtask_annotator
python3 -m pip install -r requirements.txt
```

## API Key

Set the DashScope API key before running annotation:

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

The environment variable name is configurable with `qwen.api_key_env`.

## Download Dataset

The default dataset path in `configs/annotate.yaml` is:

```yaml
dataset:
  root: "./data/lerobot_dataset"
```

Relative paths in the config are resolved from the project directory containing `configs/`, so the command works whether you run it from the project root or its parent directory.

Download a Hugging Face LeRobot-style dataset into that path:

```bash
python scripts/download_dataset.py --hf-repo-id <owner/dataset_repo>
```

Use the Hugging Face mirror:

```bash
python scripts/download_dataset.py \
  --hf-repo-id Xense/newbalance_shoe_insole_retrieval_and_packing_0611 \
  --hf-endpoint https://hf-mirror.com
```

If you see `Connection refused` through `http_proxy`/`https_proxy`, bypass the local proxy. The default `--method auto` tries `huggingface_hub`, then `huggingface-cli`, then a standard-library fallback:

```bash
python scripts/download_dataset.py \
  --hf-repo-id Xense/newbalance_shoe_insole_retrieval_and_packing_0611 \
  --hf-endpoint https://hf-mirror.com \
  --no-proxy
```

If the mirror returns `HTTP Error 403` in the standard-library fallback, use the default auto mode or the Hugging Face client path. Auto mode works even when `hf`/`huggingface-cli` is not in PATH, because it can fall back to the Python API:

```bash
python scripts/download_dataset.py \
  --hf-repo-id Xense/newbalance_shoe_insole_retrieval_and_packing_0611 \
  --hf-endpoint https://hf-mirror.com \
  --no-proxy
```

For the specific `videos/` page you provided, either pass the page URL directly:

```bash
python scripts/download_dataset.py \
  --hf-url https://huggingface.co/datasets/Xense/newbalance_shoe_insole_retrieval_and_packing_0611/tree/main/videos \
  --hf-endpoint https://hf-mirror.com
```

or download only `videos/**` by repo id:

```bash
python scripts/download_dataset.py \
  --hf-repo-id Xense/newbalance_shoe_insole_retrieval_and_packing_0611 \
  --hf-endpoint https://hf-mirror.com \
  --videos-only
```

For full LeRobot metadata, prefer downloading the whole dataset instead of only `videos/`.

For a private Hugging Face dataset, set a token first:

```bash
export HUGGINGFACE_HUB_TOKEN="your_hf_token"
python scripts/download_dataset.py --hf-repo-id <owner/private_dataset_repo> --hf-endpoint https://hf-mirror.com
```

You can also download a direct archive URL:

```bash
python scripts/download_dataset.py --url https://example.com/dataset.zip
```

Use `--output-dir /path/to/dataset` if you want a different location, then update `dataset.root` in `configs/annotate.yaml`.

## Configure The Main Camera

Edit `configs/annotate.yaml`.

The loader first tries `dataset.main_video_key`. If that key is absent, it tries `dataset.fallback_main_video_keys` in order.

For `Xense/newbalance_shoe_insole_retrieval_and_packing_0611`, the configured main camera is:

```yaml
dataset:
  main_video_key: "observation.images.head"
```

It will not automatically choose keys containing `wrist`, `side`, `left`, or `right`; those are treated as non-main camera views unless you explicitly set `dataset.main_video_key` to one of them.

Semantic decisions are made only from the selected main video. State, action, gripper values, and wrist video are not sent to the model. Dataset metadata is used only for episode length, fps, frame counts, and indices.

Check dataset compatibility without calling Qwen:

```bash
python scripts/check_dataset.py --config configs/annotate.yaml
```

Add `--probe` to inspect sample video duration/fps when OpenCV or ffprobe is available.

The annotator does not send every original frame to Qwen. It creates a temporary sampled video for the global pass at `qwen.fps_for_global` (default `1.0` FPS, width `640`), then optionally clips local boundary windows at `qwen.fps_for_refine` (default `3.0` FPS over `+/-5s`). The original video is decoded locally only to create those sampled clips and debug contact sheets.

## Staged Teacher/Student Workflow

The recommended workflow uses a strong teacher model on episode 0 to create a reusable task specification, then uses a smaller student model for later episodes.

Run both stages:

```bash
python3 scripts/run_workflow.py
```

Run only the teacher bootstrap stage:

```bash
python3 scripts/run_workflow.py bootstrap
```

This writes `outputs/subtask_annotation/task_spec.json` with the default config. The file contains `main_task`, ordered subtasks, each subtask's visual start/end signal, and ep0 reference boundaries.

Run only the student segmentation stage on selected later episodes:

```bash
python3 scripts/run_workflow.py segment --ep 3
python3 scripts/run_workflow.py segment --limit 10
```

Teacher information is passed to the student through `task_spec.json`:

1. `bootstrap` calls the teacher model on ep0.
2. The teacher output is validated and converted to `task_spec.json`.
3. `segment` loads `task_spec.json` and injects its ordered subtask names, descriptions, start signals, end signals, and ep0 reference boundaries into the student prompt.
4. The student is forced to use that closed subtask vocabulary and only predicts boundaries for the current episode.

Configure the two models in `configs/annotate.yaml`:

```yaml
workflow:
  teacher_qwen:
    provider: "dashscope"
    model: "qwen3-vl-plus"
  student_qwen:
    provider: "openai_compatible"
    model: "Qwen/Qwen2.5-VL-7B-Instruct"
    base_url: "http://127.0.0.1:8000/v1"
    api_key_env: "LOCAL_OPENAI_API_KEY"
```

For a local 7B model, serve it with an OpenAI-compatible server such as vLLM or SGLang, then point `workflow.student_qwen.base_url` at that server. If the local server does not require an API key, `LOCAL_OPENAI_API_KEY` can be left unset.

`scripts/run_workflow.py` uses `configs/annotate.yaml` by default. Use `--config other.yaml` only when running a different config file.

## Run One Episode

```bash
python3 scripts/annotate_subtasks.py --config configs/annotate.yaml --episode-index 0
```

To let Qwen name and segment subtasks directly from the episode 0 video, without using the configured `task.subtasks` list as a closed vocabulary:

```bash
python3 scripts/annotate_subtasks.py --config configs/annotate.yaml --episode-index 0 --discover-subtasks
```

In discovery mode, each segment includes the generated `subtask`, `description`, `start_signal`, and `end_signal` fields. `task.main_task` is still used as optional context when present; `task.subtasks` may be omitted or left empty.

Disable boundary refinement for a faster single-episode run:

```bash
python3 scripts/annotate_subtasks.py --config configs/annotate.yaml --episode-index 0 --no-refine
```

## Run A Dataset

```bash
python3 scripts/annotate_subtasks.py --config configs/annotate.yaml
```

Use `--limit N` for a small batch.

## Outputs

The output directory contains:

- `episode_000000.json`: per-episode annotation.
- `all_annotations.jsonl`: one JSON object per successful episode.
- `all_annotations.parquet`: compact tabular summary with `segments_json`.
- `validation_summary.csv`: validation status for every attempted episode.
- `debug/episode_000000_global_prompt.txt`: prompt sent for global segmentation.
- `debug/episode_000000_qwen_raw_response.txt`: raw model response.
- `debug/episode_000000_timeline.png`: colored segment timeline.
- `debug/episode_000000_contact_sheet.jpg`: start/mid/end frames for each segment.
- `errors/`: per-episode or boundary errors that did not stop the batch.

## Frame Convention

Segment frames use left-closed, right-open intervals:

```text
[start_frame, end_frame)
```

`start_frame = round(start_time * dataset_fps)` and `end_frame = round(end_time * dataset_fps)`. The final segment always has `end_frame == num_frames`.

## Manual QA

Open the episode JSON and compare it with:

- `debug/*_timeline.png` for segment durations and continuity.
- `debug/*_contact_sheet.jpg` for quick visual checks at each segment start, midpoint, and end.
- `debug/*_qwen_raw_response.txt` when validation fails.

## LeRobot Notes

The loader reads `meta/info.json` when available to find fps and image/video keys. It then searches video files under the dataset root for the selected main video key.

`annotation.write_lerobot_subtask_index` defaults to `false`. This minimal version does not modify the source dataset; JSON/JSONL/Parquet output is the supported path.
