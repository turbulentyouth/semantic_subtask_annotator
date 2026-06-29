#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="semantic-subtask-annotator"
PYTHON_VERSION="3.10"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_conda_env.sh [--name ENV_NAME] [--python PYTHON_VERSION]

Examples:
  bash scripts/setup_conda_env.sh
  bash scripts/setup_conda_env.sh --name subtask-annotator --python 3.10

After installation:
  conda activate semantic-subtask-annotator
  export DASHSCOPE_API_KEY="your_api_key"
  python scripts/run_workflow.py
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      ENV_NAME="${2:?missing value for --name}"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="${2:?missing value for --python}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH. Install Miniconda/Anaconda first, then rerun this script." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Project root: ${PROJECT_ROOT}"
echo "Conda environment: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Updating existing conda environment: ${ENV_NAME}"
  conda install -y -n "${ENV_NAME}" -c conda-forge \
    "python=${PYTHON_VERSION}" \
    pip \
    ffmpeg
else
  echo "Creating conda environment: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" -c conda-forge \
    "python=${PYTHON_VERSION}" \
    pip \
    ffmpeg
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

echo
echo "Environment is ready."
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Then set the DashScope key and run:"
echo '  export DASHSCOPE_API_KEY="your_api_key"'
echo "  cd ${PROJECT_ROOT}"
echo "  python scripts/download_dataset.py --hf-repo-id <owner/dataset_repo> --hf-endpoint https://hf-mirror.com"
echo "  python scripts/run_workflow.py"
