#!/usr/bin/env bash
set -euo pipefail
plant_baseline_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
plant_repo_dir="$(cd -- "$plant_baseline_dir/../.." && pwd)"
plant_venv_dir="${1:-$plant_repo_dir/.venv}"
cd "$plant_repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$plant_repo_dir/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$plant_repo_dir/.cache/python}"
command -v uv >/dev/null
uv --system-certs python install --no-bin 3.11.16
if [ ! -x "$plant_venv_dir/bin/python" ]; then
    uv venv --python 3.11.16 "$plant_venv_dir"
fi
"$plant_venv_dir/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use a Python 3.11 virtual environment"'
uv --system-certs pip install --python "$plant_venv_dir/bin/python" \
    'torch==2.3.1+cpu' --index-url https://download.pytorch.org/whl/cpu
uv --system-certs pip install --python "$plant_venv_dir/bin/python" \
    -r "$plant_baseline_dir/environment/requirements-cpu.txt"
git -C "$plant_repo_dir" submodule update --init --recursive
source "$plant_baseline_dir/environment/activate.sh" "$plant_venv_dir"
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
uv pip check --python "$plant_venv_dir/bin/python"
python "$plant_baseline_dir/smoke_inference.py"
python -m genomicreglm.cli.train \
    --data-config configs/data/maize_multiomics.yaml \
    --model-config configs/model/transformer_multimodal.yaml \
    --train-config configs/train/finetune_expression_regression.yaml \
    --synthetic --num-samples 64
