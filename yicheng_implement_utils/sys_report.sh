#!/usr/bin/env bash
set -euo pipefail

# === GPUDrive System Report ===
# Usage:
#   bash gpudrive_sys_report.sh
#   bash gpudrive_sys_report.sh baselines/ppo/config/ppo_base_sb3.yaml

YAML_PATH="${1:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="reports/gpudrive_report_${STAMP}"
mkdir -p "$OUTDIR"
echo "Writing report to: $OUTDIR"

# 0) Basic host info
{
  echo "# Host"
  date -Is
  echo "HOSTNAME: $(hostname)"
  echo
  echo "## OS"
  if command -v lsb_release >/dev/null 2>&1; then
    lsb_release -a || true
  fi
  echo
  echo "## Kernel"
  uname -a
} | tee "$OUTDIR/host.txt"

# 1) GPU, driver, and CUDA runtime
{
  echo "# NVIDIA-SMI"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    echo
    echo "# Live SM clocks / power (one shot)"
    nvidia-smi --query-gpu=timestamp,index,name,driver_version,clocks.sm,temperature.gpu,power.draw --format=csv
  else
    echo "nvidia-smi not found"
  fi
} | tee "$OUTDIR/gpu_driver.txt"

# 2) CUDA toolkit (nvcc) & CMake
{
  echo "# nvcc --version"
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
  else
    echo "nvcc not found"
  fi
  echo
  echo "# cmake --version"
  if command -v cmake >/dev/null 2>&1; then
    cmake --version
  else
    echo "cmake not found"
  fi
} | tee "$OUTDIR/cuda_cmake.txt"

# 3) Python & Conda
{
  echo "# Python"
  if command -v python >/dev/null 2>&1; then
    python -V
  fi
  echo
  echo "# Pip freeze"
  if command -v python >/dev/null 2>&1; then
    python -m pip freeze || true
  fi
  echo
  echo "# Conda env (export)"
  if command -v conda >/dev/null 2>&1; then
    conda env export || true
  else
    echo "conda not found"
  fi
} | tee "$OUTDIR/python_env.txt"

# 4) Torch / JAX quick probe (if installed)
{
  echo "# Torch/JAX probe"
  python - <<'PY' || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda.device_count:", torch.cuda.device_count())
        print("current_device:", torch.cuda.current_device())
        print("get_device_name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch probe error:", e)
try:
    import jax
    print("jax:", jax.__version__)
    print("jax devices:", jax.devices())
except Exception as e:
    print("jax probe error:", e)
PY
} | tee "$OUTDIR/dl_frameworks.txt"

# 5) GPUDrive repo state (if run inside repo)
{
  echo "# Git"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "commit: $(git rev-parse HEAD)"
    echo "branch: $(git rev-parse --abbrev-ref HEAD)"
    echo
    echo "Changed files:"
    git status --porcelain
  else
    echo "Not inside a git repo."
  fi
} | tee "$OUTDIR/git_state.txt"

# 6) Useful env vars
{
  echo "# Environment Variables"
  env | grep -E 'CUDA|CUBLAS|CUDNN|MADRONA|TORCH|OMP|MKL|OPENBLAS|LD_LIBRARY_PATH|PYTHONPATH' | sort || true
} | tee "$OUTDIR/env_vars.txt"

# 7) Optionally archive your training YAML
if [[ -n "$YAML_PATH" && -f "$YAML_PATH" ]]; then
  mkdir -p "$OUTDIR/configs"
  cp "$YAML_PATH" "$OUTDIR/configs/"
  echo "Copied YAML to $OUTDIR/configs/"
fi

echo "Done. Collected system report at: $OUTDIR"
