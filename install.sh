#!/usr/bin/env bash
# install.sh - Simple installer for the CAUA artifact
# Usage:
#   ./install.sh         # create venv and install requirements.txt
#   ./install.sh --system  # skip venv, install into current Python environment
#   ./install.sh -h      # show help
#
# Notes for reviewers:
# - This script assumes a working python3 executable is available on PATH.
# - If you prefer conda, please create a conda env manually and run:
#       pip install -r requirements.txt
# - The script does NOT install any GPU-specific torch wheel automatically.
#   If you need a CUDA-enabled torch, install it after activation using
#   instructions from https://pytorch.org/get-started/previous-versions/

set -euo pipefail

SHOW_HELP=0
USE_SYSTEM=0
VENV_DIR="venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system) USE_SYSTEM=1; shift ;;
    -h|--help) SHOW_HELP=1; shift ;;
    *) echo "Unknown option: $1"; SHOW_HELP=1; shift ;;
  esac
done

if [[ ${SHOW_HELP} -eq 1 ]]; then
  cat <<EOF
install.sh - Simple installer for the CAUA artifact

Usage:
  ./install.sh             Create a virtual environment (./venv) and install dependencies.
  ./install.sh --system    Install dependencies into the current Python environment (no venv).
  ./install.sh -h | --help Show this help message.

Notes:
  - The script installs packages from requirements.txt located in the repository root.
  - For GPU support (CUDA), please follow the official PyTorch instructions:
      https://pytorch.org/get-started/previous-versions/
  - After running this script (default mode), activate the venv before running experiments:
      source ./venv/bin/activate
EOF
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH. Please install Python 3 and retry."
  exit 1
fi

PYTHON_BIN="$(command -v python3)"
REQ_FILE="requirements.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: ${REQ_FILE} not found in repository root. Please add it and retry."
  exit 1
fi

if [[ ${USE_SYSTEM} -eq 1 ]]; then
  echo "Installing dependencies into the current Python environment..."
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r "${REQ_FILE}"
  echo ""
  echo "Installation finished (system)."
  echo "You can now run examples, e.g.:"
  echo "  python run_demo.py"
  exit 0
fi

# Default: create and use venv
if [[ -d "${VENV_DIR}" ]]; then
  echo "Virtual environment directory '${VENV_DIR}' already exists."
  echo "If you want a fresh one, remove the directory and run this script again:"
  echo "  rm -rf ${VENV_DIR}"
fi

echo "Creating virtual environment at ./${VENV_DIR} ..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "Activating virtual environment and installing dependencies..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install -r "${REQ_FILE}"

echo ""
echo "✅ Installation complete."
echo ""
echo "To use the virtual environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Example run (after activation):"
echo "  # run a quick demo or the provided run script"
echo "  bash run.sh"
echo ""
echo "If you need CUDA-enabled PyTorch, please install the appropriate wheel after activation,"
echo "for example (CUDA 11.7):"
echo "  pip install --index-url https://download.pytorch.org/whl/cu117 torch==1.13.1 torchvision==0.14.1"
