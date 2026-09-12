#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found. Install Python 3 first or set PYTHON_BIN." >&2
  exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_FULL_VERSION="$("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])')"
VENV_PACKAGE="python${PY_VERSION}-venv"

echo "Detected Python: $PYTHON_BIN $PY_FULL_VERSION"

if command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ]; then
    APT="apt-get"
  else
    APT="sudo apt-get"
  fi

  echo "Installing venv support: $VENV_PACKAGE"
  $APT update
  $APT install -y "$VENV_PACKAGE"
else
  echo "apt-get not found; skipping system package installation."
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Using existing virtual environment: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Environment ready. Activate it with: source $VENV_DIR/bin/activate"
