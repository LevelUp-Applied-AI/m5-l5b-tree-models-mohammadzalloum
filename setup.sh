#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "Starting setup in: $PROJECT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is not installed."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
else
  echo "Virtual environment already exists."
fi

source .venv/bin/activate

echo "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  python -m pip install -r requirements.txt
else
  echo "No requirements.txt found. Skipping dependency installation."
fi

if [ -f "requirements-dev.txt" ]; then
  echo "Installing development dependencies from requirements-dev.txt..."
  python -m pip install -r requirements-dev.txt
fi

if [ -f "test_environment.py" ]; then
  echo "Running environment test..."
  python test_environment.py
fi

echo
echo "Setup complete."
echo "To activate the environment later, run:"
echo "source .venv/bin/activate"
