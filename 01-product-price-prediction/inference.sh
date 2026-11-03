#!/usr/bin/env bash
set -euo pipefail
echo "--- Shell Script Start ---"

# Allow optional venv activation if present
if [ -d "venv" ]; then
  echo "--- Activate Virtual Environment ---"
  # shellcheck disable=SC1091
  source venv/bin/activate
  VENV_ACTIVE=1
else
  echo "--- No venv folder found; using system Python ---"
  VENV_ACTIVE=0
fi

# Expect: ./inference.sh --test_path /service/res/test.csv
echo "--- Run Inference ---"
python3 src/inference.py "$@"

# Deactivate virtual environment if activated
if [ "$VENV_ACTIVE" -eq 1 ]; then
  echo "--- Deactivate Virtual Environment ---"
  deactivate
fi

echo "--- Shell Script Stop ---"
