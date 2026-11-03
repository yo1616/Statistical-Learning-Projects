#!/bin/bash
set -euo pipefail

echo "--- Shell Script Start ---"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEST_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test_path)
      TEST_PATH="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --test_path /path/to/test.csv"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${TEST_PATH:-}" ]]; then
  echo "Missing required flag: --test_path /path/to/test.csv"
  exit 1
fi

if [[ ! -f "$TEST_PATH" ]]; then
  echo "Test file not found: $TEST_PATH"
  exit 1
fi

if [[ ! -d "$SCRIPT_DIR/venv" ]]; then
  echo "Python virtual environment not found at $SCRIPT_DIR/venv"
  exit 1
fi

echo "--- Activate Virtual Environment ---"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/venv/bin/activate"

echo "--- Run Inference ---"
python3 "$SCRIPT_DIR/inference.py" \
  --test_path "$TEST_PATH" \
  --output_path "$SCRIPT_DIR/predictions.csv" \
  --model_type ml \
  --config "$SCRIPT_DIR/config/config.yaml"

echo "--- Deactivate Virtual Environment ---"
deactivate

echo "--- Shell Script Stop ---"
