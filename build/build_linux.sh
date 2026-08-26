#!/usr/bin/env bash
# Build TensorScope Linux binary.
# Run from the repo root: bash build/build_linux.sh
#
# Output: dist/TensorScope/ (run dist/TensorScope/TensorScope)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi

pip install --upgrade pyinstaller
pyinstaller "$ROOT/build/tensorscope.spec" \
    --distpath "$ROOT/dist" \
    --workpath "$ROOT/build/work"

echo "Done. Binary: $ROOT/dist/TensorScope/TensorScope"
