#!/usr/bin/env bash
# Build TensorScope macOS app bundle.
# Run from the repo root: bash build/build_macos.sh
#
# Output: dist/TensorScope.app  (and dist/TensorScope/ folder)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi

pip install --upgrade pyinstaller
pyinstaller "$ROOT/build/tensorscope.spec" \
    --distpath "$ROOT/dist" \
    --workpath "$ROOT/build/work"

echo "Done. App: $ROOT/dist/TensorScope/"
