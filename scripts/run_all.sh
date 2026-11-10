#!/usr/bin/env bash
# Collect data, build features, train models, export the website payload.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python scripts/collect_data.py
.venv/bin/python scripts/build_features.py
.venv/bin/python scripts/train_and_export.py
echo
echo "Done. Start the site with:"
echo "  $ROOT/.venv/bin/python $ROOT/scripts/serve.py"
echo "Then open http://127.0.0.1:8765/"
