#!/usr/bin/env bash
# Bootstrap the workspace.
#
# Usage:
#   bash scripts/bootstrap.sh

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Bootstrapping the TinyCode workspace in ${WORKSPACE_ROOT}"

for repo in tinycode-model-builder tinycode-inference-engine; do
    echo ""
    echo "== ${repo} =="
    cd "${WORKSPACE_ROOT}/${repo}"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install pytest
done

cd "${WORKSPACE_ROOT}"
echo ""
echo "== golden fixture =="
python scripts/make_golden_fixture.py

echo ""
echo "== contract check =="
python scripts/verify_contracts.py

echo ""
echo "Done. Start the builder with: cd tinycode-model-builder && python main.py"
