#!/usr/bin/env bash
# Write unified_requirements.txt (uploaded via the step's artifact_paths).
set -euo pipefail
export UV_NO_PROGRESS=1 # spinner can't update in place in CI logs; use a heartbeat instead

echo "--- :wrench: install build toolchain"
SUDO=""; command -v sudo >/dev/null 2>&1 && SUDO=sudo
$SUDO apt-get update
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake ninja-build
pip install -q uv packaging

echo "+++ :lock: generate unified requirements + uv lock"
UV_HTTP_TIMEOUT=600 python utils/generate_unified_requirements.py --uv-lock &
pid=$!
while kill -0 "$pid" 2>/dev/null; do sleep 20; printf '· still resolving… (%ds elapsed)\n' "$SECONDS"; done
wait "$pid"

echo "--- :page_facing_up: unified_requirements.txt"
cat ./unified_requirements.txt

# Success flag for the report step (set only if everything above passed).
buildkite-agent meta-data set deps-unify-outcome passed
