#!/usr/bin/env bash
# Regenerate pyproject.toml + uv.lock; if changed, force-push the branch and open its PR. Needs GIT_PAT.
set -euo pipefail
export UV_NO_PROGRESS=1 # quiet uv's resolver spinner in CI logs

REPO="rebellions-sw/rbln_model_zoo"
BRANCH="auto/update-pyproject"

SUDO=""; command -v sudo >/dev/null 2>&1 && SUDO=sudo
$SUDO apt-get update
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake ninja-build

pip install -q uv packaging
UV_HTTP_TIMEOUT=600 python utils/generate_unified_requirements.py --uv-lock

if [ -z "$(git status --porcelain pyproject.toml uv.lock)" ]; then
  echo "No changes — skipping PR."
  exit 0
fi

git config user.name "rbln-ci[bot]"
git config user.email "rbln-ci@rebellions.ai"
git checkout -B "$BRANCH"
git add pyproject.toml uv.lock
git commit -m "chore(deps): auto-update pyproject.toml & uv.lock ($(date +%F))"
git push --force "https://x-access-token:${GIT_PAT}@github.com/${REPO}.git" "$BRANCH"

# Open the PR via REST; 422 (already open) is fine — the force-push refreshed it.
curl -fsS -X POST \
  -H "Authorization: Bearer ${GIT_PAT}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${REPO}/pulls" \
  -d "{\"title\":\"chore(deps): auto-update pyproject.toml & uv.lock\",\"head\":\"${BRANCH}\",\"base\":\"main\",\"body\":\"Auto-generated nightly; \`${BRANCH}\` is force-pushed each run.\"}" \
  || echo "PR already open — branch refreshed."
