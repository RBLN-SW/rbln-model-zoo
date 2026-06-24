#!/usr/bin/env bash
# Post the unified-requirements summary to Slack after deps-unify. Needs SLACK_BOT_TOKEN.
set -euo pipefail

pip install -q requests
# deps-unify sets this only after it succeeds; absent => treat as failure.
result=failure
[ "$(buildkite-agent meta-data get deps-unify-outcome 2>/dev/null || true)" = passed ] && result=success

buildkite-agent artifact download unified_requirements.txt . || true
python scripts/report.py \
  --slack_bot_token "$SLACK_BOT_TOKEN" \
  --github_action_url "$BUILDKITE_BUILD_URL" \
  --upstream_job_result "$result"
