#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${1:?Usage: bash scripts/submit_exp.sh <exp_id>}"

REMOTE_PORT="${REMOTE_PORT:-15600}"
REMOTE_USER="${REMOTE_USER:-jyh}"
REMOTE_HOST="${REMOTE_HOST:-59.72.109.245}"
REMOTE_DIR="${REMOTE_DIR:-/home/jyh/workplace/ClaudeProjects/GraCA}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

EXP_DIR="experiments/${EXP_ID}"
PROMPT="${EXP_DIR}/prompt.md"

if [ ! -d .git ]; then
  echo "submit_exp.sh must be run from the repository root."
  exit 1
fi

if [ ! -f "${PROMPT}" ]; then
  echo "Missing ${PROMPT}"
  exit 1
fi

echo "Syncing local branch before submit..."
git pull --ff-only

echo "Committing experiment prompt: ${EXP_ID}"
git add "${EXP_DIR}"
if git diff --cached --quiet; then
  echo "No new local experiment changes to commit."
else
  git commit -m "add experiment ${EXP_ID}"
fi

echo "Pushing experiment prompt to GitHub..."
git push

echo "Running remote experiment on ${REMOTE}:${REMOTE_DIR}"
set +e
ssh -p "${REMOTE_PORT}" "${REMOTE}" \
  "cd '${REMOTE_DIR}' && git pull --ff-only && bash scripts/run_exp.sh '${EXP_ID}'"
REMOTE_STATUS=$?
set -e

echo "Pulling remote experiment results..."
git pull --ff-only

if [ "${REMOTE_STATUS}" -eq 0 ]; then
  echo "Experiment complete: ${EXP_ID}"
else
  echo "Remote experiment exited with status ${REMOTE_STATUS}: ${EXP_ID}"
fi
echo "Read ${EXP_DIR}/result.md and ${EXP_DIR}/metrics.json for analysis."

exit "${REMOTE_STATUS}"
