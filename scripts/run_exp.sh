#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${1:?Usage: bash scripts/run_exp.sh <exp_id>}"

EXP_DIR="experiments/${EXP_ID}"
PROMPT="${EXP_DIR}/prompt.md"
LOG_DIR="${EXP_DIR}/logs"
LOG_FILE="${LOG_DIR}/claude.log"
RESULT_FILE="${EXP_DIR}/result.md"
METRICS_FILE="${EXP_DIR}/metrics.json"

if [ ! -d .git ]; then
  echo "run_exp.sh must be run from the repository root."
  exit 1
fi

if [ ! -f "${PROMPT}" ]; then
  echo "Missing ${PROMPT}"
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Running Claude Code experiment: ${EXP_ID}"
echo "Prompt: ${PROMPT}"

set +e
claude -p "$(cat "${PROMPT}")" 2>&1 | tee "${LOG_FILE}"
CLAUDE_STATUS=${PIPESTATUS[0]}
set -e

if [ ! -f "${RESULT_FILE}" ]; then
  cat > "${RESULT_FILE}" <<EOF
# Experiment Result: ${EXP_ID}

Claude Code did not create \`${RESULT_FILE}\`.

- Exit status: ${CLAUDE_STATUS}
- Log file: \`${LOG_FILE}\`

Review the log and rerun or refine \`${PROMPT}\`.
EOF
fi

if [ ! -f "${METRICS_FILE}" ]; then
  cat > "${METRICS_FILE}" <<EOF
{
  "exp_id": "${EXP_ID}",
  "status": "missing_metrics",
  "claude_exit_status": ${CLAUDE_STATUS}
}
EOF
fi

git add "${EXP_DIR}"
if git diff --cached --quiet; then
  echo "No experiment result changes to commit."
else
  git commit -m "run experiment ${EXP_ID}"
fi

git push

exit "${CLAUDE_STATUS}"
