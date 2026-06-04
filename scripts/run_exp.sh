#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${1:?Usage: bash scripts/run_exp.sh <exp_id>}"

if [[ ! "${EXP_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid exp_id: ${EXP_ID}"
  echo "Use only letters, numbers, dot, underscore, and hyphen."
  exit 1
fi

EXP_DIR="experiments/${EXP_ID}"
PROMPT="${EXP_DIR}/prompt.md"
LOG_DIR="${EXP_DIR}/logs"
LOG_FILE="${LOG_DIR}/claude.log"
RESULT_FILE="${EXP_DIR}/result.md"
METRICS_FILE="${EXP_DIR}/metrics.json"
CLAUDE_BIN="${CLAUDE_BIN:-}"
CLAUDE_ARGS="${CLAUDE_ARGS:-}"

if [ ! -d .git ]; then
  echo "run_exp.sh must be run from the repository root."
  exit 1
fi

if [ ! -f "${PROMPT}" ]; then
  echo "Missing ${PROMPT}"
  exit 1
fi

mkdir -p "${LOG_DIR}"

if [ -z "${CLAUDE_BIN}" ]; then
  if command -v claude >/dev/null 2>&1; then
    CLAUDE_BIN="$(command -v claude)"
  else
    CLAUDE_BIN="$(find "${HOME}/.vscode-server/extensions" \
      -path "*/resources/native-binary/claude" \
      -type f -perm -111 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
fi

if [ -z "${CLAUDE_BIN}" ] || [ ! -x "${CLAUDE_BIN}" ]; then
  echo "Could not find an executable Claude Code CLI."
  echo "Install claude in PATH or set CLAUDE_BIN=/absolute/path/to/claude."
  exit 1
fi

echo "Running Claude Code experiment: ${EXP_ID}"
echo "Prompt: ${PROMPT}"
echo "Claude binary: ${CLAUDE_BIN}"
echo "Claude args: ${CLAUDE_ARGS:-<none>}"

read -r -a CLAUDE_EXTRA_ARGS <<< "${CLAUDE_ARGS}"

set +e
"${CLAUDE_BIN}" "${CLAUDE_EXTRA_ARGS[@]}" -p "$(cat "${PROMPT}")" 2>&1 | tee "${LOG_FILE}"
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
