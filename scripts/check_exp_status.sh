#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${1:?Usage: bash scripts/check_exp_status.sh <exp_id>}"

if [[ ! "${EXP_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid exp_id: ${EXP_ID}"
  echo "Use only letters, numbers, dot, underscore, and hyphen."
  exit 1
fi

REMOTE_PORT="${REMOTE_PORT:-15600}"
REMOTE_USER="${REMOTE_USER:-jyh}"
REMOTE_HOST="${REMOTE_HOST:-59.72.109.245}"
REMOTE_DIR="${REMOTE_DIR:-/home/jyh/workplace/ClaudeProjects/GraCA}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

SAFE_EXP_ID="$(printf '%s' "${EXP_ID}" | tr -c 'A-Za-z0-9_-' '_')"
TMUX_SESSION="${TMUX_SESSION:-graca_claude}"
TMUX_WINDOW="${TMUX_WINDOW:-exp_${SAFE_EXP_ID}}"
EXP_DIR="experiments/${EXP_ID}"

if [ ! -d .git ]; then
  echo "check_exp_status.sh must be run from the repository root."
  exit 1
fi

echo "Checking remote tmux target: ${TMUX_SESSION}:${TMUX_WINDOW}"
set +e
ssh -p "${REMOTE_PORT}" "${REMOTE}" \
  "cd '${REMOTE_DIR}' && tmux has-session -t '${TMUX_SESSION}' 2>/dev/null && tmux list-windows -t '${TMUX_SESSION}' -F '#W' | grep -Fx '${TMUX_WINDOW}' >/dev/null"
TMUX_STATUS=$?
set -e

if [ "${TMUX_STATUS}" -eq 0 ]; then
  set +e
  ssh -p "${REMOTE_PORT}" "${REMOTE}" \
    "pgrep -af \"bash scripts/run_exp.sh ${EXP_ID}|scripts/run_exp.sh ${EXP_ID}\" >/dev/null"
  RUN_STATUS=$?
  set -e

  if [ "${RUN_STATUS}" -eq 0 ]; then
    echo "Status: running"
  else
    echo "Status: tmux window exists, but run_exp.sh is not currently running"
  fi
  echo
  echo "Recent remote log lines:"
  ssh -p "${REMOTE_PORT}" "${REMOTE}" \
    "cd '${REMOTE_DIR}' && tail -n 80 '${EXP_DIR}/logs/claude.log' 2>/dev/null || echo 'No log file yet.'"
  echo
  echo "Attach manually with:"
  echo "  ssh -p ${REMOTE_PORT} ${REMOTE}"
  echo "  tmux attach -t ${TMUX_SESSION}"
  echo "  # then use Ctrl-b w to choose window ${TMUX_WINDOW}"

  if [ "${RUN_STATUS}" -eq 0 ]; then
    exit 0
  fi

  echo
  echo "Pulling latest GitHub changes because the run process is no longer active..."
  git pull --ff-only

  if [ -f "${EXP_DIR}/result.md" ]; then
    echo
    echo "Result file found: ${EXP_DIR}/result.md"
    sed -n '1,120p' "${EXP_DIR}/result.md"
  else
    echo
    echo "No local result file found yet: ${EXP_DIR}/result.md"
  fi
  exit 0
fi

echo "Status: tmux target not found"
echo "Pulling latest GitHub changes..."
git pull --ff-only

if [ -f "${EXP_DIR}/result.md" ]; then
  echo
  echo "Result file found: ${EXP_DIR}/result.md"
  sed -n '1,120p' "${EXP_DIR}/result.md"
else
  echo
  echo "No local result file found yet: ${EXP_DIR}/result.md"
  echo "The experiment may have failed before pushing results, or the tmux window name may differ."
fi
