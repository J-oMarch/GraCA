#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${1:?Usage: bash scripts/submit_exp_tmux.sh <exp_id>}"

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

EXP_DIR="experiments/${EXP_ID}"
PROMPT="${EXP_DIR}/prompt.md"
SAFE_EXP_ID="$(printf '%s' "${EXP_ID}" | tr -c 'A-Za-z0-9_-' '_')"
TMUX_SESSION="${TMUX_SESSION:-graca_claude}"
TMUX_WINDOW="${TMUX_WINDOW:-exp_${SAFE_EXP_ID}}"
CLAUDE_ARGS_DEFAULT="--dangerously-skip-permissions --permission-mode bypassPermissions --effort max"
REMOTE_CLAUDE_ARGS="${REMOTE_CLAUDE_ARGS:-${CLAUDE_ARGS_DEFAULT}}"

if [ ! -d .git ]; then
  echo "submit_exp_tmux.sh must be run from the repository root."
  exit 1
fi

if [ ! -f "${PROMPT}" ]; then
  echo "Missing ${PROMPT}"
  exit 1
fi

echo "Syncing local branch before tmux submit..."
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

echo "Starting remote tmux experiment on ${REMOTE}:${REMOTE_DIR}"
ssh -p "${REMOTE_PORT}" "${REMOTE}" "bash -s" <<EOF
set -euo pipefail
cd "${REMOTE_DIR}"
git pull --ff-only

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH on the remote server."
  exit 1
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null && \
   tmux list-windows -t "${TMUX_SESSION}" -F '#W' | grep -Fx "${TMUX_WINDOW}" >/dev/null; then
  echo "tmux window already exists: ${TMUX_SESSION}:${TMUX_WINDOW}"
  echo "Attach with: tmux attach -t ${TMUX_SESSION}"
  exit 1
fi

RUN_CMD="cd '${REMOTE_DIR}' && CLAUDE_ARGS='${REMOTE_CLAUDE_ARGS}' bash scripts/run_exp.sh '${EXP_ID}'; status=\\\$?; echo; echo '[tmux] experiment ${EXP_ID} finished with status '\${status}; exec bash"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  tmux new-window -t "${TMUX_SESSION}" -n "${TMUX_WINDOW}" "\${RUN_CMD}"
else
  tmux new-session -d -s "${TMUX_SESSION}" -n "${TMUX_WINDOW}" "\${RUN_CMD}"
fi

tmux ls | grep "${TMUX_SESSION}"
tmux list-windows -t "${TMUX_SESSION}"
EOF

echo
echo "Remote tmux experiment started."
echo "Session: ${TMUX_SESSION}"
echo "Window: ${TMUX_WINDOW}"
echo
echo "Monitor from Mac through Codex:"
echo "  bash scripts/check_exp_status.sh ${EXP_ID}"
echo
echo "Manual server observation:"
echo "  ssh -p ${REMOTE_PORT} ${REMOTE}"
echo "  tmux attach -t ${TMUX_SESSION}"
echo "  # then use Ctrl-b w to choose window ${TMUX_WINDOW}"
echo
echo "After it finishes, pull results with:"
echo "  git pull --ff-only"
