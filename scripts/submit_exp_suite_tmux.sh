#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/submit_exp_suite_tmux.sh <exp_id> [<exp_id> ...]"
  exit 1
fi

if [ ! -d .git ]; then
  echo "submit_exp_suite_tmux.sh must be run from the repository root."
  exit 1
fi

for exp_id in "$@"; do
  if [[ ! "${exp_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid exp_id: ${exp_id}"
    echo "Use only letters, numbers, dot, underscore, and hyphen."
    exit 1
  fi

  if [ ! -f "experiments/${exp_id}/prompt.md" ]; then
    echo "Missing experiments/${exp_id}/prompt.md"
    exit 1
  fi
done

echo "Submitting ${#} experiment(s) through tmux:"
printf '  %s\n' "$@"
echo

for exp_id in "$@"; do
  echo "==> ${exp_id}"
  bash scripts/submit_exp_tmux.sh "${exp_id}"
  echo
done

echo "All submit commands completed."
