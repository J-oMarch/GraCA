#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/check_exp_suite_status.sh <exp_id> [<exp_id> ...]"
  exit 1
fi

if [ ! -d .git ]; then
  echo "check_exp_suite_status.sh must be run from the repository root."
  exit 1
fi

for exp_id in "$@"; do
  if [[ ! "${exp_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid exp_id: ${exp_id}"
    echo "Use only letters, numbers, dot, underscore, and hyphen."
    exit 1
  fi
done

for exp_id in "$@"; do
  echo "============================================================"
  echo "Experiment: ${exp_id}"
  echo "============================================================"
  bash scripts/check_exp_status.sh "${exp_id}" || true
  echo
done
