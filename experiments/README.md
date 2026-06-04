# Experiments

This directory is the only place for new Codex or ChatGPT generated experiment
prompts intended for remote Claude Code execution.

Use one directory per experiment:

```text
experiments/
  YYYY-MM-DD-short-name/
    prompt.md
    config.yaml
    notes.md
    result.md
    metrics.json
    logs/
```

Required before submission:

- `prompt.md` exists.
- The prompt states what Claude Code should inspect, change, run, and report.
- Expected outputs are listed as `result.md`, `metrics.json`, and logs.

Submit from the local Mac checkout:

```bash
bash scripts/submit_exp.sh YYYY-MM-DD-short-name
```

The remote server will run `scripts/run_exp.sh` and push results back through
Git.
