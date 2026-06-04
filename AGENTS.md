# AGENTS.md

## Repository Role

This repository is the executable GraCA/GraGE codebase. Keep research notes,
historical prompts, and reports under `docs/`; keep runnable code, configs,
tests, and experiment automation at the repository root.

## Experiment Workflow

When the user asks to design or run a new experiment:

1. Create a new directory under `experiments/` using:
   `YYYY-MM-DD-short-name`

2. Write the Claude Code task prompt to:
   `experiments/<exp_id>/prompt.md`

3. Also create, when useful:
   - `experiments/<exp_id>/config.yaml`
   - `experiments/<exp_id>/notes.md`

4. Every `prompt.md` must include:
   - experiment objective
   - relevant files or modules to inspect
   - concrete implementation or analysis tasks
   - commands Claude Code should run
   - required output contract:
     - `experiments/<exp_id>/result.md`
     - `experiments/<exp_id>/metrics.json`
     - relevant logs under `experiments/<exp_id>/logs/`

5. Do not place new experiment prompts in the repository root. All new prompts
   belong under `experiments/<exp_id>/`.

6. Do not manually upload files through VSCode. Experiment prompts and results
   must flow through Git.

7. After creating experiment files, tell the user to run:
   `bash scripts/submit_exp.sh <exp_id>`

8. Only run the submit script when the user explicitly asks, for example:
   - `submit this experiment`
   - `提交这个实验`
   - `运行这个实验`

9. The local submit script is responsible for:
   - committing `experiments/<exp_id>`
   - pushing to GitHub
   - asking the remote server to `git pull`
   - running Claude Code through `scripts/run_exp.sh`
   - pulling results back locally

10. The remote server is responsible only for:
    - `git pull`
    - running Claude Code against `experiments/<exp_id>/prompt.md`
    - writing results under `experiments/<exp_id>/`
    - committing and pushing results

11. Local analysis should read:
    - `experiments/<exp_id>/result.md`
    - `experiments/<exp_id>/metrics.json`
    - relevant logs under `experiments/<exp_id>/logs/`

## Cleanup Rules

- Prefer archiving historical prompts and reports under `docs/archive/` instead
  of deleting them.
- Keep the repository root focused on runnable project entry points.
- Do not remove datasets, result tables, or generated artifacts unless the user
  explicitly confirms the exact files or directories to delete.

## Verification

After modifying code, run the smallest relevant test or smoke command. For
workflow-only changes, run shell syntax checks for modified scripts.
