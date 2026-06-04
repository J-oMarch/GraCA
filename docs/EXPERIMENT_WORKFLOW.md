# 自动实验流程说明

本文档说明如何让 Codex/ChatGPT 生成实验 prompt，并通过 GitHub 同步到服务器，由服务器上的 Claude Code 自动识别和执行。

## 目录定位

本地代码仓库：

```bash
/Users/jo/ClaudePlace/GraCA
```

远程服务器仓库：

```bash
ssh -p 15600 jyh@59.72.109.245
/home/jyh/workplace/ClaudeProjects/GraCA
```

GitHub 仓库：

```text
https://github.com/J-oMarch/GraCA
```

## 日常工作流

1. 在 Codex 中打开本地代码仓库：

   ```bash
   /Users/jo/ClaudePlace/GraCA
   ```

2. 和 Codex/ChatGPT 讨论实验方向，例如：

   ```text
   基于当前 GraGE edge-gate 结果，生成一个验证 hybrid score 是否稳定超过 feature-only 的新实验。
   ```

3. Codex 应创建一个实验目录：

   ```text
   experiments/2026-06-04-hybrid-score-validation/
   ```

4. 最少应生成：

   ```text
   experiments/<exp_id>/prompt.md
   ```

   需要时再生成：

   ```text
   experiments/<exp_id>/config.yaml
   experiments/<exp_id>/notes.md
   ```

5. 你检查 `prompt.md` 后，在本地运行：

   ```bash
   cd /Users/jo/ClaudePlace/GraCA
   bash scripts/submit_exp.sh <exp_id>
   ```

6. 脚本会自动完成：

   ```text
   本地 git add/commit/push
   服务器 git pull
   服务器 bash scripts/run_exp.sh <exp_id>
   服务器 git add/commit/push 实验结果
   本地 git pull
   ```

7. 结果回来后，查看：

   ```text
   experiments/<exp_id>/result.md
   experiments/<exp_id>/metrics.json
   experiments/<exp_id>/logs/
   ```

## Prompt 规范

每个 `prompt.md` 应包含以下信息：

```md
# Experiment: <title>

## Objective

说明实验目标和要验证的假设。

## Repository Context

列出 Claude Code 应优先阅读的文件，例如：

- `src/grage/`
- `scripts/run_grage_hybrid_sweep.py`
- `paper_tables_grage_hybrid/`
- `tests/`

## Tasks

明确要求 Claude Code 做什么：

1. 检查当前实现。
2. 修改或新增必要代码。
3. 运行指定实验或 smoke test。
4. 汇总结果。

## Commands

列出建议执行的命令。

## Output Contract

Claude Code 必须写入：

- `experiments/<exp_id>/result.md`
- `experiments/<exp_id>/metrics.json`
- `experiments/<exp_id>/logs/`
```

## 本地提交脚本

提交并触发服务器执行：

```bash
bash scripts/submit_exp.sh <exp_id>
```

脚本默认服务器配置：

```bash
REMOTE_PORT=15600
REMOTE_USER=jyh
REMOTE_HOST=59.72.109.245
REMOTE_DIR=/home/jyh/workplace/ClaudeProjects/GraCA
```

如果未来服务器路径变化，可以临时覆盖：

```bash
REMOTE_DIR=/new/path bash scripts/submit_exp.sh <exp_id>
```

## 服务器执行脚本

服务器上实际运行：

```bash
bash scripts/run_exp.sh <exp_id>
```

它会读取：

```text
experiments/<exp_id>/prompt.md
```

然后查找并执行 Claude Code CLI：

```bash
claude -p "$(cat experiments/<exp_id>/prompt.md)"
```

如果服务器非交互 SSH 环境中没有 `claude` 命令，脚本会自动查找 VSCode
Claude Code 扩展自带的 native binary。也可以手动指定：

```bash
CLAUDE_BIN=/absolute/path/to/claude bash scripts/run_exp.sh <exp_id>
```

Claude Code 完成后，脚本会提交并推送：

```text
experiments/<exp_id>/
```

## 重要约定

- 新 prompt 不再放在仓库根目录。
- 不再通过 VSCode 手动上传 prompt。
- 不在服务器上手动复制本地文件。
- 每个实验使用独立 `exp_id`，避免结果互相覆盖。
- 旧 Prompt 和历史报告统一放在 `docs/archive/`。
- 触发远程算力前，建议先人工检查 `prompt.md`。

## 常见问题

### 如何知道当前实验 ID？

实验 ID 就是 `experiments/` 下的目录名，例如：

```text
2026-06-04-hybrid-score-validation
```

### 如果提交脚本提示缺少 prompt.md？

说明实验目录不完整，先确认：

```bash
ls experiments/<exp_id>/prompt.md
```

### 如果服务器执行失败？

查看：

```text
experiments/<exp_id>/logs/claude.log
experiments/<exp_id>/result.md
```

如果 Claude Code 没能生成结果文件，`run_exp.sh` 会写入一个失败状态的 `result.md` 和 `metrics.json`，方便本地继续分析。

### 是否可以让 Codex 自动提交？

可以，但只有你明确说“提交这个实验”或“运行这个实验”时，Codex 才应该执行：

```bash
bash scripts/submit_exp.sh <exp_id>
```

平时 Codex 只负责生成和修改 `experiments/<exp_id>/prompt.md`。
