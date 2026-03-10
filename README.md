# Dolphin MLX Toolkit

Open-source tooling to convert `ByteDance/Dolphin-v2` into an MLX-friendly bundle for Apple Silicon and run local PDF-to-markdown parsing with a Streamlit UI.

## Scope

- Converts the upstream Hugging Face model to MLX by delegating to `mlx_vlm.convert`.
- Parses PDFs locally with Dolphin-v2-MLX and assembles markdown output.
- Embeds extracted figures as inline data URIs so downloaded markdown stays standalone.
- Generates `NOTICE`, upstream license copies, a publishing checklist, and a derivative model card.
- Keeps Hugging Face publication as an explicit CLI-only step.

## Important License Distinction

This repository's **code** is MIT licensed.

The converted **model weights** are **not** MIT licensed. They remain derivative works of `ByteDance/Dolphin-v2`, which ships under the Qwen Research License. That license permits redistribution for non-commercial research/evaluation use and imposes attribution and notice requirements.

Do not describe converted Dolphin/Qwen weights as MIT licensed.

## Why This Repo Exists

The upstream Dolphin repository focuses on PyTorch / Transformers inference. This project creates a separate, public repo for:

- MLX conversion on macOS
- reproducible `uv` workflows
- a simple Streamlit PDF parser UI
- safer pre-publication checks before any Hugging Face upload

## Quickstart

```bash
uv sync
uv run python -m dolphin_mlx_toolkit.cli check
uv run streamlit run src/dolphin_mlx_toolkit/app.py
```

## CLI Usage

Preview environment:

```bash
uv run dolphin-mlx-toolkit check
```

Write the compliance bundle only:

```bash
uv run dolphin-mlx-toolkit prepare-bundle \
  --source-model ByteDance/Dolphin-v2 \
  --output-dir artifacts/dolphin-v2-mlx
```

Run local conversion:

```bash
uv run dolphin-mlx-toolkit convert \
  --source-model ByteDance/Dolphin-v2 \
  --output-dir artifacts/dolphin-v2-mlx
```

Parse a PDF with a local MLX model:

```bash
uv run dolphin-mlx-toolkit parse \
  --input-pdf test_assets/agent-lightning-rl.pdf \
  --model-dir artifacts/dolphin-v2-mlx-4bit \
  --output-dir artifacts/parsed
```

Preview Hugging Face CLI commands:

```bash
uv run dolphin-mlx-toolkit preview-publish \
  --model-dir artifacts/dolphin-v2-mlx \
  --repo-id your-user/dolphin-v2-mlx-4bit
```

Actual publication is intentionally gated and requires:

```text
I CONFIRM HF PUBLISH
```

## Streamlit UI

```bash
uv run streamlit run src/dolphin_mlx_toolkit/app.py
```

The UI lets you:

- upload a PDF
- choose the local MLX model directory
- run the two-stage Dolphin parsing pipeline locally
- preview and download the generated markdown and JSON outputs

## Notes On Fidelity

This repo ports the main two-stage document parsing flow from the upstream Dolphin project:

- page-level reading order detection
- element-level OCR / table / formula / code parsing
- markdown assembly with multi-page separators

The markdown quality still depends on the underlying model outputs. This toolkit does not yet add extra corrective heuristics beyond the upstream-style orchestration.
