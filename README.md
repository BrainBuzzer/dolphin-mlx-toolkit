# Dolphin MLX Toolkit

Open-source tooling to convert `ByteDance/Dolphin-v2` into an MLX-friendly bundle for Apple Silicon, with a Streamlit UI and conservative compliance artifacts for derivative weight publication.

## Scope

- Converts the upstream Hugging Face model to MLX by delegating to `mlx_vlm.convert`.
- Generates `NOTICE`, upstream license copies, a publishing checklist, and a derivative model card.
- Previews Hugging Face CLI publication commands.
- Blocks actual Hugging Face publishing unless you explicitly provide a confirmation string.

## Important License Distinction

This repository's **code** is MIT licensed.

The converted **model weights** are **not** MIT licensed. They remain derivative works of `ByteDance/Dolphin-v2`, which ships under the Qwen Research License. That license permits redistribution for non-commercial research/evaluation use and imposes attribution and notice requirements.

Do not describe converted Dolphin/Qwen weights as MIT licensed.

## Why This Repo Exists

The upstream Dolphin repository focuses on PyTorch / Transformers inference. This project creates a separate, public repo for:

- MLX conversion on macOS
- reproducible `uv` workflows
- a simple Streamlit UI
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

- preview the exact `mlx_vlm.convert` command
- generate compliance files before conversion
- run the conversion locally
- preview the Hugging Face CLI commands without pushing

## Notes On Fidelity

Converting the weights to MLX is only part of the work. Dolphin's full behavior also depends on its two-stage orchestration and post-processing pipeline in the upstream repo. This toolkit currently focuses on the model conversion and compliant publication workflow, not a full parity reimplementation of every PyTorch inference utility.
