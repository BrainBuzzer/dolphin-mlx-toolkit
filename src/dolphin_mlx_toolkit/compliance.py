from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download

if TYPE_CHECKING:
    from .conversion import ConversionOptions


UPSTREAM_LICENSE_NAME = "Qwen RESEARCH LICENSE AGREEMENT"
UPSTREAM_NOTICE = (
    "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, "
    "Copyright (c) Alibaba Cloud. All Rights Reserved."
)
BUILT_WITH_QWEN = "Built with Qwen"


@dataclass(slots=True)
class BundleResult:
    output_dir: Path
    files_written: list[Path]

    def to_markdown(self) -> str:
        lines = ["## Compliance Bundle"]
        for path in self.files_written:
            lines.append(f"- `{path}`")
        return "\n".join(lines)


def _read_upstream_file(source_model: str, filename: str) -> str:
    source_path = Path(source_model)
    if source_path.exists():
        return (source_path / filename).read_text()
    downloaded = hf_hub_download(repo_id=source_model, filename=filename)
    return Path(downloaded).read_text()


def _render_model_card(options: ConversionOptions) -> str:
    quantization = (
        f"{options.q_bits}-bit / group size {options.q_group_size} / mode {options.q_mode}"
        if options.quantize
        else "not quantized"
    )
    dtype = options.dtype or "upstream default"
    return dedent(
        f"""\
        ---
        tags:
          - mlx
          - multimodal
          - image-text-to-text
          - document-parsing
          - qwen2_5_vl
        license: other
        base_model:
          - {options.source_model}
        ---

        # Dolphin-v2 MLX Conversion

        This repository contains a local MLX conversion of `{options.source_model}` intended for Apple Silicon inference.

        ## Important License Notice

        The **code in this repository may be MIT-licensed**, but the **model weights are not MIT licensed**.
        The converted weights remain subject to the upstream `{UPSTREAM_LICENSE_NAME}`.

        This bundle is provided for **non-commercial research or evaluation use only** unless you separately obtain commercial rights from the upstream licensors.

        ## Required Attribution

        {BUILT_WITH_QWEN}

        ## Conversion Details

        - Source model: `{options.source_model}`
        - Quantization: `{quantization}`
        - Dtype: `{dtype}`
        - Trust remote code: `{options.trust_remote_code}`

        ## Included Compliance Files

        - `LICENSE.upstream.txt`
        - `NOTICE`
        - `UPSTREAM_MODEL_CARD.md`
        - `PUBLISHING_CHECKLIST.md`

        ## Local Usage

        ```bash
        uv run python -m mlx_vlm.generate \\
          --model . \\
          --max-tokens 512 \\
          --prompt "Parse the reading order of this document." \\
          --image /absolute/path/to/page.png
        ```

        ## Publishing Guidance

        Before publishing, confirm that:

        1. The intended release is non-commercial.
        2. The upstream license and notice files are included.
        3. Your model card prominently states `{BUILT_WITH_QWEN}`.
        4. You clearly state that the repository contains converted derivative weights.
        """
    )


def _render_notice() -> str:
    return dedent(
        f"""\
        {UPSTREAM_NOTICE}

        Additional notice for this derivative bundle:
        - This repository may contain format conversions, quantization changes, and packaging changes.
        - These changes do not replace or supersede the upstream model license.
        - Documentation for any distributed derivative model should prominently state "{BUILT_WITH_QWEN}".
        """
    )


def _render_publishing_checklist(options: ConversionOptions) -> str:
    return dedent(
        f"""\
        # Publishing Checklist

        This checklist is intentionally conservative.

        - [ ] I have confirmed my use is non-commercial research or evaluation use only.
        - [ ] I have read the upstream license in `LICENSE.upstream.txt`.
        - [ ] I have kept `NOTICE` in the bundle.
        - [ ] I have retained attribution to Qwen / Alibaba Cloud.
        - [ ] The model card prominently states "{BUILT_WITH_QWEN}".
        - [ ] The model card clearly says the weights are a converted derivative of `{options.source_model}`.
        - [ ] I am not describing the converted weights as MIT-licensed.
        - [ ] I understand Hugging Face publication can make the derivative broadly accessible.
        - [ ] I have manually reviewed the generated files before upload.
        """
    )


def write_compliance_bundle(options: ConversionOptions) -> BundleResult:
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    upstream_license = _read_upstream_file(options.source_model, "LICENSE")
    upstream_model_card = _read_upstream_file(options.source_model, "README.md")

    files = {
        output_dir / "LICENSE.upstream.txt": upstream_license,
        output_dir / "UPSTREAM_MODEL_CARD.md": upstream_model_card,
        output_dir / "NOTICE": _render_notice(),
        output_dir / "README.md": _render_model_card(options),
        output_dir / "PUBLISHING_CHECKLIST.md": _render_publishing_checklist(options),
    }

    written: list[Path] = []
    for path, contents in files.items():
        path.write_text(contents)
        written.append(path)

    return BundleResult(output_dir=output_dir, files_written=written)
