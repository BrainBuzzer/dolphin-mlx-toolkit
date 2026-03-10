from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .conversion import ConversionOptions, prepare_conversion_bundle, run_conversion
from .hf_publish import PublishOptions, preview_publish_commands, run_publish
from .parser import DolphinMLXDocumentParser, write_document_result

app = typer.Typer(
    no_args_is_help=True,
    help="Convert ByteDance/Dolphin-v2 into MLX format and parse PDFs locally.",
)


@app.command("check")
def check() -> None:
    """Print environment checks for local conversion and guarded publishing."""
    from .environment import get_environment_report

    typer.echo(get_environment_report().to_markdown())


@app.command("convert")
def convert(
    source_model: str = typer.Option(
        "ByteDance/Dolphin-v2",
        help="Hugging Face model id or local path to the original Dolphin weights.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/dolphin-v2-mlx"),
        help="Directory where the converted MLX bundle will be written.",
    ),
    quantize: bool = typer.Option(True, help="Quantize the converted model."),
    q_bits: int = typer.Option(4, min=2, max=8, help="Quantization bit-width."),
    q_group_size: int = typer.Option(64, min=32, help="Quantization group size."),
    q_mode: str = typer.Option(
        "affine",
        help="Quantization mode accepted by mlx_vlm.convert.",
    ),
    dtype: Optional[str] = typer.Option(
        "bfloat16",
        help="Target dtype. Use auto to defer to the upstream config.",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        help="Pass --trust-remote-code to mlx_vlm.convert.",
    ),
    skip_compliance_bundle: bool = typer.Option(
        False,
        help="Skip generation of README/NOTICE/license artifacts in the output folder.",
    ),
) -> None:
    """Run a local MLX conversion and prepare a compliant output bundle."""
    options = ConversionOptions(
        source_model=source_model,
        output_dir=output_dir,
        quantize=quantize,
        q_bits=q_bits,
        q_group_size=q_group_size,
        q_mode=q_mode,
        dtype=None if dtype == "auto" else dtype,
        trust_remote_code=trust_remote_code,
        write_compliance_bundle=not skip_compliance_bundle,
    )
    result = run_conversion(options)
    typer.echo(result.to_markdown())


@app.command("prepare-bundle")
def prepare_bundle(
    source_model: str = typer.Option(
        "ByteDance/Dolphin-v2",
        help="Hugging Face model id or local path to the original Dolphin weights.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/dolphin-v2-mlx"),
        help="Directory where the compliant bundle files will be written.",
    ),
    quantize: bool = typer.Option(True, help="Reflect quantization defaults in generated docs."),
    q_bits: int = typer.Option(4, min=2, max=8, help="Quantization bit-width."),
    q_group_size: int = typer.Option(64, min=32, help="Quantization group size."),
    q_mode: str = typer.Option("affine", help="Quantization mode metadata."),
    dtype: Optional[str] = typer.Option(
        "bfloat16",
        help="Target dtype. Use auto to defer to the upstream config.",
    ),
) -> None:
    """Generate model-card, license, notice, and checklist files without converting."""
    options = ConversionOptions(
        source_model=source_model,
        output_dir=output_dir,
        quantize=quantize,
        q_bits=q_bits,
        q_group_size=q_group_size,
        q_mode=q_mode,
        dtype=None if dtype == "auto" else dtype,
        trust_remote_code=False,
        write_compliance_bundle=True,
    )
    bundle = prepare_conversion_bundle(options)
    typer.echo(bundle.to_markdown())


@app.command("parse")
def parse(
    input_pdf: Path = typer.Option(..., exists=True, dir_okay=False, help="PDF to parse."),
    model_dir: Path = typer.Option(
        Path("artifacts/dolphin-v2-mlx-4bit"),
        exists=True,
        file_okay=False,
        help="Directory containing the converted Dolphin-v2 MLX model.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/parsed"),
        help="Directory where markdown and JSON outputs will be written.",
    ),
    max_pages: int = typer.Option(0, min=0, help="Use 0 to parse the full PDF."),
    layout_max_tokens: int = typer.Option(2048, min=256, max=4096),
    element_max_tokens: int = typer.Option(1024, min=128, max=4096),
    temperature: float = typer.Option(0.0, min=0.0, max=1.0),
) -> None:
    """Parse a PDF with the local MLX model and write markdown plus JSON outputs."""
    parser = DolphinMLXDocumentParser(
        model_dir,
        layout_max_tokens=layout_max_tokens,
        default_element_max_tokens=element_max_tokens,
        temperature=temperature,
    )
    result = parser.parse_pdf_path(
        input_pdf,
        max_pages=None if max_pages == 0 else max_pages,
    )
    saved = write_document_result(result, output_dir)
    typer.echo(f"Markdown written to {saved.markdown_path}")
    typer.echo(f"JSON written to {saved.json_path}")


@app.command("preview-publish")
def preview_publish(
    model_dir: Path = typer.Option(
        Path("artifacts/dolphin-v2-mlx"),
        help="Local folder that contains the converted MLX weights and compliance files.",
    ),
    repo_id: str = typer.Option(
        ...,
        help="Destination Hugging Face repo id, for example user/dolphin-v2-mlx-4bit.",
    ),
    private: bool = typer.Option(False, help="Preview private model repo creation."),
) -> None:
    """Show the Hugging Face CLI commands that would be used for publication."""
    options = PublishOptions(model_dir=model_dir, repo_id=repo_id, private=private)
    typer.echo(preview_publish_commands(options))


@app.command("publish")
def publish(
    model_dir: Path = typer.Option(
        Path("artifacts/dolphin-v2-mlx"),
        help="Local folder that contains the converted MLX weights and compliance files.",
    ),
    repo_id: str = typer.Option(
        ...,
        help="Destination Hugging Face repo id, for example user/dolphin-v2-mlx-4bit.",
    ),
    private: bool = typer.Option(False, help="Create the repo as private."),
    create_repo: bool = typer.Option(
        True,
        help="Create the destination Hugging Face repo if it does not already exist.",
    ),
    confirmation: str = typer.Option(
        "",
        help='Required safety string: type "I CONFIRM HF PUBLISH" to run.',
    ),
) -> None:
    """Publish to Hugging Face via the CLI after explicit confirmation."""
    options = PublishOptions(
        model_dir=model_dir,
        repo_id=repo_id,
        private=private,
        create_repo=create_repo,
        confirmation=confirmation,
    )
    result = run_publish(options)
    typer.echo(result.to_markdown())


def main() -> None:
    app()
