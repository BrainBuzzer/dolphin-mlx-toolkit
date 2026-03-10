from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .compliance import BundleResult, write_compliance_bundle


@dataclass(slots=True)
class ConversionOptions:
    source_model: str
    output_dir: Path
    quantize: bool = True
    q_bits: int = 4
    q_group_size: int = 64
    q_mode: str = "affine"
    dtype: Optional[str] = "bfloat16"
    trust_remote_code: bool = False
    write_compliance_bundle: bool = True


@dataclass(slots=True)
class ConversionResult:
    command: list[str]
    output_dir: Path
    compliance_bundle: Optional[BundleResult]
    return_code: int
    stdout: str
    stderr: str

    def to_markdown(self) -> str:
        lines = [
            "## Conversion Result",
            f"- Output dir: `{self.output_dir}`",
            f"- Exit code: `{self.return_code}`",
            f"- Command: `{shlex.join(self.command)}`",
        ]
        if self.compliance_bundle is not None:
            lines.append("")
            lines.append(self.compliance_bundle.to_markdown())
        if self.stdout.strip():
            lines.extend(["", "### Stdout", "```text", self.stdout.strip(), "```"])
        if self.stderr.strip():
            lines.extend(["", "### Stderr", "```text", self.stderr.strip(), "```"])
        return "\n".join(lines)


def build_conversion_command(options: ConversionOptions) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "mlx_vlm.convert",
        "--hf-path",
        options.source_model,
        "--mlx-path",
        str(options.output_dir),
    ]
    if options.quantize:
        command.extend(
            [
                "--quantize",
                "--q-bits",
                str(options.q_bits),
                "--q-group-size",
                str(options.q_group_size),
                "--q-mode",
                options.q_mode,
            ]
        )
    if options.dtype:
        command.extend(["--dtype", options.dtype])
    if options.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def prepare_conversion_bundle(options: ConversionOptions) -> BundleResult:
    options.output_dir.mkdir(parents=True, exist_ok=True)
    return write_compliance_bundle(options)


def run_conversion(options: ConversionOptions) -> ConversionResult:
    options.output_dir.parent.mkdir(parents=True, exist_ok=True)
    compliance_bundle = None
    if options.write_compliance_bundle:
        compliance_bundle = prepare_conversion_bundle(options)

    command = build_conversion_command(options)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return ConversionResult(
        command=command,
        output_dir=options.output_dir,
        compliance_bundle=compliance_bundle,
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
