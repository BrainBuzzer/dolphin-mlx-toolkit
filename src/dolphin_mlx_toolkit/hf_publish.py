from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CONFIRMATION_TEXT = "I CONFIRM HF PUBLISH"


@dataclass(slots=True)
class PublishOptions:
    model_dir: Path
    repo_id: str
    private: bool = False
    create_repo: bool = True
    confirmation: str = ""


@dataclass(slots=True)
class PublishResult:
    commands: list[list[str]]
    return_code: int
    stdout: str
    stderr: str

    def to_markdown(self) -> str:
        lines = ["## Publish Result", f"- Exit code: `{self.return_code}`"]
        lines.append("")
        lines.append("### Commands")
        for command in self.commands:
            lines.append(f"- `{shlex.join(command)}`")
        if self.stdout.strip():
            lines.extend(["", "### Stdout", "```text", self.stdout.strip(), "```"])
        if self.stderr.strip():
            lines.extend(["", "### Stderr", "```text", self.stderr.strip(), "```"])
        return "\n".join(lines)


def _hf_command_prefix() -> list[str]:
    if shutil.which("hf"):
        return ["hf"]
    return [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli"]


def build_publish_commands(options: PublishOptions) -> list[list[str]]:
    prefix = _hf_command_prefix()
    commands: list[list[str]] = []
    if options.create_repo:
        commands.append(
            [
                *prefix,
                "repos",
                "create",
                options.repo_id,
                "--repo-type",
                "model",
                "--private" if options.private else "--no-private",
                "--exist-ok",
            ]
        )
    commands.append(
        [
            *prefix,
            "upload-large-folder",
            options.repo_id,
            str(options.model_dir),
            "--repo-type",
            "model",
            "--private" if options.private else "--no-private",
        ]
    )
    return commands


def preview_publish_commands(options: PublishOptions) -> str:
    commands = build_publish_commands(options)
    lines = [
        "## Hugging Face CLI Preview",
        f"- Confirmation required to execute: `{CONFIRMATION_TEXT}`",
    ]
    for command in commands:
        lines.append(f"- `{shlex.join(command)}`")
    return "\n".join(lines)


def run_publish(options: PublishOptions) -> PublishResult:
    if options.confirmation != CONFIRMATION_TEXT:
        raise SystemExit(
            f'Publishing blocked. Re-run with --confirmation "{CONFIRMATION_TEXT}" '
            "after you manually review the output bundle."
        )
    commands = build_publish_commands(options)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    return_code = 0
    for command in commands:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout_parts.append(completed.stdout)
        stderr_parts.append(completed.stderr)
        return_code = completed.returncode
        if completed.returncode != 0:
            break
    return PublishResult(
        commands=commands,
        return_code=return_code,
        stdout="\n".join(part for part in stdout_parts if part),
        stderr="\n".join(part for part in stderr_parts if part),
    )
