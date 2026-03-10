from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(slots=True)
class EnvironmentReport:
    python: str
    uv_installed: bool
    git_installed: bool
    gh_installed: bool
    hf_cli_installed: bool
    mlx_vlm_installed: bool

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "## Environment",
                f"- Python: `{self.python}`",
                f"- uv installed: `{self.uv_installed}`",
                f"- git installed: `{self.git_installed}`",
                f"- gh installed: `{self.gh_installed}`",
                f"- hf CLI installed: `{self.hf_cli_installed}`",
                f"- mlx-vlm importable: `{self.mlx_vlm_installed}`",
            ]
        )


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _module_cli_available() -> bool:
    command = [
        sys.executable,
        "-m",
        "huggingface_hub.commands.huggingface_cli",
        "--help",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.returncode == 0


def _mlx_vlm_importable() -> bool:
    command = [sys.executable, "-c", "import mlx_vlm"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.returncode == 0


def get_environment_report() -> EnvironmentReport:
    return EnvironmentReport(
        python=sys.version.split()[0],
        uv_installed=_command_exists("uv"),
        git_installed=_command_exists("git"),
        gh_installed=_command_exists("gh"),
        hf_cli_installed=_command_exists("hf") or _module_cli_available(),
        mlx_vlm_installed=_mlx_vlm_importable(),
    )
