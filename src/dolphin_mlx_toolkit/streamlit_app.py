from __future__ import annotations

import shlex
from pathlib import Path

import streamlit as st

from .compliance import BundleResult
from .conversion import (
    ConversionOptions,
    build_conversion_command,
    prepare_conversion_bundle,
    run_conversion,
)
from .environment import get_environment_report
from .hf_publish import CONFIRMATION_TEXT, PublishOptions, preview_publish_commands


def _default_output_dir() -> Path:
    return Path.cwd() / "artifacts" / "dolphin-v2-mlx"


def _render_bundle(bundle: BundleResult) -> None:
    st.subheader("Compliance bundle")
    for path in bundle.files_written:
        st.code(str(path), language="text")


def main() -> None:
    st.set_page_config(page_title="Dolphin MLX Toolkit", layout="wide")
    st.title("Dolphin MLX Toolkit")
    st.caption(
        "Local conversion workflow for ByteDance/Dolphin-v2 to MLX with guarded Hugging Face publication."
    )

    with st.sidebar:
        st.header("Environment")
        report = get_environment_report()
        st.markdown(report.to_markdown())
        st.info(
            "This app never publishes to Hugging Face unless you separately run the guarded publish command."
        )

    with st.form("conversion_form"):
        col1, col2 = st.columns(2)
        with col1:
            source_model = st.text_input(
                "Source model",
                value="ByteDance/Dolphin-v2",
                help="Hugging Face model id or local path to the original weights.",
            )
            output_dir = st.text_input(
                "Output directory",
                value=str(_default_output_dir()),
            )
            trust_remote_code = st.checkbox("Trust remote code", value=False)
            quantize = st.checkbox("Quantize converted weights", value=True)
        with col2:
            q_bits = st.selectbox("q_bits", [3, 4, 6, 8], index=1)
            q_group_size = st.selectbox("q_group_size", [32, 64, 128], index=1)
            q_mode = st.selectbox("q_mode", ["affine", "mxfp4", "nvfp4", "mxfp8"], index=0)
            dtype = st.selectbox("dtype", ["bfloat16", "float16", "float32", "auto"], index=0)

        submitted = st.form_submit_button("Preview conversion plan")

    options = ConversionOptions(
        source_model=source_model,
        output_dir=Path(output_dir),
        quantize=quantize,
        q_bits=q_bits,
        q_group_size=q_group_size,
        q_mode=q_mode,
        dtype=None if dtype == "auto" else dtype,
        trust_remote_code=trust_remote_code,
        write_compliance_bundle=True,
    )

    st.subheader("Conversion command")
    st.code(shlex.join(build_conversion_command(options)), language="bash")

    if submitted:
        st.success("Plan generated. Review the command and the compliance checklist before running conversion.")

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Write compliance bundle"):
            bundle = prepare_conversion_bundle(options)
            _render_bundle(bundle)
    with action_col2:
        if st.button("Run local conversion"):
            with st.spinner("Running mlx_vlm.convert ..."):
                result = run_conversion(options)
            st.markdown(result.to_markdown())

    st.subheader("Hugging Face publication preview")
    repo_id = st.text_input(
        "Target Hugging Face repo id",
        value="BrainBuzzer/dolphin-v2-mlx-4bit",
    )
    private_repo = st.checkbox("Create private HF repo", value=False)
    publish_options = PublishOptions(
        model_dir=Path(output_dir),
        repo_id=repo_id,
        private=private_repo,
    )
    st.markdown(preview_publish_commands(publish_options))
    st.warning(
        f'Actual publication is gated. The CLI requires the exact confirmation string: "{CONFIRMATION_TEXT}".'
    )


if __name__ == "__main__":
    main()
