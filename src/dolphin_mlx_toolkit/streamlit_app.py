from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from .environment import get_environment_report
from .parser import DolphinMLXDocumentParser, DocumentResult


def _default_model_path() -> str:
    candidates = [
        Path.cwd() / "artifacts" / "dolphin-v2-mlx-4bit",
        Path.cwd() / "artifacts" / "dolphin-v2-mlx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


@st.cache_resource(show_spinner=False)
def _load_parser(
    model_path: str,
    layout_max_tokens: int,
    element_max_tokens: int,
    temperature: float,
) -> DolphinMLXDocumentParser:
    return DolphinMLXDocumentParser(
        model_path,
        layout_max_tokens=layout_max_tokens,
        default_element_max_tokens=element_max_tokens,
        temperature=temperature,
    )


def _render_result(result: DocumentResult, elapsed_seconds: float) -> None:
    total_elements = sum(len(page.elements) for page in result.pages)
    st.success("Parsing finished.")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Pages", result.total_pages)
    metric_col2.metric("Elements", total_elements)
    metric_col3.metric("Seconds", f"{elapsed_seconds:.1f}")

    stem = Path(result.source_name).stem
    st.download_button(
        "Download markdown",
        data=result.markdown.encode("utf-8"),
        file_name=f"{stem}.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download JSON",
        data=result.to_json().encode("utf-8"),
        file_name=f"{stem}.json",
        mime="application/json",
    )

    markdown_tab, json_tab, pages_tab = st.tabs(["Markdown", "JSON", "Pages"])
    with markdown_tab:
        st.caption("Figures are embedded as data URIs so the markdown file is standalone.")
        st.code(result.markdown or "", language="markdown")
    with json_tab:
        st.json(result.to_dict())
    with pages_tab:
        for page in result.pages:
            with st.expander(f"Page {page.page_number}"):
                st.write(f"Detected elements: {len(page.elements)}")
                st.code(page.markdown or "", language="markdown")
                st.code(page.layout_output or "", language="text")


def main() -> None:
    st.set_page_config(page_title="Dolphin MLX PDF Parser", layout="wide")
    st.title("Dolphin MLX PDF Parser")
    st.caption("Upload a PDF, run local Dolphin-v2-MLX parsing, and download the generated markdown.")

    with st.sidebar:
        st.header("Runtime")
        report = get_environment_report()
        st.markdown(report.to_markdown())
        model_path = st.text_input("MLX model directory", value=_default_model_path())
        layout_max_tokens = st.number_input(
            "Layout max tokens",
            min_value=256,
            max_value=4096,
            value=2048,
            step=256,
        )
        element_max_tokens = st.number_input(
            "Element max tokens",
            min_value=128,
            max_value=4096,
            value=1024,
            step=128,
        )
        max_pages = st.number_input(
            "Max pages to process",
            min_value=0,
            max_value=500,
            value=0,
            step=1,
            help="Use 0 to process the full PDF.",
        )
        temperature = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
        )
        st.info(
            "The app keeps the model local. It does not publish anything to Hugging Face."
        )

    uploaded_pdf = st.file_uploader("PDF file", type=["pdf"])
    run_requested = st.button("Parse PDF", type="primary", disabled=uploaded_pdf is None)

    if run_requested and uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.getvalue()
        with st.spinner("Loading MLX model ..."):
            parser = _load_parser(
                model_path=model_path,
                layout_max_tokens=int(layout_max_tokens),
                element_max_tokens=int(element_max_tokens),
                temperature=float(temperature),
            )

        started_at = time.perf_counter()
        with st.spinner("Parsing PDF ..."):
            result = parser.parse_pdf_bytes(
                pdf_bytes,
                source_name=uploaded_pdf.name,
                max_pages=None if max_pages == 0 else int(max_pages),
            )
        elapsed = time.perf_counter() - started_at
        st.session_state["last_parse"] = {"result": result, "elapsed": elapsed}

    last_parse = st.session_state.get("last_parse")
    if last_parse:
        _render_result(last_parse["result"], last_parse["elapsed"])


if __name__ == "__main__":
    main()
