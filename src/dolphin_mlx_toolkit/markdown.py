from __future__ import annotations

import re
from typing import Any


def extract_table_from_html(html_string: str) -> str:
    try:
        table_pattern = re.compile(r"<table.*?>.*?</table>", re.DOTALL)
        tables = table_pattern.findall(html_string)
        tables = [re.sub(r"<table[^>]*>", "<table>", table) for table in tables]
        return "\n".join(tables)
    except Exception as exc:
        return f"<table><tr><td>Error extracting table: {exc}</td></tr></table>"


class MarkdownConverter:
    def __init__(self) -> None:
        self.heading_levels = {
            "sec_0": "#",
            "sec_1": "##",
            "sec_2": "###",
            "sec_3": "###",
            "sec_4": "###",
            "sec_5": "###",
        }
        self.replace_dict = {
            "\\bm": "\\mathbf ",
            "\\eqno": "\\quad ",
            "\\quad": "\\quad ",
            "\\leq": "\\leq ",
            "\\pm": "\\pm ",
            "\\varmathbb": "\\mathbb ",
            "\\in fty": "\\infty",
            "\\mu": "\\mu ",
            "\\cdot": "\\cdot ",
            "\\langle": "\\langle ",
        }

    def convert(self, recognition_results: list[dict[str, Any]]) -> str:
        markdown_content: list[str] = []
        for section_count, result in enumerate(recognition_results):
            label = result.get("label", "")
            text = result.get("text", "").strip()
            if not text:
                continue

            if label in self.heading_levels:
                markdown_content.append(self._handle_heading(text, label))
            elif label == "fig":
                markdown_content.append(self._handle_figure(text, section_count))
            elif label == "tab":
                markdown_content.append(self._handle_table(text))
            elif label == "equ":
                markdown_content.append(self._handle_formula(text))
            elif label == "list":
                markdown_content.append(f"- {text.strip()}\n")
            elif label == "code":
                markdown_content.append(f"```bash\n{text}\n```\n\n")
            else:
                markdown_content.append(f"{self._handle_text(text)}\n\n")
        return "".join(markdown_content)

    def _handle_heading(self, text: str, label: str) -> str:
        level = self.heading_levels.get(label, "#")
        text = self._remove_newline_in_heading(text.strip())
        return f"{level} {self._handle_text(text)}\n\n"

    def _handle_figure(self, text: str, section_count: int) -> str:
        if text.startswith("data:image/"):
            return f"![Figure {section_count}]({text})\n\n"
        if text.startswith("!["):
            return f"{text}\n\n"
        return f"![Figure {section_count}]({text})\n\n"

    def _handle_table(self, text: str) -> str:
        return f"{extract_table_from_html(text)}\n\n"

    def _handle_formula(self, text: str) -> str:
        text = text.strip("$").rstrip("\\ ").replace(r"\upmu", r"\mu")
        for key, value in self.replace_dict.items():
            text = text.replace(key, value)
        return f"$${text}$$\n\n"

    def _handle_text(self, text: str) -> str:
        text = self._process_formulas_in_text(text)
        return self._try_remove_newline(text)

    def _process_formulas_in_text(self, text: str) -> str:
        text = text.replace(r"\upmu", r"\mu")
        for key, value in self.replace_dict.items():
            text = text.replace(key, value)
        return text

    def _remove_newline_in_heading(self, text: str) -> str:
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return text.replace("\n", "")
        return text.replace("\n", " ")

    def _try_remove_newline(self, text: str) -> str:
        text = text.strip().replace("-\n", "")
        lines = text.split("\n")
        if len(lines) == 1:
            return text

        processed_lines: list[str] = []
        for index, current in enumerate(lines[:-1]):
            current_line = current.strip()
            next_line = lines[index + 1].strip()
            if not current_line:
                processed_lines.append("\n")
                continue
            if not next_line:
                processed_lines.append(f"{current_line}\n")
                continue
            if self._is_chinese(current_line[-1]) and self._is_chinese(next_line[0]):
                processed_lines.append(current_line)
            else:
                processed_lines.append(f"{current_line} ")

        last_line = lines[-1].strip()
        if last_line:
            processed_lines.append(last_line)
        return "".join(processed_lines)

    @staticmethod
    def _is_chinese(char: str) -> bool:
        return "\u4e00" <= char <= "\u9fff"
