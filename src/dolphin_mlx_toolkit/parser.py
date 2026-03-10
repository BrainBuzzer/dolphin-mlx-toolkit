from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from PIL import Image
from mlx_vlm import generate, load
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from .markdown import MarkdownConverter

LAYOUT_PROMPT = "Parse the reading order of this document."
ELEMENT_PROMPTS = {
    "tab": ("Parse the table in the image.", 2048),
    "equ": ("Read formula in the image.", 256),
    "code": ("Read code in the image.", 1024),
    "default": ("Read text in the image.", 1024),
}


@dataclass(slots=True)
class ElementResult:
    label: str
    bbox: list[int]
    text: str
    reading_order: int
    tags: list[str]


@dataclass(slots=True)
class PageResult:
    page_number: int
    layout_output: str
    markdown: str
    elements: list[ElementResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "layout_output": self.layout_output,
            "markdown": self.markdown,
            "elements": [asdict(element) for element in self.elements],
        }


@dataclass(slots=True)
class DocumentResult:
    source_name: str
    total_pages: int
    markdown: str
    pages: list[PageResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "total_pages": self.total_pages,
            "markdown": self.markdown,
            "pages": [page.to_dict() for page in self.pages],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass(slots=True)
class SavedDocumentResult:
    markdown_path: Path
    json_path: Path


class DolphinMLXDocumentParser:
    def __init__(
        self,
        model_path: str | Path,
        *,
        layout_max_tokens: int = 2048,
        default_element_max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.model_path = str(model_path)
        self.layout_max_tokens = layout_max_tokens
        self.default_element_max_tokens = default_element_max_tokens
        self.temperature = temperature
        self.model, self.processor = load(self.model_path)
        self.markdown_converter = MarkdownConverter()

    def parse_pdf_bytes(
        self,
        pdf_bytes: bytes,
        *,
        source_name: str,
        max_pages: int | None = None,
    ) -> DocumentResult:
        images = render_pdf_bytes(pdf_bytes, max_pages=max_pages)
        return self.parse_images(images, source_name=source_name)

    def parse_pdf_path(
        self,
        pdf_path: str | Path,
        *,
        max_pages: int | None = None,
    ) -> DocumentResult:
        pdf_path = Path(pdf_path)
        return self.parse_pdf_bytes(
            pdf_path.read_bytes(),
            source_name=pdf_path.name,
            max_pages=max_pages,
        )

    def parse_images(
        self,
        images: list[Image.Image],
        *,
        source_name: str,
    ) -> DocumentResult:
        pages = [self._parse_page(image, page_number=index) for index, image in enumerate(images, start=1)]
        combined_markdown = "\n\n---\n\n".join(
            page.markdown.strip() for page in pages if page.markdown.strip()
        )
        return DocumentResult(
            source_name=source_name,
            total_pages=len(pages),
            markdown=combined_markdown.strip() + ("\n" if combined_markdown else ""),
            pages=pages,
        )

    def _parse_page(self, image: Image.Image, *, page_number: int) -> PageResult:
        layout_output = self._generate(prompt=LAYOUT_PROMPT, image=image, max_tokens=self.layout_max_tokens)
        elements = self._process_elements(layout_output, image)
        markdown = self.markdown_converter.convert([asdict(element) for element in elements])
        return PageResult(
            page_number=page_number,
            layout_output=layout_output,
            markdown=markdown,
            elements=elements,
        )

    def _process_elements(self, layout_output: str, image: Image.Image) -> list[ElementResult]:
        parsed_layout = parse_layout_string(layout_output)
        if not parsed_layout or not (layout_output.startswith("[") and layout_output.endswith("]")):
            parsed_layout = [([0.0, 0.0, float(image.size[0]), float(image.size[1])], "distorted_page", [])]
        elif len(parsed_layout) > 1 and check_bbox_overlap(parsed_layout, image):
            parsed_layout = [([0.0, 0.0, float(image.size[0]), float(image.size[1])], "distorted_page", [])]

        pending_elements: list[tuple[dict[str, Any], str, int]] = []
        results: list[ElementResult] = []
        reading_order = 0

        for bbox, label, tags in parsed_layout:
            x1, y1, x2, y2 = process_coordinates(bbox, image)
            crop = image.crop((x1, y1, x2, y2))
            if crop.size[0] <= 3 or crop.size[1] <= 3:
                reading_order += 1
                continue

            if label == "fig":
                results.append(
                    ElementResult(
                        label=label,
                        bbox=[x1, y1, x2, y2],
                        text=image_to_data_uri(crop),
                        reading_order=reading_order,
                        tags=tags,
                    )
                )
            else:
                prompt, max_tokens = prompt_for_label(label, self.default_element_max_tokens)
                pending_elements.append(
                    (
                        {
                            "label": label,
                            "bbox": [x1, y1, x2, y2],
                            "crop": crop,
                            "reading_order": reading_order,
                            "tags": tags,
                        },
                        prompt,
                        max_tokens,
                    )
                )
            reading_order += 1

        for element, prompt, max_tokens in pending_elements:
            text = self._generate(prompt=prompt, image=element["crop"], max_tokens=max_tokens)
            results.append(
                ElementResult(
                    label=element["label"],
                    bbox=element["bbox"],
                    text=text.strip(),
                    reading_order=element["reading_order"],
                    tags=element["tags"],
                )
            )

        results.sort(key=lambda item: item.reading_order)
        return results

    def _generate(self, *, prompt: str, image: Image.Image, max_tokens: int) -> str:
        formatted_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        result = generate(
            self.model,
            self.processor,
            prompt=formatted_prompt,
            image=image,
            max_tokens=max_tokens,
            temperature=self.temperature,
            verbose=False,
        )
        return result.text.strip()


def render_pdf_bytes(pdf_bytes: bytes, *, max_pages: int | None = None, target_size: int = 896) -> list[Image.Image]:
    images: list[Image.Image] = []
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = len(document)
        page_count = min(total_pages, max_pages) if max_pages else total_pages
        for page_index in range(page_count):
            page = document[page_index]
            rect = page.rect
            scale = target_size / max(rect.width, rect.height)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            images.append(image)
    finally:
        document.close()
    return images


def write_document_result(result: DocumentResult, output_dir: str | Path) -> SavedDocumentResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.source_name).stem
    markdown_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    markdown_path.write_text(result.markdown, encoding="utf-8")
    json_path.write_text(result.to_json(), encoding="utf-8")
    return SavedDocumentResult(markdown_path=markdown_path, json_path=json_path)


def prompt_for_label(label: str, default_max_tokens: int) -> tuple[str, int]:
    if label in ELEMENT_PROMPTS:
        return ELEMENT_PROMPTS[label]
    prompt, _ = ELEMENT_PROMPTS["default"]
    return prompt, default_max_tokens


def image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_layout_string(bbox_str: str) -> list[tuple[list[float], str, list[str]]]:
    parsed_results: list[tuple[list[float], str, list[str]]] = []
    segments = bbox_str.split("[PAIR_SEP]")
    flattened: list[str] = []
    for segment in segments:
        flattened.extend(segment.split("[RELATION_SEP]"))

    for segment in flattened:
        segment = segment.strip()
        if not segment:
            continue
        coord_match = re.search(r"\[(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+)\]", segment)
        labels = extract_labels_from_string(segment)
        if coord_match and labels:
            coords = [float(coord_match.group(i)) for i in range(1, 5)]
            parsed_results.append((coords, labels[0].strip(), labels[1:]))
    return parsed_results


def extract_labels_from_string(text: str) -> list[str]:
    all_matches = re.findall(r"\[([^\]]+)\]", text)
    return [match for match in all_matches if not re.match(r"^\d+,\d+,\d+,\d+$", match)]


def process_coordinates(coords: list[float], image: Image.Image) -> tuple[int, int, int, int]:
    original_width, original_height = image.size
    resized_image = resize_img(image)
    resized_height, resized_width = resized_image.size[1], resized_image.size[0]
    resized_height, resized_width = smart_resize(
        resized_height,
        resized_width,
        factor=28,
        min_pixels=784,
        max_pixels=2_560_000,
    )
    width_ratio = original_width / resized_width
    height_ratio = original_height / resized_height

    x1 = int(coords[0] * width_ratio)
    y1 = int(coords[1] * height_ratio)
    x2 = int(coords[2] * width_ratio)
    y2 = int(coords[3] * height_ratio)

    x1 = max(0, min(x1, original_width - 1))
    y1 = max(0, min(y1, original_height - 1))
    x2 = max(x1 + 1, min(x2, original_width))
    y2 = max(y1 + 1, min(y2, original_height))
    return x1, y1, x2, y2


def resize_img(image: Image.Image, max_size: int = 1600, min_size: int = 28) -> Image.Image:
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            height = int(height * (max_size / width))
            width = max_size
        else:
            width = int(width * (max_size / height))
            height = max_size
        image = image.resize((width, height))

    width, height = image.size
    if min(width, height) < min_size:
        if width < height:
            height = int(height * (min_size / width))
            width = min_size
        else:
            width = int(width * (min_size / height))
            height = min_size
        image = image.resize((width, height))
    return image


def check_bbox_overlap(
    layout_results: list[tuple[list[float], str, list[str]]],
    image: Image.Image,
    *,
    iou_threshold: float = 0.1,
    overlap_box_ratio: float = 0.25,
) -> bool:
    if len(layout_results) <= 1:
        return False
    boxes = []
    for bbox, _, _ in layout_results:
        boxes.append(process_coordinates(bbox, image))
    iou_matrix = calculate_iou_matrix(boxes)
    overlap_mask = iou_matrix > iou_threshold
    np.fill_diagonal(overlap_mask, False)
    overlap_ratio = overlap_mask.any(axis=1).sum() / len(boxes)
    return overlap_ratio > overlap_box_ratio


def calculate_iou_matrix(boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    box_array = np.array(boxes)
    areas = (box_array[:, 2] - box_array[:, 0]) * (box_array[:, 3] - box_array[:, 1])
    top_left = np.maximum(box_array[:, None, :2], box_array[None, :, :2])
    bottom_right = np.minimum(box_array[:, None, 2:], box_array[None, :, 2:])
    width_height = np.clip(bottom_right - top_left, 0, None)
    intersections = width_height[:, :, 0] * width_height[:, :, 1]
    unions = areas[:, None] + areas[None, :] - intersections
    return intersections / np.clip(unions, 1e-6, None)
