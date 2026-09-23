"""Draw extracted fields as colored boxes on the page image."""

from __future__ import annotations

from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from app.schema import BoundingBox, DocumentResult, FieldResult

# Green = high, yellow = medium, red = low.
CONFIDENCE_COLORS: dict[str, tuple[int, int, int]] = {
    "high": (22, 163, 74),
    "medium": (234, 179, 8),
    "low": (220, 38, 38),
}
_LABEL_TEXT_COLORS: dict[str, tuple[int, int, int]] = {
    "high": (255, 255, 255),
    "medium": (0, 0, 0),
    "low": (255, 255, 255),
}


def _load_font(size: int) -> ImageFont.ImageFont:
    """Default Pillow font at the given size (older Pillow ignores the size)."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: BoundingBox,
    label: str,
    confidence: str,
    font: ImageFont.ImageFont,
    line_width: int,
    image_size: tuple[int, int],
) -> None:
    """Draw one rectangle with a small label on top."""
    color = CONFIDENCE_COLORS.get(confidence, CONFIDENCE_COLORS["low"])
    draw.rectangle((box.x0, box.y0, box.x1, box.y1), outline=color, width=line_width)

    text_box = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = text_box[2] - text_box[0], text_box[3] - text_box[1]
    pad = 2
    x = min(max(box.x0, 0), max(image_size[0] - text_w - 2 * pad, 0))
    y = box.y0 - text_h - 2 * pad
    if y < 0:  # no room above the box, put the label below it
        y = box.y1
    draw.rectangle((x, y, x + text_w + 2 * pad, y + text_h + 2 * pad), fill=color)
    draw.text(
        (x + pad, y + pad - text_box[1]),
        label,
        fill=_LABEL_TEXT_COLORS.get(confidence, (255, 255, 255)),
        font=font,
    )


def draw_fields(
    image: Image.Image,
    result: DocumentResult,
    page: int = 0,
    show_line_items: bool = True,
    font_size: Optional[int] = None,
) -> Image.Image:
    """Return a copy of the page image with a box around every found field.

    Args:
        image: The processed page image (same one the OCR saw).
        result: The pipeline result for this document.
        page: Which page this image is.
        show_line_items: Also draw boxes for line item fields.
        font_size: Label size in pixels. Default scales with the image width.

    Returns:
        A new RGB image. The input image is not changed.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    size = font_size or max(12, canvas.width // 70)
    font = _load_font(size)
    line_width = max(2, canvas.width // 400)

    to_draw: list[tuple[str, FieldResult]] = [
        (name.replace("_", " "), field) for name, field in result.fields.items()
    ]
    if show_line_items:
        for index, item in enumerate(result.line_items, start=1):
            for name, field in item.items():
                to_draw.append((f"item {index} {name.replace('_', ' ')}", field))

    # Fields that point to the same box get one box with a combined label.
    rank = {"high": 0, "medium": 1, "low": 2}
    grouped: dict[tuple[float, ...], tuple[BoundingBox, list[str], str]] = {}
    for label, field in to_draw:
        if field.box is None or field.box.page != page or field.confidence is None:
            continue
        key = (field.box.x0, field.box.y0, field.box.x1, field.box.y1)
        if key not in grouped:
            grouped[key] = (field.box, [label], field.confidence)
        else:
            box, labels, confidence = grouped[key]
            labels.append(label)
            worst = max(confidence, field.confidence, key=rank.__getitem__)
            grouped[key] = (box, labels, worst)

    for box, labels, confidence in grouped.values():
        _draw_labeled_box(draw, box, " + ".join(labels), confidence, font, line_width, canvas.size)
    return canvas
