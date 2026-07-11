from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from PIL import Image, ImageDraw


RGB = Tuple[int, int, int]
DEFAULT_COLOR: RGB = (255, 0, 0)
UNSELECTED_BOX_COLOR: RGB = (180, 180, 180)


def _as_rgb(color: Any) -> RGB:
    if color is None:
        return DEFAULT_COLOR
    return tuple(int(v) for v in color[:3])  # type: ignore[index]


def _tensor_to_list(value: Any):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def _get_labels(results: Any):
    if results is None:
        return []
    if "text_labels" in results:
        return [str(label) for label in results["text_labels"]]
    if "labels" in results:
        return [str(label) for label in results["labels"]]
    return ["unknown"] * len(results.get("scores", []))


def _add_legend(image: Image.Image, items: Mapping[str, RGB], title: str) -> Image.Image:
    legend_width = 340
    row_height = 28
    padding = 14
    min_height = padding * 2 + row_height * (len(items) + 2)
    canvas_height = max(image.height, min_height)

    canvas = Image.new("RGB", (image.width + legend_width, canvas_height), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    x0 = image.width + padding
    y = padding
    draw.text((x0, y), title, fill=(0, 0, 0))
    y += row_height + 4

    for name, color in items.items():
        color = _as_rgb(color)
        draw.rectangle([x0, y + 4, x0 + 18, y + 22], fill=color, outline=(0, 0, 0))
        draw.text((x0 + 28, y + 5), name, fill=(0, 0, 0))
        y += row_height

    return canvas


def save_mask_overview(
    image: Image.Image,
    masks_by_target: Mapping[str, np.ndarray],
    target_colors: Mapping[str, RGB],
    output_path: Path,
    alpha: float = 0.45,
) -> None:
    """Save one combined overlay image with a color-coded mask per target."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    legend_items: Dict[str, RGB] = {}

    for target_name, mask_bool in masks_by_target.items():
        if mask_bool is None or int(mask_bool.sum()) == 0:
            continue
        color = _as_rgb(target_colors.get(target_name, DEFAULT_COLOR))
        color_arr = np.array(color, dtype=np.float32)
        overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color_arr
        legend_items[target_name] = color

    combined = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    combined = _add_legend(combined, legend_items, title="Mask overview")
    combined.save(output_path, quality=95)


def save_detection_overview(
    image: Image.Image,
    dino_results_by_target: Mapping[str, Any],
    selected_detections_by_target: Mapping[str, Dict[str, Any]],
    target_colors: Mapping[str, RGB],
    output_path: Path,
) -> None:
    """Save one image with all detections muted and selected detections highlighted."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drawn = image.convert("RGB").copy()
    draw = ImageDraw.Draw(drawn)
    legend_items: Dict[str, RGB] = {}

    # First draw all candidate detections in muted gray.
    for target_name, results in dino_results_by_target.items():
        if results is None:
            continue
        labels = _get_labels(results)
        for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
            x1, y1, x2, y2 = _tensor_to_list(box)
            label = labels[i] if i < len(labels) else "unknown"
            draw.rectangle([x1, y1, x2, y2], outline=UNSELECTED_BOX_COLOR, width=1)
            draw.text((x1, y1), f"{target_name}:{i} {float(score):.2f}", fill=UNSELECTED_BOX_COLOR)

    # Then draw selected detections in target-specific colors.
    for target_name, detection in selected_detections_by_target.items():
        if detection is None:
            continue
        color = _as_rgb(target_colors.get(target_name, DEFAULT_COLOR))
        legend_items[target_name] = color
        x1, y1, x2, y2 = _tensor_to_list(detection["box"])
        score = float(detection["score"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        draw.text((x1, max(0, y1 - 16)), f"SELECTED {target_name} {score:.2f}", fill=color)

    with_legend = _add_legend(drawn, legend_items, title="Selected detections")
    with_legend.save(output_path, quality=95)
