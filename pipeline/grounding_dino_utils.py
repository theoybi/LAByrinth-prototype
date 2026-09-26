from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from PIL import ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_grounding_dino(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def run_grounding_dino(
    image,
    text_prompt: str,
    processor,
    model,
    device: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
):
    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    return results


def get_labels(results: Dict[str, Any]) -> List[str]:
    """Transformers versions disagree on whether labels or text_labels exists."""
    if "text_labels" in results:
        return [str(label) for label in results["text_labels"]]
    if "labels" in results:
        return [str(label) for label in results["labels"]]
    return ["unknown"] * len(results.get("scores", []))


def calculate_box_area(box) -> float:
    x1, y1, x2, y2 = box.detach().cpu().tolist() if hasattr(box, "detach") else box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def calculate_box_center(box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box.detach().cpu().tolist() if hasattr(box, "detach") else box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def is_point_inside_box(point: Tuple[float, float], box) -> bool:
    x, y = point
    x1, y1, x2, y2 = box.detach().cpu().tolist() if hasattr(box, "detach") else box
    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a.detach().cpu().tolist() if hasattr(box_a, "detach") else box_a
    bx1, by1, bx2, by2 = box_b.detach().cpu().tolist() if hasattr(box_b, "detach") else box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_a = calculate_box_area(box_a)
    area_b = calculate_box_area(box_b)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def remove_container_boxes(results, candidate_indices: List[int], area_ratio_threshold: float = 3.0) -> List[int]:
    boxes = results["boxes"]
    indices_to_remove = set()

    for i in candidate_indices:
        box_i = boxes[i]
        area_i = calculate_box_area(box_i)

        for j in candidate_indices:
            if i == j:
                continue
            box_j = boxes[j]
            area_j = calculate_box_area(box_j)
            center_j = calculate_box_center(box_j)

            if area_i > area_j * area_ratio_threshold and is_point_inside_box(center_j, box_i):
                indices_to_remove.add(i)

    filtered = [i for i in candidate_indices if i not in indices_to_remove]
    return filtered if filtered else candidate_indices


def apply_nms(results, candidate_indices: List[int], iou_threshold: float = 0.50) -> List[int]:
    boxes = results["boxes"]
    scores = results["scores"]
    sorted_indices = sorted(candidate_indices, key=lambda i: float(scores[i]), reverse=True)
    kept_indices: List[int] = []

    for current_index in sorted_indices:
        current_box = boxes[current_index]
        should_keep = True
        for kept_index in kept_indices:
            if calculate_iou(current_box, boxes[kept_index]) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept_indices.append(current_index)

    return kept_indices


def get_selected_detection(results, selected_index: int) -> Dict[str, Any]:
    boxes = results["boxes"]
    scores = results["scores"]
    labels = get_labels(results)
    label = labels[selected_index] if selected_index < len(labels) else "unknown"
    return {
        "index": int(selected_index),
        "box": boxes[selected_index],
        "score": scores[selected_index],
        "label": label,
    }


def get_best_detection(results, *, area_ratio_threshold: float = 3.0, nms_iou_threshold: float = 0.50) -> Dict[str, Any]:
    scores = results["scores"]
    candidate_indices = list(range(len(scores)))
    if not candidate_indices:
        raise RuntimeError("GroundingDINO returned no detections.")

    candidate_indices = remove_container_boxes(
        results=results,
        candidate_indices=candidate_indices,
        area_ratio_threshold=area_ratio_threshold,
    )
    candidate_indices = apply_nms(
        results=results,
        candidate_indices=candidate_indices,
        iou_threshold=nms_iou_threshold,
    )
    best_index = max(candidate_indices, key=lambda i: float(scores[i]))
    selected = get_selected_detection(results, selected_index=best_index)
    selected["selection_method"] = "container_filter_then_nms_then_highest_score"
    selected["remaining_candidate_indices"] = [int(i) for i in candidate_indices]
    return selected


def save_detection_image(image, detection: Dict[str, Any], output_path, color=(255, 0, 0)) -> None:
    """Optional debug image for one selected detection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    x1, y1, x2, y2 = detection["box"].detach().cpu().tolist()
    draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
    draw.text((x1, y1), f"{detection['label']}: {float(detection['score']):.2f}", fill=color)
    drawn.save(output_path)


def save_all_detections_image(image, results, output_path, color=(255, 0, 0)) -> None:
    """Optional debug image for all detections for a single target."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    labels = get_labels(results)

    for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
        x1, y1, x2, y2 = box.detach().cpu().tolist()
        label = labels[i] if i < len(labels) else "unknown"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1), f"{i}: {label} {float(score):.2f}", fill=color)

    drawn.save(output_path)
