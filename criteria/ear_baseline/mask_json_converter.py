from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROUND_DIGITS = 4


# -----------------------------------------------------------------------------
# Small utilities (copied from criteria/pottery/mask_json_converter.py -- these
# are domain-agnostic, not pottery-specific, so they are reused as-is).
# -----------------------------------------------------------------------------


def _round(value: Any, digits: int = ROUND_DIGITS) -> Any:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


def _tensor_to_list(value: Any) -> List[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [float(v) for v in value]


def _get_labels(results: Any) -> List[str]:
    if results is None:
        return []
    if "text_labels" in results:
        labels = results["text_labels"]
    elif "labels" in results:
        labels = results["labels"]
    else:
        return []
    return [str(label) for label in labels]


def _largest_contour(mask_bool: np.ndarray) -> Optional[np.ndarray]:
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _shape_features(mask_bool: np.ndarray) -> Dict[str, Any]:
    """Generic 2D shape stats for one mask: this pack does not (yet) compute
    any landmark-specific geometry the way the pottery pack does for
    neck/body/handle -- that comes after this baseline shows detection and
    segmentation are reliable enough to be worth measuring precisely."""
    image_h, image_w = mask_bool.shape
    image_area = image_h * image_w
    area_px = int(mask_bool.sum())

    if area_px == 0:
        return {
            "area_px": 0,
            "area_fraction_of_image": 0.0,
            "bbox_xyxy": None,
            "bbox_width_px": None,
            "bbox_height_px": None,
            "centroid_xy": None,
            "aspect_ratio_width_over_height": None,
            "perimeter_px": None,
            "circularity": None,
            "solidity": None,
        }

    ys, xs = np.where(mask_bool)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bbox_w = int(x2 - x1 + 1)
    bbox_h = int(y2 - y1 + 1)

    contour = _largest_contour(mask_bool)
    perimeter = None
    circularity = None
    solidity = None
    if contour is not None:
        perimeter = float(cv2.arcLength(contour, closed=True))
        if perimeter > 0:
            circularity = float((4.0 * math.pi * area_px) / (perimeter ** 2))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        contour_area = float(cv2.contourArea(contour))
        if hull_area > 0:
            solidity = float(contour_area / hull_area)

    return {
        "area_px": area_px,
        "area_fraction_of_image": _round(area_px / image_area) if image_area > 0 else None,
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_width_px": bbox_w,
        "bbox_height_px": bbox_h,
        "centroid_xy": [_round(float(xs.mean()), 2), _round(float(ys.mean()), 2)],
        "aspect_ratio_width_over_height": _round(bbox_w / bbox_h) if bbox_h > 0 else None,
        "perimeter_px": _round(perimeter, 2),
        "circularity": _round(circularity),
        "solidity": _round(solidity),
    }


# -----------------------------------------------------------------------------
# Pipeline contract functions (called from main.py)
# -----------------------------------------------------------------------------


def summarize_detections_for_report(dino_results: Any, image: Any) -> List[Dict[str, Any]]:
    """Verbose per-candidate detection details for run_report.json. This is
    the number that actually answers the baseline question: how many
    candidate boxes did GroundingDINO propose for this landmark, and at what
    confidence, before the selection algorithm picks one."""
    if dino_results is None:
        return []

    boxes = dino_results.get("boxes", [])
    scores = dino_results.get("scores", [])
    labels = _get_labels(dino_results)

    image_width, image_height = image.size
    image_area = image_width * image_height

    rows: List[Dict[str, Any]] = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        box_list = _tensor_to_list(box)
        x1, y1, x2, y2 = box_list
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h
        rows.append(
            {
                "index": int(i),
                "label": str(label),
                "score": _round(float(score)),
                "box_xyxy": [_round(v, 2) for v in box_list],
                "width_px": _round(w, 2),
                "height_px": _round(h, 2),
                "area_fraction_of_image": _round(area / image_area) if image_area > 0 else None,
                "center_xy": [_round((x1 + x2) / 2.0, 2), _round((y1 + y2) / 2.0, 2)],
            }
        )
    return rows


def summarize_mask(
    *,
    mask_bool: np.ndarray,
    image: Any,
    selected_detection: Dict[str, Any],
    text_prompt: str,
    target_name: str,
    required: bool = False,
) -> Dict[str, Any]:
    """Per-target, measurement-only summary. No landmark-specific geometry
    yet -- just detection confidence + basic mask shape, which is exactly
    what's needed to judge zero-shot feasibility before writing a real rubric
    and converter for this domain."""
    mask_bool = mask_bool.astype(bool)
    selected_box = _tensor_to_list(selected_detection["box"])
    shape = _shape_features(mask_bool)

    return {
        "target": {
            "name": target_name,
            "text_prompt": text_prompt,
            "required": bool(required),
        },
        "present": bool(mask_bool.sum() > 0),
        "selected_detection": {
            "index": int(selected_detection["index"]),
            "label": str(selected_detection["label"]),
            "score": _round(float(selected_detection["score"])),
            "box_xyxy": [_round(v, 2) for v in selected_box],
            "selection_method": selected_detection.get("selection_method"),
        },
        "shape": shape,
    }


def build_summary(
    *,
    image_path: str,
    image: Any,
    criteria_name: str,
    criteria_version: str,
    rubric: Optional[Any],
    target_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the final concise measurement JSON for one image. No rubric
    judgments here (there is no rubric yet for this pack) -- this is purely
    the zero-shot detection/segmentation baseline record."""
    num_targets = len(target_summaries)
    num_present = sum(1 for t in target_summaries if t.get("present"))
    scores = [
        t["selected_detection"]["score"]
        for t in target_summaries
        if t.get("present") and t.get("selected_detection", {}).get("score") is not None
    ]

    return {
        "schema_version": "ear_baseline_measurements_v0.1",
        "criteria": {
            "name": criteria_name,
            "version": criteria_version,
            "summary": (
                "Zero-shot GroundingDINO + SAM2 baseline for temporal-bone dissection "
                "landmarks. No fine-tuning, no rubric -- detection confidence and mask "
                "shape only, to establish a 'before fine-tuning' number."
            ),
        },
        "image": {
            "path": str(image_path),
            "width_px": int(image.size[0]),
            "height_px": int(image.size[1]),
        },
        "baseline_summary": {
            "num_targets": num_targets,
            "num_detected_and_masked": num_present,
            "detection_rate": _round(num_present / num_targets) if num_targets else None,
            "mean_detection_score_of_detected": _round(sum(scores) / len(scores)) if scores else None,
            "min_detection_score_of_detected": _round(min(scores)) if scores else None,
            "max_detection_score_of_detected": _round(max(scores)) if scores else None,
        },
        "targets": target_summaries,
    }
