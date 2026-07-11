from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


ROUND_DIGITS = 4
PROFILE_SLICES = 16
CURVE_SAMPLES = 24
HANDLE_ATTACHMENT_DILATION_RADIUS_PX = 8
HANDLE_TOP_BOTTOM_EXCLUSION_FRACTION = 0.15
MIN_CORE_PIXELS_FOR_THICKNESS = 25

# In-memory cache so build_summary() can compute cross-mask measurements without
# changing main.py. The masks are not written into the JSON.
_MASK_CACHE: Dict[str, np.ndarray] = {}


# -----------------------------------------------------------------------------
# Small utilities
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


def _bbox_from_mask(mask_bool: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _bbox_dict(mask_bool: np.ndarray) -> Optional[Dict[str, Any]]:
    box = _bbox_from_mask(mask_bool)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_width_px": int(x2 - x1 + 1),
        "bbox_height_px": int(y2 - y1 + 1),
    }


def _largest_contour(mask_bool: np.ndarray) -> Optional[np.ndarray]:
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _component_shape_features(component_mask: np.ndarray) -> Dict[str, Any]:
    """Shape features for any binary component, including handle holes."""
    area_px = int(component_mask.sum())
    box = _bbox_from_mask(component_mask)
    if area_px == 0 or box is None:
        return {
            "area_px": 0,
            "bbox_xyxy": None,
            "bbox_width_px": None,
            "bbox_height_px": None,
            "aspect_ratio_width_over_height": None,
            "elongation_ratio": None,
            "perimeter_px": None,
            "circularity": None,
            "centroid_xy": None,
        }

    x1, y1, x2, y2 = box
    bbox_w = int(x2 - x1 + 1)
    bbox_h = int(y2 - y1 + 1)
    aspect = float(bbox_w / bbox_h) if bbox_h > 0 else None
    elongation = max(aspect, 1.0 / aspect) if aspect not in (None, 0) else None

    ys, xs = np.where(component_mask)
    contour = _largest_contour(component_mask)
    perimeter = None
    circularity = None
    if contour is not None:
        perimeter = float(cv2.arcLength(contour, closed=True))
        if perimeter > 0:
            circularity = float((4.0 * math.pi * area_px) / (perimeter ** 2))

    return {
        "area_px": area_px,
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_width_px": bbox_w,
        "bbox_height_px": bbox_h,
        "aspect_ratio_width_over_height": _round(aspect),
        "elongation_ratio": _round(elongation),
        "perimeter_px": _round(perimeter),
        "circularity": _round(circularity),
        "centroid_xy": [_round(xs.mean()), _round(ys.mean())],
    }


# -----------------------------------------------------------------------------
# Detection reporting for run_report.json
# -----------------------------------------------------------------------------


def summarize_detections_for_report(dino_results: Any, image: Any) -> List[Dict[str, Any]]:
    """Verbose detection details for run_report.json, not the concise summary."""
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


# -----------------------------------------------------------------------------
# Basic shape and profile measurements
# -----------------------------------------------------------------------------


def _shape_features(mask_bool: np.ndarray) -> Dict[str, Any]:
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
            "extent": None,
            "solidity": None,
        }

    ys, xs = np.where(mask_bool)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bbox_w = int(x2 - x1 + 1)
    bbox_h = int(y2 - y1 + 1)
    bbox_area = bbox_w * bbox_h

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
        "centroid_xy": [_round(xs.mean()), _round(ys.mean())],
        "aspect_ratio_width_over_height": _round(bbox_w / bbox_h) if bbox_h > 0 else None,
        "perimeter_px": _round(perimeter),
        "circularity": _round(circularity),
        "extent": _round(area_px / bbox_area) if bbox_area > 0 else None,
        "solidity": _round(solidity),
    }


def _vertical_width_profile(mask_bool: np.ndarray, num_slices: int = PROFILE_SLICES) -> List[Dict[str, Any]]:
    ys, _ = np.where(mask_bool)
    if len(ys) == 0:
        return []

    y_min = int(ys.min())
    y_max = int(ys.max())
    if y_max <= y_min:
        return []

    edges = np.linspace(y_min, y_max + 1, num_slices + 1)
    profile: List[Dict[str, Any]] = []

    for i in range(num_slices):
        start_y = int(round(edges[i]))
        end_y = int(round(edges[i + 1]))
        end_y = max(start_y + 1, end_y)

        widths: List[float] = []
        centers: List[float] = []
        lefts: List[float] = []
        rights: List[float] = []

        for y in range(start_y, min(end_y, mask_bool.shape[0])):
            xs = np.where(mask_bool[y, :])[0]
            if len(xs) == 0:
                continue
            left = float(xs.min())
            right = float(xs.max())
            widths.append(right - left + 1.0)
            centers.append((left + right) / 2.0)
            lefts.append(left)
            rights.append(right)

        y_mid = (start_y + end_y - 1) / 2.0
        y_relative = (y_mid - y_min) / max(1.0, y_max - y_min)

        profile.append(
            {
                "slice_index": int(i),
                "y_relative_top_to_bottom": _round(y_relative),
                "y_range_px": [start_y, end_y],
                "width_px": _round(np.mean(widths), 2) if widths else None,
                "left_x_px": _round(np.mean(lefts), 2) if lefts else None,
                "right_x_px": _round(np.mean(rights), 2) if rights else None,
                "center_x_px": _round(np.mean(centers), 2) if centers else None,
            }
        )

    return profile


def _valid_profile_points(profile: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [p for p in profile if p.get("width_px") not in (None, 0) and p.get("center_x_px") is not None]


def _width_profile_summary(profile: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = _valid_profile_points(profile)
    if not valid:
        return {
            "top_width_px": None,
            "middle_width_px": None,
            "bottom_width_px": None,
            "min_width_px": None,
            "max_width_px": None,
            "min_width_y_relative": None,
            "max_width_y_relative": None,
            "width_cv": None,
            "max_to_top_width_ratio": None,
        }

    widths = np.array([p["width_px"] for p in valid], dtype=float)
    max_i = int(np.argmax(widths))
    min_i = int(np.argmin(widths))
    middle_i = len(valid) // 2
    top = float(valid[0]["width_px"])
    max_width = float(widths[max_i])

    return {
        "top_width_px": _round(valid[0]["width_px"], 2),
        "middle_width_px": _round(valid[middle_i]["width_px"], 2),
        "bottom_width_px": _round(valid[-1]["width_px"], 2),
        "min_width_px": _round(widths[min_i], 2),
        "max_width_px": _round(widths[max_i], 2),
        "min_width_y_relative": _round(valid[min_i]["y_relative_top_to_bottom"]),
        "max_width_y_relative": _round(valid[max_i]["y_relative_top_to_bottom"]),
        "width_cv": _round(float(widths.std() / widths.mean())) if widths.mean() > 0 else None,
        "max_to_top_width_ratio": _round(max_width / top) if top > 0 else None,
    }


def _moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _boundary_curve_profile(mask_bool: np.ndarray, samples: int = CURVE_SAMPLES) -> List[Dict[str, Any]]:
    """Sample left/right visible boundary coordinates along vertical height."""
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return []

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox_w = max(1, x_max - x_min)
    bbox_h = max(1, y_max - y_min)

    y_values = np.linspace(y_min, y_max, samples)
    band_radius = max(1, int(round(bbox_h / (samples * 2))))

    raw_y: List[float] = []
    raw_left: List[float] = []
    raw_right: List[float] = []

    for y_float in y_values:
        y_center = int(round(y_float))
        y0 = max(0, y_center - band_radius)
        y1 = min(mask_bool.shape[0], y_center + band_radius + 1)
        band = mask_bool[y0:y1, :]
        _, band_xs = np.where(band)
        if len(band_xs) == 0:
            continue
        raw_y.append(float(y_center))
        raw_left.append(float(band_xs.min()))
        raw_right.append(float(band_xs.max()))

    if len(raw_y) < 2:
        return []

    y_arr = np.array(raw_y, dtype=float)
    left_arr = _moving_average(np.array(raw_left, dtype=float), window=3)
    right_arr = _moving_average(np.array(raw_right, dtype=float), window=3)
    center_arr = (left_arr + right_arr) / 2.0
    half_width_arr = (right_arr - left_arr) / 2.0

    # Straight endpoint baselines. Deviations from these lines are useful for
    # curvature proxies while keeping the data numerical.
    left_line = np.interp(y_arr, [y_arr[0], y_arr[-1]], [left_arr[0], left_arr[-1]])
    right_line = np.interp(y_arr, [y_arr[0], y_arr[-1]], [right_arr[0], right_arr[-1]])
    half_width_line = np.interp(
        y_arr,
        [y_arr[0], y_arr[-1]],
        [half_width_arr[0], half_width_arr[-1]],
    )

    rows: List[Dict[str, Any]] = []
    for i in range(len(y_arr)):
        y_rel = (y_arr[i] - y_min) / bbox_h
        half_width_deviation = half_width_arr[i] - half_width_line[i]
        rows.append(
            {
                "sample_index": int(i),
                "y_relative_top_to_bottom": _round(y_rel),
                "left_x_px": _round(left_arr[i], 2),
                "right_x_px": _round(right_arr[i], 2),
                "center_x_px": _round(center_arr[i], 2),
                "half_width_px": _round(half_width_arr[i], 2),
                "left_x_relative_in_bbox": _round((left_arr[i] - x_min) / bbox_w),
                "right_x_relative_in_bbox": _round((right_arr[i] - x_min) / bbox_w),
                "center_x_relative_in_bbox": _round((center_arr[i] - x_min) / bbox_w),
                "half_width_relative_to_bbox": _round(half_width_arr[i] / bbox_w),
                "left_deviation_from_endpoint_line_px": _round(left_arr[i] - left_line[i], 2),
                "right_deviation_from_endpoint_line_px": _round(right_arr[i] - right_line[i], 2),
                "half_width_deviation_from_endpoint_line_px": _round(half_width_deviation, 2),
            }
        )
    return rows


def _boundary_curve_summary(profile: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(profile) < 3:
        return {
            "sample_count": len(profile),
            "max_half_width_px": None,
            "max_half_width_y_relative": None,
            "min_half_width_px": None,
            "min_half_width_y_relative": None,
            "max_positive_half_width_deviation_px": None,
            "max_positive_half_width_deviation_y_relative": None,
            "max_negative_half_width_deviation_px": None,
            "max_negative_half_width_deviation_y_relative": None,
            "mean_positive_half_width_deviation_px": None,
            "mean_negative_half_width_deviation_px": None,
            "mean_abs_side_boundary_deviation_px": None,
        }

    half_width = np.array([p["half_width_px"] for p in profile], dtype=float)
    half_dev = np.array([p["half_width_deviation_from_endpoint_line_px"] for p in profile], dtype=float)
    left_dev = np.array([p["left_deviation_from_endpoint_line_px"] for p in profile], dtype=float)
    right_dev = np.array([p["right_deviation_from_endpoint_line_px"] for p in profile], dtype=float)
    y_rel = np.array([p["y_relative_top_to_bottom"] for p in profile], dtype=float)

    max_hw_i = int(np.argmax(half_width))
    min_hw_i = int(np.argmin(half_width))
    max_pos_i = int(np.argmax(half_dev))
    max_neg_i = int(np.argmin(half_dev))
    positive = half_dev[half_dev > 0]
    negative = half_dev[half_dev < 0]

    return {
        "sample_count": int(len(profile)),
        "max_half_width_px": _round(half_width[max_hw_i], 2),
        "max_half_width_y_relative": _round(y_rel[max_hw_i]),
        "min_half_width_px": _round(half_width[min_hw_i], 2),
        "min_half_width_y_relative": _round(y_rel[min_hw_i]),
        "max_positive_half_width_deviation_px": _round(max(0.0, float(half_dev[max_pos_i])), 2),
        "max_positive_half_width_deviation_y_relative": _round(y_rel[max_pos_i]),
        "max_negative_half_width_deviation_px": _round(min(0.0, float(half_dev[max_neg_i])), 2),
        "max_negative_half_width_deviation_y_relative": _round(y_rel[max_neg_i]),
        "mean_positive_half_width_deviation_px": _round(float(np.mean(positive)), 2) if len(positive) else 0.0,
        "mean_negative_half_width_deviation_px": _round(float(np.mean(negative)), 2) if len(negative) else 0.0,
        "mean_abs_side_boundary_deviation_px": _round(
            float((np.mean(np.abs(left_dev)) + np.mean(np.abs(right_dev))) / 2.0),
            2,
        ),
    }


# -----------------------------------------------------------------------------
# Pottery-specific measurements
# -----------------------------------------------------------------------------


def _neck_curve_measurements(width_summary: Dict[str, Any], curve_summary: Dict[str, Any]) -> Dict[str, Any]:
    top = width_summary.get("top_width_px")
    middle = width_summary.get("middle_width_px")
    bottom = width_summary.get("bottom_width_px")
    min_width = width_summary.get("min_width_px")
    endpoint_max = None
    if top is not None and bottom is not None:
        endpoint_max = max(float(top), float(bottom))

    inward_pinch_ratio = None
    inward_pinch_depth_px = None
    if endpoint_max not in (None, 0) and min_width is not None:
        inward_pinch_ratio = float(min_width) / endpoint_max
        inward_pinch_depth_px = endpoint_max - float(min_width)

    middle_to_endpoint_max_ratio = None
    if endpoint_max not in (None, 0) and middle is not None:
        middle_to_endpoint_max_ratio = float(middle) / endpoint_max

    return {
        "measurement_type": "visible_2d_neck_inward_curve_proxy",
        "top_width_px": top,
        "middle_width_px": middle,
        "bottom_width_px": bottom,
        "min_width_px": min_width,
        "min_width_y_relative": width_summary.get("min_width_y_relative"),
        "max_width_px": width_summary.get("max_width_px"),
        "max_width_y_relative": width_summary.get("max_width_y_relative"),
        "endpoint_max_width_px": _round(endpoint_max, 2),
        "inward_pinch_ratio_min_width_over_endpoint_max_width": _round(inward_pinch_ratio),
        "inward_pinch_depth_px": _round(inward_pinch_depth_px, 2),
        "middle_width_over_endpoint_max_width_ratio": _round(middle_to_endpoint_max_ratio),
        "max_negative_half_width_deviation_px": curve_summary.get("max_negative_half_width_deviation_px"),
        "mean_negative_half_width_deviation_px": curve_summary.get("mean_negative_half_width_deviation_px"),
        "measurement_limitation": "Single-view silhouette proxy; does not prove true 3D neck curvature.",
    }


def _body_curve_measurements(width_summary: Dict[str, Any], curve_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "measurement_type": "visible_2d_body_outward_curve_proxy",
        "top_width_px": width_summary.get("top_width_px"),
        "middle_width_px": width_summary.get("middle_width_px"),
        "bottom_width_px": width_summary.get("bottom_width_px"),
        "max_width_px": width_summary.get("max_width_px"),
        "max_width_y_relative": width_summary.get("max_width_y_relative"),
        "max_to_top_width_ratio": width_summary.get("max_to_top_width_ratio"),
        "width_cv": width_summary.get("width_cv"),
        "max_half_width_px": curve_summary.get("max_half_width_px"),
        "max_half_width_y_relative": curve_summary.get("max_half_width_y_relative"),
        "max_positive_half_width_deviation_px": curve_summary.get("max_positive_half_width_deviation_px"),
        "mean_positive_half_width_deviation_px": curve_summary.get("mean_positive_half_width_deviation_px"),
        "measurement_limitation": "Single-view silhouette proxy; camera angle and partial occlusion can change apparent belly fullness.",
    }


def _find_internal_holes(source_mask: np.ndarray) -> List[np.ndarray]:
    """Return connected components of enclosed background holes inside source_mask."""
    if int(source_mask.sum()) == 0:
        return []

    h, w = source_mask.shape
    source_uint8 = source_mask.astype(np.uint8)

    # Flood fill exterior background starting from all image borders.
    background = (source_uint8 == 0).astype(np.uint8)
    exterior = np.zeros_like(background, dtype=np.uint8)
    flood = background.copy()

    # Seed every border background pixel.
    seeds: List[Tuple[int, int]] = []
    for x in range(w):
        if background[0, x]:
            seeds.append((x, 0))
        if background[h - 1, x]:
            seeds.append((x, h - 1))
    for y in range(h):
        if background[y, 0]:
            seeds.append((0, y))
        if background[y, w - 1]:
            seeds.append((w - 1, y))

    # cv2.floodFill needs a padded mask. Mark all exterior background regions.
    for seed in seeds:
        x, y = seed
        if flood[y, x] == 1 and exterior[y, x] == 0:
            temp = flood.copy()
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(temp, ff_mask, seedPoint=(x, y), newVal=2)
            region = temp == 2
            exterior[region] = 1
            flood[region] = 0

    holes = (background == 1) & (exterior == 0)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)

    components: List[np.ndarray] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        components.append(labels == label)
    return components


def _expanded_bbox_mask(mask_shape: Tuple[int, int], bbox_xyxy: List[int], pad_px: int) -> np.ndarray:
    h, w = mask_shape
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, int(x1) - pad_px)
    y1 = max(0, int(y1) - pad_px)
    x2 = min(w - 1, int(x2) + pad_px)
    y2 = min(h - 1, int(y2) + pad_px)
    out = np.zeros(mask_shape, dtype=bool)
    out[y1 : y2 + 1, x1 : x2 + 1] = True
    return out


def _handle_hole_measurements(
    *,
    handle_mask: Optional[np.ndarray],
    body_mask: Optional[np.ndarray],
    whole_mask: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Measure the negative space inside/near the handle.

    Preferred source is body ∪ handle because the handle opening is usually
    bounded by handle material plus the vessel wall. If that fails, whole_object
    is tried as a fallback.
    """
    if handle_mask is None or int(handle_mask.sum()) == 0:
        return {
            "measurement_type": "handle_opening_shape_proxy",
            "handle_hole_found": False,
            "reason": "No handle mask available.",
        }

    handle_box = _bbox_from_mask(handle_mask)
    if handle_box is None:
        return {
            "measurement_type": "handle_opening_shape_proxy",
            "handle_hole_found": False,
            "reason": "Handle mask has no bounding box.",
        }
    hx1, hy1, hx2, hy2 = handle_box
    pad_px = max(5, int(round(0.10 * max(hx2 - hx1 + 1, hy2 - hy1 + 1))))
    handle_region = _expanded_bbox_mask(handle_mask.shape, [hx1, hy1, hx2, hy2], pad_px=pad_px)

    source_masks: List[Tuple[str, np.ndarray]] = []
    if body_mask is not None and int(body_mask.sum()) > 0:
        combined = np.logical_or(handle_mask, body_mask)
        source_masks.append(("body_union_handle", combined))
    if whole_mask is not None and int(whole_mask.sum()) > 0:
        source_masks.append(("whole_object", whole_mask))

    # Small closing helps if body/handle attachment has tiny segmentation gaps.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * HANDLE_ATTACHMENT_DILATION_RADIUS_PX + 1, 2 * HANDLE_ATTACHMENT_DILATION_RADIUS_PX + 1),
    )

    best_component: Optional[np.ndarray] = None
    best_source = None
    best_score = -1.0
    best_closed_used = False

    for source_name, source in source_masks:
        for closed_used, candidate_source in [
            (False, source),
            (True, cv2.morphologyEx(source.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)),
        ]:
            holes = _find_internal_holes(candidate_source)
            for component in holes:
                area = int(component.sum())
                if area <= 0:
                    continue
                overlap_with_handle_region = int(np.logical_and(component, handle_region).sum())
                if overlap_with_handle_region == 0:
                    continue
                # Prefer larger holes that sit inside/near the handle bbox.
                overlap_fraction = overlap_with_handle_region / area
                score = area * (0.5 + overlap_fraction)
                if score > best_score:
                    best_component = component
                    best_source = source_name
                    best_score = float(score)
                    best_closed_used = bool(closed_used)

    if best_component is None:
        return {
            "measurement_type": "handle_opening_shape_proxy",
            "handle_hole_found": False,
            "source_masks_tried": [name for name, _ in source_masks],
            "handle_bbox_xyxy": [hx1, hy1, hx2, hy2],
            "morphological_closing_radius_px": HANDLE_ATTACHMENT_DILATION_RADIUS_PX,
            "reason": "No enclosed background component was found inside/near the handle region. This may be a real open shape, or it may reflect segmentation gaps where the handle attaches to the body.",
        }

    hole_features = _component_shape_features(best_component)
    return {
        "measurement_type": "handle_opening_shape_proxy",
        "handle_hole_found": True,
        "source_mask_used": best_source,
        "morphological_closing_used": best_closed_used,
        "morphological_closing_radius_px": HANDLE_ATTACHMENT_DILATION_RADIUS_PX,
        "handle_bbox_xyxy": [hx1, hy1, hx2, hy2],
        "hole_area_px": hole_features["area_px"],
        "hole_bbox_xyxy": hole_features["bbox_xyxy"],
        "hole_bbox_width_px": hole_features["bbox_width_px"],
        "hole_bbox_height_px": hole_features["bbox_height_px"],
        "hole_aspect_ratio_width_over_height": hole_features["aspect_ratio_width_over_height"],
        "hole_elongation_ratio": hole_features["elongation_ratio"],
        "hole_circularity": hole_features["circularity"],
        "hole_centroid_xy": hole_features["centroid_xy"],
        "measurement_limitation": "2D negative-space proxy. It depends on body/handle masks touching well enough to enclose the handle opening.",
    }


def _morphological_skeleton(mask_bool: np.ndarray) -> np.ndarray:
    """Pure OpenCV skeletonization fallback; avoids adding scikit-image."""
    img = mask_bool.astype(np.uint8)
    skel = np.zeros_like(img, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Limit iterations to avoid infinite loops on malformed masks.
    for _ in range(1000):
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel.astype(bool)


def _handle_thickness_measurements(
    *,
    handle_mask: Optional[np.ndarray],
    body_mask: Optional[np.ndarray],
) -> Dict[str, Any]:
    if handle_mask is None or int(handle_mask.sum()) == 0:
        return {
            "measurement_type": "visible_2d_handle_thickness_proxy",
            "thickness_mean_px": None,
            "thickness_std_px": None,
            "thickness_cv": None,
            "reason": "No handle mask available.",
        }

    box = _bbox_from_mask(handle_mask)
    if box is None:
        return {
            "measurement_type": "visible_2d_handle_thickness_proxy",
            "thickness_mean_px": None,
            "thickness_std_px": None,
            "thickness_cv": None,
            "reason": "Handle mask has no bounding box.",
        }

    x1, y1, x2, y2 = box
    h = y2 - y1 + 1
    top_cut = y1 + int(round(HANDLE_TOP_BOTTOM_EXCLUSION_FRACTION * h))
    bottom_cut = y2 - int(round(HANDLE_TOP_BOTTOM_EXCLUSION_FRACTION * h))

    core = handle_mask.copy()
    # Exclude top and bottom zones because handle attachments often flare there.
    yy = np.arange(handle_mask.shape[0])[:, None]
    top_bottom_exclusion = (yy < top_cut) | (yy > bottom_cut)
    core = np.logical_and(core, ~top_bottom_exclusion)

    attachment_exclusion_used = False
    attachment_zone_px = 0
    if body_mask is not None and int(body_mask.sum()) > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * HANDLE_ATTACHMENT_DILATION_RADIUS_PX + 1, 2 * HANDLE_ATTACHMENT_DILATION_RADIUS_PX + 1),
        )
        body_dilated = cv2.dilate(body_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        attachment_zone = np.logical_and(handle_mask, body_dilated)
        attachment_zone_px = int(attachment_zone.sum())
        core = np.logical_and(core, ~body_dilated)
        attachment_exclusion_used = attachment_zone_px > 0

    core_px = int(core.sum())
    fallback_used = False
    if core_px < MIN_CORE_PIXELS_FOR_THICKNESS:
        # If exclusions were too aggressive, fall back to top/bottom exclusion only.
        core = handle_mask.copy()
        core = np.logical_and(core, ~top_bottom_exclusion)
        core_px = int(core.sum())
        fallback_used = True

    if core_px < MIN_CORE_PIXELS_FOR_THICKNESS:
        # Last fallback: use the entire visible handle mask.
        core = handle_mask.copy()
        core_px = int(core.sum())
        fallback_used = True

    distance = cv2.distanceTransform(handle_mask.astype(np.uint8), cv2.DIST_L2, 5)
    skeleton = _morphological_skeleton(handle_mask)
    sample_mask = np.logical_and(skeleton, core)
    sampling_method = "distance_transform_values_on_skeleton_after_attachment_exclusion"

    if int(sample_mask.sum()) < MIN_CORE_PIXELS_FOR_THICKNESS:
        sample_mask = core
        sampling_method = "distance_transform_values_on_core_pixels_fallback"

    values = distance[sample_mask] * 2.0
    if len(values) == 0:
        return {
            "measurement_type": "visible_2d_handle_thickness_proxy",
            "attachment_exclusion_used": attachment_exclusion_used,
            "top_bottom_exclusion_fraction": HANDLE_TOP_BOTTOM_EXCLUSION_FRACTION,
            "thickness_mean_px": None,
            "thickness_std_px": None,
            "thickness_cv": None,
            "reason": "No distance-transform values after exclusions.",
        }

    mean = float(np.mean(values))
    std = float(np.std(values))
    core_box = _bbox_dict(core)

    return {
        "measurement_type": "visible_2d_handle_thickness_proxy",
        "attachment_exclusion_used": bool(attachment_exclusion_used),
        "attachment_exclusion_radius_px": HANDLE_ATTACHMENT_DILATION_RADIUS_PX,
        "attachment_zone_px": int(attachment_zone_px),
        "top_bottom_exclusion_fraction": HANDLE_TOP_BOTTOM_EXCLUSION_FRACTION,
        "fallback_used": bool(fallback_used),
        "handle_area_px": int(handle_mask.sum()),
        "thickness_core_area_px": int(core_px),
        "thickness_sample_count_px": int(sample_mask.sum()),
        "thickness_core_bbox": core_box,
        "sampling_method": sampling_method,
        "thickness_mean_px": _round(mean, 2),
        "thickness_std_px": _round(std, 2),
        "thickness_cv": _round(std / mean) if mean > 0 else None,
        "thickness_min_px": _round(np.min(values), 2),
        "thickness_p10_px": _round(np.percentile(values, 10), 2),
        "thickness_median_px": _round(np.median(values), 2),
        "thickness_p90_px": _round(np.percentile(values, 90), 2),
        "thickness_max_px": _round(np.max(values), 2),
        "measurement_limitation": "2D distance-transform proxy measured on visible handle mask; not true 3D ceramic cross-section thickness.",
    }


def _handle_shape_measurements(summary: Dict[str, Any]) -> Dict[str, Any]:
    shape = summary["shape"]
    aspect = shape.get("aspect_ratio_width_over_height")
    elongation = None
    if aspect not in (None, 0):
        aspect_f = float(aspect)
        elongation = max(aspect_f, 1.0 / aspect_f)

    return {
        "measurement_type": "visible_2d_handle_material_shape_proxy",
        "handle_material_aspect_ratio_width_over_height": aspect,
        "handle_material_elongation_ratio": _round(elongation),
        "handle_material_circularity": shape.get("circularity"),
        "handle_material_solidity": shape.get("solidity"),
        "measurement_limitation": "This describes the handle material mask. The preferred non-circularity measurement is the handle opening/negative space, computed globally when body/whole masks are available.",
    }


# -----------------------------------------------------------------------------
# Public API used by main.py
# -----------------------------------------------------------------------------


def summarize_mask(
    *,
    mask_bool: np.ndarray,
    image: Any,
    selected_detection: Dict[str, Any],
    text_prompt: str,
    target_name: str,
    required: bool = False,
) -> Dict[str, Any]:
    """Create a per-target, measurement-only summary.

    Cross-target measurements such as neck/whole proportion and handle-hole shape
    are added later inside build_summary(), after all masks are available.
    """
    mask_bool = mask_bool.astype(bool)
    _MASK_CACHE[target_name] = mask_bool.copy()

    selected_box = _tensor_to_list(selected_detection["box"])
    shape = _shape_features(mask_bool)
    width_profile = _vertical_width_profile(mask_bool)
    width_summary = _width_profile_summary(width_profile)
    boundary_profile = _boundary_curve_profile(mask_bool)
    boundary_summary = _boundary_curve_summary(boundary_profile)

    summary: Dict[str, Any] = {
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

    # Keep detailed curve/profile values only where they matter for the rubric.
    if target_name in {"body", "neck"}:
        summary["vertical_width_profile"] = width_profile
        summary["width_profile_summary"] = width_summary
        summary["boundary_curve_profile"] = boundary_profile
        summary["boundary_curve_summary"] = boundary_summary

    if target_name == "body":
        summary["target_measurements"] = _body_curve_measurements(width_summary, boundary_summary)
    elif target_name == "neck":
        summary["target_measurements"] = _neck_curve_measurements(width_summary, boundary_summary)
    elif target_name == "handle":
        summary["target_measurements"] = _handle_shape_measurements(summary)
    else:
        summary["target_measurements"] = {
            "measurement_type": "basic_visible_2d_mask_shape",
            "measurement_limitation": "Generic target; no pottery-specific measurement attached.",
        }

    return summary


def _target_map(target_summaries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for summary in target_summaries:
        name = summary.get("target", {}).get("name")
        if name:
            out[str(name)] = summary
    return out


def _global_proportion_measurements(targets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    whole_h = targets.get("whole_object", {}).get("shape", {}).get("bbox_height_px")
    neck_h = targets.get("neck", {}).get("shape", {}).get("bbox_height_px")

    return {
        "neck_height_px": neck_h,
        "whole_object_height_px": whole_h,
        "neck_to_whole_height_ratio": _round(float(neck_h) / float(whole_h)) if neck_h not in (None, 0) and whole_h not in (None, 0) else None,
        "measurement_type": "visible_2d_bbox_height_ratio",
        "measurement_limitation": "Uses visible bounding-box heights from segmentation masks, not true 3D object dimensions.",
    }


def _add_global_handle_measurements(targets: Dict[str, Dict[str, Any]]) -> None:
    handle_summary = targets.get("handle")
    if not handle_summary:
        return

    handle_mask = _MASK_CACHE.get("handle")
    body_mask = _MASK_CACHE.get("body")
    whole_mask = _MASK_CACHE.get("whole_object")

    existing = handle_summary.get("target_measurements", {})
    handle_summary["target_measurements"] = {
        **existing,
        "handle_opening_shape": _handle_hole_measurements(
            handle_mask=handle_mask,
            body_mask=body_mask,
            whole_mask=whole_mask,
        ),
        "handle_thickness_uniformity": _handle_thickness_measurements(
            handle_mask=handle_mask,
            body_mask=body_mask,
        ),
    }


def build_summary(
    *,
    image_path: str,
    image: Any,
    criteria_name: str,
    criteria_version: str,
    rubric: Optional[Dict[str, Any]],
    target_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build final concise measurement JSON.

    The rubric argument is accepted for compatibility with main.py, but this
    converter intentionally does not generate rubric judgments.
    """
    targets = _target_map(target_summaries)
    _add_global_handle_measurements(targets)

    ordered_targets = []
    for name in ["whole_object", "neck", "body", "handle"]:
        if name in targets:
            ordered_targets.append(targets[name])
    for name, summary in targets.items():
        if name not in {"whole_object", "neck", "body", "handle"}:
            ordered_targets.append(summary)

    return {
        "schema_version": "pottery_measurements_v2.0",
        "criteria": {
            "name": criteria_name,
            "version": criteria_version,
            "summary": "Measurement-only JSON for five pottery criteria: neck proportion, body outward curve, neck inward curve, handle opening shape, and handle thickness uniformity.",
        },
        "image": {
            "path": str(image_path),
            "width_px": int(image.size[0]),
            "height_px": int(image.size[1]),
        },
        "global_measurements": {
            "neck_proportion": _global_proportion_measurements(targets),
        },
        "targets": ordered_targets,
        "not_measured_in_this_version": [
            "color_or_glaze_evenness",
            "rim_quality",
            "decoration_quality",
            "true_3d_curvature",
            "true_ceramic_wall_thickness",
        ],
        "general_limitations": [
            "All measurements are derived from single-view 2D segmentation masks.",
            "The values describe the visible silhouette and mask geometry, not true 3D form.",
            "Handle opening and thickness measurements depend strongly on clean body/handle segmentation and attachment geometry.",
        ],
    }
