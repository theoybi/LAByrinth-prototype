from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor


def load_sam2(model_id: str, device: str):
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def _post_process_masks(processor: Any, pred_masks, inputs: Any):
    """Handle small API differences across Transformers/SAM2 versions."""
    original_sizes = inputs["original_sizes"]
    reshaped_sizes = inputs.get("reshaped_input_sizes")

    try:
        if reshaped_sizes is not None:
            return processor.post_process_masks(pred_masks, original_sizes, reshaped_sizes)
    except TypeError:
        pass

    return processor.post_process_masks(pred_masks, original_sizes)


def run_sam2_with_box(image, box, processor, model, device: str) -> np.ndarray:
    box_list = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
    input_boxes = [[box_list]]

    inputs = processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    masks = _post_process_masks(processor, outputs.pred_masks.cpu(), inputs)[0]
    mask = np.asarray(masks.squeeze())

    # Squeeze can still leave multiple masks in unusual version/model combos.
    # Choose the first mask deterministically for this MVP.
    while mask.ndim > 2:
        mask = mask[0]

    mask_bool = mask > 0

    # Ensure output exactly matches original image H x W.
    expected_h, expected_w = image.size[1], image.size[0]
    if mask_bool.shape != (expected_h, expected_w):
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (expected_w, expected_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return mask_bool


def save_mask(mask_bool: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    ok = cv2.imwrite(str(output_path), mask_uint8)
    if not ok:
        raise OSError(f"Failed to write mask: {output_path}")


def save_mask_overlay(image, mask_bool: np.ndarray, output_path: Path, color=(255, 0, 0), alpha: float = 0.35) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_np = np.array(image.convert("RGB")).copy()
    if mask_bool.shape != image_np.shape[:2]:
        raise ValueError(f"Mask shape {mask_bool.shape} does not match image shape {image_np.shape[:2]}")

    overlay = image_np.copy()
    overlay[mask_bool] = color
    blended = cv2.addWeighted(image_np, 1.0 - alpha, overlay, alpha, 0)
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(output_path), blended_bgr)
    if not ok:
        raise OSError(f"Failed to write overlay: {output_path}")
