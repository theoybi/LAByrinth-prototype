from __future__ import annotations

# Keep this before importing Transformers-backed helper files.
from reporting_utils import configure_quiet_environment

configure_quiet_environment()

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from batch_utils import ProgressBar, find_images
from criteria import load_criteria_pack
from grounding_dino_utils import (
    get_best_detection,
    get_device,
    load_grounding_dino,
    run_grounding_dino,
    save_all_detections_image,
    save_detection_image,
)
from reporting_utils import RunReport, quiet_stdout_stderr, write_json
from sam2_utils import load_sam2, run_sam2_with_box, save_mask, save_mask_overlay
from visualization_utils import save_detection_overview, save_mask_overview


# Main switch: change this one variable to run a different criteria pack.
# Each pack lives in criteria/<CRITERIA_PACK>/ and owns config.py,
# mask_json_converter.py, and rubric.json.

CRITERIA_PACK = "pottery"

# Batch mode: every JPEG in INPUT_DIR is analyzed in one run. Point this at the
# folder of images you want to process (e.g. data/input/experiment3). Outputs
# are grouped under OUTPUT_ROOT / <input folder name>, one subfolder per image.
INPUT_DIR = Path("data/input") / "experiment4"
OUTPUT_ROOT = Path("outputs")

# Which files count as images. Extension matching is case-insensitive.
IMAGE_EXTENSIONS = (".jpg", ".jpeg")

GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
CONTAINER_AREA_RATIO_THRESHOLD = 3.0
NMS_IOU_THRESHOLD = 0.50

# Clear each per-image run folder before writing it, so old debug images from a
# previous run of the same image do not accumulate.
CLEAR_OUTPUT_DIR_BEFORE_RUN = True

# Clear the whole batch folder once at the start of the run.
CLEAR_BATCH_DIR_BEFORE_RUN = True

# Keeps terminal clean by suppressing model-loading progress bars and advisory warnings.
QUIET_MODEL_LOAD = True

# Print the detailed per-target log for every image. Off by default in batch
# mode so the overall progress bar stays readable; the full detail is always
# written to each image's run_report.json regardless of this setting.
VERBOSE_PER_IMAGE = False

# Keep raw masks for auditability, but inside images/raw_masks instead of the run root.
SAVE_RAW_MASKS = True

# Turn this on only when diagnosing GroundingDINO/SAM behavior.
SAVE_DEBUG_IMAGES = False


BATCH_NAME = INPUT_DIR.name
BATCH_OUTPUT_DIR = OUTPUT_ROOT / BATCH_NAME
BATCH_REPORT_OUTPUT_PATH = BATCH_OUTPUT_DIR / "batch_report.json"


def _selected_detection_for_report(detection: Dict[str, Any]) -> Dict[str, Any]:
    box = detection["box"].detach().cpu().tolist() if hasattr(detection["box"], "detach") else detection["box"]
    return {
        "index": int(detection["index"]),
        "label": str(detection["label"]),
        "score": round(float(detection["score"]), 4),
        "box_xyxy": [round(float(v), 2) for v in box],
        "selection_method": detection.get("selection_method"),
        "remaining_candidate_indices": detection.get("remaining_candidate_indices"),
    }


def process_target(
    *,
    target: Dict[str, Any],
    image: Image.Image,
    device: str,
    dino_processor: Any,
    dino_model: Any,
    sam2_processor: Any,
    sam2_model: Any,
    criteria_pack: Any,
    report: RunReport,
    raw_mask_output_dir: Path,
    debug_image_output_dir: Path,
) -> Dict[str, Any]:
    target_name = target["name"]
    text_prompt = target["prompt"]
    required = bool(target.get("required", False))

    target_report: Dict[str, Any] = {
        "name": target_name,
        "text_prompt": text_prompt,
        "required": required,
        "status": "running",
        "timings_seconds": {},
    }

    try:
        stage_start = time.perf_counter()
        dino_results = run_grounding_dino(
            image=image,
            text_prompt=text_prompt,
            processor=dino_processor,
            model=dino_model,
            device=device,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        target_report["timings_seconds"]["grounding_dino"] = round(time.perf_counter() - stage_start, 4)
        target_report["all_detections"] = criteria_pack.converter.summarize_detections_for_report(
            dino_results=dino_results,
            image=image,
        )
        target_report["num_detections"] = len(target_report["all_detections"])

        stage_start = time.perf_counter()
        selected_detection = get_best_detection(
            dino_results,
            area_ratio_threshold=CONTAINER_AREA_RATIO_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
        )
        target_report["timings_seconds"]["select_detection"] = round(time.perf_counter() - stage_start, 4)
        target_report["selected_detection"] = _selected_detection_for_report(selected_detection)

        if SAVE_DEBUG_IMAGES:
            color = criteria_pack.config.TARGET_COLORS.get(target_name, (255, 0, 0))
            save_all_detections_image(
                image=image,
                results=dino_results,
                output_path=debug_image_output_dir / f"{target_name}_all_detections.jpg",
                color=(180, 180, 180),
            )
            save_detection_image(
                image=image,
                detection=selected_detection,
                output_path=debug_image_output_dir / f"{target_name}_selected_detection.jpg",
                color=color,
            )

        stage_start = time.perf_counter()
        mask_bool = run_sam2_with_box(
            image=image,
            box=selected_detection["box"],
            processor=sam2_processor,
            model=sam2_model,
            device=device,
        )
        target_report["timings_seconds"]["sam2"] = round(time.perf_counter() - stage_start, 4)

        if SAVE_RAW_MASKS:
            mask_path = raw_mask_output_dir / f"{target_name}.png"
            save_mask(mask_bool=mask_bool, output_path=mask_path)
            target_report["raw_mask_path"] = str(mask_path)

        if SAVE_DEBUG_IMAGES:
            color = criteria_pack.config.TARGET_COLORS.get(target_name, (255, 0, 0))
            overlay_path = debug_image_output_dir / f"{target_name}_sam2_overlay.jpg"
            save_mask_overlay(
                image=image,
                mask_bool=mask_bool,
                output_path=overlay_path,
                color=color,
            )
            target_report["debug_overlay_path"] = str(overlay_path)

        stage_start = time.perf_counter()
        target_summary = criteria_pack.converter.summarize_mask(
            mask_bool=mask_bool,
            image=image,
            selected_detection=selected_detection,
            text_prompt=text_prompt,
            target_name=target_name,
            required=required,
        )
        target_report["timings_seconds"]["json_conversion"] = round(time.perf_counter() - stage_start, 4)

        target_report["mask_area_px"] = target_summary["shape"]["area_px"]
        target_report["mask_area_fraction_of_image"] = target_summary["shape"]["area_fraction_of_image"]
        target_report["status"] = "completed"
        report.add_target_report(target_report)

        return {
            "target_summary": target_summary,
            "dino_results": dino_results,
            "selected_detection": selected_detection,
            "mask_bool": mask_bool,
        }

    except Exception as error:
        target_report["status"] = "failed"
        target_report["error"] = str(error)
        report.add_target_report(target_report)
        report.add_error(f"Target failed: {target_name}", target=target_name, error=str(error))

        failure_summary = {
            "target": {
                "name": target_name,
                "text_prompt": text_prompt,
                "required": required,
            },
            "present": False,
            "error": str(error),
        }
        return {
            "target_summary": failure_summary,
            "dino_results": None,
            "selected_detection": None,
            "mask_bool": None,
        }


def process_image(
    *,
    image_path: Path,
    batch_output_dir: Path,
    criteria_pack: Any,
    device: str,
    dino_processor: Any,
    dino_model: Any,
    sam2_processor: Any,
    sam2_model: Any,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full detection/segmentation pipeline on a single image.

    Each image gets its own run folder (named after the image) with the same
    layout the single-image pipeline used to produce: images/, raw masks,
    a criteria summary JSON, and a run_report.json.
    """
    run_name = image_path.stem
    output_dir = batch_output_dir / run_name
    image_output_dir = output_dir / "images"
    raw_mask_output_dir = image_output_dir / "raw_masks"
    debug_image_output_dir = image_output_dir / "debug"
    summary_output_path = output_dir / f"{CRITERIA_PACK}_summary.json"
    run_report_output_path = output_dir / "run_report.json"

    if CLEAR_OUTPUT_DIR_BEFORE_RUN and output_dir.exists():
        shutil.rmtree(output_dir)
    image_output_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_RAW_MASKS:
        raw_mask_output_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_DEBUG_IMAGES:
        debug_image_output_dir.mkdir(parents=True, exist_ok=True)

    image_start = time.perf_counter()

    report = RunReport(
        run_name=run_name,
        criteria_pack=CRITERIA_PACK,
        output_dir=output_dir,
        image_path=image_path,
    )
    report.set_models(
        grounding_dino=GROUNDING_DINO_MODEL_ID,
        sam2=SAM2_MODEL_ID,
    )
    report.set_thresholds(
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        container_area_ratio_threshold=CONTAINER_AREA_RATIO_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
    )
    report.set_config(
        save_raw_masks=SAVE_RAW_MASKS,
        save_debug_images=SAVE_DEBUG_IMAGES,
        quiet_model_load=QUIET_MODEL_LOAD,
        clear_output_dir_before_run=CLEAR_OUTPUT_DIR_BEFORE_RUN,
        device=device,
        batch_name=BATCH_NAME,
        models_preloaded=True,
    )
    report.add_note(
        "Models are loaded once for the whole batch, so per-image model-load timings are not recorded here; "
        "see batch_report.json for the shared model-load time."
    )
    report.add_note(
        "The SAM2 sam2_video-to-sam2 advisory warning is suppressed in terminal because it is commonly emitted for compatible HF SAM2 checkpoints; verify model compatibility if segmentation fails.",
        level="warning",
    )

    with report.time_block("load_image"):
        image = Image.open(image_path).convert("RGB")
        original_path = image_output_dir / "original.jpg"
        image.save(original_path)
        report.set_output("original_image", original_path)

    if verbose:
        print(f"[LAByrinth] {run_name}: processing {len(criteria_pack.targets)} targets")

    target_summaries: List[Dict[str, Any]] = []
    dino_results_by_target: Dict[str, Any] = {}
    selected_detections_by_target: Dict[str, Dict[str, Any]] = {}
    masks_by_target: Dict[str, Any] = {}

    with report.time_block("process_all_targets"):
        for target in criteria_pack.targets:
            target_name = target["name"]
            result = process_target(
                target=target,
                image=image,
                device=device,
                dino_processor=dino_processor,
                dino_model=dino_model,
                sam2_processor=sam2_processor,
                sam2_model=sam2_model,
                criteria_pack=criteria_pack,
                report=report,
                raw_mask_output_dir=raw_mask_output_dir,
                debug_image_output_dir=debug_image_output_dir,
            )

            target_summary = result["target_summary"]
            target_summaries.append(target_summary)
            dino_results_by_target[target_name] = result["dino_results"]
            if result["selected_detection"] is not None:
                selected_detections_by_target[target_name] = result["selected_detection"]
            if result["mask_bool"] is not None:
                masks_by_target[target_name] = result["mask_bool"]

            if verbose:
                if target_summary.get("present"):
                    score = target_summary.get("selected_detection", {}).get("score")
                    area = target_summary.get("shape", {}).get("area_fraction_of_image")
                    print(f"  OK   {target_name:<13} score={score} area_frac={area}")
                else:
                    print(f"  WARN {target_name:<13} failed_or_empty")

    with report.time_block("save_visualizations"):
        mask_overview_path = image_output_dir / "mask_overview.jpg"
        detection_overview_path = image_output_dir / "detections_overview.jpg"
        if masks_by_target:
            save_mask_overview(
                image=image,
                masks_by_target=masks_by_target,
                target_colors=criteria_pack.config.TARGET_COLORS,
                output_path=mask_overview_path,
            )
            report.set_output("mask_overview", mask_overview_path)
        if dino_results_by_target:
            save_detection_overview(
                image=image,
                dino_results_by_target=dino_results_by_target,
                selected_detections_by_target=selected_detections_by_target,
                target_colors=criteria_pack.config.TARGET_COLORS,
                output_path=detection_overview_path,
            )
            report.set_output("detections_overview", detection_overview_path)

    with report.time_block("build_and_save_summary_json"):
        combined_summary = criteria_pack.converter.build_summary(
            image_path=str(image_path),
            image=image,
            criteria_name=criteria_pack.name,
            criteria_version=criteria_pack.version,
            rubric=criteria_pack.rubric,
            target_summaries=target_summaries,
        )
        write_json(combined_summary, summary_output_path)
        report.set_output("concise_summary_json", summary_output_path)

    report.set_output("run_report_json", run_report_output_path)
    status = "completed_with_errors" if report.data["errors"] else "completed"
    report.finish(status=status)
    report.save(run_report_output_path)

    num_present = sum(1 for summary in target_summaries if summary.get("present"))

    return {
        "run_name": run_name,
        "image_path": str(image_path),
        "status": status,
        "output_dir": str(output_dir),
        "summary_path": str(summary_output_path),
        "run_report_path": str(run_report_output_path),
        "num_targets": len(criteria_pack.targets),
        "num_present": num_present,
        "num_errors": len(report.data["errors"]),
        "seconds": round(time.perf_counter() - image_start, 2),
    }


def main() -> None:
    criteria_pack = load_criteria_pack(CRITERIA_PACK)

    image_paths = find_images(INPUT_DIR, IMAGE_EXTENSIONS)
    if not image_paths:
        print(
            f"[LAByrinth] no images found in {INPUT_DIR} "
            f"(looking for: {', '.join(IMAGE_EXTENSIONS)})"
        )
        return

    if CLEAR_BATCH_DIR_BEFORE_RUN and BATCH_OUTPUT_DIR.exists():
        shutil.rmtree(BATCH_OUTPUT_DIR)
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(
        f"[LAByrinth] batch={BATCH_NAME} criteria={criteria_pack.name} "
        f"device={device} images={len(image_paths)}"
    )

    # Load the models ONCE and reuse them for every image. Reloading per image
    # would dominate the runtime, so this is the key to fast batch processing.
    print("[LAByrinth] loading models (once for the whole batch)")
    model_load_start = time.perf_counter()
    with quiet_stdout_stderr(QUIET_MODEL_LOAD):
        dino_processor, dino_model = load_grounding_dino(
            model_id=GROUNDING_DINO_MODEL_ID,
            device=device,
        )
        sam2_processor, sam2_model = load_sam2(
            model_id=SAM2_MODEL_ID,
            device=device,
        )
    model_load_seconds = round(time.perf_counter() - model_load_start, 2)
    print(f"[LAByrinth] models loaded in {model_load_seconds}s")

    batch_start = time.perf_counter()
    per_image_results: List[Dict[str, Any]] = []

    progress = ProgressBar(total=len(image_paths), label=BATCH_NAME)
    for image_path in image_paths:
        progress.set_description(image_path.name)
        try:
            result = process_image(
                image_path=image_path,
                batch_output_dir=BATCH_OUTPUT_DIR,
                criteria_pack=criteria_pack,
                device=device,
                dino_processor=dino_processor,
                dino_model=dino_model,
                sam2_processor=sam2_processor,
                sam2_model=sam2_model,
                verbose=VERBOSE_PER_IMAGE,
            )
            per_image_results.append(result)
            tag = "OK  " if result["num_errors"] == 0 else "ERR "
            progress.write(
                f"  {tag} {image_path.name:<22} "
                f"present={result['num_present']}/{result['num_targets']} "
                f"errors={result['num_errors']} ({result['seconds']}s)"
            )
        except Exception as error:  # keep going even if one image blows up
            per_image_results.append(
                {
                    "run_name": image_path.stem,
                    "image_path": str(image_path),
                    "status": "failed",
                    "error": str(error),
                }
            )
            progress.write(f"  FAIL {image_path.name}: {error}")
        finally:
            progress.update(1)
    progress.close()

    num_failed = sum(1 for r in per_image_results if r.get("status") == "failed")
    num_with_errors = sum(1 for r in per_image_results if r.get("status") == "completed_with_errors")
    num_completed = sum(1 for r in per_image_results if r.get("status") == "completed")

    batch_report = {
        "batch_name": BATCH_NAME,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(BATCH_OUTPUT_DIR),
        "criteria_pack": criteria_pack.name,
        "criteria_version": criteria_pack.version,
        "device": device,
        "models": {
            "grounding_dino": GROUNDING_DINO_MODEL_ID,
            "sam2": SAM2_MODEL_ID,
        },
        "image_extensions": list(IMAGE_EXTENSIONS),
        "num_images": len(image_paths),
        "num_completed": num_completed,
        "num_completed_with_errors": num_with_errors,
        "num_failed": num_failed,
        "model_load_seconds": model_load_seconds,
        "total_seconds": round(time.perf_counter() - batch_start, 2),
        "images": per_image_results,
    }
    write_json(batch_report, BATCH_REPORT_OUTPUT_PATH)

    print(
        f"[LAByrinth] done: {num_completed} ok, "
        f"{num_with_errors} with errors, {num_failed} failed "
        f"in {batch_report['total_seconds']}s"
    )
    print(f"[LAByrinth] outputs:      {BATCH_OUTPUT_DIR}")
    print(f"[LAByrinth] batch report: {BATCH_REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
