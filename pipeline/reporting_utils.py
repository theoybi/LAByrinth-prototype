from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def configure_quiet_environment() -> None:
    """Reduce non-critical library chatter before model imports/loading."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # The SAM 2 HF checkpoints may print this compatibility warning even when the
    # checkpoint works for this MVP. Keep it in run_report notes instead of terminal.
    warnings.filterwarnings(
        "ignore",
        message=r".*sam2_video.*sam2.*",
    )

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


@contextlib.contextmanager
def quiet_stdout_stderr(enabled: bool = True) -> Iterator[None]:
    """Suppress stdout/stderr inside a block, mainly for model-loading progress bars."""
    if not enabled:
        yield
        return

    devnull_path = os.devnull
    with open(devnull_path, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _json_safe(value: Any) -> Any:
    """Convert common non-JSON objects into JSON-safe values."""
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class RunReport:
    """Collect verbose run metadata outside the concise model-facing summary JSON."""

    def __init__(
        self,
        run_name: str,
        criteria_pack: str,
        output_dir: Path,
        image_path: Path,
    ) -> None:
        self.started_at_unix = time.time()
        self._timers: Dict[str, float] = {}
        self.data: Dict[str, Any] = {
            "run_name": run_name,
            "criteria_pack": criteria_pack,
            "image_path": str(image_path),
            "output_dir": str(output_dir),
            "started_at_unix": self.started_at_unix,
            "finished_at_unix": None,
            "total_time_seconds": None,
            "status": "running",
            "models": {},
            "thresholds": {},
            "runtime": collect_runtime_info(),
            "timings_seconds": {},
            "targets": [],
            "outputs": {},
            "notes": [],
            "errors": [],
        }

    def set_models(self, **models: str) -> None:
        self.data["models"].update(models)

    def set_thresholds(self, **thresholds: float) -> None:
        self.data["thresholds"].update(thresholds)

    def set_config(self, **config: Any) -> None:
        self.data.setdefault("config", {}).update(config)

    def add_note(self, message: str, level: str = "info", **extra: Any) -> None:
        item = {"level": level, "message": message}
        item.update(extra)
        self.data["notes"].append(item)

    def add_error(self, message: str, **extra: Any) -> None:
        item = {"message": message}
        item.update(extra)
        self.data["errors"].append(item)
        self.add_note(message, level="error", **extra)

    def start(self, stage_name: str) -> None:
        self._timers[stage_name] = time.perf_counter()

    def stop(self, stage_name: str) -> float:
        start = self._timers.pop(stage_name, None)
        if start is None:
            return 0.0
        elapsed = time.perf_counter() - start
        self.data["timings_seconds"][stage_name] = round(elapsed, 4)
        return elapsed

    @contextlib.contextmanager
    def time_block(self, stage_name: str) -> Iterator[None]:
        self.start(stage_name)
        try:
            yield
        finally:
            self.stop(stage_name)

    def add_target_report(self, target_report: Dict[str, Any]) -> None:
        self.data["targets"].append(_json_safe(target_report))

    def set_output(self, name: str, path: Path) -> None:
        self.data["outputs"][name] = str(path)

    def finish(self, status: str = "completed") -> Dict[str, Any]:
        finished = time.time()
        self.data["finished_at_unix"] = finished
        self.data["total_time_seconds"] = round(finished - self.started_at_unix, 4)
        self.data["status"] = status
        return self.data

    def save(self, output_path: Path) -> None:
        write_json(self.data, output_path)


def collect_runtime_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    for package_name in ["torch", "transformers", "numpy", "opencv-python", "Pillow"]:
        try:
            info[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            pass

    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as error:
        info["torch_cuda_probe_error"] = str(error)

    return info
