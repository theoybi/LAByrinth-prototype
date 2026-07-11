from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CriteriaPack:
    name: str
    version: str
    targets: List[Dict[str, Any]]
    rubric: Any
    rubric_path: Optional[str]
    config: ModuleType
    converter: ModuleType


def _load_optional_rubric(config: ModuleType) -> Tuple[Any, Optional[str]]:
    """
    Load rubric if the criteria pack points to one.

    This is optional. Missing rubric files should not stop the vision pipeline.
    """
    raw_path = getattr(config, "RUBRIC_PATH", None)

    if raw_path is None:
        return None, None

    path = Path(raw_path)

    if not path.exists():
        return None, str(path)

    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json":
        return json.loads(text), str(path)

    # For rubric.txt or rubric.md, just return the text.
    return text, str(path)


def load_criteria_pack(pack_name: str) -> CriteriaPack:
    config = importlib.import_module(f"criteria.{pack_name}.config")
    converter = importlib.import_module(f"criteria.{pack_name}.mask_json_converter")

    rubric, rubric_path = _load_optional_rubric(config)

    return CriteriaPack(
        name=getattr(config, "CRITERIA_NAME", pack_name),
        version=getattr(config, "CRITERIA_VERSION", "unknown"),
        targets=list(config.TARGETS),
        rubric=rubric,
        rubric_path=rubric_path,
        config=config,
        converter=converter,
    )