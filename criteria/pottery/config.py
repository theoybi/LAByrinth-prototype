from __future__ import annotations

from pathlib import Path

CRITERIA_NAME = "pottery"
CRITERIA_VERSION = "0.4.0"

# Optional. The converter does not need this.
# This is just the LLM prompt/rubric you can send with the measurement JSON.
RUBRIC_PATH = Path(__file__).with_name("rubric.txt")

TARGETS = [
    {
        "name": "whole_object",
        "prompt": "entire ceramic pottery vessel or vase.",
        "required": True,
    },
    {
        "name": "body",
        "prompt": "rounded body or belly of a ceramic pottery vase.",
        "required": True,
    },
    {
        "name": "neck",
        "prompt": "neck of a ceramic pottery vase.",
        "required": True,
    },
    {
        "name": "handle",
        "prompt": "handle of a ceramic pottery vase.",
        "required": True,
    },
]

TARGET_COLORS = {
    "whole_object": (255, 215, 0),
    "body": (0, 170, 255),
    "neck": (180, 80, 255),
    "handle": (0, 220, 120),
}