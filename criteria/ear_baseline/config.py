from __future__ import annotations

from pathlib import Path

CRITERIA_NAME = "ear_baseline"
CRITERIA_VERSION = "0.1.0"

# No rubric yet. This pack exists to answer one question first: how well do
# off-the-shelf GroundingDINO + SAM2 detect and segment temporal-bone
# dissection landmarks with zero fine-tuning, before we invest in a full
# rubric + measurement schema the way we did for pottery.
RUBRIC_PATH = None

# Target list mirrors a standard cortical mastoidectomy dissection sequence
# (MacEwen's triangle -> antrum -> lateral semicircular canal -> facial ridge
# -> sigmoid sinus / tegmen as the surgical boundaries), per standard
# otologic teaching references (see run notes). Prompts are written as
# descriptive noun phrases, the style GroundingDINO responds to best, and
# deliberately span "should be easy" (large, high-contrast landmarks) to
# "likely to fail" (small, subtle landmarks) so the baseline result is a
# gradient, not just pass/fail.
#
# NOTE: these prompts describe how each landmark looks in a real drilled
# temporal bone specimen (bony textures/colors). If the test images end up
# being anatomical diagrams/illustrations instead of photos of a real or
# 3D-printed specimen, expect detection to fail even on the "easy" targets,
# since GroundingDINO was trained on natural photos, not medical illustration
# style. That mismatch would itself be a useful, reportable finding.
TARGETS = [
    {
        "name": "mastoid_cavity",
        "prompt": "drilled bony cavity of a mastoidectomy dissection specimen.",
        "required": True,
    },
    {
        "name": "lateral_semicircular_canal",
        "prompt": "dome-shaped ivory-white bony prominence at the base of the mastoid antrum (lateral semicircular canal).",
        "required": False,
    },
    {
        "name": "tegmen",
        "prompt": "smooth rounded bony roof of the mastoid cavity (tegmen mastoideum).",
        "required": False,
    },
    {
        "name": "sigmoid_sinus",
        "prompt": "smooth curved bluish-grey bony ridge along the posterior wall of the mastoid cavity (sigmoid sinus plate).",
        "required": False,
    },
    {
        "name": "facial_ridge",
        "prompt": "bony ridge below the lateral semicircular canal marking the facial nerve (facial ridge).",
        "required": False,
    },
    {
        "name": "digastric_ridge",
        "prompt": "bony ridge at the tip of the mastoid marking the digastric muscle groove (digastric ridge).",
        "required": False,
    },
]

TARGET_COLORS = {
    "mastoid_cavity": (255, 215, 0),
    "lateral_semicircular_canal": (0, 170, 255),
    "tegmen": (180, 80, 255),
    "sigmoid_sinus": (0, 220, 120),
    "facial_ridge": (255, 90, 90),
    "digastric_ridge": (255, 165, 0),
}
