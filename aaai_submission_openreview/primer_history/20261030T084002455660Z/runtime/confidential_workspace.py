"""Separate AC/SPC draft with its own configurable character limit."""
import json
from pathlib import Path

from assemble_rebuttals import ROOT, validate_review

from workspace_context import CONFIDENTIAL

STEM = CONFIDENTIAL["stem"]
FILENAME = "confidential_comment.json"
TARGET = CONFIDENTIAL["target"]


def load_confidential(path=None):
    workspace = json.loads(Path(path or ROOT / FILENAME).read_text(encoding="utf-8"))
    if workspace.get("version") != 1 or set(workspace.get("reviews", {})) != {STEM}:
        raise ValueError("Invalid confidential-comment workspace")
    review = workspace["reviews"][STEM]
    if review.get("target") != TARGET:
        raise ValueError("Invalid confidential-comment target")
    validate_review(review)
    return workspace
