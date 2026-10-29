"""Paper settings and fixed, validated output destinations for a portable editor."""
import json
import os
from pathlib import Path
import re

ENGINE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("REBUTTAL_WORKSPACE_ROOT", ENGINE.parents[1])).resolve()
CONFIG_PATH = ROOT / "workspace.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
if CONFIG.get("schema_version") != 1:
    raise ValueError("Unsupported workspace configuration")
SAFE = re.compile(r"[A-Za-z0-9_-]+\Z")
if not SAFE.fullmatch(CONFIG.get("workspace_id", "")):
    raise ValueError("Invalid workspace identity")
PAPER = CONFIG["paper"]
for field in ("title", "short_title", "venue", "submission_id"):
    if not isinstance(PAPER.get(field), str):
        raise ValueError("Invalid paper " + field)
RULES = CONFIG["rules"]
if RULES.get("response_limit_unit") != "characters":
    raise ValueError("This version supports character limits; word-limit rules need an explicit extension")
for key in ("response_character_limit", "confidential_character_limit"):
    limit = RULES.get(key)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Invalid " + key)
if type(RULES.get("allow_external_links")) is not bool:
    raise ValueError("Set allow_external_links explicitly")
REVIEW_SPECS = CONFIG["reviews"]
if not isinstance(REVIEW_SPECS, list) or not REVIEW_SPECS:
    raise ValueError("Configure at least one review")
stems, targets = set(), set()
for item in REVIEW_SPECS:
    stem, target = item.get("stem", ""), item.get("target", "")
    if not SAFE.fullmatch(stem) or not target.endswith(".md") or not SAFE.fullmatch(target[:-3]):
        raise ValueError("Review filenames must be simple local names")
    if stem in stems or target in targets:
        raise ValueError("Duplicate review or response filename")
    stems.add(stem)
    targets.add(target)
CONFIDENTIAL = CONFIG["confidential"]
if type(CONFIDENTIAL.get("enabled")) is not bool:
    raise ValueError("Set confidential.enabled explicitly")
if not SAFE.fullmatch(CONFIDENTIAL.get("stem", "")) or CONFIDENTIAL["stem"] in stems:
    raise ValueError("Invalid confidential-comment ID")
target = CONFIDENTIAL.get("target", "")
if not target.endswith(".md") or not SAFE.fullmatch(target[:-3]) or target in targets:
    raise ValueError("Invalid confidential-comment filename")
EXPECTED_TARGETS = targets
RESPONSE_LIMIT = RULES["response_character_limit"]
CONFIDENTIAL_LIMIT = RULES["confidential_character_limit"]
FORBID_LINKS = not RULES["allow_external_links"]
ASSETS = ENGINE / "review_browser"
ASSET_URL = Path(os.path.relpath(ASSETS, ROOT / "reviews")).as_posix()
BRAND = f"{PAPER['short_title']} / {PAPER['venue']} / #{PAPER['submission_id']}"
LIMIT_LABEL = f"{RESPONSE_LIMIT:,}" if RESPONSE_LIMIT is not None else "unspecified"


def browser_context(confidential=False):
    return {
        "workspaceId": CONFIG["workspace_id"],
        "workspaceFolder": ROOT.name,
        "workspaceLabel": PAPER["short_title"],
        "expectedTargets": [CONFIDENTIAL["target"]] if confidential else sorted(EXPECTED_TARGETS),
        "limit": CONFIDENTIAL_LIMIT if confidential else RESPONSE_LIMIT,
        "forbidExternalLinks": FORBID_LINKS,
    }
