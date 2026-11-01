"""Check the configured response bodies and optional confidential comment."""

import argparse
from pathlib import Path
import re


from workspace_context import ROOT, EXPECTED_TARGETS, RESPONSE_LIMIT, CONFIDENTIAL, CONFIDENTIAL_LIMIT, CONFIDENTIAL_TITLE, FORBID_LINKS
EXPECTED = EXPECTED_TARGETS
LIMIT = RESPONSE_LIMIT
LINK = re.compile(
    r"\b(?:https?://|ftp://|www\.|mailto:)"
    r"|\[[^\]\n]+\]\([^\n)]+\)"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|ai|cc|dev|ru|uk)\b",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(r"\b(?:TODO|TBD|FIXME|PLACEHOLDER)\b", re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="allow TODOs while checking the initial templates",
    )
    args = parser.parse_args()
    paths = sorted((ROOT / "rebuttals").glob("*.md"))
    actual = {p.name for p in paths}
    failed = False
    expected_files = EXPECTED | ({CONFIDENTIAL["target"]} if CONFIDENTIAL["enabled"] else set())
    if actual != expected_files:
        print("Response-file mismatch:")
        print("  Missing:", ", ".join(sorted(expected_files - actual)) or "none")
        print("  Unexpected:", ", ".join(sorted(actual - expected_files)) or "none")
        failed = True

    for path in paths:
        body = path.read_text(encoding="utf-8")
        markdown = body
        markers = len(PLACEHOLDER.findall(body))
        issues = []
        if not body.strip():
            issues.append("empty response")
        confidential = path.name == CONFIDENTIAL["target"]
        if confidential and CONFIDENTIAL_TITLE:
            prefix = f"# {CONFIDENTIAL_TITLE}\n\n"
            if body.startswith(prefix):
                body = body[len(prefix):]
            else:
                issues.append("missing or outdated confidential title")
        count = len(body.encode("utf-16-le")) // 2
        limit = CONFIDENTIAL_LIMIT if confidential else LIMIT
        if limit is not None and count > limit:
            issues.append("over character limit")
        if FORBID_LINKS and LINK.search(markdown):
            issues.append("link pattern found")
        if "\ufeff" in body:
            issues.append("byte-order mark found")
        if markers and not args.allow_placeholders:
            issues.append("unfinished placeholders")
        label = "FAIL" if issues else ("DRAFT" if markers else "OK")
        detail = "; ".join(issues) or (f"{markers} TODO markers" if markers else "no link patterns or placeholders")
        length = f"{count}/{limit} characters" if limit is not None else f"{count} characters (limit unspecified)"
        print(f"{label:5} {path.name}: {length}; {detail}")
        failed |= bool(issues)

    if not failed and args.allow_placeholders:
        print("Template checks passed. Replace TODOs and rerun without --allow-placeholders before submission.")
    elif not failed:
        print("Checks passed. Read the final wording and confirm the count in OpenReview.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
