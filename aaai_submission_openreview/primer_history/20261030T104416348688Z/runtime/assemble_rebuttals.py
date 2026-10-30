"""Assemble selected response fragments into the configured Markdown files.

Run with --check to compare without writing. Text is never truncated or paraphrased.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

from check_rebuttals import EXPECTED, LIMIT, LINK, PLACEHOLDER

from workspace_context import ROOT, REVIEW_SPECS, CONFIDENTIAL, CONFIDENTIAL_LIMIT, FORBID_LINKS
WORKSPACE = ROOT / "response_fragments.json"


def character_count(text):
    """Count UTF-16 code units, matching a browser text field, including newlines."""
    return len(text.encode("utf-16-le")) // 2


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".rebuttal-", delete=False) as stream:
            name = stream.name
            stream.write(text)
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def load_workspace(path=WORKSPACE):
    workspace = json.loads(Path(path).read_text(encoding="utf-8"))
    if workspace.get("version") != 1 or not isinstance(workspace.get("reviews"), dict):
        raise ValueError("Unsupported response workspace")
    targets = {review.get("target") for review in workspace["reviews"].values()}
    if targets != EXPECTED or set(workspace["reviews"]) != {item["stem"] for item in REVIEW_SPECS}:
        raise ValueError("The response workspace must match workspace.json review IDs and filenames")
    for review in workspace["reviews"].values():
        validate_review(review)
    return workspace


def validate_review(review):
    for field in ("intro", "closing"):
        if not isinstance(review.get(field), str) or len(review[field]) > 10000:
            raise ValueError("Invalid " + field)
    if type(review.get("revision")) is not int or review["revision"] < 0:
        raise ValueError("Invalid review revision")
    variants = review.get("frame_variants", {})
    if not isinstance(variants, dict) or set(variants) - {"intro", "closing"}:
        raise ValueError("Invalid opening/closing variants")
    for options in variants.values():
        if not isinstance(options, list) or not 1 <= len(options) <= 12:
            raise ValueError("Invalid wording options")
        ids = set()
        for option in options:
            if not isinstance(option, dict) or not isinstance(option.get("id"), str) or option["id"] in ids:
                raise ValueError("Invalid wording option ID")
            ids.add(option["id"])
            for field, limit in (("label", 200), ("text", 10000)):
                if not isinstance(option.get(field), str) or len(option[field]) > limit:
                    raise ValueError("Invalid wording option " + field)
    sections = review.get("sections")
    fragments = review.get("fragments")
    if not isinstance(sections, list) or not isinstance(fragments, list) or not fragments:
        raise ValueError("Missing response sections or fragments")
    section_ids = [section["id"] for section in sections]
    point_ids = [fragment["id"] for fragment in fragments]
    if len(set(section_ids)) != len(section_ids) or len(set(point_ids)) != len(point_ids):
        raise ValueError("Duplicate section or concern ID")
    for section in sections:
        if not isinstance(section.get("lead"), str) or len(section["lead"]) > 1000:
            raise ValueError("Invalid topic phrase")
    for fragment in fragments:
        if fragment.get("section") not in section_ids:
            raise ValueError("Unknown response section")
        for field in ("text", "notes", "joiner"):
            if not isinstance(fragment.get(field), str) or len(fragment[field]) > 50000:
                raise ValueError("Invalid fragment " + field)
        for field in ("include", "needs_check"):
            if type(fragment.get(field)) is not bool:
                raise ValueError("Invalid fragment " + field)
    topics = review.get("reply_topics", [])
    if not isinstance(topics, list) or len(topics) > 300:
        raise ValueError("Invalid feedback topics")
    topic_ids, topic_points = set(), []
    for topic in topics:
        if not isinstance(topic, dict) or not isinstance(topic.get("id"), str) or topic["id"] in topic_ids:
            raise ValueError("Invalid feedback topic ID")
        topic_ids.add(topic["id"])
        for field, limit in (("heading", 5000), ("question", 50000), ("source", 100)):
            if not isinstance(topic.get(field), str) or len(topic[field]) > limit:
                raise ValueError("Invalid feedback topic " + field)
        if 'occurrence' in topic and (type(topic['occurrence']) is not int or topic['occurrence'] < 0):
            raise ValueError('Invalid feedback occurrence')
        if not isinstance(topic.get("points"), list) or not topic["points"]:
            raise ValueError("Missing feedback concern links")
        topic_points.extend(topic["points"])
    if topics and (set(topic_points) != set(point_ids) or len(topic_points) != len(point_ids)):
        raise ValueError("Each concern must map to one feedback topic")
    if "draft" in review:
        blocks = review["draft"]
        if not isinstance(blocks, list) or len(blocks) > len(fragments) + len(topics) + 3:
            raise ValueError("Invalid linked draft")
        seen = set()
        kinds = set()
        for block in blocks:
            if not isinstance(block, dict) or block.get("kind") not in ("intro", "answer", "closing", "text", "heading"):
                raise ValueError("Invalid draft block")
            for field in ("before", "after"):
                if (not isinstance(block.get(field), str) or len(block[field]) > 50000
                        or block[field].strip()):
                    raise ValueError("Invalid draft spacing")
            if block["kind"] == "answer":
                points = block.get("points")
                if not isinstance(points, list) or not points:
                    raise ValueError("Missing linked concerns")
                for point in points:
                    if not isinstance(point, str) or point not in point_ids or point in seen:
                        raise ValueError("Unknown or duplicate linked concern")
                    seen.add(point)
                texts = {fragment["text"] for fragment in fragments if fragment["id"] in points}
                if len(texts) > 1:
                    raise ValueError("Shared answers must contain the same text")
            elif block["kind"] == "text":
                if not isinstance(block.get("text"), str) or len(block["text"]) > 50000:
                    raise ValueError("Invalid additional text")
            elif block["kind"] == "heading":
                identity = block.get("topic")
                if identity not in topic_ids or identity in kinds:
                    raise ValueError("Unknown or duplicate feedback heading")
                kinds.add(identity)
            else:
                if block["kind"] in kinds:
                    raise ValueError("Duplicate opening or closing")
                kinds.add(block["kind"])


def default_draft(review):
    """Numbered replies follow the questions in the specific-feedback field."""
    selected = {item["id"] for item in review["fragments"] if item["include"]}
    blocks = [{"kind": "intro"}] if review["intro"].strip() else []
    topics = review.get("reply_topics", [])
    if topics:
        for topic in topics:
            points = [point for point in topic["points"] if point in selected]
            if points:
                if topic["heading"].strip():
                    blocks.append({"kind": "heading", "topic": topic["id"]})
                blocks.extend({"kind": "answer", "points": [point]} for point in points)
    else:
        blocks.extend({"kind": "answer", "points": [item["id"]]} for item in review["fragments"] if item["include"])
    if review["closing"].strip():
        blocks.append({"kind": "closing"})
    result = []
    for index, block in enumerate(blocks):
        following = blocks[index + 1]["kind"] if index + 1 < len(blocks) else None
        after = "\n" if following is None else "\n\n"
        if topics and following is not None and block["kind"] in ("heading", "answer"):
            after = "\n\n" if following == "closing" else "\n" if following == "heading" else " "
        result.append(dict(block, before="", after=after))
    return result


def draft_blocks(review):
    """Read older workspaces in concern order; retain exact spacing in edited drafts.

    Blocks reference the fragment text rather than duplicate it. Several concerns
    can share one answer after an edit across their boundaries in the full draft.
    Legacy section labels and joining phrases remain in backups, but are not added.
    """
    selected = {item["id"]: item for item in review["fragments"] if item["include"]}
    if "draft" not in review:
        return default_draft(review)
    blocks = []
    for original in review["draft"]:
        block = dict(original)
        if block["kind"] == "answer":
            block["points"] = [point for point in block["points"] if point in selected]
            if not block["points"]:
                continue
        if block["kind"] == "heading":
            topic = next(topic for topic in review["reply_topics"] if topic["id"] == block["topic"])
            if not any(point in selected for point in topic["points"]):
                continue
        blocks.append(block)
    mapped = {point for block in blocks if block["kind"] == "answer" for point in block["points"]}
    points = ([point for topic in review.get("reply_topics", []) for point in topic["points"]]
              or [item["id"] for item in review["fragments"]])
    order = {point: index for index, point in enumerate(points)}

    def insert(index, block):
        if index and not blocks[index - 1]["after"].endswith("\n\n"):
            blocks[index - 1]["after"] += "\n" if blocks[index - 1]["after"].endswith("\n") else "\n\n"
        blocks.insert(index, dict(block, before="", after="\n\n" if index < len(blocks) else "\n"))

    for point in selected:
        if point not in mapped:
            index = next((i for i, block in enumerate(blocks)
                          if block["kind"] == "closing" or block["kind"] == "answer"
                          and min(order[p] for p in block["points"]) > order[point]), len(blocks))
            insert(index, {"kind": "answer", "points": [point]})
    if review["intro"].strip() and not any(block["kind"] == "intro" for block in blocks):
        insert(0, {"kind": "intro"})
    if review["closing"].strip() and not any(block["kind"] == "closing" for block in blocks):
        insert(len(blocks), {"kind": "closing"})
    return blocks


def block_text(review, block):
    if block["kind"] == "answer":
        return next(item["text"] for item in review["fragments"] if item["id"] == block["points"][0])
    if block["kind"] == "heading":
        return next(topic["heading"] for topic in review["reply_topics"] if topic["id"] == block["topic"])
    return block["text"] if block["kind"] == "text" else review[block["kind"]]


def compose(review):
    return "".join(block["before"] + block_text(review, block) + block["after"]
                   for block in draft_blocks(review))


def compile_review(review, limit=LIMIT):
    body = compose(review)
    selected = [fragment for fragment in review["fragments"] if fragment["include"]]
    omitted = [fragment["id"] for fragment in review["fragments"] if not fragment["include"]]
    issues = []
    if not body.strip() or not selected:
        issues.append("Select at least one response fragment.")
    if limit is not None and character_count(body) > limit:
        issues.append(f"The assembled draft exceeds {limit:,} characters.")
    if FORBID_LINKS and LINK.search(body):
        issues.append("The assembled draft contains a link or address.")
    if PLACEHOLDER.search(body) or "\ufeff" in body:
        issues.append("The assembled draft contains a placeholder or byte-order mark.")
    for fragment in selected:
        if not fragment["text"].strip():
            issues.append(fragment["id"] + " has an empty response.")
        if fragment["needs_check"]:
            issues.append(fragment["id"] + " still needs an evidence check.")
    return {"body": body, "characters": character_count(body), "words": len(body.split()), "limit": limit,
            "issues": issues, "can_export": not issues,
            "included": [fragment["id"] for fragment in selected], "omitted": omitted,
            "pending": [fragment["id"] for fragment in review["fragments"] if fragment["needs_check"]]}


def export_review(review, root=ROOT, check=False, limit=LIMIT):
    result = compile_review(review, limit=limit)
    path = Path(root) / "rebuttals" / review["target"]
    result["written"] = False
    if result["can_export"]:
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != result["body"]:
                result["issues"].append("The Markdown file differs from the selected fragments.")
        else:
            atomic_text(path, result["body"])
            result["written"] = True
    return result


def main():
    from confidential_workspace import load_confidential
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check current Markdown without writing")
    args = parser.parse_args()
    workspace = load_workspace()
    failed = False
    for stem, review in workspace["reviews"].items():
        result = export_review(review, check=args.check)
        print(f"{stem}: {result['characters']}/{LIMIT} characters; "
              f"{len(result['included'])}/{len(review['fragments'])} concerns included")
        if result["omitted"]:
            print("  Not included: " + ", ".join(result["omitted"]))
        if result["pending"]:
            print("  Needs evidence: " + ", ".join(result["pending"]))
        for issue in result["issues"]:
            print("  " + issue)
        failed |= bool(result["issues"])
    for stem, review in (load_confidential()["reviews"] if CONFIDENTIAL["enabled"] else {}).items():
        result = export_review(review, check=args.check, limit=CONFIDENTIAL_LIMIT)
        print(f"{stem}: {result['characters']} characters; confidential limit unspecified")
        for issue in result['issues']:
            print('  ' + issue)
        failed |= bool(result['issues'])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
