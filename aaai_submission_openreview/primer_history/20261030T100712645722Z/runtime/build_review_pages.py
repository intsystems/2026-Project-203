"""Build standalone HTML review readers from Markdown summaries and exact JSON quotes.

Run from any directory. --check verifies mappings and detects stale generated pages.
Only Python's standard library is required to build. Math assets are bundled locally.
"""

import argparse
from collections import defaultdict
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from assemble_rebuttals import load_workspace
from confidential_workspace import load_confidential, STEM as CONFIDENTIAL_STEM


from workspace_context import (ROOT, ASSETS, ASSET_URL, CONFIG, PAPER, BRAND, LIMIT_LABEL,
                               CONFIDENTIAL, browser_context)
REVIEWS = ROOT / "reviews"
FIELD_NAMES = {
    "title": "Review title", "review": "Review", "strengths": "Strengths",
    "weaknesses": "Weaknesses", "justification": "Justification",
    "feedback": "Specific Points of Feedback for Rebuttal", "rating": "Rating", "confidence": "Confidence",
}


def safe_id(value):
    return value.replace(".", "-")


def inline_markup(raw):
    """Render only the small inline Markdown subset present in the export."""
    text = escape(raw)
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    return re.sub(
        r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)",
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', text,
    )


def read_summary(stem):
    text = (REVIEWS / (stem + ".md")).read_text(encoding="utf-8")
    intro = text.split("## Brief summary\n\n", 1)[1].split("\n## Every concern", 1)[0].strip()
    concerns = text.split("## Every concern / request\n\n", 1)[1].split("\n## Response priorities", 1)[0]
    pattern = r"^\d+\.\s+\*\*((?:R\d+|AI\d*)\.\d+)\s+(?:\u2014|\?)\s+(.+?)\.\*\*\s+(.+)$"
    points = [{"id": "overview", "title": "Overall assessment", "text": intro}]
    points.extend({"id": m[0], "title": m[1], "text": m[2]} for m in re.findall(pattern, concerns, re.MULTILINE))
    if len(points) < 2:
        raise ValueError(f"No concern summaries found in {stem}.md")
    return points


def source_blocks(raw, field):
    separators = list(re.finditer(r"\n[ \t]*\n+|\n(?=\d+\.\s)", raw))
    intervals, start = [], 0
    for separator in separators:
        intervals.append((start, separator.start()))
        start = separator.end()
    intervals.append((start, len(raw)))
    section = FIELD_NAMES.get(field, field.title())
    ordinal = 0
    blocks = []
    for start, end in intervals:
        while start < end and raw[start].isspace():
            start += 1
        while end > start and raw[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        heading = re.fullmatch(r"\*\*([^*]+)\*\*", raw[start:end])
        if heading:
            section = heading.group(1)
            ordinal = 0
        else:
            ordinal += 1
        blocks.append({"start": start, "end": end, "heading": heading.group(1) if heading else None,
                       "location": section + (f" · ¶{ordinal}" if not heading else "")})
    return blocks


def prepare_review(config, note):
    points = read_summary(config["stem"])
    expected = {point["id"] for point in points}
    if set(config["points"]) != expected:
        raise ValueError(f"{config['stem']}: summary/annotation point mismatch")
    fields = {key: str(value.get('value')) if isinstance(value, dict) and 'value' in value else str(value)
              for key, value in note["content"].items()}
    blocks = {key: source_blocks(value, key) for key, value in fields.items()}
    evidence = []
    for point in points:
        entries = config["points"][point["id"]]
        if not entries:
            raise ValueError(f"No source passages for {point['id']}")
        for index, entry in enumerate(entries, 1):
            field, quote = entry[:2]
            raw = fields[field]
            occurrence = entry[2] if len(entry) == 3 else None
            if not quote or occurrence is None and raw.count(quote) != 1:
                raise ValueError(f"{config['stem']} {point['id']}: quote must match exactly once: {quote!r}")
            matches = list(re.finditer(re.escape(quote), raw))
            if occurrence is not None and (type(occurrence) is not int or not 0 <= occurrence < len(matches)):
                raise ValueError('Invalid quote occurrence: ' + point['id'])
            start = matches[occurrence or 0].start()
            end = start + len(quote)
            block = next((block for block in blocks[field] if block["start"] <= start and end <= block["end"]), None)
            if block is None or block["heading"]:
                raise ValueError(f"Passage crosses source blocks: {point['id']}: {quote!r}")
            evidence.append({"id": f"passage-{safe_id(point['id'])}-{index:02d}", "point": point["id"],
                             "field": field, "start": start, "end": end, "text": quote,
                             "location": block["location"]})
    field_order = {field: index for index, field in enumerate(fields)}
    evidence.sort(key=lambda item: (field_order[item["field"]], item["start"], item["end"]))
    return {**config, "points": points, "fields": fields, "blocks": blocks, "evidence": evidence}


def render_passage(raw, block, annotations):
    annotations = [item for item in annotations if block["start"] <= item["start"] and item["end"] <= block["end"]]
    boundaries = {block["start"], block["end"]}
    starts, ends = defaultdict(list), defaultdict(list)
    for item in annotations:
        boundaries.update((item["start"], item["end"]))
        starts[item["start"]].append(item)
        ends[item["end"]].append(item)
    positions = sorted(boundaries)
    output = []
    for i, pos in enumerate(positions):
        if ends[pos]:
            links = []
            for item in ends[pos]:
                label = "Overview" if item["point"] == "overview" else item["point"]
                title = f"Back to summary: {label}"
                links.append(f'<a class="backlink" href="#summary-{safe_id(item["point"])}" '
                             f'data-return-point="{item["point"]}" aria-label="{escape(title)}" title="{escape(title)}">↩ {label}</a>')
            output.append('<sup class="backlinks">' + "".join(links) + '</sup>')
        for item in starts[pos]:
            output.append(f'<span class="passage-anchor" id="{item["id"]}" tabindex="-1" '
                          f'data-point="{item["point"]}" aria-label="{escape(item["location"])}"></span>')
        if i == len(positions) - 1:
            continue
        end = positions[i + 1]
        text = inline_markup(raw[pos:end])
        covering = [item for item in annotations if item["start"] <= pos and end <= item["end"]]
        if covering:
            ids = " ".join(item["id"] for item in covering)
            points = " ".join(dict.fromkeys(item["point"] for item in covering))
            text = f'<mark class="source-highlight" data-points="{points}" data-evidence="{ids}">{text}</mark>'
        output.append(text)
    return "".join(output)


def summary_cards(review):
    cards = []
    for point in review["points"]:
        items = [item for item in review["evidence"] if item["point"] == point["id"]]
        label = "Overview" if point["id"] == "overview" else point["id"]
        heading = f"{label} — {point['title']}"
        text = point["text"]
        citation = re.search(r"\s+(\((?:Weakness|Question|Feedback|Suggestion|Minor issue).+\))$", text)
        if citation:
            text = escape(text[:citation.start()]) + f' <span class="summary-citation">{escape(citation.group(1))}</span>'
        else:
            text = escape(text)
        links = []
        for item in items:
            links.append(f'<li><a class="passage-link" href="#{item["id"]}" title="{escape(item["text"], quote=True)}">'
                         f'<span class="passage-location">{escape(item["location"])}</span>'
                         f'<span class="passage-preview">“{escape(item["text"])}”</span></a></li>')
        cards.append(f'''<section class="point-card" id="summary-{safe_id(point['id'])}" data-point="{point['id']}" tabindex="-1">
<h3 class="point-heading">{escape(heading)}</h3>
<div class="point-actions"><a class="point-jump" href="#{items[0]['id']}">Show in review →</a>{f'<a class="point-jump response-jump" href="#response-{safe_id(point["id"])}">Our response →</a>' if point['id'] != 'overview' else ''}</div>
<p class="point-text">{text}</p>
<details class="passage-list"><summary>{len(items)} linked passages</summary><ol>{''.join(links)}</ol></details>
</section>''')
    return "\n".join(cards)


def original_sections(review):
    sections = []
    for field, raw in review["fields"].items():
        paragraphs = []
        annotations = [item for item in review["evidence"] if item["field"] == field]
        for block in review["blocks"][field]:
            if block["heading"]:
                paragraphs.append(f'<h4 class="source-subheading">{escape(block["heading"])}</h4>')
            else:
                paragraphs.append(f'<p class="source-paragraph" data-source-field="{field}" '
                                  f'data-source-start="{block["start"]}" data-source-end="{block["end"]}">'
                                  + render_passage(raw, block, annotations) + '</p>')
        extra = ' score-section' if field in ("rating", "confidence") else ''
        sections.append(f'<section class="source-section{extra}" id="field-{field}"><h3>{FIELD_NAMES.get(field, field.title())}</h3>'
                        + "\n".join(paragraphs) + '</section>')
    return "\n".join(sections)


def review_tabs(reviews, current):
    return "".join(f'<a href="{review["stem"]}.html"'
                   + (' aria-current="page"' if review["stem"] == current else '')
                   + f'>{escape(review["name"])} · {escape(review["reviewer"])}</a>' for review in reviews) + (f'<a href="{CONFIDENTIAL_STEM}.html">Confidential AC/SPC comment</a>' if CONFIDENTIAL['enabled'] else '')


def frame_options(response, field):
    options = response.get('frame_variants', {}).get(field, [])
    if not options:
        return ''
    choices = ''.join(f'<option value="{escape(item["id"], quote=True)}">{escape(item["label"])}</option>' for item in options)
    return f'''<div class="wording-options"><label class="editor-label" for="variant-{field}">Saved wording options</label>
<select id="variant-{field}" data-variant-selector="{field}">{choices}</select>
<label class="editor-label" for="variant-text-{field}">Edit the selected option · excluded until applied</label>
<textarea id="variant-text-{field}" rows="2" data-variant-field="{field}">{escape(options[0]['text'])}</textarea>
<button type="button" data-apply-variant="{field}">Use this {'opening' if field == 'intro' else 'closing'}</button>
<p class="draft-help">Applying replaces the active wording below and updates the full draft. Editing a saved option alone leaves the draft as it is. The character counter includes only the active wording.</p></div>'''


def badges(review):
    if "rating" not in review["fields"]:
        return '<span class="badge">AI-generated review</span><span class="badge">No score supplied</span>'
    return (f'<span class="badge">Rating <strong>{escape(review["fields"]["rating"])}</strong></span>'
            f'<span class="badge">Confidence <strong>{escape(review["fields"]["confidence"])}</strong></span>')


def topic_passage(review, topic):
    raw = review['fields'][topic['source']]
    occurrence = topic.get('occurrence')
    if occurrence is None and raw.count(topic['question']) != 1:
        raise ValueError(f"{review['stem']}: feedback question must match the original review: {topic['id']}")
    matches = list(re.finditer(re.escape(topic['question']), raw))
    if not matches or occurrence is not None and (type(occurrence) is not int or not 0 <= occurrence < len(matches)):
        raise ValueError('Invalid feedback question occurrence: ' + topic['id'])
    start = matches[occurrence or 0].start()
    end = start + len(topic['question'])
    matches = [item for item in review['evidence'] if item['point'] in topic['points']
               and item['field'] == topic['source'] and start <= item['start'] < item['end'] <= end]
    if not matches:
        raise ValueError(f"{review['stem']}: feedback topic has no matching highlighted passage: {topic['id']}")
    return matches[0]['id']


def response_workspace(review):
    response = review['response']
    titles = {point['id']: point['title'] for point in review['points']}
    topics = response.get('reply_topics', [])
    topic_cards = []
    for index, topic in enumerate(topics, 1):
        passage = topic_passage(review, topic)
        links = ' '.join(f'<a href="#response-{safe_id(point)}">{point}</a>' for point in topic['points'])
        topic_cards.append(f'''<article class="response-topic" id="topic-{topic['id']}" tabindex="-1">
<div class="topic-caption">{'Specific feedback' if topic['source'] == 'feedback' else 'Additional concern' if 'feedback' in review['fields'] else 'AI review topic'} {index}<span data-topic-status="{topic['id']}"></span></div>
<label class="editor-label" for="heading-{topic['id']}">Editable heading · included in the {LIMIT_LABEL}-character limit</label>
<textarea id="heading-{topic['id']}" rows="1" class="topic-heading-editor" data-topic="{topic['id']}">{escape(topic['heading'])}</textarea>
<div class="response-links"><a href="#{passage}">Read the reviewer’s exact words →</a><span>Answers: {links}</span></div>
<details class="topic-question"><summary>Reviewer’s wording</summary><blockquote>{escape(topic['question'])}</blockquote></details></article>''')
    cards = []
    for fragment in response['fragments']:
        point = fragment['id']
        pid = safe_id(point)
        evidence = next(item for item in review['evidence'] if item['point'] == point)
        topic = next((item for item in topics if point in item['points']), None)
        topic_link = f'<a href="#topic-{topic["id"]}" data-topic-link="{topic["id"]}">{escape(topic["heading"])}</a>' if topic else ''
        cards.append(f'''<article class="response-card" id="response-{pid}" data-response-point="{point}" tabindex="-1">
<div class="response-card-top"><span class="response-id">{point}</span><span class="response-check" data-check-for="{point}">{'Needs evidence check' if fragment['needs_check'] else 'Draft answer'}</span><output class="fragment-count" data-count-for="{point}"></output></div>
<h3>{escape(titles[point])}</h3>
<div class="response-links"><a href="#summary-{pid}">↩ Concern {point}</a><a href="#{evidence['id']}">Reviewer’s exact words →</a><a href="#assembled-draft" data-assembled-link="{point}">See in assembled preview →</a></div>
<div class="response-topic-link">{topic_link}</div>
<label class="include-row"><input type="checkbox" data-point="{point}" data-field="include"{' checked' if fragment['include'] else ''}>Include this answer in the assembled draft</label>
<label class="editor-label" for="answer-{pid}">Our response</label>
<textarea class="response-editor" id="answer-{pid}" data-point="{point}" data-field="text" rows="4" spellcheck="true">{escape(fragment['text'])}</textarea>
<p class="shared-answer-note" data-shared-for="{point}" hidden><span></span><button type="button" data-separate-answer="{point}">Give this concern a separate answer</button></p>
<div class="preview-label">Preview with LaTeX</div><div class="response-preview" data-preview-for="{point}"></div>
<details class="response-notes"{' open' if fragment['needs_check'] else ''}><summary>Evidence and drafting notes · excluded from rebuttal</summary>
<label class="include-row"><input type="checkbox" data-point="{point}" data-field="needs_check"{' checked' if fragment['needs_check'] else ''}>Still needs an evidence check</label>
<label class="editor-label" for="note-{pid}">Internal notes</label><textarea id="note-{pid}" class="notes-editor" data-point="{point}" data-field="notes" rows="4">{escape(fragment['notes'])}</textarea></details></article>''')
    return f'''<section class="responses-view" id="responses" tabindex="-1" hidden>
<header class="responses-intro"><h2>Response fragments</h2><p>Edit one answer at a time or edit the full draft below. Both stay in sync. Use $...$ or \\(...\\) for inline LaTeX; $$...$$ or \\[...\\] for display equations.</p>
<a href="#assembled-draft">Go to assembled draft →</a></header>
<div class="file-mode-note" id="file-mode-note" hidden><strong>This page keeps a browser copy in localStorage (not cookies).</strong> For automatic saving to files: (1) open <code>{escape(ROOT.name)}</code> in Windows File Explorer; (2) double-click <code>start_workspace.cmd</code>; (3) edit the browser page it opens and leave the Python window running. Alternatively, click <strong>Save to repo</strong> here and choose the <code>{escape(ROOT.name)}</code> folder in Edge or Chrome.</div>
<div class="file-mode-note" id="server-update-note" hidden>The local server is using older editor code. In its Python window, press <strong>Ctrl+C</strong>, double-click <code>start_workspace.cmd</code> again, then reload this page. Your browser copy is retained.</div>
<details class="saving-details"><summary>How to run “Save to repo” and where it saves</summary><p><strong>Recommended: automatic local file saving.</strong></p><ol><li>Open the <code>{escape(ROOT.name)}</code> folder in Windows File Explorer.</li><li>Double-click <code>start_workspace.cmd</code>. It runs Python and opens <code>http://127.0.0.1:8765/reviews/index.html</code>.</li><li>Open a review in that browser page and edit. Leave the Python window open. Edits save automatically after a short pause; <strong>Save to repo</strong> saves immediately.</li></ol><p><strong>If you opened the HTML directly:</strong> in desktop Edge or Chrome, click <strong>Save to repo</strong>, select <code>{escape(ROOT.name)}</code> (or its parent repository folder), and allow the browser to write there. Click it again after later edits. The folder selection lasts for this open tab.</p><p><strong>Files written:</strong> <code>response_fragments.json</code> stores answers, editable headings and notes; <code>rebuttals/{escape(response['target'])}</code> stores the valid assembled response; <code>response_history/</code> keeps previous versions. The receipt below confirms each file save. A draft over the limit still saves to JSON; the Markdown retains its last valid text.</p><p><strong>“Repo” means files in your local repository folder.</strong> Git commit and push are separate operations. Browser autosave uses localStorage, not cookies; clearing site data removes that browser copy. File and local-server addresses have separate browser copies, so save to files or download a fragment backup before switching.</p></details>
<p class="repo-save-receipt" id="repo-save-receipt" role="status"></p>
<div class="save-conflict" id="save-conflict" role="alert" hidden><strong>Another tab or a file edit changed this review.</strong><p>Your browser edits are preserved. Choose which version to continue with.</p><button id="load-saved-version" type="button">Load version from files</button><button id="keep-browser-version" type="button">Save my browser edits instead</button></div>
<section class="feedback-plan" id="feedback-plan" tabindex="-1"><h2>Numbered response plan</h2>
<p class="draft-help">{'These topics follow “Specific Points of Feedback for Rebuttal” in the reviewer’s order. Where the reviewer did not number the questions, we number them in that order.' if 'feedback' in review['fields'] else 'The AI review has no separate specific-feedback section. These topics follow its weaknesses, suggestions and minor issues.'} Related concerns from elsewhere in the review are linked under the closest topic.</p>
<p class="draft-help">Edit each short “Re …” heading here or in the full OpenReview text. Its number, heading and colon are copied into the rebuttal and count toward the limit.</p>
{''.join(topic_cards)}<button id="arrange-feedback-topics" type="button">Arrange selected answers under these headings</button>
<p class="draft-help">This rebuilds the numbered layout using the current answers. A browser backup retains the previous layout.</p></section>
{''.join(cards)}
<section class="assembled-draft" id="assembled-draft" tabindex="-1"><div class="draft-heading"><h2>Assembled draft</h2><output id="draft-count" aria-live="polite"></output></div>
<p class="draft-help">The numbered responses follow the reviewer’s specific-feedback questions, with short editable “Re …” headings. Edit the full text freely or use the heading and answer fields above.</p>
<details class="assembly-settings"><summary>Opening and closing · editable wording options</summary>{frame_options(response, 'intro')}<label class="editor-label" for="draft-intro">Active opening</label><textarea id="draft-intro" rows="2" data-review-field="intro">{escape(response['intro'])}</textarea>{frame_options(response, 'closing')}<label class="editor-label" for="draft-closing">Active closing</label><textarea id="draft-closing" rows="2" data-review-field="closing">{escape(response['closing'])}</textarea></details>
<div class="draft-coverage" id="draft-coverage"></div><ul id="draft-issues" class="draft-issues" aria-live="polite"></ul>
<label class="editor-label" for="assembled-text">OpenReview text · edit here or in the separate answers</label><textarea id="assembled-text" rows="14" spellcheck="true" aria-describedby="assembled-edit-help"></textarea>
<p class="draft-help" id="assembled-edit-help">Edits across several answers join them into one shared answer, with every concern still linked. Its response fields then edit the same text. Use “Give this concern a separate answer” to split it again.</p>
<section class="rendered-draft" aria-labelledby="rendered-draft-heading"><h3 id="rendered-draft-heading">Full response · rendered LaTeX</h3>
<p class="draft-help">This shows the complete text above, with equations rendered and your line breaks preserved. It updates as you edit. OpenReview may use different fonts and spacing.</p>
<div id="rendered-draft" class="response-preview full-response-preview" role="region" aria-labelledby="rendered-draft-heading" tabindex="0"></div></section>
<div class="preview-heading"><h3>Assembled preview</h3><div class="draft-totals" aria-live="polite"><output id="preview-word-count" class="word-count"></output><output id="preview-character-count" class="character-count"></output></div></div>
<p class="draft-help">The numbered “Re …” headings and answer text are exported. The longer concern summaries and navigation links below are for drafting. The {LIMIT_LABEL}-character limit counts LaTeX source and every line break; words are counted by spaces and line breaks.</p>
<div id="assembled-preview" class="assembled-preview"></div>
<div class="draft-actions"><button id="download-markdown" type="button">Download {escape(response['target'])}</button><button id="download-fragments" type="button">Download fragment backup</button><label class="button-link import-backup" for="import-fragments">Restore fragment backup<input type="file" id="import-fragments" accept="application/json,.json"></label></div>
<p class="draft-help" id="export-status">The local workspace updates <code>rebuttals/{escape(response['target'])}</code> when the selected text passes the length, link and evidence checks.</p></section></section>'''


def page(review, reviews, css, script):
    data = {**browser_context(), "noteId": review["note_id"], "fields": review["fields"], "evidence": review["evidence"],
            "pointTitles": {point["id"]: point["title"] for point in review["points"]},
            "stem": review["stem"], "response": review["response"],
            "topicPassages": {topic['id']: topic_passage(review, topic) for topic in review['response'].get('reply_topics', [])}}
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    title = f'{review["name"]} · {review["reviewer"]}'
    count = len(review["points"]) - 1
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Clickable review summary and annotated original review for {escape(BRAND)}.">
<title>{escape(title)} — {escape(PAPER["short_title"])} reviews</title><link rel="stylesheet" href="{ASSET_URL}/vendor/katex/katex.min.css"><style>{css}</style></head>
<body class="review-page"><a class="skip-link" href="#original-review">Skip to original review</a>
<header class="masthead"><div class="header-identity"><p class="eyebrow">{escape(BRAND)}</p>
<h1 title="{escape(review['fields']['title'], quote=True)}">{escape(review['name'])} <span>· {escape(review['reviewer'])}</span></h1></div>
<div class="header-meta"><div class="badges">{badges(review)}</div><div class="header-actions"><a class="button-link" href="index.html">All reviews</a><a class="button-link" href="{review['stem']}.md">Markdown</a><button class="print-button" type="button">Print</button></div></div></header>
<nav class="review-tabs" aria-label="Choose a review">{review_tabs(reviews, review['stem'])}</nav>
<noscript><p class="no-script">Summary links and return links work without JavaScript. Enable JavaScript for topic highlighting and next/previous navigation.</p></noscript>
<div class="reader-layout"><aside class="summary-pane" id="summary" aria-label="Clickable summary" tabindex="-1">
<div class="pane-heading"><div class="pane-title-row"><h2>Reviewer’s concerns <span class="badge">{count}</span></h2></div>
<p>These describe the reviewer’s concerns. Follow “Our response” to edit an answer. Headings are ordinary selectable text.</p></div>
{summary_cards(review)}</aside>
<main class="original-pane" id="original-review" tabindex="-1"><div class="workspace-toolbar"><nav aria-label="Review workspace"><a href="#original-review" id="view-review" aria-current="page">Original review</a><a href="#responses" id="view-responses">Response fragments</a><a href="#assembled-draft">Assembled draft</a></nav><div class="workspace-save"><button id="save-to-repo" type="button" disabled>Save to repo</button><span id="save-status" role="status">Loading drafts…</span></div></div>
<div id="review-view"><div class="reading-toolbar">
<div class="toolbar-status" aria-live="polite"><strong id="active-point">Choose a summary point</strong><span id="passage-position">{len(review['evidence'])} linked passages in this review</span></div>
<div class="toolbar-controls"><button id="previous-passage" type="button" disabled aria-label="Previous linked passage">← Previous</button><button id="next-passage" type="button" disabled aria-label="Next linked passage">Next →</button><a class="button-link" id="back-to-summary" href="#summary">↩ Summary</a><button id="toggle-highlights" type="button" aria-pressed="true">Highlights on</button></div></div>
<div class="original-intro"><h2>Original review</h2><p>Reviewer wording from the downloaded OpenReview export. Highlights connect to the working summary.</p>
<div class="legend"><span><i></i>Mapped passage</span><span><i class="selected-key"></i>Selected point</span><span><i class="target-key"></i>Current passage</span></div></div>
{original_sections(review)}</div>{response_workspace(review)}</main></div>
<script type="application/json" id="review-data">{encoded}</script><script src="{ASSET_URL}/vendor/katex/katex.min.js"></script><script src="{ASSET_URL}/vendor/katex/contrib/auto-render.min.js"></script><script>{script}</script></body></html>
'''


def index_page(reviews, css):
    cards = []
    for review in reviews:
        title = f'{review["name"]} · {review["reviewer"]}'
        cards.append(f'''<article class="review-card"><h2><a href="{review['stem']}.html">{escape(title)}</a></h2>
<p class="card-title">{escape(review['fields']['title'])}</p><div class="badges">{badges(review)}</div>
<p class="card-overview">{escape(review['points'][0]['text'])}</p><div class="review-card-footer"><span>{len(review['points']) - 1} concerns · {len(review['evidence'])} passages</span><a href="{review['stem']}.html">Open review →</a></div></article>''')
    confidential_card = (f'<article class="review-card"><h2><a href="{CONFIDENTIAL_STEM}.html">Confidential AC/SPC comment</a></h2><p>Optional private decision summary, with separate text and reviewer references.</p></article>' if CONFIDENTIAL['enabled'] else '')
    total_points = sum(len(review["points"]) - 1 for review in reviews)
    total_evidence = sum(len(review["evidence"]) for review in reviews)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(PAPER["short_title"])} — Review navigator</title><style>{css}</style></head><body class="index-page"><main class="index-wrap">
<header class="index-header"><p class="eyebrow">{escape(PAPER["venue"])} / Submission #{escape(PAPER["submission_id"])}</p><h1>Review navigator</h1>
<p>Follow each concern to the reviewer’s exact words or an editable response. Preview LaTeX and assemble selected answers into a draft of at most {LIMIT_LABEL} characters.</p>
<p>For automatic saving to the Markdown files, run <code>start_workspace.cmd</code> in the parent folder. Direct file opening saves edits in your browser and offers downloads.</p></header>
<div class="index-stats"><div class="index-stat"><strong>{len(reviews)}</strong>reviews</div><div class="index-stat"><strong>{total_points}</strong>summary concerns</div><div class="index-stat"><strong>{total_evidence}</strong>linked passages</div></div>
<section class="review-grid" aria-label="Reviews">{''.join(cards)}{confidential_card}</section>
<footer class="index-footer">The concern summaries describe reviewer requests; the response fragments are author drafts. Original reviews are retained verbatim. Math rendering uses bundled local assets.<br><a href="../03_RESPONSE_WORKSPACE.md">Drafting plan and editor instructions</a></footer>
</main></body></html>
'''


def confidential_page(reviews, css):
    response = load_confidential()['reviews'][CONFIDENTIAL_STEM]
    references = []
    for reference in CONFIDENTIAL.get('references', []):
        stem, point, label = reference['stem'], reference['point'], reference['label']
        source = next(item for item in reviews if item['stem'] == stem)
        passage = next(item['id'] for item in source['evidence'] if item['point'] == point)
        references.append(f'<li><a href="{stem}.html#{passage}">{escape(label)}</a> <a href="{stem}.html#response-{safe_id(point)}">Our answer</a></li>')
    review = {'response': response, 'fields': {},
              'points': [{'id': 'C1', 'title': 'Decision summary for the AC/SPC'}],
              'evidence': [{'id': 'chair-references', 'point': 'C1'}]}
    editor = response_workspace(review)
    editor = editor.replace('id="responses" tabindex="-1" hidden', 'id="responses" tabindex="-1"')
    editor = editor.replace('response_fragments.json', 'confidential_comment.json')
    editor = editor.replace('Response fragments', 'Confidential comment')
    editor = editor.replace('Reviewer’s exact words →', 'Relevant review passages →')
    editor = editor.replace('Numbered response plan', 'Comment layout')
    editor = editor.replace('Our response</label>', 'Confidential comment draft</label>')
    editor = editor.replace('OpenReview text · edit here or in the separate answers', 'OpenReview confidential text · editable')
    editor = editor.replace('Edits across several answers join them into one shared answer, with every concern still linked. Its response fields then edit the same text. Use “Give this concern a separate answer” to split it again.', 'Edits here update the separate comment field above. References and internal notes stay outside the exported comment.')
    editor = editor.replace('The numbered responses follow the reviewer’s specific-feedback questions, with short editable “Re …” headings. Edit the full text freely or use the heading and answer fields', 'Edit the full confidential comment here or in its separate field')
    editor = editor.replace(f'The numbered “Re …” headings and answer text are exported. The longer concern summaries and navigation links below are for drafting. The {LIMIT_LABEL}-character limit counts LaTeX source and every line break; words are counted by spaces and line breaks.', 'Only the comment text is exported. References and notes are excluded. Confirm the separately configured confidential-comment limit before posting. Characters include LaTeX source and line breaks.')
    data = {**browser_context(True), 'stem': CONFIDENTIAL_STEM, 'response': response, 'topicPassages': {},
            'pointTitles': {'C1': 'Confidential AC/SPC comment'}, 'evidence': review['evidence'],
            'sourceFile': 'confidential_comment.json', 'confidential': True}
    encoded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    script = '\n'.join((ASSETS / name).read_text(encoding='utf-8') for name in ('draft.js', 'responses.js'))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Confidential AC/SPC comment · {escape(PAPER["short_title"])}</title><link rel="stylesheet" href="{ASSET_URL}/vendor/katex/katex.min.css"><style>{css}</style></head>
<body class="review-page confidential-page"><header class="masthead"><div class="header-identity"><p class="eyebrow">{escape(BRAND)}</p><h1>Confidential AC/SPC comment</h1></div><a class="button-link" href="index.html">All reviews</a></header>
<nav class="review-tabs" aria-label="Draft navigation">{review_tabs(reviews, CONFIDENTIAL_STEM)}</nav>
<div class="reader-layout"><aside class="summary-pane" id="summary-C1"><div class="pane-heading"><h2>Purpose and references</h2><p>A concise decision summary: correct a factual misunderstanding, connect the evidence, and make a reasoned request for acceptance.</p><p>This draft is for the AC/SPC. Saving writes local files; it does not post to OpenReview.</p><a class="button-link" href="#assembled-draft">Edit the confidential comment ↓</a></div>
<section class="chair-references" id="chair-references"><h3>Relevant review passages</h3><p>Reviewer IDs and question numbers appear in the comment. These clickable links are drafting aids and are excluded from the exported text.</p><ol>{''.join(references)}</ol>
<p>Use the internal notes to track unresolved technical items. Add decision-relevant reviewer references in <code>workspace.json</code>; the agent guide explains their format.</p>
<p><strong>Length:</strong> review responses use the limit in <code>workspace.json</code>. The confidential limit is configured separately and remains unspecified until verified against the conference rules.</p>
<p><a href="../04_CONFIDENTIAL_COMMENT_STRATEGY.md">Strategy, alternatives and source guidance</a></p></section></aside>
<main class="original-pane"><div class="workspace-toolbar"><a href="#assembled-draft">Full comment and rendered preview</a><div class="workspace-save"><button id="save-to-repo" type="button" disabled>Save to repo</button><span id="save-status" role="status">Loading draft…</span></div></div>{editor}</main></div>
<script type="application/json" id="review-data">{encoded}</script><script src="{ASSET_URL}/vendor/katex/katex.min.js"></script><script src="{ASSET_URL}/vendor/katex/contrib/auto-render.min.js"></script><script>{script}</script></body></html>'''


class AnchorAudit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.fragments = []
        self.duplicates = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicates.append(attrs["id"])
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.fragments.append(attrs["href"][1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate inputs and compare generated output without writing")
    args = parser.parse_args()
    config = json.loads((ROOT / "annotations.json").read_text(encoding="utf-8"))
    export = json.loads((ROOT / CONFIG['source']['reviews_json']).read_text(encoding='utf-8'))
    notes = {note['id']: note for note in (export['notes'] if isinstance(export, dict) else export)}
    css = (ASSETS / "reader.css").read_text(encoding="utf-8")
    script = '\n'.join((ASSETS / name).read_text(encoding="utf-8")
                       for name in ("reader.js", "draft.js", "responses.js"))
    reviews = [prepare_review(item, notes[item["note_id"]]) for item in config["reviews"]]
    workspace = load_workspace()
    for review in reviews:
        review['response'] = workspace['reviews'][review['stem']]
        if {fragment['id'] for fragment in review['response']['fragments']} != {point['id'] for point in review['points'] if point['id'] != 'overview'}:
            raise ValueError('Summary/response concern mismatch: ' + review['stem'])
    outputs = {review["stem"] + ".html": page(review, reviews, css, script) for review in reviews}
    outputs["index.html"] = index_page(reviews, css)
    if CONFIDENTIAL['enabled']:
        outputs[CONFIDENTIAL_STEM + '.html'] = confidential_page(reviews, css)
    for name, content in outputs.items():
        audit = AnchorAudit()
        audit.feed(content)
        missing = set(audit.fragments) - audit.ids
        if missing or audit.duplicates:
            raise ValueError(f"{name}: missing fragments {missing}; duplicate IDs {audit.duplicates}")
        path = REVIEWS / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"{name} is missing or stale; rerun without --check")
        else:
            path.write_text(content, encoding="utf-8")
        print(f"{'Checked' if args.check else 'Built'} {name}")
    print(f"{len(reviews)} reviews; {sum(len(r['points']) - 1 for r in reviews)} concerns; "
          f"{sum(len(r['evidence']) for r in reviews)} exact passage mappings; all fragment links resolve.")


if __name__ == "__main__":
    main()
