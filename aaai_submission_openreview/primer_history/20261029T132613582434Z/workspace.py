"""Initialize, build, serve or validate a paper workspace beside this primer.

Python standard library only. No downloads, API credentials or submission actions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

PRIMER = Path(__file__).resolve().parent
RUNTIME = PRIMER / "runtime"


def write_new(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def value(raw):
    return raw.get("value") if isinstance(raw, dict) and "value" in raw else raw


def fields(note):
    return {key: str(value(raw)) for key, raw in note.get("content", {}).items()}


def paragraphs(text):
    # Match the reader's paragraph boundaries, retaining exact source text.
    return [part.strip() for part in re.split(r"\n[ \t]*\n+|\n(?=\d+\.\s)", text)
            if part.strip() and not re.fullmatch(r"\*\*[^*]+\*\*", part.strip())]


def initialize(args, root):
    source = Path(args.source)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    raw = source.read_bytes()
    export = json.loads(raw)
    notes = export.get("notes") if isinstance(export, dict) else export
    if not isinstance(notes, list):
        raise ValueError("Expected a JSON list of notes or an object with a notes list")
    reviews = []
    for note in notes:
        invitations = note.get("invitations", []) + ([note["invitation"]] if note.get("invitation") else [])
        if any(item.endswith('/Official_Review') or item.endswith('/AI_Review') for item in invitations):
            reviews.append(note)
    if not reviews:
        raise ValueError("No Official_Review or AI_Review notes found. Add an explicit importer for this export format")
    reviews.sort(key=lambda n: (any('AI_Review' in x for x in n.get('invitations', [])), n.get('number', 0), n['id']))
    relative_source = Path(os.path.relpath(source, root)).as_posix()
    config = {
        "schema_version": 1, "workspace_id": uuid.uuid4().hex,
        "paper": {"title": args.title, "short_title": args.short_title, "venue": args.venue, "submission_id": args.submission},
        "rules": {"response_limit_unit": "characters", "response_character_limit": args.limit,
                  "confidential_character_limit": None, "allow_external_links": False, "deadline": args.deadline,
                  "source": "Author-supplied conference announcement; confirm other settings before drafting."},
        "source": {"reviews_json": relative_source, "sha256": hashlib.sha256(raw).hexdigest()},
        "reviews": [], "confidential": {"enabled": True, "stem": "confidential_ac", "target": "confidential_ac_comment.md", "references": []},
        "port": args.port,
    }
    annotations = {"schema_version": 1, "reviews": []}
    responses = {"version": 1, "reviews": {}}
    outputs = {}
    human, ai = 0, 0
    for note in reviews:
        signatures = note.get('signatures', [])
        is_ai = any('AI_Reviewer' in s for s in signatures) or any('/AI_Review' in s for s in note.get('invitations', []))
        if is_ai:
            ai += 1
            prefix = 'AI' if ai == 1 else f'AI{ai}'
            name = f'AI review{f" {ai}" if ai > 1 else ""}'
        else:
            human += 1
            prefix, name = f'R{human}', f'Human review {human}'
        reviewer = signatures[0].rsplit('/', 1)[-1] if signatures else note['id']
        reviewer = reviewer.removeprefix('Program_Committee_')
        slug = re.sub(r'[^A-Za-z0-9_-]', '_', reviewer)
        stem = f'review_{prefix}_{slug}'
        target = f'rebuttal_{prefix}_{slug}.md'
        spec = {'stem': stem, 'target': target, 'note_id': note['id'], 'name': name, 'reviewer': reviewer, 'kind': 'ai' if is_ai else 'human'}
        config['reviews'].append(spec)
        content = fields(note)
        main = next((key for key in ('review', 'summary', 'justification', 'strengths', 'feedback') if paragraphs(content.get(key, ''))), None)
        if main is None:
            raise ValueError('No readable review text in ' + note['id'])
        mapping = {'overview': [[main, paragraphs(content[main])[0], 0]]}
        fragments, topics, summaries = [], [], []
        imported = [(key, quote) for key in ('feedback', 'weaknesses') if key in content
                    for quote in paragraphs(content[key])]
        if not imported:
            imported = [(main, quote) for quote in paragraphs(content[main])]
        if not imported:
            raise ValueError('No feedback passages in ' + note['id'])
        # Count repeated identical paragraphs explicitly, without rewriting quotes.
        occurrences = {}
        for ordinal, (key, quote) in enumerate(imported, 1):
            occurrence = occurrences.get((key, quote), 0)
            occurrences[key, quote] = occurrence + 1
            point, topic = f'{prefix}.{ordinal}', f'Q{ordinal}'
            mapping[point] = [[key, quote, occurrence]]
            summaries.append(f'{ordinal}. **{point} \u2014 Summary needed for imported {key} passage {ordinal}.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.')
            fragments.append({'id': point, 'text': '', 'notes': 'Imported passage only. Read the paper and code, record evidence, then draft and verify this answer.',
                              'include': True, 'needs_check': True, 'joiner': ' ', 'section': 'answers'})
            topics.append({'id': topic, 'heading': f'{ordinal}. Re [name the specific concern]:', 'question': quote,
                           'source': key, 'occurrence': occurrence, 'points': [point]})
        response = {'target': target, 'revision': 0, 'intro': '', 'closing': '', 'sections': [{'id': 'answers', 'lead': ''}],
                    'fragments': fragments, 'reply_topics': topics,
                    'frame_variants': {
                        'intro': [{'id': 'specific', 'label': 'Tailor to this review', 'text': ''},
                                  {'id': 'brief', 'label': 'Brief thanks', 'text': 'Thank you for your careful review.'}],
                        'closing': [{'id': 'none', 'label': 'No closing', 'text': ''},
                                    {'id': 'conditional', 'label': 'Conditional score request' if not is_ai else 'Thanks without a score request',
                                     'text': 'If these answers resolve your concerns, would you consider raising your score?' if not is_ai else 'Thank you for the detailed feedback.'}]}}
        responses['reviews'][stem] = response
        annotations['reviews'].append({key: val for key, val in spec.items() if key not in ('target',)} | {'points': mapping})
        original = '\n\n'.join(f'### {key}\n\n{text}' for key, text in content.items())
        outputs[f'reviews/{stem}.md'] = f'# {name} \u2014 {reviewer}\n\nNote ID: `{note["id"]}`. Source: `{relative_source}`.\n\n## Brief summary\n\nImported review. Scientific summaries and answers have not been prepared.\n\n## Every concern / request\n\n' + '\n\n'.join(summaries) + '\n\n## Response priorities\n\nRead `AGENT_START.md` and prepare the evidence map before drafting.\n\n## Original review\n\n' + original + '\n'
    comment = {'target': 'confidential_ac_comment.md', 'revision': 0, 'intro': '', 'closing': '',
               'sections': [{'id': 'comment', 'lead': ''}], 'reply_topics': [],
               'fragments': [{'id': 'C1', 'text': '', 'notes': 'Optional: determine whether a factual correction or a concise synthesis would help the AC/SPC. Verify the field limit and readers.',
                              'include': True, 'needs_check': True, 'joiner': ' ', 'section': 'comment'}]}
    outputs.update({'workspace.json': json_text(config), 'annotations.json': json_text(annotations),
                    'response_fragments.json': json_text(responses),
                    'confidential_comment.json': json_text({'version': 1, 'reviews': {'confidential_ac': comment}})})
    inventory = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in ('.pdf', '.zip', '.json', '.tex'):
            inventory.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    outputs['inputs_manifest.json'] = json_text({'created_utc': datetime.now(timezone.utc).isoformat(), 'files': inventory})
    for path in (PRIMER / 'templates').glob('*.md'):
        outputs[path.name] = path.read_text(encoding='utf-8')
    primer_relative = Path(os.path.relpath(PRIMER, root)).as_posix()
    outputs['start_workspace.cmd'] = '@echo off\ncd /d "%~dp0"\necho Keep this window open while editing. Saving writes local files; Git is separate.\npython -X utf8 "' + primer_relative.replace('/', '\\') + '\\workspace.py" serve --open\nif errorlevel 1 pause\n'
    # Preflight every output before writing any of the paper's editable data.
    collisions = [name for name in outputs if (root / name).exists()]
    if collisions:
        raise ValueError('Initialization never overwrites existing files: ' + ', '.join(collisions))
    for name, text in outputs.items():
        write_new(root / name, text)
    print(f'Initialized {root}: {human} human review(s), {ai} AI review(s).')
    print('Imported passages need author-written summaries and verified answers; no rebuttal bodies were invented.')


def engine_module(root, name):
    os.environ['REBUTTAL_WORKSPACE_ROOT'] = str(root)
    sys.path.insert(0, str(RUNTIME))
    return importlib.import_module(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=PRIMER.parent, help='paper workspace (default: parent of primer)')
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='import reviews and create editable, explicitly unfinished scaffolds')
    for key in ('source', 'title', 'short-title', 'venue', 'submission'):
        init.add_argument('--' + key, required=True)
    init.add_argument('--limit', type=int, default=1500)
    init.add_argument('--deadline', default='Confirm the response deadline and timezone')
    init.add_argument('--port', type=int, default=0)
    build = commands.add_parser('build', help='build HTML and export only verified response text')
    build.add_argument('--check', action='store_true')
    serve = commands.add_parser('serve', help='open the local editor and save edits to this paper workspace')
    serve.add_argument('--open', action='store_true')
    serve.add_argument('--port', type=int)
    commands.add_parser('check', help='validate mappings, draft coverage, exact exports and generated pages')
    args = parser.parse_args()
    root = args.workspace.resolve()
    if not root.is_dir():
        raise ValueError('The paper workspace directory must already exist')
    if root not in PRIMER.parents:
        raise ValueError('Copy the primer folder inside the paper workspace before initializing or serving it')
    if args.command == 'init':
        initialize(args, root)
        sys.argv = ['build_review_pages.py']
        engine_module(root, 'build_review_pages').main()
        return 0
    if args.command in ('build', 'check'):
        context = engine_module(root, 'workspace_context')
        current = root / context.CONFIG['source']['reviews_json']
        if hashlib.sha256(current.read_bytes()).hexdigest() != context.CONFIG['source']['sha256']:
            raise ValueError('Original review export changed. Preserve the old file and reconcile note IDs/mappings before deliberately updating its recorded hash')
        sys.argv = [str(RUNTIME / 'build_review_pages.py')] + (['--check'] if args.command == 'check' or args.check else [])
        engine_module(root, 'build_review_pages').main()
        assembly = engine_module(root, 'assemble_rebuttals')
        check = args.command == 'check' or args.check
        unfinished = []
        for stem, review in assembly.load_workspace()['reviews'].items():
            result = assembly.export_review(review, check=check)
            print(f'{stem}: {result["characters"]} characters; {len(result["included"])}/{len(review["fragments"])} concerns selected')
            for issue in result['issues']:
                print('  ' + issue)
            if result['omitted']:
                print('  Omitted: ' + ', '.join(result['omitted']))
            if result['pending']:
                print('  Needs evidence: ' + ', '.join(result['pending']))
            if result['issues']:
                unfinished.append(stem)
        if context.CONFIDENTIAL['enabled']:
            comment = next(iter(engine_module(root, 'confidential_workspace').load_confidential()['reviews'].values()))
            result = assembly.export_review(comment, check=check, limit=context.CONFIDENTIAL_LIMIT)
            print('Confidential comment: ' + ('ready' if not result['issues'] else '; '.join(result['issues'])))
        if args.command == 'check' and unfinished:
            print('Drafts are unfinished. The editor remains usable; final submission checks have not passed.')
            return 1
        return 0
    if args.command == 'serve':
        module = engine_module(root, 'serve_workspace')
        context = engine_module(root, 'workspace_context')
        sys.argv = ['serve_workspace.py', '--port', str(args.port if args.port is not None else context.CONFIG.get('port', 0))] + (['--open'] if args.open else [])
        module.main()
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print('Workspace error:', error, file=sys.stderr)
        raise SystemExit(2)
