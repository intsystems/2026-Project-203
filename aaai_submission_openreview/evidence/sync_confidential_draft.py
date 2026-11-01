"""Import the approved Markdown into the editor. Dry-run unless --apply is given."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error('Choose --apply or --dry-run')
    config_path = ROOT / 'workspace.json'
    source_path = ROOT / 'confidential_comment.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    source = json.loads(source_path.read_text(encoding='utf-8'))
    target = ROOT / 'rebuttals' / config['confidential']['target']
    raw = target.read_bytes()
    heading, body = raw.decode('utf-8').split('\n\n', 1)
    if not heading.startswith('# ') or '\n' in heading:
        raise ValueError('Expected a single H1 title followed by a blank line')
    title = heading[2:]
    count = lambda value: len(value.encode('utf-16-le')) // 2
    title_limit = config['rules'].get('confidential_title_character_limit', 250)
    body_limit = config['rules']['confidential_character_limit']
    if count(title) > title_limit or count(body) > body_limit:
        raise ValueError('Approved title or body exceeds its separate field limit')
    stem = config['confidential']['stem']
    review = source['reviews'][stem]
    if len(review['fragments']) != 1 or review['reply_topics']:
        raise ValueError('Import requires the existing single-fragment comment')
    print(f'Title: {count(title)}/{title_limit}; body: {count(body)}/{body_limit}')
    print('Approved Markdown SHA-256:', hashlib.sha256(raw).hexdigest())
    if not args.apply:
        print('Dry-run: no files changed. Use --apply after reviewing these counts.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = ROOT / 'response_history' / ('before_confidential_sync_' + stamp)
    backup.mkdir(parents=True)
    for path in (config_path, source_path, target):
        shutil.copy2(path, backup / path.name)
    history = ROOT / 'response_history' / stem
    history.mkdir(parents=True, exist_ok=True)
    (history / f"{review['revision']:05d}-{stamp}.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    config['confidential']['export_title'] = title
    config['rules']['confidential_title_character_limit'] = title_limit
    review['revision'] += 1
    review.update(intro='', closing='')
    review['draft'] = [{'kind': 'answer', 'points': [review['fragments'][0]['id']], 'before': '', 'after': ''}]
    review['sections'][0]['lead'] = ''
    fragment = review['fragments'][0]
    fragment.update(text=body, include=True, needs_check=False, joiner='')
    fragment['notes'] += '\nApproved Markdown imported with separate title/body limits; previous drafts preserved in response_history.'
    sys.path.insert(0, str(ROOT / 'primer' / 'runtime'))
    from assemble_rebuttals import atomic_text
    for path, data in ((config_path, config), (source_path, source)):
        atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    if target.read_bytes() != raw:
        raise ValueError('The approved Markdown changed during import')
    print('Imported; backup:', backup.relative_to(ROOT).as_posix())


if __name__ == '__main__':
    main()
