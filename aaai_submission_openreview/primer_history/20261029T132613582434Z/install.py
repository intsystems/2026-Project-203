"""Copy or update only the primer software, preserving all paper workspace data.

python primer/install.py --destination C:/path/to/paper/aaai_submission_openreview/primer
Add --update for an existing primer. A complete prior copy is archived first.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parent


def checked_child(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents or Path(relative).is_absolute():
        raise ValueError('Unsafe primer manifest path: ' + relative)
    return path


def install(destination, update=False):
    destination = destination.resolve()
    if destination == SOURCE or SOURCE in destination.parents or destination in SOURCE.parents:
        raise ValueError('Source and destination primer folders must be separate')
    if destination.name != 'primer':
        raise ValueError('The destination must be the paper workspace / primer folder')
    manifest = json.loads((SOURCE / 'primer_manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = checked_child(SOURCE, relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Primer source differs from its manifest: ' + relative + '. Run refresh_manifest.py after reviewing engine changes')
    if destination.exists():
        if not update:
            raise ValueError('Destination exists; inspect it and use --update to archive the previous primer before replacing software')
        if not (destination / 'primer_manifest.json').is_file():
            raise ValueError('Destination has no primer manifest; refusing to treat other files as an installed primer')
        if destination.is_symlink() or any(path.is_symlink() for path in destination.rglob('*')):
            raise ValueError('Review symbolic links in the existing primer before updating')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        history = destination.parent / 'primer_history' / stamp
        shutil.copytree(destination, history)
        print('Archived previous primer:', history)
    else:
        destination.mkdir(parents=True)
    for relative in [*manifest['files'], 'primer_manifest.json']:
        source = checked_child(SOURCE, relative)
        target = checked_child(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f'Installed primer {manifest["version"]}: {destination}')
    print('Paper JSON, summaries, answers, exports and their history are outside the primer and were not changed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--update', action='store_true')
    args = parser.parse_args()
    try:
        install(args.destination, args.update)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, 'Install error: ' + str(error) + '\n')
