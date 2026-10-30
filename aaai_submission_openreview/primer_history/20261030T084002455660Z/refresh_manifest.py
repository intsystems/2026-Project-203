"""Record the reviewed primer files for copying and future updates.

Does not include paper data, cache files or generated test output.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = '0.1.0'


def refresh():
    files = {}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name == 'primer_manifest.json' or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'schema_version': 1, 'version': VERSION, 'files': files}
    (ROOT / 'primer_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Primer {VERSION}: {len(files)} files recorded.')


if __name__ == '__main__':
    refresh()
