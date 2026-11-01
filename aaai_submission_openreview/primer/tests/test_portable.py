"""Exercise transfer, variable reviews/rules, saving and non-destructive updates.

python primer/tests/test_portable.py
All mutations and local servers use temporary synthetic paper folders.
"""
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1]


def run(root, *args):
    return subprocess.run([sys.executable, '-X', 'utf8', str(root/'primer'/'workspace.py'), *args],
                          capture_output=True, text=True, encoding='utf-8', timeout=25)


def notes(humans, ai=False, duplicate=False):
    output = []
    for i in range(1, humans + 1):
        question = 'How does the stated assumption affect the result?'
        if duplicate and i == 1:
            question += '\n\n' + question
        output.append({'id': f'note{i}', 'number': i,
                       'signatures': [f'Venue/Program_Committee_Rev{i}'],
                       'invitations': ['Venue/Official_Review'],
                       'content': {'title': {'value': f'Review {i}'}, 'review': {'value': 'A useful contribution.'},
                                   'weaknesses': {'value': 'Clarify the assumption and give evidence.'},
                                   'feedback': {'value': question}, 'rating': {'value': '6'}, 'confidence': {'value': '3'}}})
    if ai:
        output.append({'id': 'ai_note', 'number': 1, 'signatures': ['Venue/AI_Reviewer'],
                       'invitations': ['Venue/AI_Review'],
                       'content': {'title': 'AI review', 'review': '**Assessment**\n\nThe contribution is clear.\n\n**Questions**\n\nPlease explain the mathematical bound.'}})
    # Non-review notes must not turn into response editors.
    output.append({'id': 'comment_note', 'number': 99, 'invitations': ['Venue/Official_Comment'], 'content': {'comment': 'Administrative update'}})
    return {'notes': output}


class PortableChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='portable-rebuttal-')
        self.root = Path(self.temp.name)
        shutil.copytree(SOURCE, self.root/'primer', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        result = subprocess.run([sys.executable, str(self.root/'primer'/'refresh_manifest.py')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def tearDown(self):
        self.temp.cleanup()

    def initialize(self, humans=2, ai=True, duplicate=False, limit=700, name='Paper Alpha'):
        (self.root/'reviews.json').write_text(json.dumps(notes(humans, ai, duplicate)), encoding='utf-8')
        result = run(self.root, 'init', '--source', 'reviews.json', '--title', name, '--short-title', name,
                     '--venue', 'Test Conference', '--submission', '123', '--limit', str(limit))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads((self.root/'workspace.json').read_text(encoding='utf-8'))

    def test_two_papers_different_review_counts_limits_and_identifiers(self):
        config = self.initialize(humans=5, ai=True, limit=900)
        self.assertEqual(len(config['reviews']), 6)
        self.assertEqual(config['rules']['response_character_limit'], 900)
        html = (self.root/'reviews'/'index.html').read_text(encoding='utf-8')
        self.assertIn('<strong>6</strong>reviews', html)
        first = config['reviews'][0]
        page = (self.root/'reviews'/(first['stem']+'.html')).read_text(encoding='utf-8')
        self.assertIn('Paper Alpha', page)
        self.assertIn('900-character limit', page)
        self.assertIn('../primer/runtime/review_browser/vendor/katex/', page)
        self.assertIn(config['workspace_id'], page)
        self.assertNotIn('ImprovEvolve', page)
        self.assertNotIn('nmgc', page)
        self.assertFalse((self.root/'rebuttals').exists(), 'Unverified scaffold was exported')
        with tempfile.TemporaryDirectory(prefix='portable-paper-beta-') as other:
            beta = Path(other)
            shutil.copytree(self.root/'primer', beta/'primer')
            (beta/'reviews.json').write_text(json.dumps(notes(1)), encoding='utf-8')
            result = run(beta, 'init', '--source', 'reviews.json', '--title', 'Paper Beta', '--short-title', 'Beta', '--venue', 'OtherConf', '--submission', '987', '--limit', '321')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            second = json.loads((beta/'workspace.json').read_text(encoding='utf-8'))
            self.assertEqual(len(second['reviews']), 1)
            self.assertNotEqual(config['workspace_id'], second['workspace_id'])
            self.assertIn('321-character limit', (beta/'reviews'/(second['reviews'][0]['stem']+'.html')).read_text(encoding='utf-8'))

    def test_duplicate_quotes_and_source_hash_guard(self):
        self.initialize(duplicate=True)
        result = run(self.root, 'build', '--check')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        (self.root/'reviews.json').write_text('{}', encoding='utf-8')
        result = run(self.root, 'check')
        self.assertEqual(result.returncode, 2)
        self.assertIn('Original review export changed', result.stderr)

    def test_init_refuses_to_overwrite_and_unfinished_check_fails(self):
        self.initialize()
        path = self.root/'response_fragments.json'
        before = path.read_bytes()
        result = run(self.root, 'init', '--source', 'reviews.json', '--title', 'Another', '--short-title', 'Other', '--venue', 'X', '--submission', '1')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(path.read_bytes(), before)
        result = run(self.root, 'check')
        self.assertEqual(result.returncode, 1)
        self.assertIn('Drafts are unfinished', result.stdout)

    def test_configured_limit_confidential_limit_save_conflict_and_backup(self):
        config = self.initialize(humans=1, ai=False, limit=250)
        config['rules']['confidential_character_limit'] = 100
        config['rules']['confidential_title_character_limit'] = 250
        config['confidential']['export_title'] = 'Private decision summary'
        config['confidential']['references'] = [{'stem': config['reviews'][0]['stem'], 'point': 'R1.1', 'label': 'Rev1 question 1'}]
        (self.root/'workspace.json').write_text(json.dumps(config), encoding='utf-8')
        result = run(self.root, 'build')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        chair_html = (self.root/'reviews'/'confidential_ac.html').read_text(encoding='utf-8')
        self.assertIn('Rev1 question 1', chair_html)
        self.assertIn('"limit": 100', chair_html)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        process = subprocess.Popen([sys.executable, '-X', 'utf8', str(self.root/'primer'/'workspace.py'), 'serve', '--port', str(port)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            for _ in range(80):
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.2):
                        break
                except OSError:
                    time.sleep(.05)
            def request(method, stem, payload=None):
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
                connection.request(method, '/api/reviews/'+stem, body=json.dumps(payload) if payload is not None else None,
                                   headers={'Content-Type':'application/json','X-Rebuttal-Workspace':'1'})
                response = connection.getresponse()
                status, data = response.status, json.loads(response.read())
                connection.close()
                return status, data
            stem = config['reviews'][0]['stem']
            status, data = request('GET', stem)
            self.assertEqual(status, 200)
            self.assertEqual(data['result']['limit'], 250)
            review = data['review']
            review['intro'] = 'Thank you.'
            for item in review['fragments']:
                item['include'] = False
            review['fragments'][0].update(text='The paper proves the stated claim under its explicit assumption.', include=True, needs_check=False)
            review['reply_topics'][0]['heading'] = '1. Re the assumption:'
            status, saved = request('PUT', stem, {'base_revision': review['revision'], 'review': review})
            self.assertEqual(status, 200)
            self.assertTrue(saved['result']['written'])
            target = self.root/'rebuttals'/review['target']
            valid = target.read_bytes()
            status, _ = request('PUT', stem, {'base_revision': review['revision'], 'review': review})
            self.assertEqual(status, 409)
            review['revision'] = saved['revision']
            review['fragments'][0]['text'] = 'x'*251
            status, saved = request('PUT', stem, {'base_revision': review['revision'], 'review': review})
            self.assertEqual(status, 200)
            self.assertFalse(saved['result']['written'])
            self.assertEqual(target.read_bytes(), valid)
            self.assertTrue(list((self.root/'response_history'/stem).glob('*.json')))
            _, chair = request('GET', 'confidential_ac')
            self.assertEqual(chair['result']['limit'], 100)
            comment = chair['review']
            comment['fragments'][0].update(text='Private synthesis for the chairs.', needs_check=False)
            status, saved = request('PUT', 'confidential_ac', {'base_revision': comment['revision'], 'review': comment})
            self.assertEqual(status, 200)
            self.assertTrue(saved['result']['written'])
            self.assertEqual(saved['result']['characters'], len(saved['result']['body']))
            self.assertTrue((self.root/'rebuttals'/comment['target']).read_text(encoding='utf-8').startswith('# Private decision summary\n\n'))
        finally:
            process.terminate()
            process.wait(timeout=8)

    def test_update_archives_primer_and_preserves_author_data(self):
        self.initialize()
        marker = self.root/'response_fragments.json'
        before = marker.read_bytes()
        with tempfile.TemporaryDirectory(prefix='portable-primer-update-') as location:
            upstream = Path(location)/'primer'
            shutil.copytree(SOURCE, upstream, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            result = subprocess.run([sys.executable, str(upstream/'refresh_manifest.py')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            changed = self.root/'primer'/'README.md'
            changed.write_text('Locally edited primer guide.', encoding='utf-8')
            result = subprocess.run([sys.executable, str(upstream/'install.py'), '--destination', str(self.root/'primer'), '--update'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(marker.read_bytes(), before)
            archives = list((self.root/'primer_history').glob('*/README.md'))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_text(encoding='utf-8'), 'Locally edited primer guide.')

    def test_confidential_title_survives_rebuild_with_separate_limits(self):
        config = self.initialize(humans=1, ai=False, limit=321, name='Title Test')
        config['rules'].update(confidential_character_limit=40, confidential_title_character_limit=10)
        config['confidential']['export_title'] = 'A title 😀'
        (self.root/'workspace.json').write_text(json.dumps(config), encoding='utf-8')
        comments = json.loads((self.root/'confidential_comment.json').read_text(encoding='utf-8'))
        review = comments['reviews']['confidential_ac']
        fragment = review['fragments'][0]
        review['draft'] = [{'kind': 'answer', 'points': [fragment['id']], 'before': '', 'after': ''}]
        fragment.update(text='x'*40, needs_check=False)
        path = self.root/'confidential_comment.json'
        path.write_text(json.dumps(comments), encoding='utf-8')
        target = self.root/'rebuttals'/config['confidential']['target']
        result = run(self.root, 'build')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = '# A title 😀\n\n' + 'x'*40
        self.assertEqual(target.read_text(encoding='utf-8'), expected)
        self.assertIn('Short title', (self.root/'reviews'/'confidential_ac.html').read_text(encoding='utf-8'))
        fragment['text'] += 'x'
        path.write_text(json.dumps(comments), encoding='utf-8')
        self.assertEqual(run(self.root, 'build').returncode, 0)
        self.assertEqual(target.read_text(encoding='utf-8'), expected)
        result = run(self.root, 'check')
        self.assertIn('exceeds 40', result.stdout)
        config['confidential']['export_title'] += 'x'
        (self.root/'workspace.json').write_text(json.dumps(config), encoding='utf-8')
        result = run(self.root, 'build')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('title exceeds', result.stderr)
        self.assertEqual(target.read_text(encoding='utf-8'), expected)

    def test_final_check_rejects_invalid_confidential_text_but_allows_empty_optional_field(self):
        config = self.initialize(humans=1, ai=False)
        config['rules']['confidential_character_limit'] = 100
        (self.root/'workspace.json').write_text(json.dumps(config), encoding='utf-8')
        path = self.root/'response_fragments.json'
        workspace = json.loads(path.read_text(encoding='utf-8'))
        for review in workspace['reviews'].values():
            for fragment in review['fragments']:
                fragment['include'] = False
            review['fragments'][0].update(text='The theorem uses the stated assumption.', include=True, needs_check=False)
            review['reply_topics'][0]['heading'] = '1. Re the assumption:'
        path.write_text(json.dumps(workspace), encoding='utf-8')
        comment_path = self.root/'confidential_comment.json'
        comments = json.loads(comment_path.read_text(encoding='utf-8'))
        comment = comments['reviews']['confidential_ac']['fragments'][0]
        comment.update(text='x'*101, needs_check=False)
        comment_path.write_text(json.dumps(comments), encoding='utf-8')
        self.assertEqual(run(self.root, 'build').returncode, 0)
        self.assertEqual(run(self.root, 'check').returncode, 1)
        comment.update(text='', needs_check=True)
        comment_path.write_text(json.dumps(comments), encoding='utf-8')
        self.assertEqual(run(self.root, 'build').returncode, 0)
        result = run(self.root, 'check')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
