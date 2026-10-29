"""Optional relocated-browser smoke test; requires Edge and websocket-client.

python primer/tests/check_browser.py
Runtime users do not need this development dependency. All edits target temp files.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import build_opener, ProxyHandler, Request

import websocket
from test_portable import SOURCE, notes, run


class Browser:
    def __init__(self, connection):
        self.connection, self.serial, self.errors = connection, 0, []

    def call(self, method, params=None):
        self.serial += 1
        serial = self.serial
        self.connection.send(json.dumps({'id': serial, 'method': method, 'params': params or {}}))
        while True:
            result = json.loads(self.connection.recv())
            if result.get('method') == 'Runtime.exceptionThrown':
                self.errors.append(result['params'])
            if result.get('id') != serial:
                continue
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result.get('result', {})

    def evaluate(self, expression):
        result = self.call('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True})
        if 'exceptionDetails' in result:
            raise AssertionError(result['exceptionDetails'])
        return result['result'].get('value')

    def wait(self, expression):
        for _ in range(100):
            if self.evaluate(expression):
                return
            time.sleep(.08)
        raise AssertionError('Browser condition timed out: ' + expression)

    def load(self, url):
        self.call('Page.navigate', {'url': url})
        self.wait("document.readyState === 'complete' && Boolean(document.getElementById('assembled-text')) && !document.getElementById('assembled-text').disabled")


def main():
    edge = Path(os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)'))/'Microsoft'/'Edge'/'Application'/'msedge.exe'
    if not edge.is_file():
        raise RuntimeError('Installed Edge not found')
    with tempfile.TemporaryDirectory(prefix='portable-browser-') as directory:
        root = Path(directory)
        paper = root/'paper_data'
        paper.mkdir()
        shutil.copytree(SOURCE, paper/'primer', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (paper/'reviews.json').write_text(json.dumps(notes(2, ai=True)), encoding='utf-8')
        result = run(paper, 'init', '--source', 'reviews.json', '--title', 'Relocated Paper', '--short-title', 'Relocated', '--venue', 'OtherConf', '--submission', '321', '--limit', '380')
        assert result.returncode == 0, result.stdout + result.stderr
        config = json.loads((paper/'workspace.json').read_text(encoding='utf-8'))
        stem = config['reviews'][0]['stem']
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        server = subprocess.Popen([sys.executable, '-X', 'utf8', str(paper/'primer'/'workspace.py'), 'serve', '--port', str(port)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        browser_process = connection = None
        try:
            for _ in range(80):
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.2):
                        break
                except OSError:
                    time.sleep(.05)
            profile = root/'edge_profile'
            browser_process = subprocess.Popen([str(edge), '--headless=new', '--disable-gpu', '--in-process-gpu', '--no-first-run',
                '--disable-extensions', '--disable-background-networking', '--disable-sync', '--disable-component-update',
                '--remote-debugging-port=0', '--remote-allow-origins=http://localhost', '--user-data-dir='+str(profile), 'about:blank'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            port_file = profile/'DevToolsActivePort'
            for _ in range(150):
                if port_file.is_file():
                    break
                time.sleep(.1)
            debug_port = int(port_file.read_text(encoding='utf-8').splitlines()[0])
            opener = build_opener(ProxyHandler({}))
            target = json.loads(opener.open(Request(f'http://127.0.0.1:{debug_port}/json/new?about:blank', method='PUT'), timeout=5).read())
            connection = websocket.create_connection(target['webSocketDebuggerUrl'], timeout=15, origin='http://localhost', http_no_proxy=['localhost','127.0.0.1'])
            browser = Browser(connection)
            browser.call('Page.enable')
            browser.call('Runtime.enable')
            browser.load(f'http://127.0.0.1:{port}/reviews/{stem}.html#responses')
            assert browser.evaluate("JSON.parse(document.getElementById('review-data').textContent).limit") == 380
            browser.evaluate(r'''{
              const data = JSON.parse(document.getElementById('review-data').textContent);
              for (const input of document.querySelectorAll('[data-field="include"]')) {
                input.checked = input.dataset.point === 'R1.1'; input.dispatchEvent(new Event('input',{bubbles:true}));
              }
              const check = document.querySelector('[data-point="R1.1"][data-field="needs_check"]');
              check.checked = false; check.dispatchEvent(new Event('input',{bubbles:true}));
              const heading = document.getElementById('heading-Q1');
              heading.value = '1. Re the assumption:'; heading.dispatchEvent(new Event('input',{bubbles:true}));
              const answer = document.getElementById('answer-R1-1');
              answer.value = 'We verified $x^2 \ge 0$ under the stated assumption.';
              answer.dispatchEvent(new Event('input',{bubbles:true}));
            }''')
            browser.wait("document.getElementById('save-status').textContent.includes('Markdown updated')")
            assert browser.evaluate("Boolean(document.querySelector('#rendered-draft .katex'))")
            assert browser.evaluate("document.getElementById('draft-count').textContent.includes('/ 380')")
            assert browser.evaluate("Object.keys(localStorage).some(key => key.startsWith('rebuttal-v1:'))")
            browser.evaluate("{const full=document.getElementById('assembled-text'); full.value=full.value.replace('We verified','We explicitly verified'); full.dispatchEvent(new Event('input',{bubbles:true}));}")
            browser.wait("document.getElementById('save-status').textContent.includes('Markdown updated')")
            assert browser.evaluate("document.getElementById('answer-R1-1').value.startsWith('We explicitly verified')")
            browser.evaluate("{const select=document.getElementById('variant-intro'); select.value='brief'; select.dispatchEvent(new Event('input',{bubbles:true})); document.querySelector('[data-apply-variant=\"intro\"]').click();}")
            browser.wait("document.getElementById('save-status').textContent.includes('Markdown updated')")
            assert browser.evaluate("document.getElementById('assembled-text').value.startsWith('Thank you for your careful review.')")
            target_file = paper/'rebuttals'/config['reviews'][0]['target']
            valid = target_file.read_bytes()
            browser.evaluate("{const input=document.getElementById('answer-R1-1'); window.previousAnswer=input.value; input.value='x'.repeat(381); input.dispatchEvent(new Event('input',{bubbles:true}));}")
            browser.wait("document.getElementById('save-status').textContent.includes('needs attention')")
            assert target_file.read_bytes() == valid
            browser.evaluate("{const input=document.getElementById('answer-R1-1'); input.value=window.previousAnswer; input.dispatchEvent(new Event('input',{bubbles:true}));}")
            browser.wait("document.getElementById('save-status').textContent.includes('Markdown updated')")
            for width, height, mobile in [(1440,1000,False),(390,844,True)]:
                browser.call('Emulation.setDeviceMetricsOverride', {'width':width,'height':height,'deviceScaleFactor':1,'mobile':mobile})
                assert browser.evaluate('document.documentElement.scrollWidth <= innerWidth')
            # Rebuild to embed saved revision before testing a directly opened page.
            result = run(paper, 'build')
            assert result.returncode == 0, result.stdout + result.stderr
            current = json.loads((paper/'response_fragments.json').read_text(encoding='utf-8'))
            browser.load((paper/'reviews'/(stem+'.html')).as_uri()+'#responses')
            exercise = r'''(async initial => {
              const data = JSON.parse(document.getElementById('review-data').textContent);
              const files = new Map([['response_fragments.json', JSON.stringify(initial)]]);
              const folder = (prefix='') => ({name:data.workspaceFolder,
                async getDirectoryHandle(name) {return folder(prefix+name+'/');},
                async getFileHandle(name, options={}) {
                  const key=prefix+name;
                  if (!files.has(key) && !options.create) throw new Error('Missing');
                  return {async getFile(){return {text:async()=>files.get(key)};},async createWritable(){let pending;return {async write(text){pending=text;},async close(){files.set(key,pending);},async abort(){}};}};
                }});
              window.showDirectoryPicker=async()=>({name:'parent',async getFileHandle(){throw new Error('Select child');},async getDirectoryHandle(name){if(name!==data.workspaceFolder)throw new Error('Wrong child');return folder();}});
              const input=document.getElementById('answer-R1-1'); input.value+=' Native save.'; input.dispatchEvent(new Event('input',{bubbles:true}));
              document.getElementById('save-to-repo').click();
              for(let i=0;i<100 && document.getElementById('save-to-repo').disabled;i++)await new Promise(resolve=>setTimeout(resolve,30));
              if(!document.getElementById('save-status').textContent.includes('Markdown updated'))throw new Error(document.getElementById('save-status').textContent);
              const saved=JSON.parse(files.get('response_fragments.json'));
              if(saved.reviews[data.stem].fragments[0].text!==input.value)throw new Error('Native answer lost');
              for(const stem of Object.keys(initial.reviews))if(stem!==data.stem && JSON.stringify(initial.reviews[stem])!==JSON.stringify(saved.reviews[stem]))throw new Error('Other review changed');
              if(files.get('rebuttals/'+data.response.target)!==document.getElementById('assembled-text').value)throw new Error('Native export differs');
              return true;
            })'''
            assert browser.evaluate(exercise+'('+json.dumps(current)+')')
            assert not browser.errors, browser.errors
            print('Relocated browser checks passed: math, both editors, variants, 380-character limit, autosave, native folder save, 3 reviews, desktop/mobile and separate paper storage.')
            browser.call('Browser.close')
        finally:
            if connection:
                connection.close()
            if browser_process:
                try:
                    browser_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    browser_process.terminate()
                    browser_process.wait(timeout=5)
            server.terminate()
            server.wait(timeout=5)


if __name__ == '__main__':
    main()
