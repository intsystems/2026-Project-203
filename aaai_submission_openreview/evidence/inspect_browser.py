"""Read-only Edge inspection of this paper's actual generated workspace."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import build_opener, ProxyHandler, Request

import websocket

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'primer/tests'))
from check_browser import Browser


def main():
    edge=Path(os.environ.get('PROGRAMFILES(X86)',r'C:\Program Files (x86)'))/'Microsoft/Edge/Application/msedge.exe'
    assert edge.is_file()
    config=json.loads((BASE/'workspace.json').read_text(encoding='utf-8'))
    sources=[BASE/'response_fragments.json',BASE/'annotations.json',BASE/'confidential_comment.json',BASE/'workspace.json']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    opener=build_opener(ProxyHandler({}))
    with tempfile.TemporaryDirectory(prefix='signmuon-rebuttal-browser-') as directory:
        temporary=Path(directory).resolve()
        # The context manager's recursive cleanup is confined to this named temp directory.
        assert Path(tempfile.gettempdir()).resolve() in temporary.parents
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
        flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
        server=subprocess.Popen([sys.executable,'-X','utf8',str(BASE/'primer/workspace.py'),'serve','--port',str(port)],
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=flags)
        browser_process=connection=None
        try:
            for _ in range(100):
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.2):break
                except OSError:time.sleep(.05)
            profile=temporary/'profile'
            browser_process=subprocess.Popen([str(edge),'--headless=new','--no-sandbox','--disable-features=RendererCodeIntegrity','--disable-gpu','--in-process-gpu',
                '--no-first-run','--disable-extensions','--disable-background-networking','--disable-sync',
                '--disable-component-update','--remote-debugging-port=0','--remote-allow-origins=http://localhost',
                '--user-data-dir='+str(profile),'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=flags)
            port_file=profile/'DevToolsActivePort'
            for _ in range(150):
                if port_file.is_file():break
                time.sleep(.1)
            debug_port=int(port_file.read_text(encoding='utf-8').splitlines()[0])
            target=json.loads(opener.open(Request(f'http://127.0.0.1:{debug_port}/json/new?about:blank',method='PUT'),timeout=5).read())
            connection=websocket.create_connection(target['webSocketDebuggerUrl'],timeout=15,
                                                  origin='http://localhost',http_no_proxy=['localhost','127.0.0.1'])
            browser=Browser(connection)
            browser.call('Page.enable');browser.call('Runtime.enable')
            browser.call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1000,deviceScaleFactor=1,mobile=False))
            report=[]
            for spec in config['reviews']+[dict(stem='confidential_ac',target='confidential_ac_comment.md')]:
                browser.load(f'http://127.0.0.1:{port}/reviews/{spec["stem"]}.html#responses')
                browser.wait("document.querySelector('#rendered-draft') && document.querySelector('#rendered-draft').textContent.trim().length > 0")
                data=browser.evaluate(r'''(() => {
                    const source=document.getElementById('assembled-text').value;
                    const links=[...document.querySelectorAll('a[href^="#"]')];
                    return {source,characters:source.length,countLabel:document.getElementById('draft-count').textContent,
                        renderedMath:document.querySelectorAll('#rendered-draft .katex').length,
                        draftMathErrors:[...document.querySelectorAll('#rendered-draft .katex-error')].map(x=>x.textContent),
                        sourceMathErrors:[...document.querySelectorAll('.katex-error')].map(x=>x.textContent),
                        brokenAnchors:links.filter(a=>a.hash && !document.getElementById(decodeURIComponent(a.hash.slice(1)))).map(a=>a.hash),
                        selected:[...document.querySelectorAll('[data-field="include"]')].filter(x=>x.checked).length,
                        pending:[...document.querySelectorAll('[data-field="needs_check"]')].filter(x=>x.checked).map(x=>x.dataset.point),
                        overflow:document.documentElement.scrollWidth>innerWidth,
                        saveStatus:document.getElementById('save-status').textContent};
                })()''')
                expected=(BASE/'rebuttals'/spec['target']).read_text(encoding='utf-8')
                assert data.pop('source')==expected,spec['stem']
                assert not data['draftMathErrors'] and not data['brokenAnchors'],data
                if spec['stem']!='confidential_ac':assert data['characters']<=1500
                assert not data['overflow'],data
                browser.call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
                data['mobileOverflow']=browser.evaluate('document.documentElement.scrollWidth>innerWidth')
                assert not data['mobileOverflow'],data
                browser.call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1000,deviceScaleFactor=1,mobile=False))
                if spec['stem']=='review_R3_Ueq5':
                    browser.evaluate("document.getElementById('rendered-draft').scrollIntoView({block:'start'})")
                    screenshot=browser.call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
                    (BASE/'evidence/browser_R3.png').write_bytes(base64.b64decode(screenshot['data']))
                report.append(dict(stem=spec['stem'],**data))
            browser.call('Page.navigate',{'url':f'http://127.0.0.1:{port}/reviews/index.html'})
            browser.wait("document.readyState==='complete'")
            index=browser.evaluate("({title:document.title,reviewLinks:[...document.querySelectorAll('a')].map(a=>a.getAttribute('href')),overflow:document.documentElement.scrollWidth>innerWidth})")
            assert not browser.errors,browser.errors
            after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
            assert before==after,'Read-only inspection changed author data'
            output=dict(checked_utc=datetime.now(timezone.utc).isoformat(),pages=report,index=index,
                        runtime_exceptions=browser.errors,author_data_unchanged=True,
                        scope='Actual generated pages, draft/export equality, UTF-16 counts, draft math, links, desktop/mobile layout. No edits or saves.')
            (BASE/'evidence/browser_inspection.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print('PASS:',len(report),'actual editor pages; exact exports, counts, rendered math, local anchors, desktop/mobile layout, no JS exceptions; author data unchanged.')
            for row in report:print(row['stem'],row['characters'],'chars;',row['renderedMath'],'draft math expressions;',len(row['sourceMathErrors']),'source math errors')
        finally:
            if connection:connection.close()
            if browser_process:
                browser_process.terminate();browser_process.wait(timeout=15)
            server.terminate();server.wait(timeout=10)


if __name__=='__main__':
    main()
