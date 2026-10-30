"""Serve the local response editor and save edits into this folder.

python aaai_submission_openreview/serve_workspace.py --open
"""

import argparse
import copy
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit
import webbrowser

from assemble_rebuttals import (ROOT, WORKSPACE, atomic_text, compile_review,
                               export_review, load_workspace, validate_review)
from confidential_workspace import FILENAME, STEM, load_confidential

from workspace_context import RESPONSE_LIMIT, CONFIDENTIAL_LIMIT, CONFIDENTIAL

LOCK = threading.Lock()


class WorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, root=ROOT):
        self.root = Path(root).resolve()
        self.workspace_file = self.root / WORKSPACE.name
        super().__init__(address, partial(WorkspaceHandler, directory=str(self.root)))


class WorkspaceHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        if self.path.startswith('/api/') or args and str(args[1]) != '200':
            super().log_message(format, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        super().end_headers()

    def json_response(self, status, value):
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def allowed_host(self):
        host = self.headers.get("Host", "")
        expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        return host in expected

    def review_stem(self):
        path = urlsplit(self.path).path
        return path[len('/api/reviews/'):] if path.startswith('/api/reviews/') else None

    def workspace_for(self, stem):
        if stem == STEM and CONFIDENTIAL["enabled"]:
            path = self.server.root / FILENAME
            return load_confidential(path), path, CONFIDENTIAL_LIMIT
        return load_workspace(self.server.workspace_file), self.server.workspace_file, RESPONSE_LIMIT

    def do_GET(self):
        if not self.allowed_host():
            self.json_response(403, {"error": "Use the local workspace address."})
            return
        stem = self.review_stem()
        if stem is not None:
            with LOCK:
                workspace, _, limit = self.workspace_for(stem)
                review = workspace["reviews"].get(stem)
                if review is None:
                    self.json_response(404, {"error": "Unknown review."})
                    return
                self.json_response(200, {"review": review, "result": compile_review(review, limit=limit), "editor_format": 3})
            return
        if self.path == '/':
            self.send_response(302)
            self.send_header("Location", "/reviews/index.html")
            self.end_headers()
            return
        super().do_GET()

    def do_PUT(self):
        origin = self.headers.get("Origin")
        if (not self.allowed_host() or
                origin is not None and origin != "http://" + self.headers.get("Host", "") or
                self.headers.get("X-Rebuttal-Workspace") != "1" or
                self.headers.get("Content-Type", "").split(';')[0] != "application/json"):
            self.json_response(403, {"error": "Save edits from the local response editor."})
            return
        stem = self.review_stem()
        if stem is None:
            self.json_response(404, {"error": "Unknown endpoint."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1_000_000:
                raise ValueError("Unexpected request size")
            incoming = json.loads(self.rfile.read(size))
            with LOCK:
                workspace, workspace_file, limit = self.workspace_for(stem)
                current = workspace["reviews"].get(stem)
                if current is None:
                    self.json_response(404, {"error": "Unknown review."})
                    return
                if incoming.get("base_revision") != current["revision"]:
                    self.json_response(409, {"error": "This review was changed in another tab or on disk. Your edits are still in this browser; reconcile them before saving.",
                                              "review": current})
                    return
                changed = incoming.get("review", {})
                updated = self.merge_edit(current, changed)
                validate_review(updated)
                updated["revision"] += 1
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                history = self.server.root / "response_history" / stem / f"{current['revision']:05d}-{stamp}.json"
                atomic_text(history, json.dumps(current, ensure_ascii=False, indent=2) + "\n")
                workspace["reviews"][stem] = updated
                atomic_text(workspace_file, json.dumps(workspace, ensure_ascii=False, indent=2) + "\n")
                result = export_review(updated, root=self.server.root, limit=limit)
                self.json_response(200, {"revision": updated["revision"], "result": result})
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            self.json_response(400, {"error": str(error)})
        except OSError as error:
            self.json_response(500, {"error": "Could not save the files: " + str(error)})

    @staticmethod
    def merge_edit(current, changed):
        updated = copy.deepcopy(current)
        # Paths, concern IDs and section IDs are defined on disk, never by a request.
        for field in ("intro", "closing"):
            updated[field] = changed[field]
        if "frame_variants" in changed:
            expected = current.get("frame_variants", {})
            incoming = changed["frame_variants"]
            if not isinstance(incoming, dict) or set(incoming) != set(expected):
                raise ValueError("Wording option groups changed")
            for field, options in expected.items():
                if [item["id"] for item in incoming[field]] != [item["id"] for item in options]:
                    raise ValueError("Wording option IDs or order changed")
                for option, edit in zip(updated["frame_variants"][field], incoming[field]):
                    option["text"] = edit["text"]
                    option["label"] = edit["label"]
        if [item["id"] for item in changed["sections"]] != [item["id"] for item in current["sections"]]:
            raise ValueError("Topic IDs or order changed")
        for section, edit in zip(updated["sections"], changed["sections"]):
            section["lead"] = edit["lead"]
        if [item["id"] for item in changed["fragments"]] != [item["id"] for item in current["fragments"]]:
            raise ValueError("Concern IDs or order changed")
        for fragment, edit in zip(updated["fragments"], changed["fragments"]):
            for field in ("text", "notes", "joiner", "include", "needs_check", "section"):
                fragment[field] = edit[field]
        if "reply_topics" in changed:
            expected = current.get("reply_topics", [])
            incoming = changed["reply_topics"]
            if not isinstance(incoming, list) or [item["id"] for item in incoming] != [item["id"] for item in expected]:
                raise ValueError("Feedback topic IDs or order changed")
            for topic, edit in zip(updated["reply_topics"], incoming):
                if topic.get('occurrence') != edit.get('occurrence'):
                    raise ValueError('Reviewer question occurrence cannot change')
                for field in ("points", "question", "source"):
                    if topic[field] != edit[field]:
                        raise ValueError("Reviewer wording and feedback links cannot change")
                topic["heading"] = edit["heading"]
        # Older browser backups have no linked layout; they use concern order.
        updated.pop("draft", None)
        if "draft" in changed:
            updated["draft"] = copy.deepcopy(changed["draft"])
        return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--open", action="store_true", help="open the review navigator")
    args = parser.parse_args()
    workspace = load_workspace()
    for review in workspace["reviews"].values():
        result = export_review(review)
        if result["issues"]:
            print(review['target'] + ': draft needs preparation; see coverage and evidence flags in the editor.', flush=True)
    for review in (load_confidential()["reviews"] if CONFIDENTIAL["enabled"] else {}).values():
        export_review(review, limit=CONFIDENTIAL_LIMIT)
    server = WorkspaceServer(("127.0.0.1", args.port))
    url = f"http://127.0.0.1:{server.server_port}/reviews/index.html"
    print("Response workspace: " + url, flush=True)
    print("Edits save to response_fragments.json or confidential_comment.json; valid drafts save to rebuttals/. Ctrl+C stops the server.", flush=True)
    print("Keep this window open while editing in the browser. Save to repo writes local files; Git commit/push is separate.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
