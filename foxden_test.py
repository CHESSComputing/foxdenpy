"""Spins up a tiny local HTTP server that mimics the FOXDEN frontend's
/record and /dm endpoints, then exercises the real foxden package
against it end-to-end.

Mock tree:

    /                     (dataset root)
    |-- 1/
    |   |-- meta/
    |   |   `-- 1/
    |   |       |-- scalars.txt
    |   |       `-- positions.txt
    |   `-- data.txt
    `-- 2/
        `-- info.txt

(4 files total, matching FOXDEN's documented example of fetching
path=/5/meta/1&file=scalars.txt -- same shape, shallower numbers so the
test tree stays small.)
"""
import concurrent.futures
import json
import logging
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

sys.path.insert(0, ".")
import foxden  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")

DID = "/beamline=3a/btr=shanks-4838-a/cycle=2026-1/sample_name=ti7-19-tomo"

RECORD_RESPONSE = {
    "http_code": 200,
    "service_code": 0,
    "service": "DataDiscovery",
    "status": "",
    "error": "",
    "results": {
        "nrecords": 1,
        "records": [
            {
                "did": DID,
                "sample_name": "ti7-19-tomo",
                "beamline": ["3A"],
            }
        ],
    },
}

# path -> listing (mirrors FOXDEN's plain-array /dm response shape).
# File entries here follow the dev-instance convention observed in
# practice: `path` is the FULL path including the filename (not just
# the parent directory), e.g. path="1/meta/1/scalars.json" for a file
# named "scalars.json" -- this is what foxden.dm.resolve_fetch_path()
# has to normalize back into a (dir, file) pair.
LISTINGS = {
    None: [
        {"did": DID, "esc_did": "", "name": "1", "is_dir": True, "path": "1"},
        {"did": DID, "esc_did": "", "name": "2", "is_dir": True, "path": "2"},
    ],
    "1": [
        {"did": DID, "esc_did": "", "name": "meta", "is_dir": True, "path": "1/meta"},
        {"did": DID, "esc_did": "", "name": "data.txt", "is_dir": False, "path": "1/data.txt"},
    ],
    "1/meta": [
        {"did": DID, "esc_did": "", "name": "1", "is_dir": True, "path": "1/meta/1"},
    ],
    "1/meta/1": [
        {"did": DID, "esc_did": "", "name": "scalars.json", "is_dir": False, "path": "1/meta/1/scalars.json"},
        {"did": DID, "esc_did": "", "name": "positions.txt", "is_dir": False, "path": "1/meta/1/positions.txt"},
    ],
    "2": [
        {"did": DID, "esc_did": "", "name": "info.txt", "is_dir": False, "path": "2/info.txt"},
    ],
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default logging
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        assert self.headers.get("Authorization") == "Bearer test-token", self.headers
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/record":
            assert qs.get("did", [""])[0] == DID, qs
            self._json(RECORD_RESPONSE)
            return

        if parsed.path == "/dm":
            assert qs.get("did", [""])[0] == DID, qs
            path = qs.get("path", [None])[0]
            file = qs.get("file", [None])[0]
            if file is not None:
                body = f"CONTENT:{path}/{file}".encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(LISTINGS.get(path, []))
            return

        self.send_response(404)
        self.end_headers()

    def _json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_concurrent_timeout_does_not_hang():
    """Regression test: a slow/unresponsive file among several
    concurrently-downloaded files must not hang the whole batch --
    each request is bounded by its own (connect, read) timeout, and
    each worker thread uses its own Session so one slow request can't
    starve the others."""
    import time

    from foxden.config import FoxdenConfig
    from foxden.client import FoxdenClient
    from foxden.dm import DMClient

    class SlowHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            file = qs.get("file", [None])[0]
            if file == "slow.txt":
                time.sleep(5)  # far longer than the 1s test timeout below
            body = f"CONTENT:{file}".encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    cfg = FoxdenConfig(
        url=f"http://127.0.0.1:{port}",
        connect_timeout=1.0,
        timeout=1.0,  # read timeout much shorter than the 5s sleep above
    )
    http = FoxdenClient(cfg)
    dm = DMClient(http, cfg.url)

    names = ["a.txt", "slow.txt", "b.txt", "c.txt"]
    started = time.monotonic()
    results = {}
    errors = {}

    def fetch(name):
        try:
            results[name] = dm.get_file("did", "path", name)
        except Exception as exc:  # noqa: BLE001
            errors[name] = exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, n) for n in names]
        concurrent.futures.wait(futures, timeout=10)

    elapsed = time.monotonic() - started
    server.shutdown()

    assert elapsed < 8, f"batch took {elapsed:.1f}s -- looks hung, not just timed out"
    assert "slow.txt" in errors, "expected the slow request to raise a timeout error"
    assert results.get("a.txt") == b"CONTENT:a.txt"
    assert results.get("b.txt") == b"CONTENT:b.txt"
    assert results.get("c.txt") == b"CONTENT:c.txt"
    print(
        f"concurrent timeout OK: batch finished in {elapsed:.1f}s, "
        f"slow.txt raised {type(errors['slow.txt']).__name__}"
    )


def main():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    ds = foxden.open(DID, url=f"http://127.0.0.1:{port}", token="test-token")

    # metadata
    md = ds.metadata
    assert md["sample_name"] == "ti7-19-tomo", md
    print("metadata OK:", md["did"])

    # shallow listing
    top = ds.list()
    assert {e.name for e in top} == {"1", "2"}, top
    print("list() OK:", [e.name for e in top])

    # full recursive tree
    tree = ds.tree()
    print("tree:\n" + tree.pretty())

    all_files = ds.files()
    names = sorted(e.path for e in all_files)
    expected = sorted(
        [
            "1/data.txt",
            "1/meta/1/scalars.json",
            "1/meta/1/positions.txt",
            "2/info.txt",
        ]
    )
    assert names == expected, names
    print("files() OK:", names)

    # scoped listing under a subpath
    meta_files = ds.files(path="1/meta/1")
    assert {e.name for e in meta_files} == {"scalars.json", "positions.txt"}, meta_files
    print("files(path=...) OK:", [e.name for e in meta_files])

    # resolve_fetch_path correctly strips the filename back off entry.path
    scalars_entry = next(e for e in meta_files if e.name == "scalars.json")
    dirpath, filename = foxden.resolve_fetch_path(scalars_entry)
    assert (dirpath, filename) == ("1/meta/1", "scalars.json"), (dirpath, filename)
    print("resolve_fetch_path() OK:", (dirpath, filename))

    # ds.fetch(entry) uses that normalization automatically
    data = ds.fetch(scalars_entry)
    assert data == b"CONTENT:1/meta/1/scalars.json", data
    print("fetch(entry) OK:", data)

    # ds.get_file(path, file) is the raw form for manual (dir, file) pairs
    data2 = ds.get_file("1/meta/1", "scalars.json")
    assert data2 == data
    print("get_file() OK:", data2)

    # chunked batch fetch over the whole dataset
    batches = list(ds.training_data(batch_size=3))
    assert [len(b) for b in batches] == [3, 1], batches
    total = sum(len(b) for b in batches)
    assert total == 4
    for batch in batches:
        for item in batch:
            assert item["data"].decode() == f"CONTENT:{item['path']}", item
    print("training_data() OK:", [[i["name"] for i in b] for b in batches])

    # chunked batch fetch scoped to a subpath, with a custom loader
    sub_batches = list(
        ds.training_data(path="1/meta/1", batch_size=1, loader=lambda e, d: d.decode())
    )
    assert len(sub_batches) == 2 and all(len(b) == 1 for b in sub_batches)
    print("training_data(path=...) OK:", sub_batches)

    # include/exclude filename filtering
    txt_only = ds.files(include=["*.txt"])
    assert {e.name for e in txt_only} == {"data.txt", "positions.txt", "info.txt"}, txt_only
    print("files(include=...) OK:", [e.name for e in txt_only])

    no_json = ds.files(exclude=["*.json"])
    assert "scalars.json" not in {e.name for e in no_json}
    assert len(no_json) == 3, no_json
    print("files(exclude=...) OK:", [e.name for e in no_json])

    filtered_batches = list(ds.training_data(batch_size=10, exclude=["*.json"]))
    assert sum(len(b) for b in filtered_batches) == 3
    print("training_data(exclude=...) OK")

    server.shutdown()

    # resolve_fetch_path() must also handle FOXDEN's *documented*
    # convention, where a file entry's `path` is already directory-only
    # (no filename suffix) -- e.g. path="5/meta/1", name="scalars.txt".
    doc_style = foxden.DMEntry(
        did=DID, name="scalars.txt", path="5/meta/1", is_dir=False
    )
    assert foxden.resolve_fetch_path(doc_style) == ("5/meta/1", "scalars.txt")
    # a root-level file with no parent directory at all
    root_style = foxden.DMEntry(did=DID, name="readme.txt", path="readme.txt", is_dir=False)
    assert foxden.resolve_fetch_path(root_style) == ("", "readme.txt")
    print("resolve_fetch_path() dir-only + root-level conventions OK")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
    test_concurrent_timeout_does_not_hang()
