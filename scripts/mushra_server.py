"""Serve webMUSHRA and collect its results, with no PHP and no root.

    python scripts/mushra_server.py --root ~/webMUSHRA
    # then http://localhost:8000/?config=emt12.yaml

    python scripts/mushra_server.py --root ~/webMUSHRA --flatten   # rebuild CSVs

WHY THIS EXISTS. webMUSHRA ships a PHP service for result collection and the
box has no PHP and no sudo. The service is not doing anything PHP is needed
for: it accepts ONE form-encoded POST field, `sessionJSON`, holding the whole
session, and writes CSVs under results/<testId>/. That is a page of Python, and
replacing it removes the only dependency standing between the test and running.

Everything else about webMUSHRA is untouched -- the config, the randomisation,
the anchors, the finish page. Only the endpoint behind `remoteService` changes,
and the config still names it service/write.php because this server answers any
POST regardless of path.

THE RAW JSON IS WRITTEN FIRST AND UNCONDITIONALLY, before anything tries to
interpret it. Listening-test data is not reproducible: a listener who has done
a 25-minute session will not do it again because a flattener hit a key it did
not expect. So each session lands as results/<testId>/session_<uuid>.json
verbatim, and the CSV is derived from it afterwards. If the CSV is wrong or the
schema turns out to differ from what this assumes, --flatten rebuilds every CSV
from the stored JSONs with no session lost. THE JSON IS THE RECORD; the CSV is
a convenience.

THE FLATTENING IS SCHEMA-AGNOSTIC for the same reason. Rather than hardcoding
webMUSHRA's response fields -- which are not documented and vary by page type --
each trial's responses are flattened by whatever keys they actually carry, and
the column set is the union over the file. A page type this has never seen
produces a wider CSV, not a crash.

BINDS TO LOCALHOST BY DEFAULT. Reach it over an SSH tunnel:

    ssh -L 8000:localhost:8000 <user>@<host>

which needs no firewall change and exposes nothing. --host 0.0.0.0 opens it to
the network, which on a box holding the project's data should be a deliberate
decision rather than a default.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import uuid as _uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs


def safe(s: str) -> str:
    """webMUSHRA's testId reaches the filesystem; keep it to a path segment.

    The character class alone is not enough: it permits "." and "..", which as
    a whole segment walk out of the results directory. Only matters once the
    server is bound to something other than localhost, which --host allows.
    """
    out = re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))[:64]
    return "unnamed" if out.strip(".") == "" else out


def flatten(session: dict) -> tuple[list[str], list[dict]]:
    """One row per (trial, response). Columns are whatever the data carries."""
    base = {"testId": session.get("testId", ""),
            "uuid": session.get("uuid", "")}
    part = session.get("participant") or {}
    for n, r in zip(part.get("name") or [], part.get("response") or []):
        base[f"participant_{safe(n)}"] = r

    rows: list[dict] = []
    for trial in session.get("trials") or []:
        t = dict(base)
        for k, v in trial.items():
            if k != "responses" and not isinstance(v, (list, dict)):
                t[f"trial_{k}"] = v
        resp = trial.get("responses")
        if not isinstance(resp, list) or not resp:
            rows.append(t)
            continue
        for r in resp:
            row = dict(t)
            if isinstance(r, dict):
                for k, v in r.items():
                    row[k] = v if not isinstance(v, (list, dict)) else json.dumps(v)
            else:
                row["response"] = r
            rows.append(row)

    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols, rows


def write_csv(out: Path, sessions: list[dict]) -> int:
    """All sessions for one testId into one CSV, columns unioned across them."""
    cols: list[str] = []
    rows: list[dict] = []
    for s in sessions:
        c, r = flatten(s)
        for k in c:
            if k not in cols:
                cols.append(k)
        rows += r
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def store(results: Path, session: dict) -> Path:
    """Raw JSON first, then the CSV. Never the other way round."""
    d = results / safe(session.get("testId", "unnamed"))
    d.mkdir(parents=True, exist_ok=True)
    uid = safe(session.get("uuid") or _uuid.uuid4().hex)
    raw = d / f"session_{uid}.json"
    raw.write_text(json.dumps(session, indent=1))
    sessions = []
    for f in sorted(d.glob("session_*.json")):
        try:
            sessions.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            print(f"  ! {f.name} is not valid JSON, skipped in the CSV")
    n = write_csv(d / "mushra.csv", sessions)
    print(f"  stored {raw}  ->  {d/'mushra.csv'} ({len(sessions)} session(s), "
          f"{n} rows)")
    return raw


class Handler(SimpleHTTPRequestHandler):
    results: Path = Path("results")

    def do_POST(self):                                   # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n).decode("utf-8", "replace")
        try:
            # webMUSHRA posts form-encoded with a single sessionJSON field. A
            # raw JSON body is accepted too so a changed client still lands.
            fields = parse_qs(body)
            raw = fields.get("sessionJSON", [None])[0] or body
            session = json.loads(raw)
        except (ValueError, KeyError) as e:
            # Never drop a session because it could not be parsed -- park it.
            self.results.mkdir(parents=True, exist_ok=True)
            p = self.results / f"unparsed_{_uuid.uuid4().hex}.txt"
            p.write_text(body)
            print(f"  ! could not parse a submission ({e}); raw body -> {p}")
            self.send_response(200)
            self.end_headers()
            return
        try:
            store(self.results, session)
        except OSError as e:
            print(f"  ! could not write results: {e}")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *a):
        if "POST" in (a[0] if a else ""):
            super().log_message(fmt, *a)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, required=True,
                   help="the webMUSHRA checkout to serve")
    p.add_argument("--results", type=Path, default=None,
                   help="default <root>/results")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="localhost",
                   help="localhost, reached over an SSH tunnel. 0.0.0.0 "
                        "exposes it to the network.")
    p.add_argument("--flatten", action="store_true",
                   help="rebuild every CSV from the stored JSONs and exit")
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    if not (root / "index.html").exists():
        raise SystemExit(f"{root} has no index.html -- is that the webMUSHRA "
                         f"checkout?")
    results = (args.results or root / "results").expanduser().resolve()

    if args.flatten:
        for d in sorted(x for x in results.iterdir() if x.is_dir()):
            sessions = [json.loads(f.read_text())
                        for f in sorted(d.glob("session_*.json"))]
            if sessions:
                n = write_csv(d / "mushra.csv", sessions)
                print(f"{d.name}: {len(sessions)} session(s), {n} rows")
        return 0

    results.mkdir(parents=True, exist_ok=True)
    Handler.results = results
    srv = ThreadingHTTPServer((args.host, args.port),
                              partial(Handler, directory=str(root)))
    print(f"serving {root}\nresults -> {results}")
    print(f"  http://{args.host}:{args.port}/?config=<yourconfig>.yaml")
    if args.host == "localhost":
        print("  (localhost only -- tunnel in with "
              "ssh -L {p}:localhost:{p} <user>@<host>)".format(p=args.port))
    print("\nEvery submission prints a line here. If you finish a session and "
          "see nothing,\nthe finish page is not set to writeResults: true.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
