#!/usr/bin/env python3
"""Export one n8n execution's node-by-node data (the migration parity harness).

Reads the n8n SQLite database inside the running container (read-only) and
writes data/parity/exec_<id>.json with {"meta": ..., "runData": ...}.

  .venv/bin/python tools/n8n_export.py            # latest MIRA execution
  .venv/bin/python tools/n8n_export.py 2084
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "parity"
WORKFLOW_ID = "iz3yMcSlkWIQhRmn"  # Memory Innovation Research Assistant
CONTAINER = "n8n"

_QUERY = r'''
import sqlite3, sys, json
c = sqlite3.connect("file:/home/node/.n8n/database.sqlite?mode=ro", uri=True)
eid = sys.argv[1]
if eid == "latest":
    eid = c.execute("select id from execution_entity where workflowId=? order by id desc limit 1", (sys.argv[2],)).fetchone()[0]
row = c.execute("select e.id, e.status, e.mode, e.startedAt, e.stoppedAt, d.data from execution_entity e join execution_data d on d.executionId=e.id where e.id=?", (eid,)).fetchone()
sys.stdout.write(json.dumps({"id": row[0], "status": row[1], "mode": row[2], "startedAt": row[3], "stoppedAt": row[4]}) + "\n")
sys.stdout.write(row[5])
'''


def flatted_parse(text: str):
    """Decode n8n's execution storage (the `flatted` format): a JSON array
    where every string inside an object/array is the index of another entry."""
    arr = json.loads(text)
    memo: dict[int, object] = {}

    def res(i: int):
        if i in memo:
            return memo[i]
        v = arr[i]
        if isinstance(v, list):
            out: list = []
            memo[i] = out
            out.extend(res(int(x)) if isinstance(x, str) else x for x in v)
            return out
        if isinstance(v, dict):
            out_d: dict = {}
            memo[i] = out_d
            for k, x in v.items():
                out_d[k] = res(int(x)) if isinstance(x, str) else x
            return out_d
        memo[i] = v
        return v

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
    return res(0)


def export(exec_id: str = "latest") -> Path:
    raw = subprocess.run(["docker", "exec", "-i", CONTAINER, "python3", "-", exec_id, WORKFLOW_ID],
                         input=_QUERY, capture_output=True, text=True, check=True).stdout
    meta_line, data = raw.split("\n", 1)
    meta = json.loads(meta_line)
    run = flatted_parse(data)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"exec_{meta['id']}.json"
    path.write_text(json.dumps({"meta": meta, "runData": run["resultData"]["runData"],
                                "lastNode": run["resultData"].get("lastNodeExecuted"),
                                "error": run["resultData"].get("error")}, default=str))
    print(f"{path}  status={meta['status']}  nodes={len(run['resultData']['runData'])}")
    return path


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else "latest")
