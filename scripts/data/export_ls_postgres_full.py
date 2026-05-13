#!/usr/bin/env python3
"""Export full Label Studio annotations from the on-host Postgres container.

Target output: JSONL where each line is one task with annotations in the
exact shape `kv_llm.kv_nsp.extract_label_studio_pairs` expects:

    {
      "task_id": 12345,
      "project_id": 12,
      "ocr_text": "...",
      "category": "...",
      "image": "...",
      "annotations": [
        {"was_cancelled": false, "result": [
          {"type": "labels", "id": "abc", "value": {"start": 0, "end": 7, "text": "...", "labels": ["键名"]}},
          {"type": "labels", "id": "def", "value": {...}, "labels": ["值"]},
          {"type": "relation", "from_id": "abc", "to_id": "def"}
        ]}
      ]
    }

Only project IDs that use the 键名/值/医院名称 schema are included:
{10, 11, 12, 14, 18, 19, 21, 22, 23, 24}. Total expected ~16,687 labeled tasks.

This script SHELLS OUT to `docker exec labelstudio_pg psql` because:
  1. Postgres is only exposed inside the labelstudio Docker network
  2. SQL native JSON aggregation is faster than pulling rows over psycopg
  3. No extra Python deps required
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


KV_PROJECTS = (10, 11, 12, 14, 18, 19, 21, 22, 23, 24)


def build_sql(project_ids: tuple[int, ...]) -> str:
    ids = ",".join(str(p) for p in project_ids)
    return f"""
        COPY (
          SELECT json_build_object(
            'task_id', t.id,
            'project_id', t.project_id,
            'ocr_text', t.data->>'ocr_text',
            'category', t.data->>'category',
            'image', t.data->>'image',
            'annotations', (
              SELECT json_agg(json_build_object(
                'completion_id', tc.id,
                'was_cancelled', tc.was_cancelled,
                'result', tc.result::json
              ) ORDER BY tc.id)
              FROM task_completion tc
              WHERE tc.task_id = t.id AND tc.was_cancelled = false
            )
          )::text
          FROM task t
          WHERE t.is_labeled = true
            AND t.project_id IN ({ids})
        ) TO STDOUT
    """


def run_psql_copy(container: str, db: str, user: str, sql: str, out_path: Path) -> int:
    cmd = [
        "docker", "exec", "-i", container,
        "psql", "-U", user, "-d", db, "-A", "-t", "-c", sql,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("w", encoding="utf-8") as fout:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            # psql -A -t echoes a "COPY n" line at the end of stdout for COPY TO STDOUT? actually
            # in -A -t mode COPY rows arrive as plain JSON lines. Defensive: skip any line that
            # doesn't parse as JSON.
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            rows += 1
        ret = proc.wait()
        if ret != 0:
            err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            sys.stderr.write(f"[ERR] psql exited {ret}\n{err}\n")
            return ret
    print(f"[OK] wrote {rows} rows to {out_path}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--container", default="labelstudio_pg")
    p.add_argument("--db", default="labelstudio")
    p.add_argument("--user", default="lsuser")
    p.add_argument("--output", default="/data/ocean/code/dapt/data_full/ls_kv_tasks.jsonl")
    args = p.parse_args()

    sql = build_sql(KV_PROJECTS)
    out = Path(args.output)
    sys.exit(run_psql_copy(args.container, args.db, args.user, sql, out))


if __name__ == "__main__":
    main()
