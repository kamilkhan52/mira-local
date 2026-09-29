#!/usr/bin/env python3
"""Merge the memory and optical LightRAG working dirs into a combined one.

See docs/superpowers/specs/2026-07-22-combined-graph-chatbot-design.md §3.
"""
import argparse
from datetime import datetime, timezone
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mira.graph_merge import (
    MergeError, build_provenance_multi, load_working_dir, run_merge,
)
from mira.merge_cli import make_openai_embedder, parse_env_file

LIGHTRAG_ROOT = Path("/Users/Eddie/Documents/n8n_memory_research_agent/lightrag")


def restart_combined(compose_file: Path) -> None:
    """Recreate Combined so its bind mount follows an atomic directory swap."""
    subprocess.run([
        "docker", "compose", "-f", str(compose_file),
        "up", "-d", "--force-recreate", "lightrag-combined",
    ], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memory-dir", type=Path,
                    default=LIGHTRAG_ROOT / "working_dir")
    ap.add_argument("--optical-dir", type=Path,
                    default=LIGHTRAG_ROOT / "working_dir_optical")
    ap.add_argument("--storage-dir", type=Path,
                    help="optionally chain this Storage working dir into the merge")
    ap.add_argument("--output-dir", type=Path,
                    default=LIGHTRAG_ROOT / "working_dir_combined")
    ap.add_argument("--env-file", type=Path,
                    default=LIGHTRAG_ROOT / ".env.lightrag")
    ap.add_argument("--compose-file", type=Path,
                    default=LIGHTRAG_ROOT / "docker-compose.lightrag.yml")
    ap.add_argument("--dry-run", action="store_true",
                    help="print merge report, write nothing")
    ap.add_argument("--skip-restart", action="store_true",
                    help="do not restart the lightrag-combined container")
    args = ap.parse_args()

    if args.dry_run:
        embed_fn = None
    else:
        env = parse_env_file(args.env_file)
        key = env.get("EMBEDDING_BINDING_API_KEY")
        if not key:
            print(f"ERROR: EMBEDDING_BINDING_API_KEY not found in "
                  f"{args.env_file}", file=sys.stderr)
            return 2
        embed_fn = make_openai_embedder(
            key, env.get("EMBEDDING_BINDING_HOST", "https://openrouter.ai/api/v1"),
            env.get("EMBEDDING_MODEL", "text-embedding-3-small"))

    try:
        intermediate = args.output_dir.with_name(
            f"{args.output_dir.name}.chain-tmp")
        first_output = intermediate if args.storage_dir else args.output_dir
        report = run_merge(args.memory_dir, args.optical_dir, first_output,
                           embed_fn=embed_fn, dry_run=args.dry_run)
        if args.storage_dir and not args.dry_run:
            report = run_merge(intermediate, args.storage_dir, args.output_dir,
                               embed_fn=embed_fn, dry_run=False)
            provenance = build_provenance_multi(
                {
                    "memory": load_working_dir(args.memory_dir),
                    "optical": load_working_dir(args.optical_dir),
                    "storage": load_working_dir(args.storage_dir),
                },
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            (args.output_dir / "provenance.json").write_text(
                json.dumps(provenance, ensure_ascii=False, sort_keys=True))
            shutil.rmtree(intermediate)
    except MergeError as e:
        print(f"MERGE ABORTED: {e}", file=sys.stderr)
        return 1

    print(f"nodes={report.nodes} edges={report.edges} "
          f"shared_entities={report.shared_entities} "
          f"shared_relations={report.shared_relations} "
          f"shared_chunks={report.shared_chunks} "
          f"reembedded={report.reembedded}")
    if args.dry_run:
        if args.storage_dir:
            print("dry run: storage chaining requires a real run; "
                  "preflighted memory+optical only")
        print("dry run: nothing written")
        return 0
    print(f"written: {report.output_dir}")

    if args.skip_restart:
        return 0
    restart_combined(args.compose_file)
    import httpx
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            r = httpx.get("http://127.0.0.1:9623/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                print("lightrag-combined healthy on :9623")
                return 0
        except httpx.HTTPError:
            pass
        time.sleep(3)
    print("ERROR: lightrag-combined did not become healthy within 120s",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
