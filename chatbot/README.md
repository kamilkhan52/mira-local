# Combined Graph Chatbot

Chat over the merged memory + optical + storage LightRAG graph. Spec:
`docs/superpowers/specs/2026-07-22-combined-graph-chatbot-design.md`.

## Refresh the merged graph

    python merge_graphs.py --dry-run    # inspect counts first (memory+optical only)
    python merge_graphs.py \
      --storage-dir /Users/Eddie/Documents/n8n_memory_research_agent/lightrag/working_dir_storage

Storage is chained in as a second merge pass, so `--dry-run` preflights only
memory+optical; omitting `--storage-dir` produces a two-domain graph.

Sources are read-only; output goes to
`/Users/Eddie/Documents/n8n_memory_research_agent/lightrag/working_dir_combined`
(previous version kept as `.bak-<date>`, 2 most recent retained).

## Run the gateway

    cp chatbot/.env.example chatbot/.env   # once; fill LIGHTRAG_API_KEY from
                                           # lightrag/.env.lightrag.combined (main checkout)
    set -a; source chatbot/.env; set +a
    python -m uvicorn chatbot.app:app --factory \
      --host "$CHATBOT_BIND" --port "$CHATBOT_PORT" --workers 1

One worker only — the rate limiter is in-process.

## Exhaustive research API

Set `EXHAUSTIVE_RETRIEVAL_ENABLED=true`, `OPENROUTER_API_KEY`, and the three
`EXHAUSTIVE_{MEMORY,OPTICAL,STORAGE}_DIR` paths to enable independent,
graph-first scans. Remote callers must also present
`X-Research-Token: $EXHAUSTIVE_RESEARCH_TOKEN`; loopback callers are exempt.
The socket peer address—not `X-Forwarded-For`—is the authorization and
ownership boundary.

Long-running chat and hypothesis requests use the same durable job protocol:

- `POST /api/research/chat`
- `POST /api/research/hypotheses`
- `GET /api/research/active`
- `GET /api/research/{job_id}`
- `GET /api/research/{job_id}/events`
- `DELETE /api/research/{job_id}`

Every accepted request independently pins and validates the memory, optical,
and storage snapshots, scores every node and edge, then applies relevance
thresholds. It sends every deduplicated evidence chunk attached to selected
typed graph regions through Gemini 2.5 Flash and reserves Sonnet 4.6 for final
synthesis. Job records distinguish estimated cost (including reserve) from
provider-reported actual cost.

Operational setup, acceptance commands, failure diagnosis, and rollback are in
[`docs/EXHAUSTIVE_RETRIEVAL_RUNBOOK.md`](../docs/EXHAUSTIVE_RETRIEVAL_RUNBOOK.md).

## Expose on the tailnet

    tailscale serve --bg --https=443 http://127.0.0.1:8090
    tailscale serve status     # shows the https://<machine>.<tailnet>.ts.net URL

Only tailnet members can reach it; TLS is Tailscale's. To stop:
`tailscale serve --https=443 off`.

## Manual E2E checklist

1. `python merge_graphs.py` succeeds; `curl -s 127.0.0.1:9623/health` healthy.
2. Ask a memory-only question ("What is the memory wall?") — sources mostly
   `memory` badges.
3. Ask an optical-only question ("What is co-packaged optics?") — mostly
   `optical` badges.
4. Ask a storage-only question ("What are Zoned Namespaces (ZNS)?") — mostly
   `storage` badges.
5. Ask a bridging question ("How does co-packaged optics relate to HBM
   memory-wall pressure?") — answer cites multi-domain entities, badged as one
   chip per source domain (`memory` + `optical`); legacy two-domain entities may
   still show a single `both` chip.
6. Two browsers streaming simultaneously — both complete.
7. Send >RATE_PER_MIN requests in a minute — UI shows rate-limit message,
   recovers after cooldown.
8. `docker stop lightrag-combined` mid-stream — UI shows error banner with
   retry; partial answer preserved. `docker start lightrag-combined` after.
9. DevTools network tab: no requests leave the origin.

## Security posture

- Gateway binds 127.0.0.1; users reach it only through Tailscale serve.
- LightRAG :9623 binds 127.0.0.1 + requires X-API-Key; its mutation
  endpoints are not proxied — users cannot touch them.
- All markdown sanitized with DOMPurify; CSP default-src 'self'.
- Per-IP rate + concurrency limits cap OpenRouter spend.
