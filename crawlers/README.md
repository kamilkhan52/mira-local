# Micron – EE Times Crawler

Crawls [EE Times Memory Designline](https://www.eetimes.com/category/news-analysis/designline/memory-designline/) and can POST results to an n8n webhook via Node’s native `fetch()`.

## Sending to n8n (webhook)

The crawler **defaults to the production webhook** (`http://localhost:5678/webhook/eetimes-crawl`). Ensure the workflow **"EE Times Crawl 2026-01-01 to 01-29"** is **Active** in n8n (toggle on, top right, then save).

- **Production** (default): No env var needed. Uses `/webhook/eetimes-crawl`; workflow must be Active.
- **Test URL**: Set `N8N_WEBHOOK_URL="http://localhost:5678/webhook-test/eetimes-crawl"` for quick test (no activation; expires when you leave the workflow).
- **Other host**: Set `N8N_WEBHOOK_URL` to the Production URL from the Webhook node in n8n.

### Run (production by default)

```bash
node crawl-eetimes-list.js
# or with pnpm
pnpm exec node crawl-eetimes-list.js
```

### Option 2: HTTPS with self-signed certificate

If n8n runs on HTTPS locally, tell Node to allow the certificate:

```bash
export NODE_TLS_REJECT_UNAUTHORIZED='0'
N8N_WEBHOOK_URL="https://localhost:5678/webhook/ee-times-crawl-2026" node crawl-eetimes-list.js
```

## Test the webhook with curl

**HTTP (localhost):** Use the same URL as `N8N_WEBHOOK_URL` (copy from Webhook node in n8n).

```bash
curl -X POST http://localhost:5678/webhook/eetimes-crawl \
  -H "Content-Type: application/json" \
  -d '[{"url": "https://test.com", "listTitle": "Manual Test", "content": "Test content"}]'
```

**HTTPS with self-signed cert:** add `-k` (insecure):

```bash
curl -k -X POST https://localhost:5678/webhook/ee-times-crawl-2026 \
  -H "Content-Type: application/json" \
  -d '[{"url": "https://test.com", "listTitle": "Manual Test", "content": "Test content"}]'
```

## Other env vars

- `OUTPUT_PATH` – Also write JSON to this path (e.g. `../configs/eetimes-latest.json`).
- `MAX_ARTICLES` – Limit number of articles to crawl.
- `DATE_FROM` / `DATE_TO` – Filter list by date (if supported by the crawler).
