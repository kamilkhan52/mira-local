# News crawlers

TypeScript crawlers for EE Times (Memory Designline), SemiAnalysis, TrendForce
and Digitimes. The Python pipeline (`mira/media.py`, `mira/realtime.py`) runs
them with `npx tsx <crawler>.ts` and reads the JSON they write.

Setup (once): `npm ci --ignore-scripts` in this directory (`make setup` does it).
They drive a locally installed Chrome/Chromium through `puppeteer-core`; set
`PUPPETEER_EXECUTABLE_PATH` or `CHROME_PATH` if Chrome is not in a standard location.

Environment (set by the Python caller):

| Variable | Meaning |
|---|---|
| `DATE_FROM`, `DATE_TO` | Inclusive article date window (YYYY-MM-DD) |
| `MAX_ARTICLES` | Cap on articles (0 = no cap) |
| `OUTPUT_PATH` | Where to write the JSON result |
| `OUTPUT_DIR` | Directory for timestamped copies (defaults under the data dir) |
| `LIST_URL` | EE Times list page override |
| `CRAWLER_WEBHOOK_URL` | Optional: also POST results to this URL. Off by default. (`N8N_WEBHOOK_URL` is accepted as a deprecated alias.) |

Manual run:

```bash
DATE_FROM=2026-09-20 DATE_TO=2026-09-28 MAX_ARTICLES=3 OUTPUT_PATH=/tmp/ee.json npx tsx ee-times-crawler.ts
```

`generate-pdf.ts` renders the report HTML to PDF with the same local Chrome.
