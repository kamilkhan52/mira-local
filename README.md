# MIRA (local)

Research digests from arXiv and industry news, plus hourly realtime alerts,
orchestrated with **Prefect 3** on your own machine. This project replaces the
n8n deployment: no n8n, no Docker, no cloud accounts.

- **Digests** (`digest` flow): arXiv fetch → first-page PDFs → affiliation and
  classification LLM stages (with a per-paper cache) → completeness gate,
  thresholds, rank and cap → LLM selection → full-text deep analysis → news
  crawl and summaries → report and trend analysis → HTML, PDF and email.
  This is a faithful port of the live n8n workflow; the prompt-building and
  report nodes are tested for byte-level parity against the original n8n
  JavaScript (`tests/test_port_*`).
- **MIRA Live** (`realtime` flow): new papers and news triaged by Jev
  (TypeSafe), with the top items summarized for each profile's preferences
  and emailed to the subscribers in `configs/realtime.json`.

The Jev evaluation (what was tested, results, recommendations, setup) is in
[`docs/jev-report.html`](docs/jev-report.html); open it in a browser.

## Why this stack

| Requirement | How it is met |
|---|---|
| Free for commercial use at a large company | Prefect is Apache-2.0 with no enterprise-only code in the OSS repo. Every dependency is permissive (`make licenses`; the one flag, `text-unidecode`, is dual Artistic/GPL and used under the Artistic License). PyMuPDF (AGPL) was replaced by pypdfium2/pypdf. |
| Fully local, no telemetry | The Prefect server binds to `127.0.0.1` and stores state in SQLite under `data/`. Analytics, cloud telemetry, resource metrics and promotional UI content are off (`deploy/prefect.env`); the server made no outbound connections in testing. |
| Nice development experience | Plain Python functions with `@flow` and `@task`. Runs locally with `make`, has a UI with run history, logs, retries and parameter forms for manual runs, and is covered by pytest. |
| Minimal infrastructure | Two processes (server and serve), with no Postgres, Redis or Kubernetes. |

**Data leaving the machine.** Orchestration is local, but the pipeline itself
calls external services: arXiv and the news sites (public content), the LLM
endpoint, and Jev (realtime triage and the optional pre-screen). To keep
prompts inside company infrastructure, set `MIRA_LLM_BASE_URL` to an approved
OpenAI-compatible gateway or a local model server (vLLM, Ollama, LM Studio).
The realtime flow can run without Jev by leaving it unscheduled.

## Setup

Requires Python 3.11 with [uv](https://docs.astral.sh/uv/), Node 20+, and a
local Chrome or Chromium (for the crawlers and the PDF).

```bash
make setup              # uv sync + npm ci in crawlers/
cp .env.example .env    # then fill in the LLM key, SMTP, recipients, TYPESAFE_API_KEY
make test
```

## Running

```bash
make server             # Prefect UI and API at http://127.0.0.1:4200
make serve              # registers deployments and executes runs (second terminal)
```

- **Scheduled runs** are defined in `configs/schedules.json`. Everything starts
  **disabled** so this can run beside n8n without duplicate emails. To enable
  one, set `"enabled": true` and restart `make serve`. The live n8n schedule
  was the CXL monthly digest (1st of the month, 06:00 America/Los_Angeles).
- **Manual runs**: in the UI, open Deployments → `digest-manual` → Run →
  Custom run. The parameters replace the n8n backfill webhook: `profile`,
  `mode`, `current_date`, `lookback_days`, `max_limit`, `test_mode`,
  `llm_cache_bypass`, `trend_enabled`, `send_email`, `recipients`,
  `jev_prescreen` and `pdf`.
- **From the CLI**, without the server:
  ```bash
  .venv/bin/python run.py --profile cxl-research --mode monthly --test-mode --no-email
  .venv/bin/python run.py --trigger "CXL Monthly Trigger" --current-date 2026-09-01
  .venv/bin/python run_realtime.py --dry-run
  ```

Outputs go to `data/` (override with `MIRA_DATA_DIR`), using the same layout
as the n8n `/report-files`:

- `data/report-files/prod/<profile>/`: report records; trend history reads these
- `data/report-files/tests/<profile>/`: test-mode records
- `data/report-files/cache/<profile>/<stage>/`: the per-paper LLM cache, file-compatible with n8n's
- `data/realtime-state/`: realtime ledgers and alert HTML
- `data/prefect/`: Prefect's SQLite database

To carry over history (for trend analysis) and the LLM cache, copy the old
`report-files/prod` and `report-files/cache` directories into
`data/report-files/`.

## Always-on deployment

- **Linux VM**: `deploy/mira-prefect-server.service` and
  `deploy/mira-flows.service` (systemd). Adjust `User=` and the paths.
- **macOS**: `deploy/com.mira.server.plist` and `deploy/com.mira.flows.plist`
  (launchd). Keep `MIRA_DATA_DIR` on the internal disk, because launchd jobs
  cannot write to external volumes without Full Disk Access.
- For access from other machines, keep the server on `127.0.0.1` and use an
  SSH tunnel or a reverse proxy with company SSO. The OSS server has no user
  accounts; `PREFECT_SERVER_API_AUTH_STRING` adds basic auth.

## Cutover from n8n

1. Copy `report-files/prod` and `report-files/cache` from the n8n host into `data/report-files/`.
2. Fill in `.env` (SMTP, recipients, LLM endpoint) and run one digest with
   `test_mode=true` and `send_email=false`. Compare it with the latest n8n report.
3. Deactivate the n8n workflow, then set `"enabled": true` for
   `cxl-research-monthly` (and any others) in `configs/schedules.json` and restart serve.
4. For realtime alerts, enable `realtime-hourly` here and unload the old
   `com.mira.realtime` launchd job from the n8n repo, so alerts are not sent twice.

## Differences from the n8n workflow

**Improvements over n8n**
- The PDF is attached to the email; n8n built it but never attached it.
- Trend-analysis and PDF failures are logged and the report still goes out;
  in n8n either one stopped the run.
- Selected paper IDs are matched tolerantly (URL prefix and version stripped).
- A failed deep analysis is flagged in the report instead of failing the whole run.

**n8n quirks kept for parity** (commented in the code; fix if you like)
- Double spaces in the subject line.
- The teal email palette regardless of profile (set `email_use_profile_colors: true` to use the profile's colors).
- A 60-day trend window.
- Prior reports containing the word "test" are excluded from trend history.
- `trend_enabled` is effectively always on.

## Maintenance

- **Tests**: `make test`. Parity tests run the original n8n JavaScript under
  node, and skip when node is missing.
- **Licenses**: `make licenses` (add `--strict` in CI to fail on anything that needs review).
- **Models** are set in the profile config's `llm_models` block; the endpoint
  and key come from `.env`.
