# MIRA — local research digests. `make help` for targets.
.PHONY: help setup server serve test digest realtime licenses

help:
	@echo "setup      install Python deps (uv) and crawler deps (npm ci)"
	@echo "server     start the local Prefect server + UI on http://127.0.0.1:4200"
	@echo "serve      register deployments from configs/schedules.json and run flows"
	@echo "test       run the test suite"
	@echo "digest     one-off digest without Prefect:  make digest PROFILE=cxl-research MODE=monthly ARGS='--test-mode'"
	@echo "realtime   one-off realtime run:              make realtime ARGS='--dry-run'"
	@echo "licenses   print the license of every installed Python/Node dependency"

setup:
	uv sync
	cd crawlers && npm ci --ignore-scripts

server:
	bin/mira-env prefect server start

serve:
	bin/mira-env python serve.py

test:
	.venv/bin/python -m pytest -q

digest:
	.venv/bin/python run.py --profile $(PROFILE) --mode $(MODE) $(ARGS)

realtime:
	.venv/bin/python run_realtime.py $(ARGS)

licenses:
	.venv/bin/python scripts/license_report.py
