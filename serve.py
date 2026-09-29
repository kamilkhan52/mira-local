#!/usr/bin/env python3
"""Serve MIRA's flows to the local Prefect server (replaces n8n's triggers).

Registers one deployment per entry in configs/schedules.json (cron schedules,
active only when `enabled`), plus unscheduled `digest-manual` and
`realtime-manual` deployments for runs started from the UI or CLI with custom
parameters (the n8n backfill webhook's role). Runs until stopped; flow runs
execute in this process. Start the server first (`make server`).
"""
from __future__ import annotations

import json

from prefect import serve
from prefect.schedules import Cron

from mira.paths import CONFIG_DIR
from mira_flows.digest import digest_flow
from mira_flows.realtime import realtime_flow

FLOWS = {"digest": digest_flow, "realtime": realtime_flow}


def build_deployments() -> list:
    cfg = json.loads((CONFIG_DIR / "schedules.json").read_text())
    tz = cfg.get("timezone", "UTC")
    deployments = []
    for d in cfg["deployments"]:
        flow = FLOWS[d["flow"]]
        deployments.append(flow.to_deployment(
            name=d["name"],
            schedules=[Cron(d["cron"], timezone=tz, active=bool(d.get("enabled")))],
            parameters=d.get("parameters", {}),
            description=d.get("description"),
            tags=[d["flow"], "scheduled"],
            concurrency_limit=1,
        ))
    deployments.append(digest_flow.to_deployment(
        name="digest-manual", tags=["digest", "manual"], concurrency_limit=2,
        description="Run any profile/mode with overrides (date, lookback, test mode, cache bypass...)."))
    deployments.append(realtime_flow.to_deployment(
        name="realtime-manual", tags=["realtime", "manual"], concurrency_limit=1,
        description="Run the realtime monitor now (optionally dry-run)."))
    return deployments


if __name__ == "__main__":
    # pause_on_shutdown=False: schedules.json, not the last shutdown, decides
    # whether a schedule is active.
    serve(*build_deployments(), pause_on_shutdown=False)
