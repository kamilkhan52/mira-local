"""Run-history ledger for discovery recommendations (spec §3 Stage 2).

Keyed by canonical unordered pair. Statuses are user-maintained via
`discover.py --set-status`. A corrupt ledger is backed up and replaced —
never silently overwritten (spec §5).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import date, datetime
from pathlib import Path

STATUSES = ("new", "pursuing", "rejected", "published")
SEEN_DEMOTION = 0.9  # mild demotion for previously recommended, still-'new' pairs


def pair_key(topic_a: str, topic_c: str) -> str:
    return " || ".join(sorted((topic_a, topic_c)))


def parse_pair(text: str) -> tuple[str, str] | None:
    """Split a user-supplied pair: "A|B" or the displayed key form "A || B"."""
    sep = "||" if "||" in text else "|"
    a, found, c = text.partition(sep)
    if not found or not a.strip() or not c.strip():
        return None
    return tuple(sorted((a.strip(), c.strip())))


class Ledger:
    def __init__(self, path: Path, entries: dict[str, dict], warning: str | None = None):
        self.path = Path(path)
        self.entries = entries
        self.warning = warning

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        path = Path(path)
        if not path.exists():
            return cls(path, {})
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError("ledger root is not a JSON object")
            # Validate every entry before accepting the data
            required_keys = {"first_recommended", "last_recommended", "best_score", "status", "times_recommended"}
            for key, entry in data.items():
                if not isinstance(entry, dict):
                    raise ValueError(f"malformed ledger entry {key!r}")
                missing_keys = required_keys - set(entry.keys())
                if missing_keys:
                    raise ValueError(f"malformed ledger entry {key!r}")
                if entry["status"] not in STATUSES:
                    raise ValueError(f"malformed ledger entry {key!r}")
            return cls(path, data)
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            backup = path.with_name(f"{path.stem}.bak-{datetime.now():%Y%m%d-%H%M%S}")
            path.rename(backup)
            return cls(path, {}, warning=f"ledger corrupt ({exc}); backed up to "
                                         f"{backup.name}, starting fresh")

    def entry(self, topic_a: str, topic_c: str) -> dict | None:
        return self.entries.get(pair_key(topic_a, topic_c))

    def status(self, topic_a: str, topic_c: str) -> str:
        e = self.entry(topic_a, topic_c)
        return e["status"] if e else "unseen"

    def multiplier(self, topic_a: str, topic_c: str) -> float:
        e = self.entry(topic_a, topic_c)
        return SEEN_DEMOTION if e and e["status"] == "new" else 1.0

    def upsert(self, topic_a: str, topic_c: str, score: float, run_date: date) -> None:
        key = pair_key(topic_a, topic_c)
        e = self.entries.get(key)
        if e is None:
            self.entries[key] = {
                "first_recommended": run_date.isoformat(),
                "last_recommended": run_date.isoformat(),
                "best_score": score,
                "status": "new",
                "times_recommended": 1,
            }
        else:
            e["last_recommended"] = run_date.isoformat()
            e["best_score"] = max(e["best_score"], score)
            e["times_recommended"] += 1

    def set_status(self, key: str, status: str) -> bool:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        if key not in self.entries:
            return False
        self.entries[key]["status"] = status
        return True

    def partition(self, candidates) -> tuple[list, list, list]:
        """Split candidates by ledger status (spec §3 Stage 2 table)."""
        eligible, pursuing, suppressed = [], [], []
        for c in candidates:
            status = self.status(c.topic_a, c.topic_c)
            if status in ("rejected", "published"):
                suppressed.append((pair_key(c.topic_a, c.topic_c), status))
            elif status == "pursuing":
                pursuing.append(c)
            else:
                eligible.append(c)
        return eligible, pursuing, suppressed

    def summary(self) -> str:
        counts = Counter(e["status"] for e in self.entries.values())
        parts = ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s])
        return f"{len(self.entries)} tracked ({parts})" if self.entries else "empty"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, indent=2, sort_keys=True))
        os.replace(tmp, self.path)
