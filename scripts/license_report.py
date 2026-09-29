#!/usr/bin/env python3
"""License inventory of every installed Python distribution and crawler Node
package, flagging copyleft / source-available terms for legal review.

  .venv/bin/python scripts/license_report.py            # table + flags
  .venv/bin/python scripts/license_report.py --strict   # exit 1 if anything is flagged
"""
from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Terms that need legal review for internal commercial use. LGPL is flagged
# for visibility (usually fine when used unmodified as a library).
FLAG = re.compile(r"\b(A?GPL|LGPL|SSPL|BUSL|Business Source|Elastic|Commons Clause|"
                  r"Sustainable Use|CC-BY-NC|proprietary|EUPL|OSL|RPL)\b", re.I)


def python_licenses() -> list[tuple[str, str, str]]:
    rows = []
    for dist in metadata.distributions():
        md = dist.metadata
        name = md.get("Name") or "?"
        lic = md.get("License-Expression") or ""
        if not lic:
            classifiers = [c.split("::")[-1].strip() for c in md.get_all("Classifier") or []
                           if c.startswith("License ::")]
            lic = "; ".join(classifiers) or (md.get("License") or "").splitlines()[0][:80] if (
                classifiers or md.get("License")) else "UNKNOWN"
        rows.append(("py", f"{name} {dist.version}", lic or "UNKNOWN"))
    return rows


def node_licenses() -> list[tuple[str, str, str]]:
    rows = []
    nm = ROOT / "crawlers" / "node_modules"
    for pkg in sorted(nm.glob("*/package.json")) + sorted(nm.glob("@*/*/package.json")):
        try:
            data = json.loads(pkg.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        lic = data.get("license") or data.get("licenses") or "UNKNOWN"
        if isinstance(lic, (dict, list)):
            lic = json.dumps(lic)
        rows.append(("node", f"{data.get('name')} {data.get('version')}", str(lic)))
    return rows


def main() -> int:
    rows = sorted(set(python_licenses() + node_licenses()), key=lambda r: (r[0], r[1].lower()))
    flagged = [r for r in rows if FLAG.search(r[2]) or r[2] == "UNKNOWN"]
    width = max(len(r[1]) for r in rows)
    for eco, name, lic in rows:
        mark = "  <-- review" if (eco, name, lic) in flagged else ""
        print(f"{eco:4} {name:<{width}}  {lic}{mark}")
    print(f"\n{len(rows)} packages, {len(flagged)} flagged for review")
    return 1 if (flagged and "--strict" in sys.argv) else 0


if __name__ == "__main__":
    sys.exit(main())
