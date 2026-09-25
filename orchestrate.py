#!/usr/bin/env python3
"""Run a benchmark plan: rounds x scenarios x engines, alternating engine order per round.

After each cell the thermal canary (fixed ~2K prompt, 128-token decode) is compared
with that engine's best canary so far in the run; if it is more than
`canary_tolerance` slower the cell is flagged, the machine rests, and the cell is
re-run once. The summarizer uses the last attempt of each cell.

  python orchestrate.py --plan plans/day.json --out results/<run>/rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def last_canary(rows_path: Path, engine: str, kv: str, scenario: str, rnd: int, attempt: int) -> float | None:
    if not rows_path.exists():
        return None
    value = None
    for line in rows_path.read_text().splitlines():
        r = json.loads(line)
        if (r["phase"] == "canary" and r["engine"] == engine and r["kv"] == kv and r["scenario"] == scenario
                and r["round"] == rnd and r.get("attempt", 1) == attempt):
            value = r.get("decode_tok_s")
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    out = Path(a.out)
    log = out.parent / "orchestrate.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    best: dict[str, float] = {}
    tol = float(plan.get("canary_tolerance", 0.03))

    def note(msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with log.open("a") as fh:
            fh.write(line + "\n")

    for rnd, order in enumerate(plan["rounds"], 1):
        for cell in plan["cells"]:
            for pos, engine in enumerate(order, 1):
                kv = cell.get("kv", "q8")
                for attempt in (1, 2):
                    note(f"round {rnd} {cell['scenario']} {kv} {engine} attempt {attempt}")
                    cmd = [sys.executable, str(ROOT / "runner.py"), "--engine", str(ROOT / "engines" / f"{engine}.json"),
                           "--scenario", str(ROOT / "scenarios" / f"{cell['scenario']}.json"), "--out", str(out),
                           "--round", str(rnd), "--order-pos", str(pos), "--kv", kv, "--attempt", str(attempt)]
                    rc = subprocess.run(cmd, cwd=ROOT).returncode
                    if rc != 0:
                        note(f"  runner exited {rc}")
                    canary = last_canary(out, engine, kv, cell["scenario"], rnd, attempt)
                    key = f"{engine}/{kv}"
                    ref = best.get(key)
                    if canary:
                        best[key] = max(ref or 0.0, canary)
                    slow = bool(canary and ref and canary < ref * (1 - tol))
                    note(f"  canary {canary} ref {ref} slow={slow}")
                    time.sleep(plan.get("cooldown_s", 90))
                    if not slow:
                        break
                    note(f"  canary slow: resting {plan.get('rest_s', 300)} s and re-running once")
                    time.sleep(plan.get("rest_s", 300))
    note("plan done")


if __name__ == "__main__":
    main()
