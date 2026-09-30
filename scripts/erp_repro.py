#!/usr/bin/env python3
"""Short-prompt head-to-head in the style of the "Splash on M1, part 2" post (benchmark 5).

For one engine: npanj's five splash-plus prompts on a loop for 5 minutes (250 tokens,
temperature 0, reasoning xhigh), then a rest, then one cold 44K prompt. macmon samples
temperature, power, GPU clock and fans every second the whole time; the engine's
process-tree footprint is sampled every 2 s.

  python scripts/erp_repro.py --engine engines/erp/splash.json --out results/erp-repro --round 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner import Engine, build_context, expand, local_vars, port_pids, stream_chat  # noqa: E402

# npanj/splash-plus scripts/run_all_models_standard_bench.py, verbatim
PROMPTS = [
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your step-by-step algebraic derivation, followed by the final answer.",
    "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping intervals. It must be clean, optimal in O(N log N) time, handle empty lists and negative coordinates, and include type annotations and docstring.",
    "Three friends (Alice, Bob, Charlie) are sitting in a row of 3 chairs numbered 1 to 3 from left to right. Constraints:\n1. Alice never sits next to Bob.\n2. Charlie is to the right of Alice (higher chair number).\nWho sits in chair 1, chair 2, and chair 3? Give a logical deduction step by step.",
    "Explain clearly the respective roles and architectural differences between FlashAttention, PagedAttention, and Speculative Decoding in high-performance LLM inference engines.",
    "In exactly three clear, professional sentences, explain why memory bandwidth (rather than raw compute TFLOPS) is the dominant latency bottleneck during autoregressive token generation in transformer LLMs.",
]
PREFILL_TOKENS = 44 * 1024


def tree_pids(root: int) -> list[int]:
    out = subprocess.run(["pgrep", "-g", str(root)], capture_output=True, text=True).stdout.split()
    pids = {int(p) for p in out if p.isdigit()} | {root}
    kids = subprocess.run(["pgrep", "-P", str(root)], capture_output=True, text=True).stdout.split()
    pids |= {int(p) for p in kids if p.isdigit()}
    return sorted(pids)


def footprint_bytes(pids: list[int]) -> int:
    total = 0
    for pid in pids:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "Footprint:" in line:
                num, unit = line.split("Footprint:")[1].split("(")[0].split()[:2]
                total += float(num) * {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit]
                break
    return int(total)


class Monitor(threading.Thread):
    """macmon every second plus the engine footprint every 2 s, tagged with the current phase."""

    def __init__(self, path: Path, root_pid: int):
        super().__init__(daemon=True)
        self.path, self.root_pid, self.phase, self.stop_event = path, root_pid, "idle", threading.Event()

    def run(self) -> None:
        proc = subprocess.Popen(["macmon", "pipe", "-i", "1000"], stdout=subprocess.PIPE, text=True)
        fp_at = 0.0
        fp = None
        with self.path.open("a") as fh:
            for line in proc.stdout:
                if self.stop_event.is_set():
                    break
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                now = time.time()
                if now - fp_at >= 2.0:
                    fp, fp_at = footprint_bytes(tree_pids(self.root_pid)), now
                row = {"t": now, "phase": self.phase, "gpu_temp": d["temp"]["gpu_temp_avg"],
                       "cpu_temp": d["temp"]["cpu_temp_avg"], "all_power": d["all_power"],
                       "gpu_power": d["gpu_power"], "sys_power": d.get("sys_power"),
                       "gpu_freq_mhz": d["gpu_freq_mhz"], "gpu_active": d["gpu_active_ratio"],
                       "fans_rpm": [f["rpm"] for f in d.get("fans", [])], "footprint": fp,
                       "ram_usage": d["memory"]["ram_usage"]}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
        proc.kill()


def payload(spec: dict, text: str, max_tokens: int) -> dict:
    p = {"model": spec["model_id"], "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
         "temperature": 0.0, "top_p": 1.0, "stream": True, "stream_options": {"include_usage": True}}
    p.update(spec.get("erp_request", {}))
    return p


def request(engine: Engine, spec: dict, text: str, max_tokens: int, timeout: float) -> dict:
    r = stream_chat(engine.base, payload(spec, text, max_tokens), timeout)
    usage = r["usage"] or {}
    n = usage.get("completion_tokens") or 0
    dec = (r["last_token_s"] - r["ttft_s"]) if r["ttft_s"] is not None and r["last_token_s"] else None
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": n, "ttft_s": r["ttft_s"],
            "e2e_s": r["e2e_s"], "decode_tok_s": (n - 1) / dec if dec and n > 1 else None,
            "gen_s": dec, "finish_reason": r["finish_reason"], "error": r["error"], "text": r["text"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--loop-s", type=float, default=300)
    ap.add_argument("--rest-s", type=float, default=180)
    ap.add_argument("--prefill-tokens", type=int, default=PREFILL_TOKENS)
    a = ap.parse_args()
    spec = json.loads(Path(a.engine).read_text())
    out = Path(a.out)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cell = out / "cells" / f"{stamp}-{spec['id']}-r{a.round}"
    cell.mkdir(parents=True)
    from tokenizers import Tokenizer

    vars_ = local_vars()
    tok = Tokenizer.from_file(str(Path(expand(spec["tokenizer_dir"], vars_)) / "tokenizer.json"))
    engine = Engine(spec, cell, vars_)
    rows = out / "rows.jsonl"
    mon = None

    def emit(row: dict) -> None:
        text = row.pop("text", "")
        (cell / f"{row['phase']}-{row.get('i', 0)}.txt").write_text(text)
        with rows.open("a") as fh:
            fh.write(json.dumps({"engine": spec["id"], "round": a.round, "stamp": stamp, **row}) + "\n")
        print(json.dumps({k: row.get(k) for k in ("phase", "i", "prompt_tokens", "completion_tokens",
                                                  "ttft_s", "decode_tok_s", "error")}), flush=True)

    try:
        load_s = engine.start()
        mon = Monitor(cell / "monitor.jsonl", port_pids(engine.port)[0])
        mon.start()
        emit({"phase": "warmup", **request(engine, spec, "Say hello in five words.", 16, 600), "load_s": load_s})
        mon.phase = "loop"
        t_end = time.time() + a.loop_s
        i = 0
        while time.time() < t_end:
            r = request(engine, spec, PROMPTS[i % len(PROMPTS)], 250, 900)
            emit({"phase": "loop", "i": i, "prompt_id": i % len(PROMPTS), **r, "t_end": time.time()})
            i += 1
            if r["error"]:
                break
        mon.phase = "rest"
        time.sleep(a.rest_s)
        mon.phase = "prefill44k"
        context, n = build_context(tok, a.prefill_tokens, 200)
        r = request(engine, spec, context + "\n\nIn one sentence, what are the records above?", 16, 5400)
        emit({"phase": "prefill44k", "context_tokens": n, **r})
        mon.phase = "after"
        time.sleep(15)
    finally:
        if mon:
            mon.stop_event.set()
            mon.join(timeout=5)
        engine.stop()


if __name__ == "__main__":
    main()
