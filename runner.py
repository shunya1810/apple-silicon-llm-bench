#!/usr/bin/env python3
"""Multi-engine, multi-turn long-context benchmark over the OpenAI-compatible HTTP API.

One cell = (engine, scenario). Each cell starts the engine fresh (engine-side
caches wiped), sends a short warmup, a thermal canary, then the scenario's
turns as one growing chat. Every turn is streamed and timed on the client, so
all engines are measured the same way; engine-reported stats are attached as
`engine_stats` when the adapter knows how to read them.

  python runner.py --engine engines/mtplx-fork.json --scenario scenarios/mt-8k.json \
      --out results/<run>/rows.jsonl --round 1 --order-pos 2

Engines and scenarios are JSON files, so adding an engine is one new file
(see engines/README.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def local_vars() -> dict:
    """Machine-specific paths (config/local.json, not committed) + ROOT, for ${VAR} expansion."""
    p = ROOT / "config" / "local.json"
    vars_ = json.loads(p.read_text()) if p.exists() else {}
    return {**vars_, "ROOT": str(ROOT)}

# ---------------------------------------------------------------- prompt

STATUSES = ("nominal", "degraded", "recovering", "offline", "calibrating")
NEEDLE = "Record NEEDLE: maintenance override for zone Q42 uses code 7391-ALPHA; this is not telemetry.\n"
NEEDLE_ANSWER = "7391-ALPHA"


def build_context(tokenizer, target_tokens: int, reserve_tokens: int) -> tuple[str, int]:
    """Deterministic synthetic telemetry with one needle line at 50% depth."""

    lines = [
        f"Record {i:06d}: sensor={((i * 7919) % 997) / 10:.1f} "
        f"status={STATUSES[(i * 31) % len(STATUSES)]} "
        f"zone={chr(65 + (i * 13) % 26)}{(i * 17) % 90 + 10}.\n"
        for i in range(target_tokens // 20 + 16)
    ]
    budget = max(1, target_tokens - reserve_tokens - len(tokenizer.encode(NEEDLE).ids))
    lo, hi = 1, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.encode("".join(lines[:mid])).ids) <= budget:
            lo = mid
        else:
            hi = mid - 1
    body = lines[:lo]
    body.insert(len(body) // 2, NEEDLE)
    text = "".join(body)
    return text, len(tokenizer.encode(text).ids)


# ---------------------------------------------------------------- engine process


def port_pids(port: int) -> list[int]:
    out = subprocess.run(["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def free_port(port: int) -> None:
    for sig, wait in ((signal.SIGINT, 60), (signal.SIGKILL, 20)):
        for pid in port_pids(port):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.time() + wait
        while time.time() < deadline and port_pids(port):
            time.sleep(1)
    if port_pids(port):
        raise RuntimeError(f"port {port} still busy")


def expand(value, env: dict):
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: str(env[m.group(1)]), value)
    if isinstance(value, list):
        return [expand(v, env) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, env) for k, v in value.items()}
    return value


def get_json(url: str, timeout: float = 30.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


class Engine:
    def __init__(self, spec: dict, cell_dir: Path, vars_: dict):
        self.spec = expand(spec, {**os.environ, **vars_})
        self.port = int(self.spec["port"])
        self.base = f"http://127.0.0.1:{self.port}"
        self.cell_dir = cell_dir
        self.proc = None

    def start(self) -> float:
        free_port(self.port)
        for path in self.spec.get("wipe_before_start", []):
            shutil.rmtree(path, ignore_errors=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", **self.spec.get("env", {}))
        log = (self.cell_dir / "server.log").open("w")
        t0 = time.time()
        self.proc = subprocess.Popen(self.spec["cmd"], stdout=log, stderr=subprocess.STDOUT, env=env,
                                     cwd=self.spec.get("cwd"), start_new_session=True)
        deadline = t0 + float(self.spec.get("start_timeout_s", 900))
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"engine exited early ({self.proc.returncode}), see {log.name}")
            try:
                get_json(self.base + self.spec.get("health_path", "/v1/models"), timeout=5)
                return time.time() - t0
            except Exception:
                time.sleep(2)
        raise RuntimeError("engine did not become healthy")

    def stop(self) -> None:
        try:
            free_port(self.port)
        finally:
            if self.proc and self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass  # group already gone (the child re-exec'd into its own session)

    def version(self) -> dict:
        out = {}
        for key, cmd in self.spec.get("version_cmds", {}).items():
            out[key] = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        return out

    def stats(self, log_offset: int) -> dict | None:
        """Engine-reported stats for the request that just finished (adapter specific)."""
        kind = self.spec.get("stats")
        try:
            if kind == "mtplx-metrics":
                latest = get_json(self.base + "/metrics", timeout=60).get("latest") or {}
                keys = ("prompt_tokens", "cached_tokens", "new_prefill_tokens", "completion_tokens",
                        "ttft_s", "prefill_tok_s", "decode_tok_s", "verify_calls", "accepted_drafts",
                        "drafted_tokens", "accepted_by_depth", "peak_memory_bytes", "cache_source",
                        "paged_kv_quant_mode", "verify_time_s", "draft_time_s")
                return {k: latest.get(k) for k in keys}
            if kind == "omlx-log":
                text = (self.cell_dir / "server.log").read_text(errors="replace")[log_offset:]
                out = {}
                m = re.findall(r"MTP\[\d+\] finish=\S+ tokens=(\d+) cycles=(\d+) tok/cycle=([\d.]+) "
                               r"accept=(\d+)/(\d+)", text)
                if m:
                    t, c, tpc, a, d = m[-1]
                    out.update(tokens=int(t), cycles=int(c), tok_per_cycle=float(tpc),
                               accepted_drafts=int(a), drafted_tokens=int(d))
                m = re.findall(r"Chat completion: .*? (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\), prompt: (\d+)",
                               text)
                if m:
                    out["server_line"] = dict(zip(("completion_tokens", "total_s", "tok_s", "prompt_tokens"),
                                                  map(float, m[-1])))
                m = re.findall(r"(?i)cache hit[^\n]*?(\d+)\s*(?:/\s*\d+\s*)?tokens", text)
                if m:
                    out["cache_hit_tokens_log"] = int(m[-1])
                return out
        except Exception as exc:
            return {"error": repr(exc)}
        return None

    def log_size(self) -> int:
        p = self.cell_dir / "server.log"
        return p.stat().st_size if p.exists() else 0


# ---------------------------------------------------------------- telemetry


def vm_bytes() -> dict:
    out, page = {}, 16384
    for line in subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        for key, name in (("Pages wired down", "wired"), ("Pages occupied by compressor", "compressed")):
            if line.startswith(key):
                out[name] = int(line.split(":")[1].strip().rstrip(".")) * page
    return out


def tree_rss(root_pid: int) -> int:
    """RSS of the engine process group (MTPLX serve spawns the server as a child)."""
    ps = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    kids: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in ps.splitlines():
        pid, ppid, r = map(int, line.split())
        kids.setdefault(ppid, []).append(pid)
        rss[pid] = r * 1024
    total, stack = 0, [root_pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack.extend(kids.get(p, []))
    return total


class Sampler(threading.Thread):
    def __init__(self, pid: int, path: Path, interval: float = 1.0):
        super().__init__(daemon=True)
        self.pid, self.path, self.interval = pid, path, interval
        self.stop_event = threading.Event()
        self.window = {"rss": 0, "wired": 0}

    def reset_window(self) -> dict:
        w, self.window = self.window, {"rss": 0, "wired": 0}
        return w

    def run(self) -> None:
        with self.path.open("a") as fh:
            while not self.stop_event.is_set():
                try:
                    s = {"t": time.time(), "rss": tree_rss(self.pid), **vm_bytes()}
                    self.window["rss"] = max(self.window["rss"], s["rss"])
                    self.window["wired"] = max(self.window["wired"], s.get("wired", 0))
                    fh.write(json.dumps(s) + "\n")
                    fh.flush()
                except Exception:
                    pass
                self.stop_event.wait(self.interval)


# ---------------------------------------------------------------- request


def stream_chat(base: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    first = last = None
    parts: list[str] = []
    chunks = 0
    usage = finish = error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                error = obj.get("error") or error
                usage = obj.get("usage") or usage
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = "".join(delta.get(k) or "" for k in ("content", "reasoning_content", "reasoning"))
                    if piece:
                        now = time.perf_counter() - t0
                        first = now if first is None else first
                        last = now
                        chunks += 1
                        parts.append(piece)
                    finish = ch.get("finish_reason") or finish
    except Exception as exc:
        error = repr(exc)
    text = "".join(parts)
    return {"e2e_s": time.perf_counter() - t0, "ttft_s": first, "last_token_s": last, "chunks": chunks,
            "text": text, "usage": usage, "finish_reason": finish, "error": error}


def run_turn(engine: Engine, spec: dict, messages: list, max_tokens: int, timeout: float, tokenizer) -> dict:
    payload = {"model": engine.spec["model_id"], "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.0, "top_p": 1.0, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    payload.update(spec.get("request_extra", {}))
    offset = engine.log_size()
    r = stream_chat(engine.base, payload, timeout)
    usage = r["usage"] or {}
    completion = usage.get("completion_tokens") or len(tokenizer.encode(r["text"]).ids)
    decode_s = (r["last_token_s"] - r["ttft_s"]) if r["ttft_s"] is not None and r["last_token_s"] else None
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens_usage": details.get("cached_tokens"),
        "completion_tokens": completion,
        "ttft_s": r["ttft_s"],
        "e2e_s": r["e2e_s"],
        "decode_tok_s": (completion - 1) / decode_s if decode_s and completion > 1 else None,
        "stream_chunks": r["chunks"],
        "finish_reason": r["finish_reason"],
        "error": r["error"],
        "text": r["text"],
        "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
        "engine_stats": engine.stats(offset),
    }


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--order-pos", type=int, default=0)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--kv", default=None, help="override the engine's kv variant (e.g. fp16)")
    a = ap.parse_args()

    from tokenizers import Tokenizer

    spec = json.loads(Path(a.engine).read_text())
    scen = json.loads(Path(a.scenario).read_text())
    kv = a.kv or spec.get("default_kv", "q8")
    variant = spec["kv_variants"][kv]
    spec = {**spec, **{k: v for k, v in variant.items() if k != "cmd_extra"},
            "cmd": spec["cmd"] + variant.get("cmd_extra", [])}
    model_dir = expand(spec["tokenizer_dir"], {**os.environ, **local_vars()})
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cell_dir = out.parent / "cells" / f"{stamp}-{spec['id']}-{kv}-{scen['id']}-r{a.round}a{a.attempt}"
    cell_dir.mkdir(parents=True)

    context, ctx_tokens = build_context(tok, scen["context_tokens"], scen.get("reserve_tokens", 400))
    engine = Engine(spec, cell_dir, local_vars())
    base_row = {"run_stamp": stamp, "engine": spec["id"], "engine_label": spec["label"], "kv": kv,
                "kv_impl": spec.get("kv_impl"), "mtp_impl": spec.get("mtp_impl"),
                "model_artifact": spec.get("model_artifact"), "scenario": scen["id"],
                "context_target": scen["context_tokens"], "context_tokens_raw": ctx_tokens,
                "round": a.round, "order_pos": a.order_pos, "attempt": a.attempt}
    sampler = None
    try:
        load_s = engine.start()
        base_row.update(engine_version=engine.version(), load_s=load_s)
        root_pid = port_pids(engine.port)[0]
        sampler = Sampler(root_pid, cell_dir / "memory.jsonl")
        sampler.start()
        timeout = float(scen.get("turn_timeout_s", 3600))

        def emit(row: dict) -> None:
            text = row.pop("text", "")
            (cell_dir / f"{row['phase']}-t{row.get('turn', 0)}.txt").write_text(text)
            with out.open("a") as fh:
                fh.write(json.dumps({**base_row, **row}) + "\n")
            print(json.dumps({k: row.get(k) for k in ("phase", "turn", "prompt_tokens", "cached_tokens_usage",
                                                      "ttft_s", "decode_tok_s", "e2e_s", "peak_wired",
                                                      "needle_ok", "error")}), flush=True)

        # warmup (JIT/compile), excluded from results but recorded
        sampler.reset_window()
        w = run_turn(engine, spec, [{"role": "user", "content": "Say hello in five words."}], 16, 600, tok)
        emit({"phase": "warmup", **w, **{f"peak_{k}": v for k, v in sampler.reset_window().items()}})

        # thermal canary: fixed ~2K prompt, decode speed is the machine-state probe
        canary_ctx, _ = build_context(tok, 2048, 200)
        c = run_turn(engine, spec, [{"role": "user", "content": canary_ctx + scen["canary_question"]}],
                     128, 900, tok)
        emit({"phase": "canary", **c, **{f"peak_{k}": v for k, v in sampler.reset_window().items()}})

        messages: list = []
        for i, q in enumerate(scen["turns"], 1):
            content = (context + q) if i == 1 else q
            messages.append({"role": "user", "content": content})
            sampler.reset_window()
            r = run_turn(engine, spec, messages, scen["max_tokens"], timeout, tok)
            peaks = sampler.reset_window()
            r.update(phase="turn", turn=i, peak_rss=peaks["rss"], peak_wired=peaks["wired"],
                     needle_ok=(NEEDLE_ANSWER in r["text"]) if scen.get("needle_turn") == i else None)
            messages.append({"role": "assistant", "content": r["text"]})
            emit(r)
            if r["error"]:
                break
    finally:
        if sampler:
            sampler.stop_event.set()
        engine.stop()


if __name__ == "__main__":
    main()
