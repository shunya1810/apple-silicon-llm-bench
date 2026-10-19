#!/usr/bin/env python3
"""Start each engine, send turn 1 of a scenario with 8 output tokens, print prompt_tokens (same prompt check)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runner
from tokenizers import Tokenizer
scen = json.loads(Path(sys.argv[1]).read_text())
for eng in sys.argv[2:]:
    spec = json.loads(Path(eng).read_text())
    v = spec["kv_variants"][spec.get("default_kv", "q8")]
    spec = {**spec, **{k: x for k, x in v.items() if k != "cmd_extra"}, "cmd": spec["cmd"] + v.get("cmd_extra", [])}
    tok = Tokenizer.from_file(str(Path(runner.expand(spec["tokenizer_dir"], runner.local_vars())) / "tokenizer.json"))
    ctx, _ = runner.build_context(tok, scen["context_tokens"], scen.get("reserve_tokens", 400))
    cell = Path(sys.argv[1]).resolve().parent.parent / "results" / "2026-10-b2-smoke" / "parity" / f"{time.strftime('%H%M%S')}-{spec['id']}"
    cell.mkdir(parents=True, exist_ok=True)
    e = runner.Engine(spec, cell, runner.local_vars())
    try:
        load = e.start()
        r = runner.run_turn(e, spec, [{"role": "user", "content": ctx + scen["turns"][0]}], 8, 900, tok,
                            scen.get("sampling"), True, scen.get("reasoning_effort"))
        print(json.dumps({"engine": spec["id"], "load_s": round(load, 1), "prompt_tokens": r["prompt_tokens"],
                          "ttft_s": r["ttft_s"], "error": r["error"], "text": r["text"][:60]}), flush=True)
    except Exception as exc:
        print(json.dumps({"engine": spec["id"], "error": repr(exc)}), flush=True)
    finally:
        e.stop()
