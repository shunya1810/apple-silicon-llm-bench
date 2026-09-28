# Apple Silicon LLM engine benchmark: long-context, multi-turn

English | [日本語](README.ja.md)

How fast is a **3-turn conversation over a long prompt** on a Mac, depending on the
inference engine? On a MacBook Pro with M1 Max and 64 GB, five inference engines serve Qwen3.8-27B with speculative decoding and 8-bit KV cache
(TensorFold has no 8-bit KV and runs bf16 KV), measured the same way over
each engine's OpenAI-compatible streaming API. The three MTPLX/oMLX engines load the
**same model files** with MTP (depth 3); Splash and TensorFold need their own packages
(4-bit, group-64 weights plus a DFlash2 draft model), so their columns also reflect a
different quantization and a different speculative method.

| engine | version | notes |
|---|---|---|
| **MTPLX fork** | [`shunya1810/MTPLX@0cd2a73`](https://github.com/shunya1810/MTPLX/tree/m1max-longctx) | M1-family long-context branch, maintained by the author of this benchmark; re-run on 2026-09-26–28 after it turned context-copy off on M1 (`40b6113`), lowered its M1 memory defaults (`4fe8067`), and fused small projections and pruned the draft head on M1 (`bba1b7b`), and enlarged its M1 Metal command buffers (`0cd2a73`); the earlier rows stay in the raw data as `mtplx-fork-16751dc`, `mtplx-fork-40b6113`, `mtplx-fork-4fe8067` and `mtplx-fork-bba1b7b` |
| **MTPLX upstream** | [`youssofal/MTPLX@1de2b1c`](https://github.com/youssofal/MTPLX/commit/1de2b1c049136ed117af0c6712baaadd81820b51) | `main` at run time; the fork's base |
| **oMLX** | [0.6.4](https://github.com/jundot/omlx/releases/tag/v0.6.4) | latest stable at run time |
| **Splash 1.1.0-m1** | [paperniuk/splash 1.1.0-m1](https://github.com/paperniuk/splash/releases/tag/1.1.0-m1) | community M1/M2 build of [incoai/splash](https://github.com/incoai/splash) 1.1.0 with Apple7/8 kernels; model `incoai/Qwen3.8-27B-Splash` (4-bit g64 + DFlash2 draft), INT8 KV; replaces the 1.0.2-m1 release and a source build of its head measured earlier (raw data: `splash-m1-102`, `splash-src`) |
| **TensorFold** | [0.3.5.1](https://github.com/ashhart/TensorFold/releases/tag/v0.3.5.1) | Python/MLX engine with exact (token-equality) draft acceptance; model `Vontra/Qwen3.8-27B-MLX-4bit` (4-bit g64) + `z-lab/Qwen3.8-27B-DFlash2` draft (4-bit at load), bf16 KV (no 8-bit option); release of 2026-09-28, the first that loads on M1 Max |

Prompt lengths 2K–64K ran in 2 rounds each (median shown); 128K ran once. Splash ran
after the other three, in its own two rounds and one 128K run; TensorFold ran last, the same way.

## Highlights (8-bit KV)

- **Decode:** at 128K the MTPLX fork decodes at **19.6 tok/s**, oMLX at 7.4 and MTPLX
  upstream at 4.9 (mean of the three turns); at 64K it is 24.3, 10.7 and 8.7. The fork's
  lead over the other two grows with the prompt: +33–56% at 2K–8K, 1.9–2.0× at 32K,
  2.3–2.8× at 64K and 2.6–4.0× at 128K.
- **Follow-up turns:** time to first token for turns 2 and 3 at 128K is **2.2 s** on the
  fork, 3.5–4.2 s on upstream, and 1.6 min / 7.9 s on oMLX (64K: 1.4 s, 2.1 s, 1.0 min /
  4.8 s). oMLX keeps its prefix cache on SSD in 4,096-token blocks and, for this model
  family, can save a prompt only up to the last block boundary it crossed, so the next
  turn re-reads the rest (up to ~4K tokens).
- **First turn:** up to 64K, reading the prompt cold takes about the same time on the
  MTPLX/oMLX engines (9.8–10.2 min at 64K, 11.4–11.7 s at 2K): the prefill is bound by the
  same MLX quantized matrix multiply (about 7.9 TFLOPS at 2K on this GPU). At 128K
  attention becomes a large share and they separate: 24.7 min (fork), 26.3 min (oMLX),
  30.5 min (upstream).
- **Whole conversation at 128K:** 25.3 min (fork), 29.6 min (oMLX), 33.0 min (upstream).
- **Memory:** peak wired memory at 128K is 43.1 GB (fork), 48.5 GB (oMLX), 52.7 GB
  (upstream); at 64K 37.8, 39.6 and 48.4 GB. The fork's `4fe8067` defaults (MLX buffer
  cache capped at 1 GiB, two saved states per conversation, MMA prefill attention from
  the first chunk) took 3.4–6.4 GB off its 32K–128K peaks at the same speed (`40b6113`:
  37.8 / 44.1 / 48.0 GB; `4fe8067`: 34.2 / 37.7 / 44.6 GB).
- **Fork `bba1b7b` against `4fe8067`:** decode +5.4% (2K), +6.3% (8K), +3.3% (32K),
  +3.2% (64K) and +1.7% (128K). The M1 draft read a 715 MB draft head three times per
  round; the pruned FR-Spec head (a code-ranked 64K list plus Japanese tokens, 42% of the
  vocabulary) was not reaching that path. Small attention/GDN projections are now fused.
  Acceptance and memory are unchanged; turns 2–3 at 2K–8K generate different text (the
  fused lane changes the follow-up prefill's rounding).
- **Fork `0cd2a73` against `bba1b7b`:** decode +9.6% (2K), +6.3% (8K), +8.9% (32K),
  +7.5% (64K) and +6.5% (128K). On M1 the fork now lets MLX put up to 150 operations
  and 1,000 MB into one Metal command buffer (a round was split into about 100 buffers,
  and the GPU sat idle at each boundary: 13% of a round), and evaluates the prefill every
  four layers so the larger buffers do not raise its peak. Cold prefill at 128K is 4%
  faster, peak memory is the same or lower, and all 18 turns generate the same text as
  `bba1b7b`.
- **Splash 1.1.0-m1:** decode 33.1 / 32.5 / 27.1 / 24.1 / 18.7 tok/s at 2K / 8K / 32K /
  64K / 128K (mean of the turns), against 31.9 / 30.3 / 27.0 / 24.3 / 19.6 on the fork:
  4–7% ahead at 2K–8K, even at 32K–64K, 5% behind at 128K. It is fastest on turn 2
  (39.2 tok/s at 2K, 22.8 at 128K, against 32.2 and 20.8 on the fork) and slower on turns
  1 and 3 from 32K up (128K: 15.9 / 17.4 against 19.0 / 19.0). Its cold prefill is
  slower from 32K (64K: 10.9 min against 9.8; 128K: 28.5 min against 24.7) and follow-up
  turns re-read ~300 tokens (TTFT 2.6–6.3 s against 0.6–2.2 s on the fork); the whole
  128K conversation takes 29.4 min against 25.3. It uses the least memory: 22–26 GB wired
  at every length, against 30–43 GB on the fork.
- **Splash 1.1.0-m1 against the earlier Splash builds:** it finishes 128K, which the
  1.0.2-m1 release could not (reading 128K cold outlasted that release's 30-minute request
  deadline; 1.1.0 defaults to 10,000 s and splits the prefill into GPU commands of about
  5 s). Decode is +14% (8K), +12% (32K) and +25% (64K) over 1.0.2-m1, and +9% (64K) and
  +18% (128K) over a source build of the port's earlier head (equal at 2K–32K); cold
  prefill at 64K takes 10.9 min against 12.8 (1.0.2-m1) and 12.3 (source build), and 34.3
  min at 128K on the source build against 28.5. Those rows stay in the raw data.
- **TensorFold 0.3.5.1 (bf16 KV):** decode 27.7 / 25.5 / 16.9 / 11.3 / 6.4 tok/s at 2K /
  8K / 32K / 64K / 128K (mean of the turns): 13–16% behind the fork at 2K–8K and a third
  of it at 128K. Cold prefill is the slowest of the five up to 64K (64K: 11.9 min) and
  follow-up turns take 3.5–5.3 s to the first token up to 64K. At 128K its default prompt
  cache (an eighth of RAM, 8 GiB) cannot hold the conversation, so turns 2 and 3 re-read
  all 131K tokens (29.3–29.4 min each) and the conversation takes 89.6 min; a larger
  `--prompt-cache-gib` may avoid that, but every engine runs at its defaults here. Peak
  wired memory 22.6–37.2 GB. It accepts a draft token only when it equals the target's own
  sample (exact decoding, output independent of drafting), which works at temperature 0
  but accepts fewer tokens when sampling.
- **Output check:** every engine quoted the needle line correctly in turn 3 in every
  completed run.

## Decode speed

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-dark.svg?v=tensorfold">
  <img alt="Decode tok/s vs prompt length, five engines" src="charts/decode-light.svg?v=tensorfold">
</picture>

## Follow-up turns

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/ttft-followup-dark.svg?v=tensorfold">
  <img alt="Time to first token for turns 2-3, log scale" src="charts/ttft-followup-light.svg?v=tensorfold">
</picture>

## The whole conversation

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/conversation-dark.svg?v=tensorfold">
  <img alt="Three turns end to end per engine and prompt length" src="charts/conversation-light.svg?v=tensorfold">
</picture>

## Memory

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/memory-dark.svg?v=tensorfold">
  <img alt="Peak wired memory vs prompt length" src="charts/memory-light.svg?v=tensorfold">
</picture>

## Tables

Median of 2 rounds per cell for 2K–64K; 128K is a single run.
`needle` counts the runs whose turn 3 quoted the needle line correctly. For Splash, the
2K fp16 cells use its BF16 KV (`--kv-format bf16`). `cached` is the engine-reported
`usage.prompt_tokens_details.cached_tokens`. Wired memory is system-wide.

| context | engine | T1 TTFT (cold) | T2 TTFT | T3 TTFT | decode T1 / T2 / T3 (tok/s) | 3-turn total | cached T2 / T3 | peak wired | needle |
|---|---|---|---|---|---|---|---|---|---|
| 2K | MTPLX fork | 11.6 s | 0.61 s | 0.62 s | 31.5 / 32.2 / 32.1 | 34.3 s | 1,936 / 2,243 | 30.0 GB | 2/2 |
| 2K | MTPLX upstream | 11.5 s | 0.64 s | 0.64 s | 25.1 / 22.9 / 23.2 | 41.8 s | 1,936 / 2,243 | 34.0 GB | 2/2 |
| 2K | oMLX | 11.4 s | 13.5 s | 15.4 s | 20.4 / 23.5 / 21.0 | 1.2 min | 0 / 0 | 26.9 GB | 2/2 |
| 2K | Splash 1.1.0-m1 | 12.0 s | 2.56 s | 2.76 s | 31.3 / 39.2 / 28.8 | 37.8 s | 1,664 / 1,952 | 21.9 GB | 2/2 |
| 2K | TensorFold (bf16 KV) | 15.7 s | 3.47 s | 3.48 s | 26.1 / 28.9 / 28.0 | 47.0 s | 1,673 / 1,980 | 22.9 GB | 2/2 |
| 8K | MTPLX fork | 1.0 min | 0.72 s | 0.71 s | 30.0 / 32.2 / 28.8 | 1.4 min | 8,089 / 8,396 | 30.7 GB | 2/2 |
| 8K | MTPLX upstream | 54.2 s | 0.76 s | 0.77 s | 23.5 / 23.2 / 21.7 | 1.4 min | 8,089 / 8,396 | 35.5 GB | 2/2 |
| 8K | oMLX | 56.4 s | 30.0 s | 2.69 s | 20.2 / 19.8 / 18.3 | 2.1 min | 4,096 / 8,192 | 33.2 GB | 2/2 |
| 8K | Splash 1.1.0-m1 | 56.8 s | 2.91 s | 2.79 s | 31.1 / 37.4 / 29.1 | 1.4 min | 7,808 / 8,128 | 22.1 GB | 2/2 |
| 8K | TensorFold (bf16 KV) | 1.2 min | 3.68 s | 3.69 s | 24.6 / 26.0 / 26.0 | 1.8 min | 7,826 / 8,132 | 22.6 GB | 2/2 |
| 32K | MTPLX fork | 4.2 min | 0.97 s | 0.98 s | 27.5 / 26.9 / 26.7 | 4.7 min | 32,665 / 32,972 | 33.7 GB | 2/2 |
| 32K | MTPLX upstream | 4.1 min | 1.38 s | 1.37 s | 17.1 / 12.5 / 12.0 | 5.0 min | 32,665 / 32,972 | 40.8 GB | 2/2 |
| 32K | oMLX | 4.4 min | 44.6 s | 3.45 s | 14.4 / 14.0 / 14.4 | 6.0 min | 28,672 / 32,768 | 37.0 GB | 2/2 |
| 32K | Splash 1.1.0-m1 | 4.5 min | 3.57 s | 3.57 s | 23.3 / 29.1 / 28.7 | 5.1 min | 32,384 / 32,704 | 22.9 GB | 2/2 |
| 32K | TensorFold (bf16 KV) | 5.3 min | 4.31 s | 4.31 s | 15.4 / 18.9 / 16.5 | 6.1 min | 32,402 / 32,709 | 27.1 GB | 2/2 |
| 64K | MTPLX fork | 9.8 min | 1.36 s | 1.36 s | 23.8 / 25.3 / 23.9 | 10.3 min | 65,419 / 65,726 | 37.8 GB | 2/2 |
| 64K | MTPLX upstream | 9.9 min | 2.07 s | 2.12 s | 8.8 / 8.7 / 8.4 | 11.3 min | 65,419 / 65,726 | 48.4 GB | 2/2 |
| 64K | oMLX | 10.2 min | 1.0 min | 4.76 s | 10.9 / 10.9 / 10.4 | 12.4 min | 61,440 / 65,536 | 39.6 GB | 2/2 |
| 64K | Splash 1.1.0-m1 | 10.9 min | 4.28 s | 4.90 s | 23.2 / 29.9 / 19.1 | 11.5 min | 65,152 / 65,440 | 24.1 GB | 2/2 |
| 64K | TensorFold (bf16 KV) | 11.9 min | 5.25 s | 5.19 s | 10.8 / 13.1 / 10.0 | 13.1 min | 65,156 / 65,463 | 33.2 GB | 2/2 |
| 128K | MTPLX fork | 24.7 min | 2.15 s | 2.15 s | 19.0 / 20.8 / 19.0 | 25.3 min | 130,969 / 131,276 | 43.1 GB | 1/1 |
| 128K | MTPLX upstream | 30.5 min | 4.16 s | 3.49 s | 4.6 / 5.1 / 5.0 | 33.0 min | 130,969 / 131,276 | 52.7 GB | 1/1 |
| 128K | oMLX | 26.3 min | 1.6 min | 7.92 s | 7.6 / 7.8 / 6.9 | 29.6 min | 126,976 / 131,072 | 48.5 GB | 1/1 |
| 128K | Splash 1.1.0-m1 | 28.5 min | 6.27 s | 6.20 s | 15.9 / 22.8 / 17.4 | 29.4 min | 130,688 / 131,008 | 26.2 GB | 1/1 |
| 128K | TensorFold (bf16 KV) | 29.2 min | 29.3 min | 29.4 min | 6.3 / 6.5 / 6.3 | 89.6 min | 0 / 0 | 37.2 GB | 1/1 |

**2K: fp16 KV vs 8-bit KV** (decode tok/s, mean of turns 1–3)

| engine | fp16 KV | 8-bit KV | change |
|---|---|---|---|
| MTPLX fork | 29.9 | 31.9 | +6.6% |
| MTPLX upstream | 24.8 | 23.7 | -4.5% |
| oMLX | 23.1 | 21.6 | -6.4% |
| Splash 1.1.0-m1 | 33.5 | 33.1 | -1.3% |

Raw rows: [`results/2026-09-m1max-64gb/rows.jsonl`](results/2026-09-m1max-64gb/rows.jsonl) ·
per-turn CSV: [`summary.csv`](results/2026-09-m1max-64gb/summary.csv) ·
environment: [`environment.json`](results/2026-09-m1max-64gb/environment.json) ·
generated text of every turn: `results/2026-09-m1max-64gb/cells/*/`.

## Why Splash is fastest on turn 2

Splash reports its draft and verify counters on `/status`, so each turn can be broken
down into verify passes (one DFlash2 draft of 7 tokens plus one verify of 8 positions).
Splash 1.1.0-m1, round 1, 8-bit KV:

| context | turn | decode tok/s | verify passes | drafted / pass | accepted / pass | tokens / pass | acceptance | ms / pass |
|---|---|---|---|---|---|---|---|---|
| 2K | 1 | 31.2 | 75 | 7.0 | 2.24 | 3.25 | 32% | 105 |
| 2K | 2 | 39.2 | 63 | 7.0 | 3.06 | 4.06 | 44% | 105 |
| 2K | 3 | 28.8 | 60 | 7.0 | 1.98 | 3.00 | 28% | 105 |
| 8K | 1 | 31.1 | 76 | 7.0 | 2.34 | 3.36 | 33% | 109 |
| 8K | 2 | 37.1 | 63 | 7.0 | 3.06 | 4.06 | 44% | 111 |
| 8K | 3 | 29.3 | 57 | 7.0 | 2.16 | 3.18 | 31% | 109 |
| 32K | 1 | 24.0 | 86 | 7.0 | 1.98 | 2.98 | 28% | 125 |
| 32K | 2 | 29.1 | 71 | 7.0 | 2.59 | 3.61 | 37% | 125 |
| 32K | 3 | 28.7 | 47 | 7.0 | 2.55 | 3.57 | 36% | 126 |
| 64K | 1 | 23.2 | 76 | 7.0 | 2.37 | 3.37 | 34% | 146 |
| 64K | 2 | 28.8 | 57 | 7.0 | 3.47 | 4.49 | 50% | 157 |
| 64K | 3 | 19.1 | 66 | 7.0 | 1.79 | 2.80 | 26% | 148 |
| 128K | 1 | 15.9 | 82 | 7.0 | 2.12 | 3.12 | 30% | 197 |
| 128K | 2 | 22.8 | 60 | 7.0 | 3.27 | 4.27 | 47% | 189 |
| 128K | 3 | 17.4 | 57 | 7.0 | 2.25 | 3.26 | 32% | 189 |

- **Turn 2 is the only turn where the draft is accepted often:** 37–50% of drafted tokens,
  against 26–36% on turns 1 and 3, so a pass commits 3.6–4.5 tokens instead of 2.8–3.4. The
  cost of a pass does not depend on the turn, so the extra tokens are the whole difference.
  The turn-2 answer repeats the comparison pattern set up in turn 1, which a 7-token draft
  can follow further than the 3-token MTP draft; the MTPLX fork commits 2.8–3.0 tokens per
  round on turn 2 and 2.7–2.9 on turns 1 and 3, with a ceiling of 4.
- **Splash verifies twice as many positions per pass at a similar cost at short context:**
  105 ms per pass at 2K for 8 positions and a 7-token draft, against 88–91 ms per round on
  the fork for 4 positions and a 3-token MTP draft.
- **The pass cost grows faster with the prompt:** from 105 ms (2K) to 189–197 ms (128K)
  on Splash, against 88–91 to 140–142 ms per round on the fork. Each pass attends over the
  whole history for 8 positions instead of 4, which most likely explains why Splash falls
  behind on turns 1 and 3 at long context.
- 1.1.0-m1 passes cost less than those of the source build measured earlier (64K: 146–157
  against 163–166 ms; 128K: 189–197 against 218–225 ms), which is where its gain at long
  context comes from; acceptance is about the same.

## How it was measured

Short version (full details in [METHODOLOGY.md](METHODOLOGY.md)):

- Turn 1: a synthetic telemetry log of the target length with one "needle" line at 50%
  depth, plus a question. Turns 2 and 3 add the previous answer and a new question
  (~300 tokens); turn 3 asks for the needle line. At most 256 generated tokens per turn.
- `temperature 0`, thinking off. These speeds compare engines on the same deterministic
  workload; with sampling on, MTP acceptance and so decode speed change.
- Same model files for the MTPLX and oMLX engines: oMLX reads the MTPLX checkpoint
  through a symlinked, text-only view ([`scripts/prepare_omlx.py`](scripts/prepare_omlx.py));
  no weight is converted. Splash reads its own package, `incoai/Qwen3.8-27B-Splash`.
- 8-bit KV is each engine's own implementation (MTPLX affine q8 paged KV, oMLX
  TurboQuant 8-bit). Everything else is at engine defaults, including the prefix cache
  (MTPLX: RAM; oMLX: SSD, 4,096-token blocks).
- Every cell restarts the engine with its caches wiped, then runs a warmup and a
  thermal canary. Engine order is reversed in round 2. A cell whose canary was >3%
  slower than that engine's best was re-run after a 5-minute rest (this happened once).
  128K ran once, in the order upstream → oMLX → fork (the fork last, on the warmest
  machine); its canaries were 1.7% (fork), 2.5% (upstream) and 4.8% (oMLX) below each
  engine's best canary of the 2K–64K run, and those cells were not re-run. The fork's
  column was later re-run alone at `40b6113`, `4fe8067`, `bba1b7b` and `0cd2a73` with the same plans
  (128K canary 0.2% below its 2K–64K best in the `bba1b7b` run; in the `0cd2a73` run the
  8K cell of round 1 stayed 5% slow on its re-run, just after a reboot, and is kept).
- Round-to-round decode difference: median 1.3%, largest 10% (oMLX, 64K turn 3, where the
  two runs generated different text). Both MTPLX arms produced byte-identical text in both
  rounds in all 15 of their turns; oMLX in 12 of 15.

## Reproduce

```bash
cp config/local.example.json config/local.json   # set your paths
python3 scripts/prepare_omlx.py                   # oMLX model view + base paths
./run_day.sh                                      # 2K-64K, two rounds (~3 h)
./run_night.sh                                    # 128K, one round (~3 h)
python3 scripts/summarize.py results/2026-09-m1max-64gb
```

MTPLX arms run from git worktrees of the two commits under `work/worktrees/`
(`git worktree add --detach work/worktrees/mtplx-<commit> <commit>` in an MTPLX clone),
with the MTPLX app's Python environment. `runner.py` and `summarize.py` need only the
standard library and `tokenizers`.

## Adding an engine

An engine is one JSON file in [`engines/`](engines/README.md): its start command, port,
KV modes and how to read its own stats. Add it to a plan in `plans/` and to the engine
list in `scripts/summarize.py`.

## License

MIT for the code in this repository. Model and engines are under their own licenses.
