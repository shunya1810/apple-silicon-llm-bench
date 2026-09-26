# Apple Silicon LLM engine benchmark: long-context, multi-turn

English | [日本語](README.ja.md)

How fast is a **3-turn conversation over a long prompt** on a Mac, depending on the
inference engine? On a MacBook Pro with M1 Max and 64 GB, four inference engines (plus a
source build of one of them) serve Qwen3.8-27B with speculative decoding and 8-bit KV cache, measured the same way over
each engine's OpenAI-compatible streaming API. The three MTPLX/oMLX engines load the
**same model files** with MTP (depth 3); Splash needs its own package (the
mlx-community 4-bit, group-64 weights plus a DFlash2 draft model), so its column also
reflects a different quantization and a different speculative method.

| engine | version | notes |
|---|---|---|
| **MTPLX fork** | [`shunya1810/MTPLX@4fe8067`](https://github.com/shunya1810/MTPLX/tree/m1max-longctx) | M1-family long-context branch, maintained by the author of this benchmark; re-run on 2026-09-26 after it turned context-copy off on M1 (`40b6113`) and then lowered its M1 memory defaults (`4fe8067`); the earlier rows stay in the raw data as `mtplx-fork-16751dc` and `mtplx-fork-40b6113` |
| **MTPLX upstream** | [`youssofal/MTPLX@1de2b1c`](https://github.com/youssofal/MTPLX/commit/1de2b1c049136ed117af0c6712baaadd81820b51) | `main` at run time; the fork's base |
| **oMLX** | [0.6.4](https://github.com/jundot/omlx/releases/tag/v0.6.4) | latest stable at run time |
| **Splash M1 build** | [paperniuk/splash 1.0.2-m1](https://github.com/paperniuk/splash/releases/tag/1.0.2-m1) | community M1/M2 build of [incoai/splash](https://github.com/incoai/splash) with Apple7 kernels; model `incoai/Qwen3.8-27B-Splash` (4-bit g64 + DFlash2 draft), INT8 KV |
| **Splash source (HEAD)** | [paperniuk/splash@5967821](https://github.com/paperniuk/splash/tree/apple7-m1-kernels) + [a4f7e96](https://github.com/shunya1810/splash/tree/m1-analysis-metrics) | the M1 fork's unreleased head, built locally for analysis (one round; tables and the section below only) |

Prompt lengths 2K–64K ran in 2 rounds each (median shown); 128K ran once. Splash ran
after the other three, in its own two rounds; its 128K cell could not complete (see
below).

## Highlights (8-bit KV)

- **Decode:** at 128K the MTPLX fork decodes at **18.1 tok/s**, oMLX at 7.4 and MTPLX
  upstream at 4.9 (mean of the three turns); at 64K it is 21.9, 10.7 and 8.7. The fork's
  lead over the other two grows with the prompt: +16–38% at 2K–8K, about 1.7× at 32K,
  2.0–2.5× at 64K and 2.4–3.7× at 128K.
- **Follow-up turns:** time to first token for turns 2 and 3 at 128K is **2.1 s** on the
  fork, 3.5–4.2 s on upstream, and 1.6 min / 7.9 s on oMLX (64K: 1.4 s, 2.1 s, 1.0 min /
  4.8 s). oMLX keeps its prefix cache on SSD in 4,096-token blocks and, for this model
  family, can save a prompt only up to the last block boundary it crossed, so the next
  turn re-reads the rest (up to ~4K tokens).
- **First turn:** up to 64K, reading the prompt cold takes about the same time on the
  MTPLX/oMLX engines (9.9–10.2 min at 64K, 11.4–11.7 s at 2K): the prefill is bound by the
  same MLX quantized matrix multiply (about 7.9 TFLOPS at 2K on this GPU). At 128K
  attention becomes a large share and they separate: 25.1 min (fork), 26.3 min (oMLX),
  30.5 min (upstream).
- **Whole conversation at 128K:** 25.9 min (fork), 29.6 min (oMLX), 33.0 min (upstream).
- **Memory:** peak wired memory at 128K is 44.3 GB (fork), 48.5 GB (oMLX), 52.7 GB
  (upstream); at 64K 38.0, 39.6 and 48.4 GB. The fork's `4fe8067` defaults (MLX buffer
  cache capped at 1 GiB, two saved states per conversation, MMA prefill attention from
  the first chunk) took 3.4–6.1 GB off its 32K–128K peaks at the same speed (`40b6113`:
  37.8 / 44.1 / 48.0 GB).
- **Splash M1 build (2K–64K):** its decode is fastest on turn 2 (41.6 tok/s at 2K against
  29.0 on the fork, 31.9 against 29.2 at 8K); on turns 1 and 3 it is close to the fork
  up to 32K (−6% to +13%) and slower at 64K (18.7 / 16.5 against 21.8 / 21.4). Its cold prefill is
  slower (32K: 5.0 min against 4.2; 64K: 12.8 min against 9.9) and follow-up turns
  re-read ~300 tokens (TTFT 2.7–6.3 s against 0.6–1.4 s on the fork). It uses the least
  memory: 22–25 GB wired at every length, against 31–44 GB on the fork. **At 128K it
  could not start the conversation:** reading 128K cold takes longer than Splash's
  30-minute request deadline on this GPU (64K took 12.8 min). Three attempts: the first
  stopped after 23 minutes with a Metal command-buffer error
  (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`); the second ran while another
  GPU load slowed the machine to a third and is not counted; the third, on an idle machine
  (canary 26.8 tok/s), was cut off by Splash at exactly 30.0 minutes (`request timed out`).
  No documented setting changes that deadline in this release.
- **Splash source (HEAD):** the M1 fork's unreleased head, built from source, raises that
  deadline to 10,000 s and adds newer Apple7/8 attention kernels. It finishes 128K (first
  turn 34.3 min, decode 14.4 / 18.5 / 14.6 tok/s) and decodes 10–15% faster than the
  release from 8K up (64K: 22.0 against 19.3 tok/s, mean of the turns). It ran one round
  only, so the charts show it beside the release, which needs no build.
- **Output check:** every engine quoted the needle line correctly in turn 3 in every
  completed run.

## Decode speed

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-dark.svg">
  <img alt="Decode tok/s vs prompt length, five engine builds" src="charts/decode-light.svg">
</picture>

## Follow-up turns

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/ttft-followup-dark.svg">
  <img alt="Time to first token for turns 2-3, log scale" src="charts/ttft-followup-light.svg">
</picture>

## The whole conversation

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/conversation-dark.svg">
  <img alt="Three turns end to end per engine and prompt length" src="charts/conversation-light.svg">
</picture>

## Memory

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/memory-dark.svg">
  <img alt="Peak wired memory vs prompt length" src="charts/memory-light.svg">
</picture>

## Tables

Median of 2 rounds per cell for 2K–64K; 128K and the Splash source build are single runs.
`needle` counts the runs whose turn 3 quoted the needle line correctly. For Splash, the
2K fp16 cells use its BF16 KV (`--kv-format bf16`). `cached` is the engine-reported
`usage.prompt_tokens_details.cached_tokens`. Wired memory is system-wide.

| context | engine | T1 TTFT (cold) | T2 TTFT | T3 TTFT | decode T1 / T2 / T3 (tok/s) | 3-turn total | cached T2 / T3 | peak wired | needle |
|---|---|---|---|---|---|---|---|---|---|
| 2K | MTPLX fork | 11.6 s | 0.62 s | 0.62 s | 27.5 / 29.0 / 26.3 | 37.6 s | 1,936 / 2,243 | 30.8 GB | 2/2 |
| 2K | MTPLX upstream | 11.5 s | 0.64 s | 0.64 s | 25.1 / 22.9 / 23.2 | 41.8 s | 1,936 / 2,243 | 34.0 GB | 2/2 |
| 2K | oMLX | 11.4 s | 13.5 s | 15.4 s | 20.4 / 23.5 / 21.0 | 1.2 min | 0 / 0 | 26.9 GB | 2/2 |
| 2K | Splash M1 build | 12.1 s | 2.87 s | 2.68 s | 27.6 / 41.6 / 29.6 | 39.0 s | 1,664 / 1,984 | 21.7 GB | 2/2 |
| 2K | Splash source (HEAD) | 12.2 s | 2.81 s | 2.62 s | 28.6 / 35.9 / 32.3 | 39.2 s | 1,664 / 1,984 | 21.1 GB | 1/1 |
| 8K | MTPLX fork | 55.1 s | 0.69 s | 0.73 s | 26.7 / 29.2 / 24.3 | 1.4 min | 8,089 / 8,396 | 31.4 GB | 2/2 |
| 8K | MTPLX upstream | 54.2 s | 0.76 s | 0.77 s | 23.5 / 23.2 / 21.7 | 1.4 min | 8,089 / 8,396 | 35.5 GB | 2/2 |
| 8K | oMLX | 56.4 s | 30.0 s | 2.69 s | 20.2 / 19.8 / 18.3 | 2.1 min | 4,096 / 8,192 | 33.2 GB | 2/2 |
| 8K | Splash M1 build | 59.6 s | 3.21 s | 3.05 s | 26.6 / 31.9 / 27.1 | 1.5 min | 7,808 / 8,128 | 21.6 GB | 2/2 |
| 8K | Splash source (HEAD) | 58.6 s | 3.06 s | 2.91 s | 28.5 / 40.1 / 29.2 | 1.4 min | 7,808 / 8,128 | 21.4 GB | 1/1 |
| 32K | MTPLX fork | 4.2 min | 0.97 s | 0.98 s | 24.6 / 24.0 / 23.4 | 4.8 min | 32,665 / 32,972 | 34.4 GB | 2/2 |
| 32K | MTPLX upstream | 4.1 min | 1.38 s | 1.37 s | 17.1 / 12.5 / 12.0 | 5.0 min | 32,665 / 32,972 | 40.8 GB | 2/2 |
| 32K | oMLX | 4.4 min | 44.6 s | 3.45 s | 14.4 / 14.0 / 14.4 | 6.0 min | 28,672 / 32,768 | 37.0 GB | 2/2 |
| 32K | Splash M1 build | 5.0 min | 4.43 s | 4.18 s | 23.2 / 27.2 / 22.1 | 5.7 min | 32,384 / 32,704 | 23.1 GB | 2/2 |
| 32K | Splash source (HEAD) | 4.9 min | 3.97 s | 3.96 s | 25.3 / 29.1 / 27.1 | 5.5 min | 32,384 / 32,704 | 22.2 GB | 1/1 |
| 64K | MTPLX fork | 9.9 min | 1.36 s | 1.37 s | 21.8 / 22.4 / 21.4 | 10.4 min | 65,419 / 65,726 | 38.0 GB | 2/2 |
| 64K | MTPLX upstream | 9.9 min | 2.07 s | 2.12 s | 8.8 / 8.7 / 8.4 | 11.3 min | 65,419 / 65,726 | 48.4 GB | 2/2 |
| 64K | oMLX | 10.2 min | 1.0 min | 4.76 s | 10.9 / 10.9 / 10.4 | 12.4 min | 61,440 / 65,536 | 39.6 GB | 2/2 |
| 64K | Splash M1 build | 12.8 min | 5.68 s | 6.34 s | 18.7 / 22.5 / 16.5 | 13.6 min | 65,152 / 65,440 | 23.4 GB | 2/2 |
| 64K | Splash source (HEAD) | 12.3 min | 5.01 s | 5.78 s | 20.7 / 27.1 / 18.3 | 13.0 min | 65,152 / 65,440 | 23.3 GB | 1/1 |
| 128K | MTPLX fork | 25.1 min | 2.14 s | 2.13 s | 17.3 / 18.9 / 18.1 | 25.9 min | 130,969 / 131,276 | 44.3 GB | 1/1 |
| 128K | MTPLX upstream | 30.5 min | 4.16 s | 3.49 s | 4.6 / 5.1 / 5.0 | 33.0 min | 130,969 / 131,276 | 52.7 GB | 1/1 |
| 128K | oMLX | 26.3 min | 1.6 min | 7.92 s | 7.6 / 7.8 / 6.9 | 29.6 min | 126,976 / 131,072 | 48.5 GB | 1/1 |
| 128K | Splash M1 build | — | — | — | — / — / — | — | — / — | 24.7 GB | — |
| 128K | Splash source (HEAD) | 34.3 min | 8.02 s | 7.81 s | 14.4 / 18.5 / 14.6 | 35.3 min | 130,688 / 131,008 | 25.5 GB | 1/1 |

**2K: fp16 KV vs 8-bit KV** (decode tok/s, mean of turns 1–3)

| engine | fp16 KV | 8-bit KV | change |
|---|---|---|---|
| MTPLX fork | 26.3 | 27.6 | +4.7% |
| MTPLX upstream | 24.8 | 23.7 | -4.5% |
| oMLX | 23.1 | 21.6 | -6.4% |
| Splash M1 build | 33.8 | 33.0 | -2.5% |
| Splash source (HEAD) | — | 32.3 | — |

Raw rows: [`results/2026-09-m1max-64gb/rows.jsonl`](results/2026-09-m1max-64gb/rows.jsonl) ·
per-turn CSV: [`summary.csv`](results/2026-09-m1max-64gb/summary.csv) ·
environment: [`environment.json`](results/2026-09-m1max-64gb/environment.json) ·
generated text of every turn: `results/2026-09-m1max-64gb/cells/*/`.

## Why Splash is fastest on turn 2

A source build of Splash's M1 fork at its current head
([paperniuk/splash@5967821](https://github.com/paperniuk/splash/tree/apple7-m1-kernels),
built locally with [one added status counter](https://github.com/shunya1810/splash/tree/m1-analysis-metrics))
reports its draft and verify counters, so each turn can be broken down into verify passes
(one DFlash2 draft of 7 tokens plus one verify of 8 positions). One round, 8-bit KV:

| context | turn | decode tok/s | verify passes | drafted / pass | accepted / pass | tokens / pass | acceptance | ms / pass |
|---|---|---|---|---|---|---|---|---|
| 2K | 1 | 28.6 | 84 | 7.0 | 2.04 | 3.05 | 29% | 107 |
| 2K | 2 | 35.9 | 67 | 7.0 | 2.82 | 3.82 | 40% | 107 |
| 2K | 3 | 32.3 | 52 | 7.0 | 2.44 | 3.46 | 35% | 107 |
| 8K | 1 | 28.5 | 81 | 7.0 | 2.16 | 3.16 | 31% | 111 |
| 8K | 2 | 40.1 | 57 | 7.0 | 3.49 | 4.49 | 50% | 113 |
| 8K | 3 | 29.2 | 56 | 7.0 | 2.27 | 3.29 | 32% | 113 |
| 32K | 1 | 25.3 | 76 | 7.0 | 2.36 | 3.37 | 34% | 133 |
| 32K | 2 | 29.1 | 66 | 7.0 | 2.86 | 3.88 | 41% | 133 |
| 32K | 3 | 27.1 | 48 | 7.0 | 2.62 | 3.65 | 38% | 136 |
| 64K | 1 | 20.7 | 76 | 7.0 | 2.37 | 3.37 | 34% | 163 |
| 64K | 2 | 27.1 | 58 | 7.0 | 3.40 | 4.41 | 49% | 164 |
| 64K | 3 | 18.3 | 59 | 7.0 | 2.00 | 3.02 | 29% | 166 |
| 128K | 1 | 14.4 | 81 | 7.0 | 2.16 | 3.16 | 31% | 220 |
| 128K | 2 | 18.5 | 62 | 7.0 | 3.13 | 4.13 | 45% | 225 |
| 128K | 3 | 14.6 | 54 | 7.0 | 2.15 | 3.17 | 31% | 218 |

- **Turn 2 is the only turn where the draft is accepted often:** 40–50% of drafted tokens,
  against 29–38% on turns 1 and 3, so a pass commits 3.8–4.5 tokens instead of 3.0–3.7. The
  cost of a pass does not depend on the turn, so the extra tokens are the whole difference.
  The turn-2 answer repeats the comparison pattern set up in turn 1, which a 7-token draft
  can follow further than the 3-token MTP draft; the MTPLX fork commits 2.9–3.0 tokens per
  round on turn 2 and 2.6–2.9 on turns 1 and 3, with a ceiling of 4.
- **Splash verifies twice as many positions per pass at the same cost at short context:**
  107 ms per pass at 2K for 8 positions and a 7-token draft, against about 100 ms per
  round on the fork for 4 positions and a 3-token MTP draft.
- **The pass cost grows faster with the prompt:** from 107 ms (2K) to 220 ms (128K) on
  Splash, against 100 to 157 ms per round on the fork. Each pass attends over the whole
  history for 8 positions instead of 4, which most likely explains why Splash falls
  behind from 64K on turns 1 and 3.
- This source build finishes 128K (first turn 34.3 min, then 8.0 / 7.8 s to first token):
  its server's `--request-timeout` default is 10,000 s, where the 1.0.2-m1 release had
  1,800 s. It also carries two newer Apple7/8 attention commits; at 64K it decodes at
  20.7 / 27.1 / 18.3 tok/s against 18.7 / 22.5 / 16.5 for the release.

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
  column was later re-run alone at `40b6113` and at `4fe8067` with the same plans (128K
  canary 0.1% below its 2K–64K best in the `4fe8067` run).
- Round-to-round decode difference: median 0.9%, largest 10% (oMLX, 64K turn 3, where the
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
