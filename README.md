# Apple Silicon LLM engine benchmark: long-context, multi-turn

English | [日本語](README.ja.md)

<!-- b2:start -->
## Latest: newest releases, thinking on, same weights (2026-10-18/19)

All four engines at their newest release on 2026-10-18, on the same MacBook Pro (M1 Max,
64 GB), with **thinking on** and the **same body weights** for every engine. Each cell is a
3-turn conversation over a synthetic telemetry prompt (the same as the sections below), up
to 4,096 generated tokens per turn, thinking tokens included.

| engine | version | weights |
|---|---|---|
| **MTPLX M1 fork** | [`v2.12.2-m1.10`](https://github.com/shunya1810/MTPLX/releases/tag/v2.12.2-m1.10) (code `25927ca3`) | MTPLX pack below, MTP depth from the pack (27B: 3; A3B: 1), `--profile turbo`, q8 KV |
| **MTPLX upstream** | [`v2.12.2`](https://github.com/youssofal/MTPLX/releases/tag/v2.12.2) (`9882703f`, `main` at run time) | same pack and flags as the fork |
| **oMLX** | [0.7.1rc1](https://github.com/jundot/omlx/releases/tag/v0.7.1rc1) (wheel, mlx 0.32.4) | same pack as MTPLX through a text-only symlinked view (MTP head renamed for mlx-lm), MTP with 3 draft tokens, **fp16 KV** (see below) |
| **Splash** | [paperniuk/splash 1.3.2-m1.1](https://github.com/paperniuk/splash/releases/tag/1.3.2-m1.1) | `mlx-community/Qwen3.8-27B-4bit` / `mlx-community/Qwen3.6-35B-A3B-4bit` as published, plus the DFlash2 draft Splash selects, INT8 KV |
| oMLX + oQ4e (extra row) | oMLX 0.7.1rc1 | `Jundot/Qwen3.8-27B-oQ4e-fp16-mtp` / `Jundot/Qwen3.6-35B-A3B-oQ4e-mtp`: oMLX's own mixed 4/5/6-bit quantization, **different weights** from the other rows |

**Same weights.** Splash loads the mlx-community 4-bit checkpoints unchanged. The MTPLX and
oMLX rows load the same 4-bit codes with Qwen's native MTP head added (MTPLX needs it; the
mlx-community checkpoints drop it):
- **27B**: `mlx-community/Qwen3.8-27B-4bit` (4-bit, group 64, every layer) with the bf16
  floats (scales, biases, norms) cast to fp16, as MTPLX's forge does for M1, plus the MTP
  head of `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` (INT4, group 64). The packed
  codes are byte-identical to mlx-community's.
- **A3B**: `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16` as published. Its body is
  tensor-for-tensor identical to `mlx-community/Qwen3.6-35B-A3B-4bit` after the same fp16
  cast (all 2,090 tensors compared: 512 packed, 1,578 float); the extra size is the MoE MTP
  head and the vision tower.

**Conditions.** Thinking on, `reasoning_effort` medium on every engine (without it Splash
renders the template's xhigh instruction and the prompts differ by 42 tokens), temperature
1.0, top_p 0.95, top_k 20, seed 7 (the models' `generation_config`). Each engine was checked
to tokenize the first turn to the same 1,678 tokens at 2K. Turns 2–3 send the previous
answer back without its thinking (engines differ in what they do with an echoed
`reasoning_content`; with the answer alone every engine builds the same next prompt). Every
cell starts the engine fresh; one run per cell. The 256K row uses a 246,000-token prompt so
three 4,096-token turns fit the 262,144-token window.

**Read with care.**
- This telemetry text is easy to predict: the M1 fork's MTP drafts 3.5–3.7 tokens per verify
  round here, against 3.1 in the author's real coding-agent sessions. Decode speeds are
  higher than in agent use for every speculative engine (Splash's DFlash2 included), and
  1–2 tok/s differences are within run-to-run noise.
- At temperature 1.0 each engine thinks for a different number of tokens, so whole-
  conversation times are not compared, and the needle answer (turn 3) mostly shows whether
  the thinking ended within 4,096 tokens.
- **oMLX runs fp16 KV**: 0.7.1rc1's TurboQuant 8-bit KV fails on M1 (`sdpa_vector_2pass`
  quantized kernel asks for 960 threads per threadgroup; M1's limit is 896), so oMLX keeps
  more KV in memory than the 8-bit rows.
- **oMLX on the 27B re-reads the whole prompt every turn** (its prefix cache cannot restore a
  hybrid model's recurrent state where the history changes), so its 27B 246K cell, and the
  oQ4e row's 27B 128K / 246K and A3B 246K cells, ran turn 1 only (each further turn would
  have taken about an hour). On the A3B it reuses part of the prompt up to 128K.
- **MTPLX upstream failed the 27B at 246K**: its memory guard stopped turn 1 mid-prefill
  (sustained critical memory pressure, 58.4 GB wired).
- **The M1 fork rows are v2.12.2-m1.10 (re-run 2026-10-20); the other engines' rows are from
  2026-10-18/19.** On v2.12.2-m1.9 the fork's 27B turn 3 at 32K / 128K / 246K re-read part
  (3,061 tokens) or all (246K: 248,809 tokens, 51 min) of the prompt, because turn 2's
  thinking hit the 4,096-token cap and the save of turn 2's state had not finished when turn
  3 arrived; at 246K that turn also drafted far worse (2.2 instead of 3.5 tokens per round,
  decode 12.8 tok/s). v2.12.2-m1.10 fixes both: turn 3 starts after 1.6 s (32K), 4.1 s
  (128K) and 3.3 s (246K), and decodes at 18.2 tok/s (246K, 3.0 tokens per round).
  The fork's A3B turn 1 at 246K read 11% slower in this re-run than on v2.12.2-m1.9 (973 vs
  871 s); run back to back on one Mac state the two versions prefilled the same 246K prompt
  at 291 and 289 tok/s, so that gap is run-to-run variation.

### Qwen3.8-27B (dense)

**Turn 1, cold prompt: time to first token (prefill speed)**

| context | MTPLX M1 fork m1.9 | MTPLX upstream 2.12.2 | oMLX 0.7.1rc1 | Splash 1.3.2-m1.1 | oMLX 0.7.1rc1 + Jundot oQ4e |
|---|---|---|---|---|---|
| 2K | 8 s (200 tok/s) | 11 s (148 tok/s) | 11 s (148 tok/s) | 8 s (214 tok/s) | 11 s (147 tok/s) |
| 8K | 33 s (237 tok/s) | 52 s (150 tok/s) | 50 s (157 tok/s) | 37 s (211 tok/s) | 50 s (156 tok/s) |
| 32K | 151 s (215 tok/s) | 237 s (136 tok/s) | 236 s (138 tok/s) | 184 s (176 tok/s) | 237 s (137 tok/s) |
| 64K | 380 s (171 tok/s) | 565 s (115 tok/s) | 554 s (118 tok/s) | 455 s (143 tok/s) | 556 s (117 tok/s) |
| 128K | 995 s (131 tok/s) | 1,649 s (79 tok/s) | 1,435 s (91 tok/s) | 1,263 s (103 tok/s) | 1,439 s (91 tok/s) |
| 246K | 3,062 s (80 tok/s) | failed | 3,762 s (65 tok/s) | 3,534 s (70 tok/s) | 3,770 s (65 tok/s) |

**Decode, tok/s (turns 1 / 2 / 3, thinking tokens included)**

| context | MTPLX M1 fork m1.9 | MTPLX upstream 2.12.2 | oMLX 0.7.1rc1 | Splash 1.3.2-m1.1 | oMLX 0.7.1rc1 + Jundot oQ4e |
|---|---|---|---|---|---|
| 2K | 42.1 / 42.3 / 41.6 | 31.4 / 27.8 / 30.1 | 34.3 / 35.1 / 35.3 | 48.8 / 57.3 / 57.2 | 33.5 / 35.2 / 33.6 |
| 8K | 40.5 / 43.0 / 40.5 | 29.5 / 29.3 / 25.7 | 31.6 / 31.5 / 30.5 | 47.4 / 59.4 / 58.9 | 29.7 / 33.2 / 30.9 |
| 32K | 36.8 / 35.8 / 36.9 | 16.0 / 17.1 / 15.6 | 24.7 / 24.2 / 24.6 | 41.4 / 43.0 / 30.1 | 23.0 / 24.2 / 21.1 |
| 64K | 30.4 / 33.6 / 32.5 | 10.1 / 11.0 / 10.6 | 18.1 / 21.1 / 19.1 | 36.0 / 33.6 / 23.1 | 17.8 / 19.7 / 19.4 |
| 128K | 29.6 / 27.2 / 26.2 | 6.0 / 6.1 / 5.9 | 11.5 / 13.2 / 11.8 | 31.7 / 25.0 / 17.5 | 11.0 / - / - |
| 246K | 20.9 / 21.0 / 18.2 | failed | 6.6 / - / - | 14.4 / 14.3 / 17.2 | 6.7 / - / - |

**Turns 2 / 3: time to first token, s**

| context | MTPLX M1 fork m1.9 | MTPLX upstream 2.12.2 | oMLX 0.7.1rc1 | Splash 1.3.2-m1.1 | oMLX 0.7.1rc1 + Jundot oQ4e |
|---|---|---|---|---|---|
| 2K | 3.8 / 3.7 | 4.1 / 4.4 | 13.4 / 13.3 | 3.3 / 0.9 | 2.8 / 2.9 |
| 8K | 1.3 / 1.5 | 1.0 / 1.3 | 25.7 / 25.7 | 3.6 / 1.0 | 27.2 / 27.4 |
| 32K | 1.3 / 1.6 | 2.3 / 2.6 | 32.7 / 32.7 | 3.3 / 1.2 | 32.8 / 32.9 |
| 64K | 2.4 / 2.8 | 3.0 / 3.0 | 42.9 / 42.6 | 1.2 / 1.4 | 46.4 / 45.7 |
| 128K | 3.6 / 4.1 | 6.2 / 7.3 | 1,437.2 / 1,438.7 | 1.8 / 1.5 | - / - |
| 246K | 3.3 / 3.3 | failed | - / - | 9.0 / 10.8 | - / - |

**Peak wired memory**

| context | MTPLX M1 fork m1.9 | MTPLX upstream 2.12.2 | oMLX 0.7.1rc1 | Splash 1.3.2-m1.1 | oMLX 0.7.1rc1 + Jundot oQ4e |
|---|---|---|---|---|---|
| 2K | 25.7 GB | 28.7 GB | 24.2 GB | 24.0 GB | 23.5 GB |
| 8K | 28.0 GB | 30.5 GB | 28.0 GB | 24.3 GB | 29.1 GB |
| 32K | 30.8 GB | 33.1 GB | 30.4 GB | 25.1 GB | 31.1 GB |
| 64K | 33.6 GB | 43.2 GB | 36.4 GB | 26.1 GB | 36.6 GB |
| 128K | 36.3 GB | 49.7 GB | 51.6 GB | 28.2 GB | 50.7 GB |
| 246K | 44.4 GB | 58.4 GB (failed) | 56.8 GB | 32.5 GB | 57.7 GB |

### Qwen3.6-35B-A3B (MoE)

**Turn 1, cold prompt: time to first token (prefill speed)**

| context | MTPLX M1 fork m1.9 | MTPLX upstream 2.12.2 | oMLX 0.7.1rc1 | Splash 1.3.2-m1.1 | oMLX 0.7.1rc1 + Jundot oQ4e |
|---|---|---|---|---|---|
| 2K | 2 s (950 tok/s) | 2 s (864 tok/s) | 2 s (1,002 tok/s) | 1 s (1,133 tok/s) | 2 s (728 tok/s) |
| 8K | 7 s (1,072 tok/s) | 8 s (940 tok/s) | 7 s (1,048 tok/s) | 7 s (1,047 tok/s) | 12 s (639 tok/s) |
| 32K | 32 s (1,005 tok/s) | 43 s (760 tok/s) | 44 s (741 tok/s) | 44 s (729 tok/s) | 74 s (438 tok/s) |
| 64K | 93 s (697 tok/s) | 121 s (538 tok/s) | 122 s (533 tok/s) | 126 s (516 tok/s) | 217 s (301 tok/s) |
| 128K | 283 s (461 tok/s) | 482 s (271 tok/s) | 381 s (343 tok/s) | 429 s (305 tok/s) | 703 s (186 tok/s) |
| 246K | 973 s (253 tok/s) | 1,670 s (147 tok/s) | 1,162 s (211 tok/s) | 1,363 s (180 tok/s) | 2,200 s (112 tok/s) |

**Decode, tok/s (turns 1 / 2 / 3, thinking tokens included)**

| context | MTPLX M1 fork m1.9 | MTPLX upstream 2.12.2 | oMLX 0.7.1rc1 | Splash 1.3.2-m1.1 | oMLX 0.7.1rc1 + Jundot oQ4e |
|---|---|---|---|---|---|
| 2K | 113.8 / 113.5 / 115.4 | 61.0 / 60.4 / 64.0 | 105.2 / 102.1 / 112.8 | 122.6 / 122.3 / 127.3 | 92.8 / 92.2 / 92.3 |
| 8K | 114.1 / 113.7 / 116.6 | 54.0 / 60.4 / 56.4 | 103.7 / 90.1 / 84.9 | 106.0 / 116.1 / 109.0 | 84.9 / 78.6 / 85.2 |
| 32K | 97.4 / 99.1 / 98.4 | 43.9 / 44.1 / 44.7 | 65.1 / 74.7 / 65.8 | 91.7 / 88.8 / 88.8 | 64.3 / 59.2 / 76.9 |
| 64K | 76.6 / 73.0 / 81.2 | 33.4 / 37.9 / 32.4 | 49.3 / 50.9 / 49.7 | 69.2 / 66.4 / 82.9 | 47.0 / 46.0 / 54.0 |
| 128K | 64.9 / 61.7 / 61.6 | 23.0 / 24.4 / 24.0 | 33.5 / 33.1 / 34.3 | 49.3 / 48.9 / 52.6 | 31.0 / 32.0 / 31.6 |
| 246K | 48.5 / 46.2 / 45.9 | 11.5 / 13.2 / 12.6 | 19.4 / 21.2 / 20.1 | 34.1 / 33.9 / 35.8 | 16.6 / - / - |

**Turns 2 / 3: time to first token, s**

| context | MTPLX M1 fork m1.9 | MTPLX upstream 2.12.2 | oMLX 0.7.1rc1 | Splash 1.3.2-m1.1 | oMLX 0.7.1rc1 + Jundot oQ4e |
|---|---|---|---|---|---|
| 2K | 0.6 / 1.0 | 0.4 / 0.7 | 1.8 / 1.9 | 0.4 / 0.2 | 2.5 / 2.6 |
| 8K | 0.4 / 0.5 | 0.5 / 0.5 | 3.9 / 4.0 | 0.6 / 0.5 | 6.7 / 7.4 |
| 32K | 1.1 / 1.0 | 1.3 / 1.4 | 7.9 / 8.1 | 0.8 / 0.3 | 12.6 / 12.6 |
| 64K | 1.5 / 1.6 | 1.2 / 1.2 | 11.5 / 11.3 | 1.7 / 1.3 | 22.5 / 1.0 |
| 128K | 1.5 / 2.0 | 3.8 / 3.5 | 19.9 / 19.5 | 3.1 / 2.0 | 39.6 / 41.4 |
| 246K | 3.8 / 5.2 | 6.8 / 11.5 | 1,163.7 / 1,164.5 | 4.1 / 1.0 | - / - |

**Peak wired memory**

| context | MTPLX M1 fork m1.9 | MTPLX upstream 2.12.2 | oMLX 0.7.1rc1 | Splash 1.3.2-m1.1 | oMLX 0.7.1rc1 + Jundot oQ4e |
|---|---|---|---|---|---|
| 2K | 28.2 GB | 26.8 GB | 27.0 GB | 26.0 GB | 27.4 GB |
| 8K | 28.4 GB | 27.6 GB | 28.8 GB | 26.2 GB | 29.3 GB |
| 32K | 29.0 GB | 32.2 GB | 31.0 GB | 26.4 GB | 31.5 GB |
| 64K | 29.9 GB | 35.4 GB | 31.6 GB | 26.7 GB | 32.3 GB |
| 128K | 31.0 GB | 39.0 GB | 35.1 GB | 27.4 GB | 35.8 GB |
| 246K | 34.6 GB | 49.6 GB | 44.8 GB | 28.6 GB | 44.2 GB |

### What the numbers say

- **Short prompts: Splash.** At 2K Splash is the fastest engine on both models (27B decode
  57 vs 42 tok/s on turns 2–3, A3B 122–127 vs 114–115), and its DFlash2 draft commits 5–6.6
  tokens per verify round where MTP depth 3 commits at most 4 (the fork's 27B rounds are
  25–40% shorter, but carry 3.5 tokens). Splash also uses the least memory at every length
  (27B 24–33 GB, A3B 26–29 GB).
- **Long prompts: the M1 fork.** From 8K on, the fork reads prompts fastest on both models,
  and the gap grows with length: 27B 1.12× Splash at 8K, 1.27× at 128K; A3B 1.02× at 8K,
  1.51× at 128K, 1.40× at 246K. Its decode holds up deeper too: the 27B is level with Splash
  at 64K and ahead from 128K (turns 2–3: 26–27 vs 25 and 17.5 tok/s; 246K: 18–21 vs 14–17), the A3B level
  at 8K and ahead from 32K (32K: 97–99 vs 89–92 tok/s; 128K: 62–65 vs 49–53; 246K: 46–49 vs 34–36). Splash's 27B decode drops
  on later turns from 32K on (32K turn 3: 30 tok/s; 64K: 23; 128K: 17.5).
- **Against MTPLX upstream 2.12.2** (same weights, same flags), the fork reads prompts
  1.1–1.7× faster and decodes 1.3–4.9× faster (27B 128K: 26–30 vs 6 tok/s; A3B 246K: 46–49 vs
  12–13), with less memory from 32K on. Upstream could not finish the 27B at 246K.
- **oMLX 0.7.1rc1** sits between upstream and the fork on decode, but on the 27B it re-reads
  the whole prompt on every turn (turn 2 starts after 13 s at 2K, 43 s at 64K, 24 min at
  128K) and runs fp16 KV on M1. On the A3B it reuses the prompt up to 128K (turn 2: 2–20 s)
  and re-reads it at 246K.
- **oMLX's own oQ4e quantization** (extra row, different weights) runs about as fast as
  oMLX on the shared weights for the 27B (decode within ±10%), and slower for the A3B
  (prefill 0.5–0.75×, decode 0.8–1.0×): its 5–6-bit layers cost bandwidth. Its quality is
  not measured here.

Raw rows: [`results/2026-10-m1max-64gb-b2/`](results/2026-10-m1max-64gb-b2/), summary
`python scripts/summarize_b2.py results/2026-10-m1max-64gb-b2 --readme 27b` (or `a3b`),
plans `plans/b2-27b.json` / `plans/b2-a3b.json`, run script `run_b2.sh`.
<!-- b2:end -->

## Earlier runs: thinking off (2026-09 to 2026-10-14)

How fast is a **3-turn conversation over a long prompt** on a Mac, depending on the
inference engine? On a MacBook Pro with M1 Max and 64 GB, five inference engines serve
Qwen3.8-27B with speculative decoding and 8-bit KV cache (TensorFold has no 8-bit KV and
runs bf16 KV), measured the same way over each engine's OpenAI-compatible streaming API.
A sixth column runs MTPLX upstream with the M1 fork's defaults set through environment
variables, to separate what the M1 fork's code adds from what its settings add. The
MTPLX and oMLX engines load the **same model files** with MTP (depth 3); Splash and
TensorFold need their own packages (4-bit, group-64 weights plus a DFlash2 draft model),
so their columns also reflect a different quantization and a different speculative method.

> **Disclosure:** the MTPLX M1 fork is maintained by the author of this benchmark. Its
> column runs the published release tag with the same command line as upstream, and every
> number below comes from `rows.jsonl` through `scripts/summarize.py`.

| engine | version | notes |
|---|---|---|
| **MTPLX M1 fork** | [`v2.12.2-m1.1`](https://github.com/shunya1810/MTPLX/releases/tag/v2.12.2-m1.1) (`c121efb`, [`shunya1810/MTPLX`](https://github.com/shunya1810/MTPLX/tree/m1max-longctx)); 256K: [`v2.12.2-m1.2`](https://github.com/shunya1810/MTPLX/releases/tag/v2.12.2-m1.2) | unofficial M1-family long-context build of MTPLX on upstream 2.12.2; m1.1 adds a Neural Engine prefill and fewer decode round trips. Measured with Splash 1.3.0-m1.1 on 2026-10-13. m1.1's memory guard refused the 256K prompt, so the 256K spot check runs m1.2 (the same code plus M1 admission and paged-prefill fixes; [details](#splash-130-m11-and-the-m1-fork-v2122-m11)). Earlier rows: `mtplx-fork-m1-2122` (v2.12.2-m1, 2026-10-08), `mtplx-fork-m1.2` (2026-09-30), `mtplx-fork-m1` (first release; [why it decoded faster](#why-the-m1-fork-trades-decode-for-prompt-reading)), `mtplx-fork-16751dc`, `-40b6113`, `-4fe8067`, `-bba1b7b`, `-0cd2a73` |
| **MTPLX upstream** | [`youssofal/MTPLX@1de2b1c`](https://github.com/youssofal/MTPLX/commit/1de2b1c049136ed117af0c6712baaadd81820b51) | `main` at run time (still its head on 2026-09-29); the M1 fork's base. Its first run (2026-09-25/26) stays in the raw data as `mtplx-upstream-0925` |
| **MTPLX upstream + env** | upstream `1de2b1c` | the same code with every M1-fork default that upstream exposes as an environment variable: `MTPLX_CONTEXT_COPY=0`, `MTPLX_FUSE_PROJ=gdn,attn`, `MTPLX_FRSPEC_DRAFT=1 MTPLX_FRSPEC_LEGACY=1 MTPLX_FRSPEC_VOCAB=builtin:qwen38-code-64k`, `MTPLX_MLX_CACHE_LIMIT=1G`, `MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES=2` ([engine file](engines/mtplx-upstream-tuned-nobuf.json)); the M1 fork's larger Metal command buffers are left out (see below) |
| **oMLX** | [0.7.0](https://github.com/jundot/omlx/releases/tag/v0.7.0) | latest stable at run time; measured alone on 2026-10-07 (see below). The 0.6.4 run of 2026-09-25/26 stays in the raw data as `omlx-064` ([what changed](#omlx-070-against-064)) |
| **Splash 1.3.0-m1.1** | [paperniuk/splash 1.3.0-m1.1](https://github.com/paperniuk/splash/releases/tag/1.3.0-m1.1) | community M1/M2 build of [incoai/splash](https://github.com/incoai/splash) 1.3.0 with Apple7/8 kernels and a Neural Engine prefill; model `incoai/Qwen3.8-27B-Splash` (4-bit g64 + DFlash2 draft), INT8 KV. Measured on 2026-10-13 (upstream Splash 1.3.1 had no M1 fork yet). Earlier rows: `splash-m1-121` (1.2.1-m1, 2026-10-08, [what changed](#splash-130-m11-and-the-m1-fork-v2122-m11)), `splash-m1-110`, `splash-m1-102`, `splash-src` |
| **TensorFold** | [0.3.5.1](https://github.com/ashhart/TensorFold/releases/tag/v0.3.5.1) | Python/MLX engine with exact (token-equality) draft acceptance; model `Vontra/Qwen3.8-27B-MLX-4bit` (4-bit g64) + `z-lab/Qwen3.8-27B-DFlash2` draft (4-bit at load), bf16 KV (no 8-bit option); release of 2026-09-28, the first that loads on M1 Max |

Prompt lengths 2K–64K ran in 2 rounds each (median shown); 128K ran once. MTPLX upstream
and upstream + env ran together, interleaved, on 2026-09-29; the M1 fork v2.12.2-m1.1 and
Splash 1.3.0-m1.1 ran together, interleaved, on 2026-10-13; oMLX 0.7.0 ran alone on
2026-10-07 and TensorFold in its own session, the same way. Before each later session a
2K run of the M1 fork confirmed the machine was in the same state (canary 30.3–31.9 tok/s).

## Highlights (8-bit KV)

**Whole 3-turn conversation** (turn 1 reads the prompt cold, turns 2–3 add ~300 tokens
each, at most 256 generated tokens per turn):

| engine | 2K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| MTPLX M1 fork (m1.1) | **33 s** | **61 s** | **3.3 min** | **7.3 min** | **18.4 min** |
| MTPLX upstream | 42 s | 86 s | 5.0 min | 11.3 min | 30.6 min |
| MTPLX upstream + env | 38 s | 84 s | 4.9 min | 11.3 min | 32.2 min |
| oMLX | 61 s | 95 s | 5.1 min | 10.9 min | 27.0 min |
| Splash 1.3.0-m1.1 | 35 s | 68 s | 3.8 min | 8.7 min | 23.1 min |
| TensorFold (bf16 KV) | 47 s | 106 s | 6.1 min | 13.1 min | 89.6 min |

**Decode**, mean of the three turns, tok/s:

| engine | 2K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| MTPLX M1 fork (m1.1) | 29.4 | 29.2 | **28.1** | **25.5** | **19.8** |
| MTPLX upstream | 23.6 | 22.5 | 14.0 | 8.7 | 5.0 |
| MTPLX upstream + env | 27.2 | 25.8 | 15.3 | 7.8 | 4.6 |
| oMLX | 21.6 | 20.2 | 14.3 | 10.6 | 7.1 |
| Splash 1.3.0-m1.1 | **33.6** | **29.6** | 25.4 | 21.4 | 14.3 |
| TensorFold (bf16 KV) | 27.7 | 25.5 | 17.0 | 11.3 | 6.4 |

- **The M1 fork's whole conversation is the shortest at every length**: 5% shorter than
  Splash's at 2K, 10% at 8K, 13% at 32K, 15% at 64K and 20% at 128K. Both engines now read
  prompts with help from the Neural Engine; the M1 fork reads 8K+ prompts 7–20% faster,
  starts follow-up turns 3–5× sooner, and from 32K up also decodes faster. Splash decodes
  14% faster at 2K, the two are even at 8K, and the M1 fork leads by 11% at 32K, 19% at
  64K and 38% at 128K. When a turn generates much more than it reads (chat, whole-file
  code generation), Splash's 2K decode lead decides instead.
- **Both new releases read prompts much faster** ([details](#splash-130-m11-and-the-m1-fork-v2122-m11)):
  the M1 fork's whole conversation is 6% shorter at 2K and 23–28% at 8K–128K than the
  previous release's (cold prompt reading 1.36–1.49× faster from 8K, 1.23× at 2K; decode
  +10% at 32K–64K), Splash 1.3.0-m1.1's 8% shorter at 2K and 20–33% at 8K–128K than
  1.2.1-m1's (1.46–1.51× faster prompt reading; decode within 3% except 8K, −8%, and
  128K, −5%).
- **Sampling** (temperature 1.0, top-p 0.95, top-k 20, up to 512 tokens): decode is
  31.6 / 28.3 / 23.9 tok/s on the M1 fork and 31.5 / 25.8 / 18.5 on Splash at 2K / 32K /
  64K: even at 2K, the M1 fork 10% ahead at 32K and 29% at 64K. Upstream:
  23.2 / 14.2 / 8.9 ([details](#sampled-decoding-temperature-10)).
- **What the M1 fork's settings are worth on upstream:** upstream + env is 15%
  faster than upstream at 2K–8K and 9% at 32K, but 10% slower at 64K and 8% at 128K
  (turning context-copy drafting off costs upstream's slower long-context rounds more
  than it saves). The M1 fork's lead over upstream + env (9% at 2K, 1.7× at 32K, 3.0×
  at 64K, 4.2× at 128K) is therefore its code: M1 attention kernels for verify, draft
  and prefill, and the session-bank fixes ([what the build changes](https://github.com/shunya1810/MTPLX/blob/m1max-longctx/docs/m1max-longctx/CHANGES.ja.md)).
- **First turn (cold prefill):** Splash reads 2K fastest (8.3 s against 9.0 s); from 8K the
  M1 fork reads fastest: 36 s, 2.8 min, 6.9 min and 17.8 min at 8K / 32K / 64K / 128K
  (Splash 39 s, 3.2, 8.0 and 22.1 min; upstream 53 s, 4.0, 9.9 and 28.1 min; oMLX 128K 25.1 min).
- **Follow-up turns:** time to first token for turns 2 and 3 at 128K is **1.8 s** on the
  M1 fork, 3.5 s on upstream, 4.2 s on upstream + env, 6.4 / 6.2 s on Splash (it
  re-reads ~300 tokens), 10.9 / 7.8 s on oMLX and 29 min on TensorFold (below). oMLX keeps
  its prefix cache on SSD in 4,096-token blocks; 0.7.0 also saves the end of the first
  prompt, so turn 2 re-reads only the new ~300 tokens, but turn 3 resumes from the last
  block boundary and re-reads up to ~4K tokens (0.6.4 did that on both turns: 1.6 min /
  7.9 s).
- **Memory:** Splash uses the least, 24–28 GB wired at every length. The M1 fork peaks at
  32–45 GB (its Neural Engine prefill adds 1–2 GB over v2.12.2-m1), upstream at 34–52 GB, upstream + env at 30–53 GB, oMLX at 30–48 GB (peak
  system-wide wired memory; Splash maps its weights differently, so compare with care).
- **The command-buffer setting is not an upstream win.** MLX's Metal command buffers
  raised to 150 ops / 1,000 MB (the first M1 fork's default) raised upstream's peak by
  12–15 GB, slowed its 32K–64K cold prefill by 12–16% and ran out of GPU memory at 128K,
  for no decode gain; those rows stay in the raw data as `mtplx-upstream-tuned`.
- **TensorFold 0.3.5.1 (bf16 KV):** 6–13% behind the M1 fork's decode at 2K–8K, a third
  of it at 128K. At 128K its default prompt cache (an eighth of RAM, 8 GiB) cannot hold
  the conversation, so turns 2 and 3 re-read all 131K tokens and the conversation takes
  89.6 min; a larger `--prompt-cache-gib` may avoid that, but every engine runs at its
  defaults here. It accepts a draft token only when it equals the target's own sample
  (exact decoding), which works at temperature 0 but accepts fewer tokens when sampling.
  (TensorFold 1.0.1 is out; not re-measured.)
- **oMLX 0.7.0** (replacing 0.6.4): the whole conversation is 9–24% shorter, mostly
  because turn 2 now reuses the whole first prompt ([details](#omlx-070-against-064)).
- **256K (three engines, one run each):** the M1 fork (m1.2) read the prompt in 58.5 min,
  Splash 1.3.0-m1.1 in 67.1 and oMLX in 69.8; they decoded at 14.2, 10.2 and 4.1 tok/s and
  started follow-up turns in 2.9 s, 9.8–10.3 s and 13.5–19.4 s. Splash used 32.7 GB, the
  M1 fork 50.1 GB and oMLX 57.3 GB. m1.1 refused the prompt at its memory guard (m1.2
  fixes the M1 bills), and oMLX ran only with its guard raised to `aggressive`
  ([details](#256k-spot-check)).
- **Short prompts with reasoning on** (the Splash part 2 post's benchmark): Splash
  decodes 29% faster and uses a third less energy per token; the M1 fork reads a 44K
  prompt 20% faster ([details](#short-prompts-the-splash-part-2-conditions)).
- **Output check:** every engine quoted the needle line correctly in turn 3 in every
  completed run.

### Why the M1 fork trades decode for prompt reading

The first release (`v2.12.0-m1`) raised MLX's Metal command-buffer limits on M1. In this benchmark that made
decode 3–9% faster but cold prompt reading 2–9% slower at 2K–32K on an idle Mac (13–21%
slower when the Mac was busy or power-limited, in a separate test), and no setting kept one without the other. In a
long-context turn most of the time is prompt reading, so from `v2.12.0-m1.1` the build
leaves MLX's default. Rough rule for a turn that reads N new tokens and generates M:
m1.2 is faster when M is below about N/8 on an idle Mac (about 0.6 N on a busy one);
for generation-heavy use, `MTPLX_MLX_COMMAND_BUFFER_MB=1000` restores the m1 behaviour.
The m1 rows (`mtplx-fork-m1`): decode 31.8 / 31.2 / 27.0 / 24.3 / 19.6 tok/s, whole
conversation 35 s / 78 s / 4.7 min / 10.9 min / 25.3 min at 2K / 8K / 32K / 64K / 128K.

## Decode speed

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-dark.svg?v=1013">
  <img alt="Decode tok/s vs prompt length, six engines" src="charts/decode-light.svg?v=1013">
</picture>


The same, extended to 256K for the three engines of the spot check (the M1 fork's 256K point is v2.12.2-m1.2, because m1.1's memory guard refused the prompt; oMLX's ran with its memory guard at `aggressive`):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-256k-dark.svg?v=1013">
  <img alt="Decode tok/s from 2K to 256K for the MTPLX M1 fork, MTPLX upstream, oMLX and Splash 1.3.0-m1.1" src="charts/decode-256k-light.svg?v=1013">
</picture>

## Follow-up turns

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/ttft-followup-dark.svg?v=1013">
  <img alt="Time to first token for turns 2-3, log scale" src="charts/ttft-followup-light.svg?v=1013">
</picture>

## The whole conversation

The last panel is the 256K spot check (one run each; the M1 fork there is v2.12.2-m1.2, and oMLX ran with its memory guard at `aggressive`).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/conversation-dark.svg?v=1013">
  <img alt="Three turns end to end per engine and prompt length" src="charts/conversation-light.svg?v=1013">
</picture>

## Memory

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/memory-dark.svg?v=1013">
  <img alt="Peak wired memory vs prompt length" src="charts/memory-light.svg?v=1013">
</picture>

## Tables

Median of 2 rounds per cell for 2K–64K; 128K is a single run.
`needle` counts the runs whose turn 3 quoted the needle line correctly. For Splash, the
2K fp16 cells use its BF16 KV (`--kv-format bf16`). `cached` is the engine-reported
`usage.prompt_tokens_details.cached_tokens`. Wired memory is system-wide.

| context | engine | T1 TTFT (cold) | T2 TTFT | T3 TTFT | decode T1 / T2 / T3 (tok/s) | 3-turn total | cached T2 / T3 | peak wired | needle |
|---|---|---|---|---|---|---|---|---|---|
| 2K | MTPLX M1 fork (m1.1) | 9.02 s | 0.55 s | 0.53 s | 29.0 / 31.9 / 27.3 | 33.3 s | 1,936 / 2,243 | 31.6 GB | 2/2 |
| 2K | MTPLX upstream | 11.4 s | 0.63 s | 0.64 s | 25.0 / 22.8 / 23.1 | 41.8 s | 1,936 / 2,243 | 34.0 GB | 2/2 |
| 2K | MTPLX upstream + env | 11.4 s | 0.64 s | 0.63 s | 24.5 / 31.6 / 25.4 | 37.5 s | 1,928 / 2,234 | 29.8 GB | 2/2 |
| 2K | oMLX | 11.4 s | 13.6 s | 4.69 s | 21.2 / 22.9 / 20.8 | 1.0 min | 0 / 1,673 | 29.8 GB | 2/2 |
| 2K | Splash 1.3.0-m1.1 | 8.34 s | 2.72 s | 2.64 s | 28.9 / 40.7 / 31.3 | 34.8 s | 1,664 / 1,952 | 24.1 GB | 2/2 |
| 2K | TensorFold (bf16 KV) | 15.7 s | 3.47 s | 3.48 s | 26.1 / 28.9 / 28.0 | 47.0 s | 1,673 / 1,980 | 22.9 GB | 2/2 |
| 8K | MTPLX M1 fork (m1.1) | 36.5 s | 0.59 s | 0.58 s | 28.6 / 30.8 / 28.3 | 1.0 min | 8,089 / 8,396 | 33.2 GB | 2/2 |
| 8K | MTPLX upstream | 53.2 s | 0.74 s | 0.77 s | 22.9 / 23.0 / 21.5 | 1.4 min | 8,089 / 8,396 | 35.5 GB | 2/2 |
| 8K | MTPLX upstream + env | 55.5 s | 0.78 s | 0.79 s | 25.1 / 27.2 / 25.2 | 1.4 min | 8,089 / 8,396 | 31.3 GB | 2/2 |
| 8K | oMLX | 54.5 s | 2.95 s | 2.58 s | 20.7 / 21.9 / 17.8 | 1.6 min | 7,826 / 8,192 | 33.6 GB | 2/2 |
| 8K | Splash 1.3.0-m1.1 | 38.9 s | 2.83 s | 2.74 s | 28.1 / 31.7 / 29.0 | 1.1 min | 7,808 / 8,128 | 24.3 GB | 2/2 |
| 8K | TensorFold (bf16 KV) | 1.2 min | 3.68 s | 3.69 s | 24.6 / 26.0 / 26.0 | 1.8 min | 7,826 / 8,132 | 22.6 GB | 2/2 |
| 32K | MTPLX M1 fork (m1.1) | 2.8 min | 0.81 s | 0.80 s | 27.5 / 28.9 / 27.9 | 3.3 min | 32,665 / 32,972 | 36.2 GB | 2/2 |
| 32K | MTPLX upstream | 4.0 min | 1.32 s | 1.34 s | 17.3 / 12.7 / 12.0 | 5.0 min | 32,665 / 32,972 | 42.5 GB | 2/2 |
| 32K | MTPLX upstream + env | 4.0 min | 1.31 s | 1.34 s | 19.8 / 14.5 / 11.6 | 4.9 min | 32,665 / 32,972 | 38.1 GB | 2/2 |
| 32K | oMLX | 4.2 min | 4.73 s | 3.54 s | 14.4 / 14.1 / 14.3 | 5.1 min | 32,402 / 32,768 | 36.3 GB | 2/2 |
| 32K | Splash 1.3.0-m1.1 | 3.2 min | 3.56 s | 3.48 s | 24.3 / 28.5 / 23.2 | 3.8 min | 32,384 / 32,704 | 25.2 GB | 2/2 |
| 32K | TensorFold (bf16 KV) | 5.3 min | 4.31 s | 4.31 s | 15.4 / 18.9 / 16.5 | 6.1 min | 32,402 / 32,709 | 27.1 GB | 2/2 |
| 64K | MTPLX M1 fork (m1.1) | 6.9 min | 1.12 s | 1.13 s | 24.4 / 28.7 / 23.4 | 7.3 min | 65,419 / 65,726 | 40.1 GB | 2/2 |
| 64K | MTPLX upstream | 9.9 min | 2.07 s | 2.09 s | 8.8 / 8.8 / 8.4 | 11.3 min | 65,419 / 65,726 | 48.6 GB | 2/2 |
| 64K | MTPLX upstream + env | 9.7 min | 2.09 s | 2.12 s | 8.2 / 6.9 / 8.2 | 11.3 min | 65,419 / 65,726 | 44.5 GB | 2/2 |
| 64K | oMLX | 9.7 min | 6.77 s | 4.99 s | 10.6 / 11.0 / 10.1 | 10.9 min | 65,156 / 65,536 | 41.3 GB | 2/2 |
| 64K | Splash 1.3.0-m1.1 | 8.0 min | 4.27 s | 4.50 s | 19.2 / 26.4 / 18.8 | 8.7 min | 65,152 / 65,440 | 26.5 GB | 2/2 |
| 64K | TensorFold (bf16 KV) | 11.9 min | 5.25 s | 5.19 s | 10.8 / 13.1 / 10.0 | 13.1 min | 65,156 / 65,463 | 33.2 GB | 2/2 |
| 128K | MTPLX M1 fork (m1.1) | 17.8 min | 1.83 s | 1.83 s | 18.3 / 22.1 / 18.9 | 18.4 min | 130,969 / 131,276 | 44.9 GB | 1/1 |
| 128K | MTPLX upstream | 28.1 min | 3.53 s | 3.50 s | 4.7 / 5.1 / 5.0 | 30.6 min | 130,969 / 131,276 | 52.4 GB | 1/1 |
| 128K | MTPLX upstream + env | 29.5 min | 4.19 s | 4.16 s | 4.4 / 4.9 / 4.5 | 32.2 min | 130,969 / 131,276 | 52.7 GB | 1/1 |
| 128K | oMLX | 25.1 min | 10.9 s | 7.84 s | 7.3 / 7.4 / 6.6 | 27.0 min | 130,706 / 131,072 | 48.0 GB | 1/1 |
| 128K | Splash 1.3.0-m1.1 | 22.1 min | 6.42 s | 6.19 s | 12.9 / 17.0 / 13.0 | 23.1 min | 130,688 / 131,008 | 28.4 GB | 1/1 |
| 128K | TensorFold (bf16 KV) | 29.2 min | 29.3 min | 29.4 min | 6.3 / 6.5 / 6.3 | 89.6 min | 0 / 0 | 37.2 GB | 1/1 |

**2K: fp16 KV vs 8-bit KV** (decode tok/s, mean of turns 1–3)

| engine | fp16 KV | 8-bit KV | change |
|---|---|---|---|
| MTPLX M1 fork (m1.1) | 29.1 | 29.4 | +1.2% |
| MTPLX upstream | 24.8 | 23.6 | -5.0% |
| MTPLX upstream + env | 28.1 | 27.2 | -3.3% |
| oMLX | 23.4 | 21.6 | -7.4% |
| Splash 1.3.0-m1.1 | 33.5 | 33.6 | +0.5% |
Raw rows: [`results/2026-09-m1max-64gb/rows.jsonl`](results/2026-09-m1max-64gb/rows.jsonl) ·
per-turn CSV: [`summary.csv`](results/2026-09-m1max-64gb/summary.csv) ·
environment: [`environment.json`](results/2026-09-m1max-64gb/environment.json) ·
generated text of every turn: `results/2026-09-m1max-64gb/cells/*/`.

## Splash 1.3.0-m1.1 and the M1 fork v2.12.2-m1.1

Both ran together, interleaved, on 2026-10-13 with the same plans as the 2026-10-08 run of
their previous releases, which stay in the raw data (`splash-m1-121`, `mtplx-fork-m1-2122`).
Both releases added a Neural Engine (ANE) prefill: part of each layer's MLP runs on the
ANE while the GPU computes the rest. Each cell shows old → new.

**Splash 1.2.1-m1 → 1.3.0-m1.1**

| | 2K | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|
| 3-turn total | 38.0 → 34.8 s | 85.1 → 68.0 s | 5.4 → 3.8 min | 12.6 → 8.7 min | 34.3 → 23.1 min | 102.3 → 68.6 min |
| T1 TTFT (cold) | 12.1 → 8.3 s | 57.9 → 38.9 s | 4.8 → 3.2 min | 11.9 → 8.0 min | 33.3 → 22.1 min | 100.6 → 67.1 min |
| decode, mean of turns (tok/s) | 34.5 → 33.6 | 32.1 → 29.6 | 25.5 → 25.4 | 21.3 → 21.4 | 15.0 → 14.3 | 9.5 → 10.2 |
| peak wired | 23.2 → 24.1 GB | 23.5 → 24.3 GB | 24.3 → 25.2 GB | 25.4 → 26.5 GB | 27.6 → 28.4 GB | 31.8 → 32.7 GB |

- Cold prompt reading is 1.46–1.51× faster at every length, and the whole conversation
  8% (2K) to 33% (128K) shorter; decode is within 3% except at 8K (−8%) and 128K (−5%).
- Its 2K canary read 22.7–23.8 tok/s through the session, against 28.7–29.6 for 1.2.1-m1
  (the canary is a single cold 1.9K-token turn; the 2K cells' decode is within 3%). Three
  Splash cells (64K round 1 and both sampled 64K rounds) read more than 3% below the
  canary reference and were re-run once; the re-runs are used.

**M1 fork v2.12.2-m1 → v2.12.2-m1.1** (128K: v2.12.2-m1 refused turn 2, so the old value
is v2.12.0-m1.2's)

| | 2K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| 3-turn total | 35.3 → 33.3 s | 79.0 → 61.0 s | 4.6 → 3.3 min | 9.9 → 7.3 min | 24.9 → 18.4 min |
| T1 TTFT (cold) | 11.1 → 9.0 s | 54.5 → 36.5 s | 4.1 → 2.8 min | 9.3 → 6.9 min | 24.2 → 17.8 min |
| T2 / T3 TTFT | 0.53 → 0.54 s | 0.60 → 0.58 s | 0.86 → 0.81 s | 1.21 → 1.13 s | 2.10 → 1.83 s |
| decode, mean of turns (tok/s) | 29.6 → 29.4 | 29.2 → 29.2 | 25.6 → 28.1 | 23.1 → 25.5 | 19.0 → 19.8 |
| peak wired | 31.0 → 31.6 GB | 31.3 → 33.2 GB | 34.0 → 36.2 GB | 37.5 → 40.1 GB | 43.5 → 44.9 GB |

- Cold prompt reading is 1.36–1.49× faster from 8K (1.23× at 2K), decode 10% faster at
  32K–64K (fewer host round trips per decode round, an in-place draft cache), and the
  whole conversation 23–28% shorter from 8K. Peak memory is 1–2.6 GB higher (the ANE
  buffers). v2.12.2-m1.1 finished 128K, which v2.12.2-m1 had refused at turn 2.
- **256K needed m1.2.** m1.1's memory guard refused the 258K-token first turn (54.7 GiB
  projected against 48 GiB) although the same conversation ran without swap at a 53.2 GB
  peak with `--allow-swap`. v2.12.2-m1.2 corrects the M1 admission bills, prefills prompts
  of 32K+ straight into the 8-bit pages (same speed, less memory) and allows the engine
  13/16 of RAM on M1 (52 GiB on 64 GB); it ran the 256K conversation at its defaults with
  a 50.1 GB peak. Its code differs from m1.1 only for prompts of 32K+ and in admission, so
  the 2K–128K columns keep the m1.1 run.
- The fork's 2K canaries read 31.0–32.2 tok/s through the session (v2.12.2-m1: 30.3–30.9).

## oMLX 0.7.0 against 0.6.4

The same plans, model view and engine settings; 0.6.4 ran on 2026-09-25/26 (raw rows
`omlx-064`), 0.7.0 on 2026-10-07. Each cell shows 0.6.4 → 0.7.0.

| | 2K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| 3-turn total | 71 → **61 s** | 125 → **95 s** | 6.0 → **5.1 min** | 12.4 → **10.9 min** | 29.6 → **27.0 min** |
| T1 TTFT (cold) | 11.4 → 11.4 s | 56.4 → 54.5 s | 4.4 → 4.2 min | 10.2 → 9.7 min | 26.3 → 25.1 min |
| T2 TTFT | 13.5 → 13.6 s | 30.0 → **2.95 s** | 44.6 → **4.73 s** | 1.0 min → **6.77 s** | 1.6 min → **10.9 s** |
| T3 TTFT | 15.4 → **4.69 s** | 2.69 → 2.58 s | 3.45 → 3.54 s | 4.76 → 4.99 s | 7.92 → 7.84 s |
| decode, mean of turns (tok/s) | 21.6 → 21.6 | 19.4 → 20.2 | 14.3 → 14.3 | 10.7 → 10.6 | 7.4 → 7.1 |
| peak wired | 26.9 → 29.8 GB | 33.2 → 33.6 GB | 37.0 → 36.3 GB | 39.6 → 41.3 GB | 48.5 → 48.0 GB |

- **The whole conversation is 9–24% shorter, mostly from the follow-up turns.** 0.7.0
  also saves the prefix cache at the end of the first prompt, so turn 2 reads only the
  ~300 new tokens instead of everything after the last 4,096-token block (cached tokens
  at 32K: 32,402 instead of 28,672). Turn 3 still resumes from the last block boundary,
  as in 0.6.4. At 2K, turn 2 still re-reads the whole prompt and turn 3 reuses the first
  one.
- **Cold prompt reading is 3–6% faster from 8K up; decode is within 5% of 0.6.4** at
  every length.
- **At 128K, oMLX's whole conversation (27.0 min) is now shorter than upstream's (30.6)
  and those of Splash 1.1.0-m1 (29.4) and 1.2.1-m1 (34.3)**; Splash 1.3.0-m1.1's is 23.1 and
  the M1 fork's 18.4 (both measured later).
- **The rebuilt memory guard** (on by default, tier `balanced`) paused the 128K first
  turn once and then read it in 1,024-token chunks instead of 4,096 (the 4,096 tokens
  read before the pause show as `cached` on that turn); it was still faster than 0.6.4.
  At 256K it rejects the first turn outright ([below](#256k-spot-check)).
- **Output:** the 2K fp16-KV and 32K texts are byte-identical to 0.6.4's; the other cells
  differ in some turns (0.7.0's own two 2K 8-bit rounds differ too), and every run quoted
  the needle line correctly.

## 256K spot check

One run per engine at 258,048 context tokens (the prompt reaches 258K tokens by turn 3,
inside each engine's 262,144-token window): the MTPLX M1 fork at **v2.12.2-m1.2** and
Splash 1.3.0-m1.1 on 2026-10-13, oMLX 0.7.0 on 2026-10-07. The two MTPLX upstream columns
and TensorFold were not run: at 128K they already decode at 4.6–6.4 tok/s, and a 256K
first turn would take well over an hour each.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/spot256k-dark.svg?v=1013">
  <img alt="256K spot check: first-turn prefill, follow-up TTFT, decode and peak memory for the MTPLX M1 fork (m1.2), oMLX 0.7.0 and Splash 1.3.0-m1.1" src="charts/spot256k-light.svg?v=1013">
</picture>

| | Splash 1.3.0-m1.1 | MTPLX M1 fork (m1.2) | oMLX 0.7.0 (memory guard aggressive) |
|---|---|---|---|
| T1 TTFT (cold) | 67.1 min | **58.5 min** | 69.8 min |
| T2 / T3 TTFT | 9.8 s / 10.3 s | **2.9 s / 2.9 s** | 19.4 s / 13.5 s |
| decode T1 / T2 / T3 (tok/s) | 9.1 / 13.3 / 8.3 | 14.4 / 14.4 / 13.8 | 4.1 / 4.5 / 3.8 |
| decode, mean of turns | 10.2 | **14.2** | 4.1 |
| 3-turn total | 68.6 min | **59.4 min** | 73.1 min |
| peak wired | **32.7 GB** | 50.1 GB | 57.3 GB |
| needle | 1/1 | 1/1 | 1/1 |

- **The M1 fork needed m1.2 at 256K.** v2.12.2-m1.1's memory guard refused the first
  turn before reading it (54.7 GiB projected against its 48 GiB limit). Run with
  `--allow-swap`, m1.1 finished the same conversation without touching swap (58.9 min,
  14.4 tok/s, 3.0 s follow-ups, 53.2 GB peak; rows `mtplx-fork-allowswap`). m1.2 prices
  the M1 fork's prefill as measured, reads prompts of 32K+ straight into 8-bit pages and
  lets the engine use 13/16 of RAM on M1, and ran it at its defaults.
- **Splash 1.3.0-m1.1 read the prompt 33% faster than 1.2.1-m1** (67.1 against 100.6 min)
  and decoded 7% faster (10.2 against 9.5 tok/s).
- **oMLX cannot run 256K at its defaults.** 0.7.0's prefill memory guard (tier
  `balanced`) rejected the first turn with HTTP 400 before reading it (~46.1 GB predicted
  peak against a 39.4 GB ceiling; the row is kept in the raw data). The column above is a
  second run with `memory_guard_tier: aggressive`, the only non-default oMLX setting in
  this benchmark. Even then the guard paused the first turn once and, about 40 minutes
  in, cut its prefill chunks from 4,096 to 1,024 tokens (the tokens read before a pause
  are counted as `cached`); peak wired memory reached 57.3 GB.
- **Memory:** Splash keeps the whole 256K conversation in 32.7 GB; the M1 fork needs
  50.1 GB. About 6 GB of the difference is its model files (8-bit embeddings, output
  layer and last 8 MLPs), most of the rest a second copy of the conversation that its
  session bank holds for fast follow-up turns.
- Single runs. 2K canaries before each run: Splash 23.8 tok/s, the M1 fork 30.9, oMLX 21.0.

## Qwen3.6-35B-A3B (MoE)

The same 3-turn conversations on the mixture-of-experts Qwen3.6-35B-A3B (35B parameters,
about 3B active per token), 8-bit KV, temperature 0. The MTPLX M1 fork
[`v2.12.2-m1.5`](https://github.com/shunya1810/MTPLX/releases/tag/v2.12.2-m1.5) serves
`Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16` with its MTP head (depth 1);
Splash 1.3.0-m1.1 serves `incoai/Qwen3.6-35B-A3B-Splash` (4-bit + DFlash2 draft). The two
ran together, interleaved, on the afternoon of 2026-10-14 (2K–64K two rounds, 128K once);
the fork's v2.12.2-m1.4 row is the run of that morning, for reference. Tables:
`scripts/summarize_a3b.py`, raw rows in `results/2026-09-m1max-64gb-qwen36-a3b/`.

| Context | Engine | Whole conversation | Turn 1 TTFT | Turns 2–3 TTFT | Cold prefill | Decode (mean of 3 turns) | Peak wired | Rounds |
|---|---|---|---|---|---|---|---|---|
| 2K | **MTPLX M1 fork (m1.5)** | **9.3 s** | 1.5 s | 0.7 s | 1109 tok/s | **111.8 tok/s** | 27.0 GB | 2 |
| 2K | Splash 1.3.0-m1.1 | 9.5 s | 1.8 s | 0.5 s | 947 tok/s | 96.1 tok/s | 26.1 GB | 2 |
| 2K | MTPLX M1 fork (m1.4) | 10.9 s | 1.6 s | 0.7 s | 1068 tok/s | 90.1 tok/s | 26.5 GB | 2 |
| 8K | **MTPLX M1 fork (m1.5)** | **14.4 s** | 6.4 s | 0.8 s | 1218 tok/s | **110.8 tok/s** | 27.1 GB | 2 |
| 8K | Splash 1.3.0-m1.1 | 18.1 s | 8.5 s | 0.6 s | 918 tok/s | 80.8 tok/s | 26.4 GB | 2 |
| 8K | MTPLX M1 fork (m1.4) | 17.3 s | 7.2 s | 0.8 s | 1088 tok/s | 87.6 tok/s | 27.6 GB | 2 |
| 32K | **MTPLX M1 fork (m1.5)** | **41.3 s** | 30.7 s | 1.1 s | 1054 tok/s | **86.4 tok/s** | 28.2 GB | 2 |
| 32K | Splash 1.3.0-m1.1 | 60.7 s | 48.1 s | 0.9 s | 674 tok/s | 63.8 tok/s | 26.5 GB | 2 |
| 32K | MTPLX M1 fork (m1.4) | 47.3 s | 35.5 s | 1.2 s | 914 tok/s | 77.1 tok/s | 29.5 GB | 2 |
| 64K | **MTPLX M1 fork (m1.5)** | **92.7 s** | 80.2 s | 1.6 s | 813 tok/s | **76.5 tok/s** | 29.1 GB | 2 |
| 64K | Splash 1.3.0-m1.1 | 2.5 min | 2.2 min | 1.3 s | 488 tok/s | 57.7 tok/s | 26.8 GB | 2 |
| 64K | MTPLX M1 fork (m1.4) | 108.2 s | 94.2 s | 1.7 s | 692 tok/s | 69.4 tok/s | 29.8 GB | 2 |
| 128K | **MTPLX M1 fork (m1.5)** | **4.3 min** | 4.0 min | 2.5 s | 543 tok/s | **61.2 tok/s** | 30.5 GB | 1 |
| 128K | Splash 1.3.0-m1.1 | 7.7 min | 7.4 min | 2.1 s | 296 tok/s | 43.9 tok/s | 27.5 GB | 1 |
| 128K | MTPLX M1 fork (m1.4) | 4.9 min | 4.5 min | 2.9 s | 480 tok/s | 56.7 tok/s | 30.7 GB | 1 |

- **The M1 fork is now ahead at every length but on memory:** the whole conversation
  takes 2–44% less time (9.3 against 9.5 s at 2K, 14.4 against 18.1 s at 8K, 4.3 against
  7.7 min at 128K), decode is 16–39% faster (111.8 against 96.1 tok/s at 2K, 61.2 against
  43.9 at 128K) and cold prefill 17–83% faster.
- **m1.5 against m1.4:** decode +24% at 2K, +26% at 8K, +12% at 32K, +10% at 64K, +8% at
  128K; conversations 12–17% shorter. m1.5 keeps the MTP rounds on the GPU (the next round
  is queued before the host reads the previous one; greedy text identical, sampled
  requests too) and runs part of the prefill on the Neural Engine; see the fork's
  [CHANGES.ja.md](https://github.com/shunya1810/MTPLX/blob/m1max-longctx/docs/m1max-longctx/CHANGES.ja.md).
- **Memory:** Splash peaks at 26–28 GB, the fork at 27–31 GB.
- 256K was not re-run; on 2026-10-14 m1.4 took 17.9 min (42.8 tok/s) against Splash's
  27.0 min (21.5 tok/s).

### A3B, short prompts with thinking on

The conditions of Splash's own numbers (`docs/performance.md`: short coding prompts,
reasoning on, greedy, 1,024-token cap) plus a sampled variant (temperature 1.0, top_p
0.95, top_k 20, 2,048 tokens), four prompts (pelican SVG, heptagon, LRU cache, ISO 8601
parser), each engine twice, alternating (`scripts/think_short.py`, raw rows in
`results/2026-10-m1max-64gb-a3b-think/`). Every run hit the token cap; decode counts
reasoning tokens.

| Decode (tok/s) | MTPLX M1 fork (m1.5) | Splash 1.3.0-m1.1 |
|---|---|---|
| greedy, 1,024 tokens | 120.8 | **132.0** |
| sampled, 2,048 tokens | 114.6 | **131.6** |

- **Here Splash is 9–15% faster.** Per prompt it ranges from even (heptagon, ISO parser:
  117–120 against 120–124 greedy) to 15–26% (pelican, LRU). Thinking text and boilerplate
  code are what Splash's DFlash2 block draft guesses best; the fork's MTP head drafts one
  token per round (at most 2 tokens per round, 1.7–1.9 here).
- Splash's README figures (210 tok/s for this model) are from an M5 Pro, not comparable
  with an M1 Max.

## Short prompts, the Splash part 2 conditions

A repeat of benchmark 5 from the "Splash on M1, part 2" post, for the MTPLX M1 fork
(v2.12.0-m1.2) and Splash 1.1.0-m1 only: npanj's five splash-plus prompts on a
5-minute loop (250 tokens, temperature 0, reasoning xhigh; both engines see the same
prompt token counts), a 3-minute rest, then one cold 44K prompt. ABBA, two runs each,
2026-09-30, macmon every second ([`scripts/erp_repro.py`](scripts/erp_repro.py),
[`run_erp.sh`](run_erp.sh), raw data in
[`results/2026-09-m1max-64gb-erp/`](results/2026-09-m1max-64gb-erp/)). Means of the two runs:

| | MTPLX M1 fork v2.12.0-m1.2 | Splash 1.1.0-m1 |
|---|---|---|
| decode, 5-minute loop (mean per request) | 31.5 tok/s | **40.7 tok/s** |
| GPU temperature while looping (mean / max) | 88.4 / 94.5 °C | 90.1 / 92.8 °C |
| time above 80 °C while looping | 94% | 93% |
| package power while looping | 47.1 W | 42.0 W |
| energy per generated token | 1.74 J | **1.17 J** |
| fans while looping (mean of both) | 3,400 rpm | 1,900 rpm |
| cold 44K prompt | **364 s (123 tok/s)** | 438 s (103 tok/s) |
| GPU temperature during that prefill (mean / max) | 85.2 / 97.7 °C | 90.8 / 94.1 °C |
| engine footprint after the 44K prompt | 29.0 GB (peak 31.1) | see note |

- The same picture as the Splash post: on short prompts Splash decodes 29% faster and
  uses a third less energy per token, and the MLX-based engine reads a long prompt ~20%
  faster. The GPU clock stayed at ~1,295 MHz from start to end of every loop (no
  throttling), on the 100 W adapter with the battery at 93–95%.
- Temperatures are not comparable with that post's: here the fans ran on macOS's default
  control, which let both engines sit near 90 °C; that post used a fan curve reaching
  full speed at 80 °C, under which only Splash stayed below 80. The MTPLX build's GPU
  sensors peaked higher during prefill (97–98 °C), as that post observed for MTPLX.
- Energy per token here is package power over the whole loop (including the short
  prefills) divided by generated tokens.
- Splash maps its weights from disk, so `footprint` sees only ~4.9 GB of it; its wired
  memory in the long-context runs above was 22–26 GB.

## Thinking on (reasoning xhigh, long generations)

Three tasks with thinking on, reasoning xhigh, the model's sampling settings (temperature
1.0, top-p 0.95, top-k 20) and no practical output cap, on the MTPLX M1 fork at
v2.12.0-m1.3 and Splash 1.1.0-m1 (2026-09-30/10-01, [`scripts/pelican.py`](scripts/pelican.py),
[`run_thinking2.sh`](run_thinking2.sh), raw data and every output in
[`results/pelican/`](results/pelican/)):

| task | engine | runs | generated tokens | total time | decode tok/s |
|---|---|---|---|---|---|
| "Generate an SVG of a pelican riding a bicycle" | Splash 1.1.0-m1 | 3 | 30.2K–46.2K | 17.6–26.1 min | 27.9–29.5 |
| | MTPLX M1 fork | 3 | 29.5K–31.7K | 16.7–18.3 min | 28.9–29.6 |
| | MTPLX M1 fork + `MTPLX_MLX_COMMAND_BUFFER_MB=1000` | 3 | 26.9K–35.6K | 14.0–19.0 min | 30.3–31.9 |
| 20 balls in a spinning heptagon (Python) | Splash 1.1.0-m1 | 2 | 62.6K–63.5K | 42.6–46.2 min | 22.9–24.5 |
| | MTPLX M1 fork | 2 | 47.2K–56.8K | 31.9–39.1 min | 24.3–24.7 |
| Review a 29.7K-token source file | Splash 1.1.0-m1 | 2 | 40.4K–51.2K | 34.1–47.1 min | 19.7–20.0 |
| | MTPLX M1 fork | 2 | 31.8K–34.5K | 26.8–27.0 min | 21.5–22.9 |

<img alt="Nine pelican SVGs: three each from Splash 1.1.0-m1, the MTPLX M1 fork and the fork with larger command buffers" src="charts/pelican-grid.png" width="720">

- Once the reasoning runs to tens of thousands of tokens, the fork decodes as fast as
  Splash or slightly faster; Splash's lead on short generations does not carry over.
- Reasoning length varies a lot from run to run at temperature 1.0, so compare decode
  speed rather than total time. The second code-review run of each engine reused the
  first run's prompt cache (cold prompt read: Splash 264.6 s, the fork 228.8 s).
- All four heptagon programs compile (617–721 lines); none was run here (they open a
  tkinter window). None of the four code reviews found the restore-point bug fixed in
  v2.12.0-m1.2; three flagged a reservation leak in `cancel_pending()` instead (not yet
  verified).
- Every fork run before v2.12.0-m1.3 failed at about 16,530 generated tokens (non-finite
  logits; upstream 1de2b1c as well): the compiled-verify cache could not grow past the
  16,384 new tokens reserved up front. The failed runs are in
  `results/pelican-xhigh-aborted/` and `results/pelican/*fork-m1.2*`; the reproduction is
  [`scripts/nan_repro.py`](scripts/nan_repro.py) with its results in `results/nan-repro/`.

## Sampled decoding (temperature 1.0)

Greedy decoding at 256 tokens is a matched workload, not how the model is normally run.
The same conversations at 2K, 32K and 64K were repeated at the model's own sampling
settings (`generation_config.json`: temperature 1.0, top-p 0.95, top-k 20), up to 512
tokens per turn, 8-bit KV, two rounds with the order reversed (the upstream columns on
2026-09-29; the M1 fork v2.12.2-m1.1 and Splash 1.3.0-m1.1 together on 2026-10-13). Sampled text differs between runs, so turns 2 and 3 see different
prompts; acceptance is the engine's own count (MTPLX: accepted / drafted MTP tokens,
Splash: accepted / drafted DFlash2 tokens, 7 per pass) and is not comparable across the
two methods.

| context | engine | T1 TTFT (cold) | decode T1 / T2 / T3 (tok/s) | decode mean | draft acceptance T1 / T2 / T3 | tokens T1 / T2 / T3 | needle |
|---|---|---|---|---|---|---|---|
| 2K | MTPLX M1 fork (m1.1) | 9.00 s | 30.2 / 33.1 / 31.7 | **31.6** | 58% / 67% / 63% | 286 / 326 / 186 | 2/2 |
| 2K | MTPLX upstream | 11.4 s | 23.1 / 25.3 / 21.2 | **23.2** | 58% / 77% / 59% | 314 / 413 / 200 | 2/2 |
| 2K | MTPLX upstream + env | 11.6 s | 24.8 / 27.0 / 27.3 | **26.4** | 56% / 64% / 66% | 294 / 328 / 188 | 2/2 |
| 2K | Splash 1.3.0-m1.1 | 8.41 s | 30.4 / 34.5 / 29.6 | **31.5** | 33% / 39% / 32% | 278 / 328 / 182 | 2/2 |
| 32K | MTPLX M1 fork (m1.1) | 2.8 min | 28.0 / 28.3 / 28.5 | **28.3** | 62% / 63% / 66% | 351 / 311 / 174 | 2/2 |
| 32K | MTPLX upstream | 4.0 min | 17.1 / 12.9 / 12.7 | **14.2** | 56% / 66% / 63% | 276 / 322 / 202 | 2/2 |
| 32K | MTPLX upstream + env | 4.1 min | 17.5 / 12.7 / 12.4 | **14.2** | 53% / 61% / 62% | 292 / 321 / 172 | 2/2 |
| 32K | Splash 1.3.0-m1.1 | 3.2 min | 24.1 / 27.3 / 26.0 | **25.8** | 33% / 39% / 37% | 316 / 312 / 186 | 2/2 |
| 64K | MTPLX M1 fork (m1.1) | 6.9 min | 21.7 / 25.0 / 25.1 | **23.9** | 48% / 61% / 62% | 293 / 274 / 182 | 2/2 |
| 64K | MTPLX upstream | 9.7 min | 8.3 / 9.0 / 9.3 | **8.9** | 55% / 70% / 60% | 322 / 330 / 208 | 2/2 |
| 64K | MTPLX upstream + env | 10.2 min | 7.6 / 8.8 / 8.1 | **8.2** | 56% / 69% / 61% | 295 / 298 / 204 | 2/2 |
| 64K | Splash 1.3.0-m1.1 | 7.9 min | 17.5 / 20.0 / 17.9 | **18.5** | 30% / 35% / 30% | 278 / 320 / 188 | 2/2 |

- **The two are even at 2K (31.6 against 31.5 tok/s); the M1 fork leads at 32K (28.3
  against 25.8) and 64K (23.9 against 18.5).** With v2.12.2-m1 and 1.2.1-m1 (2026-10-08)
  the rows read 29.2 / 25.5 / 22.8 for the M1 fork and 32.4 / 23.8 / 20.1 for Splash
  (`mtplx-fork-m1-2122`, `splash-m1-121` in the raw data).
- **Sampling moves Splash more than the MTPLX engines** against their greedy speed: Splash
  is 6–14% slower than greedy at 2K and 64K; the M1 fork is within 7% and faster at 2K.
- **Upstream + env** is 14% faster than upstream at 2K, the same at 32K and 8% slower at
  64K, as with greedy decoding.
- The model ends turns on its own, so each engine generated a different number of tokens
  (turn 1: 266–358); every engine quoted the needle line in every run.

Raw rows: [`results/2026-09-m1max-64gb-sampling/rows.jsonl`](results/2026-09-m1max-64gb-sampling/rows.jsonl).

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
  can follow further than the 3-token MTP draft; the MTPLX M1 fork commits 2.8–3.0 tokens per
  round on turn 2 and 2.7–2.9 on turns 1 and 3, with a ceiling of 4 (verify plus draft time
  per round, m1.2).
- **Splash verifies twice as many positions per pass at a similar cost at short context:**
  105 ms per pass at 2K for 8 positions and a 7-token draft, against 93–98 ms per round on
  the M1 fork for 4 positions and a 3-token MTP draft.
- **The pass cost grows faster with the prompt:** from 105 ms (2K) to 189–197 ms (128K)
  on Splash, against 93–98 to 141–144 ms per round on the M1 fork. Each pass attends over the
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
  workload; the sampled scenario above repeats 2K–64K at temperature 1.0.
- Same model files for the MTPLX and oMLX engines: oMLX reads the MTPLX checkpoint
  through a symlinked, text-only view ([`scripts/prepare_omlx.py`](scripts/prepare_omlx.py));
  no weight is converted. Splash reads its own package, `incoai/Qwen3.8-27B-Splash`.
- 8-bit KV is each engine's own implementation (MTPLX affine q8 paged KV, oMLX
  TurboQuant 8-bit). Everything else is at engine defaults, including the prefix cache
  (MTPLX: RAM; oMLX: SSD, 4,096-token blocks).
- Every cell restarts the engine with its caches wiped, then runs a warmup and a
  thermal canary. Engine order is reversed in round 2. A cell whose canary was >3%
  slower than that engine's best was re-run after a 5-minute rest.
- MTPLX upstream and upstream + env ran together on 2026-09-29 (none needed a re-run).
  128K ran once; upstream + env with command buffers ran out of memory there, and the
  published upstream + env (without them) ran its own 128K afterwards. Their 128K canaries
  were within 2.4% of each engine's best 2K–64K canary.
- The M1 fork (v2.12.0-m1.2) ran alone on 2026-09-30 with the same plans, after a 2K
  probe had to reach 29.9 tok/s on its canary (it read 30.6). A first attempt that
  afternoon was stopped because the machine ran ~5% slow (canary 28.0–29.1); its rows
  are in `rows-fork-m12-aborted-20260930.jsonl`. In the run used, the 2K and 8K cells of
  round 2 had canaries 3.1% below the run's best and were re-run once; the re-runs stayed
  3% below and are used. Its 128K canary was 2.6% below that best. The first release (m1)
  ran with the other two MTPLX columns on 2026-09-29 and stays as `mtplx-fork-m1`.
- The M1 fork v2.12.2-m1.1 and Splash 1.3.0-m1.1 ran together on 2026-10-13 with the same
  plans, interleaved (round 1 Splash first, round 2 reversed; 128K Splash first), then the
  sampled scenario and both 256K runs ([`run_refresh_1013.sh`](run_refresh_1013.sh)); the
  M1 fork's 256K row is v2.12.2-m1.2's ([`run_256k_m12b.sh`](run_256k_m12b.sh)). The
  fork's canaries read 31.0–32.2 tok/s, Splash's 22.7–23.8; three Splash cells were re-run
  once (above). The previous releases ran the same way on 2026-10-08
  ([`run_refresh_1008.sh`](run_refresh_1008.sh)).
- oMLX 0.7.0 ran alone on 2026-10-07 with the same plans, after the M1 fork's 2K probe
  read 31.1 tok/s (gate 29.9). Its 8-bit canaries stayed within 0.3% of its best (21.0
  tok/s) and no cell was re-run. Splash and TensorFold ran in earlier sessions with the
  same plans (details and canaries in [METHODOLOGY.md](METHODOLOGY.md)).
- Round-to-round decode difference in the 2026-09-29 run: median 0.7%. The largest is one
  turn of upstream + env at 64K (5.2 against 8.7 tok/s on byte-identical text; its verify
  time was 47 s against 27 s in the other round), which pulls that cell's turn-2 median
  to 6.9. All MTPLX columns produced byte-identical text across rounds in every turn.

## Reproduce

```bash
cp config/local.example.json config/local.json   # set your paths
python3 scripts/prepare_omlx.py                   # oMLX model view + base paths
./run_day.sh                                      # 2K-64K, two rounds (~3 h)
./run_night.sh                                    # 128K, one round (~3 h)
./run_fair.sh                                     # the three MTPLX columns + sampled scenario
./run_refresh_1013.sh                             # M1 fork v2.12.2-m1.1 + Splash 1.3.0-m1.1 (~6 h)
./run_256k_m12b.sh                                # M1 fork v2.12.2-m1.2 at 256K (~1 h)
./run_omlx070.sh                                  # oMLX 0.7.0: 2K-128K (~2 h); 256K is rejected at defaults
./run_omlx070_256k_aggressive.sh                  # oMLX 0.7.0 256K, memory guard aggressive (~75 min)
python3 scripts/summarize.py results/2026-09-m1max-64gb
python3 scripts/summarize_sampling.py results/2026-09-m1max-64gb-sampling
```

MTPLX arms run from git worktrees under `work/worktrees/` (`mtplx-1de2b1c`, `mtplx-v2.12.0-m1`)
(`git worktree add --detach work/worktrees/mtplx-<commit> <commit>` in an MTPLX clone),
with the MTPLX app's Python environment. `runner.py` and `summarize.py` need only the
standard library and `tokenizers`.

## Adding an engine

An engine is one JSON file in [`engines/`](engines/README.md): its start command, port,
KV modes and how to read its own stats. Add it to a plan in `plans/` and to the engine
list in `scripts/summarize.py`.

## License

MIT for the code in this repository. Model and engines are under their own licenses.
