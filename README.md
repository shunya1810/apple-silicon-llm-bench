# Apple Silicon LLM engine benchmark: long-context, multi-turn

English | [日本語](README.ja.md)

How fast is a **3-turn conversation over a long prompt** on a Mac, depending on the
inference engine? On a MacBook Pro with M1 Max and 64 GB, five inference engines serve
Qwen3.8-27B with speculative decoding and 8-bit KV cache (TensorFold has no 8-bit KV and
runs bf16 KV), measured the same way over each engine's OpenAI-compatible streaming API.
A sixth column runs MTPLX upstream with the M1 build's defaults set through environment
variables, to separate what the M1 build's code adds from what its settings add. The
MTPLX and oMLX engines load the **same model files** with MTP (depth 3); Splash and
TensorFold need their own packages (4-bit, group-64 weights plus a DFlash2 draft model),
so their columns also reflect a different quantization and a different speculative method.

> **Disclosure:** the MTPLX M1 build is maintained by the author of this benchmark. Its
> column runs the published release tag with the same command line as upstream, and every
> number below comes from `rows.jsonl` through `scripts/summarize.py`.

| engine | version | notes |
|---|---|---|
| **MTPLX M1 build** | [`v2.12.0-m1`](https://github.com/shunya1810/MTPLX/releases/tag/v2.12.0-m1) (`6be2b28`, [`shunya1810/MTPLX`](https://github.com/shunya1810/MTPLX/tree/m1max-longctx)) | unofficial M1-family long-context build of MTPLX 2.12.0. The 256K spot check runs its successor [`v2.12.0-m1.1`](https://github.com/shunya1810/MTPLX/releases/tag/v2.12.0-m1.1) (larger command buffers off: prompt reading at upstream speed, decode 6–8% lower; [below](#256k-spot-check-two-engines)). Earlier rounds of this build stay in the raw data as `mtplx-fork-16751dc`, `-40b6113`, `-4fe8067`, `-bba1b7b` and `-0cd2a73` |
| **MTPLX upstream** | [`youssofal/MTPLX@1de2b1c`](https://github.com/youssofal/MTPLX/commit/1de2b1c049136ed117af0c6712baaadd81820b51) | `main` at run time (still its head on 2026-09-29); the M1 build's base. Its first run (2026-09-25/26) stays in the raw data as `mtplx-upstream-0925` |
| **MTPLX upstream + env** | upstream `1de2b1c` | the same code with every M1-build default that upstream exposes as an environment variable: `MTPLX_CONTEXT_COPY=0`, `MTPLX_FUSE_PROJ=gdn,attn`, `MTPLX_FRSPEC_DRAFT=1 MTPLX_FRSPEC_LEGACY=1 MTPLX_FRSPEC_VOCAB=builtin:qwen38-code-64k`, `MTPLX_MLX_CACHE_LIMIT=1G`, `MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES=2` ([engine file](engines/mtplx-upstream-tuned-nobuf.json)); the M1 build's larger Metal command buffers are left out (see below) |
| **oMLX** | [0.6.4](https://github.com/jundot/omlx/releases/tag/v0.6.4) | latest stable at run time |
| **Splash 1.1.0-m1** | [paperniuk/splash 1.1.0-m1](https://github.com/paperniuk/splash/releases/tag/1.1.0-m1) | community M1/M2 build of [incoai/splash](https://github.com/incoai/splash) 1.1.0 with Apple7/8 kernels; model `incoai/Qwen3.8-27B-Splash` (4-bit g64 + DFlash2 draft), INT8 KV; replaces the 1.0.2-m1 release and a source build of its head measured earlier (raw data: `splash-m1-102`, `splash-src`) |
| **TensorFold** | [0.3.5.1](https://github.com/ashhart/TensorFold/releases/tag/v0.3.5.1) | Python/MLX engine with exact (token-equality) draft acceptance; model `Vontra/Qwen3.8-27B-MLX-4bit` (4-bit g64) + `z-lab/Qwen3.8-27B-DFlash2` draft (4-bit at load), bf16 KV (no 8-bit option); release of 2026-09-28, the first that loads on M1 Max |

Prompt lengths 2K–64K ran in 2 rounds each (median shown); 128K ran once. The three MTPLX
columns ran together, interleaved, on 2026-09-29; oMLX ran on 2026-09-25/26, Splash and
TensorFold each in their own session afterwards, the same way.

## Highlights (8-bit KV)

Decode is the mean of the three turns, in tok/s, at 2K / 8K / 32K / 64K / 128K.

| engine | 2K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| MTPLX M1 build | 31.8 | 31.2 | 27.0 | **24.3** | **19.6** |
| MTPLX upstream | 23.6 | 22.5 | 14.0 | 8.7 | 5.0 |
| MTPLX upstream + env | 27.2 | 25.8 | 15.3 | 7.8 | 4.6 |
| oMLX | 21.6 | 19.4 | 14.3 | 10.7 | 7.4 |
| Splash 1.1.0-m1 | **33.1** | **32.5** | **27.1** | 24.1 | 18.7 |
| TensorFold (bf16 KV) | 27.7 | 25.5 | 17.0 | 11.3 | 6.4 |

- **Short prompts belong to Splash, long ones are a tie with the M1 build.** Splash
  1.1.0-m1 decodes 4% faster than the M1 build at 2K–8K; at 32K–64K the two are within 1%,
  and at 128K the M1 build is 5% ahead. Both decode 1.9× (32K) to 2.5–2.6× (128K) as fast as oMLX.
- **Sampling changes the order a little.** At the model's own sampling settings
  (temperature 1.0, top-p 0.95, top-k 20, up to 512 tokens), decode is
  32.3 / 27.3 / 24.1 tok/s on the M1 build and 32.7 / 25.8 / 22.0 on Splash at 2K / 32K / 64K:
  even at 2K, the M1 build 6% ahead at 32K and 10% at 64K. Upstream: 23.2 / 14.2 / 8.9
  ([details](#sampled-decoding-temperature-10)).
- **What the M1 build's settings are worth on upstream:** upstream + env is 15%
  faster than upstream at 2K–8K and 9% at 32K, but 10% slower at 64K and 8% at 128K
  (turning context-copy drafting off costs upstream's slower long-context rounds more
  than it saves). The M1 build's lead over upstream + env (17% at 2K, 1.8× at 32K, 3.1×
  at 64K, 4.3× at 128K) is therefore its code: M1 attention kernels for verify, draft
  and prefill, and the session-bank fixes ([what the build changes](https://github.com/shunya1810/MTPLX/blob/m1max-longctx/docs/m1max-longctx/CHANGES.ja.md)).
- **The command-buffer setting is not an upstream win.** The M1 build also raises MLX's
  Metal command buffers to 150 ops / 1,000 MB, paired with a prefill change that is code.
  Given to upstream through `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` alone, it
  raised peak memory by 12–15 GB over upstream + env without it, slowed the 32K–64K cold
  prefill by 12–16% and ran out of GPU memory at 128K, for no decode gain; those rows stay in the raw data as
  `mtplx-upstream-tuned`.
- **Follow-up turns:** time to first token for turns 2 and 3 at 128K is **2.2 s** on the
  M1 build, 3.5 s on upstream, 4.2 s on upstream + env, 6.2 s on Splash (it re-reads ~300
  tokens), 29 min on TensorFold (below) and 1.6 min / 7.9 s on oMLX. oMLX keeps its
  prefix cache on SSD in 4,096-token blocks and, for this model family, can save a prompt
  only up to the last block boundary it crossed, so the next turn re-reads the rest (up
  to ~4K tokens).
- **First turn (cold prefill):** up to 64K the engines are close (64K: 9.7–11.9 min, TensorFold
  slowest). The M1 build is 1–5% slower than upstream at 2K–32K in its
  cleaner round (larger command buffers and the prefill evaluated every four layers); its
  64K median of 10.4 min mixes 11.1 min in a round with other load on the machine and 9.7
  min in the other. At 128K attention dominates: 24.6 min (M1 build), 26.3 (oMLX), 28.1
  (upstream), 28.5 (Splash), 29.2 (TensorFold), 29.5 (upstream + env).
- **Whole conversation at 128K:** 25.3 min (M1 build), 29.4 (Splash), 29.6 (oMLX), 30.6
  (upstream), 32.2 (upstream + env), 89.6 (TensorFold).
- **Memory:** Splash uses the least, 22–26 GB wired at every length. The M1 build peaks at
  30–46 GB, upstream at 34–52 GB, upstream + env at 30–53 GB, oMLX at 27–49 GB (peak
  system-wide wired memory; Splash maps its weights differently, so compare with care).
- **TensorFold 0.3.5.1 (bf16 KV):** 13–18% behind the M1 build at 2K–8K, a third of it at
  128K. At 128K its default prompt cache (an eighth of RAM, 8 GiB) cannot hold the
  conversation, so turns 2 and 3 re-read all 131K tokens and the conversation takes 89.6
  min; a larger `--prompt-cache-gib` may avoid that, but every engine runs at its defaults
  here. It accepts a draft token only when it equals the target's own sample (exact
  decoding), which works at temperature 0 but accepts fewer tokens when sampling.
- **256K (two engines, one run each):** the M1 build (m1.1) read the prompt in 67.9 min
  against 87.3 on Splash, decoded 14.5 against 12.9 tok/s and started follow-up turns in
  3.6 s against 9.3–10.1 s; Splash used 30.4 GB against 51.8 GB
  ([details](#256k-spot-check-two-engines)).
- **Not measured:** GPU temperature, power and energy per token. MTPLX engines have been
  reported to run hotter than Splash on M1 during prefill; this benchmark does not check
  that.
- **Output check:** every engine quoted the needle line correctly in turn 3 in every
  completed run.

## Decode speed

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-dark.svg?v=fair0929">
  <img alt="Decode tok/s vs prompt length, six engines" src="charts/decode-light.svg?v=fair0929">
</picture>

## Follow-up turns

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/ttft-followup-dark.svg?v=fair0929">
  <img alt="Time to first token for turns 2-3, log scale" src="charts/ttft-followup-light.svg?v=fair0929">
</picture>

## The whole conversation

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/conversation-dark.svg?v=fair0929">
  <img alt="Three turns end to end per engine and prompt length" src="charts/conversation-light.svg?v=fair0929">
</picture>

## Memory

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/memory-dark.svg?v=fair0929">
  <img alt="Peak wired memory vs prompt length" src="charts/memory-light.svg?v=fair0929">
</picture>

## Tables

Median of 2 rounds per cell for 2K–64K; 128K is a single run.
`needle` counts the runs whose turn 3 quoted the needle line correctly. For Splash, the
2K fp16 cells use its BF16 KV (`--kv-format bf16`). `cached` is the engine-reported
`usage.prompt_tokens_details.cached_tokens`. Wired memory is system-wide.

| context | engine | T1 TTFT (cold) | T2 TTFT | T3 TTFT | decode T1 / T2 / T3 (tok/s) | 3-turn total | cached T2 / T3 | peak wired | needle |
|---|---|---|---|---|---|---|---|---|---|
| 2K | MTPLX M1 build | 12.4 s | 0.62 s | 0.61 s | 31.4 / 32.0 / 31.8 | 35.2 s | 1,936 / 2,243 | 30.4 GB | 2/2 |
| 2K | MTPLX upstream | 11.4 s | 0.63 s | 0.64 s | 25.0 / 22.8 / 23.1 | 41.8 s | 1,936 / 2,243 | 34.0 GB | 2/2 |
| 2K | MTPLX upstream + env | 11.4 s | 0.64 s | 0.63 s | 24.5 / 31.6 / 25.4 | 37.5 s | 1,928 / 2,234 | 29.8 GB | 2/2 |
| 2K | oMLX | 11.4 s | 13.5 s | 15.4 s | 20.4 / 23.5 / 21.0 | 1.2 min | 0 / 0 | 26.9 GB | 2/2 |
| 2K | Splash 1.1.0-m1 | 12.0 s | 2.56 s | 2.76 s | 31.3 / 39.2 / 28.8 | 37.8 s | 1,664 / 1,952 | 21.9 GB | 2/2 |
| 2K | TensorFold (bf16 KV) | 15.7 s | 3.47 s | 3.48 s | 26.1 / 28.9 / 28.0 | 47.0 s | 1,673 / 1,980 | 22.9 GB | 2/2 |
| 8K | MTPLX M1 build | 54.6 s | 0.69 s | 0.68 s | 30.9 / 33.0 / 29.7 | 1.3 min | 8,089 / 8,396 | 30.9 GB | 2/2 |
| 8K | MTPLX upstream | 53.2 s | 0.74 s | 0.77 s | 22.9 / 23.0 / 21.5 | 1.4 min | 8,089 / 8,396 | 35.5 GB | 2/2 |
| 8K | MTPLX upstream + env | 55.5 s | 0.78 s | 0.79 s | 25.1 / 27.2 / 25.2 | 1.4 min | 8,089 / 8,396 | 31.3 GB | 2/2 |
| 8K | oMLX | 56.4 s | 30.0 s | 2.69 s | 20.2 / 19.8 / 18.3 | 2.1 min | 4,096 / 8,192 | 33.2 GB | 2/2 |
| 8K | Splash 1.1.0-m1 | 56.8 s | 2.91 s | 2.79 s | 31.1 / 37.4 / 29.1 | 1.4 min | 7,808 / 8,128 | 22.1 GB | 2/2 |
| 8K | TensorFold (bf16 KV) | 1.2 min | 3.68 s | 3.69 s | 24.6 / 26.0 / 26.0 | 1.8 min | 7,826 / 8,132 | 22.6 GB | 2/2 |
| 32K | MTPLX M1 build | 4.2 min | 0.96 s | 0.98 s | 27.4 / 26.8 / 26.6 | 4.7 min | 32,665 / 32,972 | 33.8 GB | 2/2 |
| 32K | MTPLX upstream | 4.0 min | 1.32 s | 1.34 s | 17.3 / 12.7 / 12.0 | 5.0 min | 32,665 / 32,972 | 42.5 GB | 2/2 |
| 32K | MTPLX upstream + env | 4.0 min | 1.31 s | 1.34 s | 19.8 / 14.5 / 11.6 | 4.9 min | 32,665 / 32,972 | 38.1 GB | 2/2 |
| 32K | oMLX | 4.4 min | 44.6 s | 3.45 s | 14.4 / 14.0 / 14.4 | 6.0 min | 28,672 / 32,768 | 37.0 GB | 2/2 |
| 32K | Splash 1.1.0-m1 | 4.5 min | 3.57 s | 3.57 s | 23.3 / 29.1 / 28.7 | 5.1 min | 32,384 / 32,704 | 22.9 GB | 2/2 |
| 32K | TensorFold (bf16 KV) | 5.3 min | 4.31 s | 4.31 s | 15.4 / 18.9 / 16.5 | 6.1 min | 32,402 / 32,709 | 27.1 GB | 2/2 |
| 64K | MTPLX M1 build | 10.4 min | 1.37 s | 1.39 s | 23.8 / 25.3 / 23.8 | 10.9 min | 65,419 / 65,726 | 36.3 GB | 2/2 |
| 64K | MTPLX upstream | 9.9 min | 2.07 s | 2.09 s | 8.8 / 8.8 / 8.4 | 11.3 min | 65,419 / 65,726 | 48.6 GB | 2/2 |
| 64K | MTPLX upstream + env | 9.7 min | 2.09 s | 2.12 s | 8.2 / 6.9 / 8.2 | 11.3 min | 65,419 / 65,726 | 44.5 GB | 2/2 |
| 64K | oMLX | 10.2 min | 1.0 min | 4.76 s | 10.9 / 10.9 / 10.4 | 12.4 min | 61,440 / 65,536 | 39.6 GB | 2/2 |
| 64K | Splash 1.1.0-m1 | 10.9 min | 4.28 s | 4.90 s | 23.2 / 29.9 / 19.1 | 11.5 min | 65,152 / 65,440 | 24.1 GB | 2/2 |
| 64K | TensorFold (bf16 KV) | 11.9 min | 5.25 s | 5.19 s | 10.8 / 13.1 / 10.0 | 13.1 min | 65,156 / 65,463 | 33.2 GB | 2/2 |
| 128K | MTPLX M1 build | 24.6 min | 2.15 s | 2.15 s | 19.0 / 20.9 / 19.0 | 25.3 min | 130,969 / 131,276 | 46.1 GB | 1/1 |
| 128K | MTPLX upstream | 28.1 min | 3.53 s | 3.50 s | 4.7 / 5.1 / 5.0 | 30.6 min | 130,969 / 131,276 | 52.4 GB | 1/1 |
| 128K | MTPLX upstream + env | 29.5 min | 4.19 s | 4.16 s | 4.4 / 4.9 / 4.5 | 32.2 min | 130,969 / 131,276 | 52.7 GB | 1/1 |
| 128K | oMLX | 26.3 min | 1.6 min | 7.92 s | 7.6 / 7.8 / 6.9 | 29.6 min | 126,976 / 131,072 | 48.5 GB | 1/1 |
| 128K | Splash 1.1.0-m1 | 28.5 min | 6.27 s | 6.20 s | 15.9 / 22.8 / 17.4 | 29.4 min | 130,688 / 131,008 | 26.2 GB | 1/1 |
| 128K | TensorFold (bf16 KV) | 29.2 min | 29.3 min | 29.4 min | 6.3 / 6.5 / 6.3 | 89.6 min | 0 / 0 | 37.2 GB | 1/1 |

**2K: fp16 KV vs 8-bit KV** (decode tok/s, mean of turns 1–3)

| engine | fp16 KV | 8-bit KV | change |
|---|---|---|---|
| MTPLX M1 build | 29.2 | 31.8 | +8.6% |
| MTPLX upstream | 24.8 | 23.6 | -5.0% |
| MTPLX upstream + env | 28.1 | 27.2 | -3.3% |
| oMLX | 23.1 | 21.6 | -6.4% |
| Splash 1.1.0-m1 | 33.5 | 33.1 | -1.3% |

Raw rows: [`results/2026-09-m1max-64gb/rows.jsonl`](results/2026-09-m1max-64gb/rows.jsonl) ·
per-turn CSV: [`summary.csv`](results/2026-09-m1max-64gb/summary.csv) ·
environment: [`environment.json`](results/2026-09-m1max-64gb/environment.json) ·
generated text of every turn: `results/2026-09-m1max-64gb/cells/*/`.

## 256K spot check (two engines)

The two engines that lead at 128K, one run each at 258,048 context tokens (the prompt
reaches 258K tokens by turn 3, inside both engines' 262,144-token window), on
2026-09-29: Splash 1.1.0-m1 first, then the MTPLX M1 build at **v2.12.0-m1.1** (see
the note below the table), so the M1 build ran on the warmer machine. The other engines
were not run: at 128K they already decode at 4.6–7.4 tok/s, and a 256K first turn would
take well over an hour each.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/spot256k-dark.svg?v=spot256">
  <img alt="256K spot check: first-turn prefill, follow-up TTFT, decode and peak memory for Splash 1.1.0-m1 and the MTPLX M1 build m1.1" src="charts/spot256k-light.svg?v=spot256">
</picture>

| | Splash 1.1.0-m1 | MTPLX M1 build m1.1 |
|---|---|---|
| T1 TTFT (cold) | 87.3 min | **67.9 min** |
| T2 / T3 TTFT | 9.3 s / 10.1 s | **3.6 s / 3.6 s** |
| decode T1 / T2 / T3 (tok/s) | 11.4 / 14.9 / 12.5 | 14.5 / 14.9 / 14.2 |
| decode, mean of turns | 12.9 | **14.5** |
| 3-turn total | 88.5 min | **68.9 min** |
| peak wired | **30.4 GB** | 51.8 GB |
| needle | 1/1 | 1/1 |

- Single runs. Each engine's 2K canary before the run: Splash 29.2 tok/s (3.7% below its
  best of the day), the M1 build 30.0 tok/s (2% below its best m1.1 canary), so the machine
  was slightly slower for Splash.
- **v2.12.0-m1.1** (released 2026-09-29) turns off the larger Metal command buffers
  that v2.12.0-m1 set on M1: they gave decode +6–7% but slowed cold prefill by 2–5% on an
  idle machine and 13–21% on a busy or power-limited one. m1.1 reads prompts at upstream
  speed (8K: 54.5 s against 54.3 s upstream and 67.1 s on m1, measured one after
  another) and decodes 6–8% below m1, with identical text. The 2K–128K tables above are
  v2.12.0-m1.

## Sampled decoding (temperature 1.0)

Greedy decoding at 256 tokens is a matched workload, not how the model is normally run.
The same conversations at 2K, 32K and 64K were repeated at the model's own sampling
settings (`generation_config.json`: temperature 1.0, top-p 0.95, top-k 20), up to 512
tokens per turn, 8-bit KV, two rounds with the order reversed (2026-09-29; Splash in the
same session). Sampled text differs between runs, so turns 2 and 3 see different
prompts; acceptance is the engine's own count (MTPLX: accepted / drafted MTP tokens,
Splash: accepted / drafted DFlash2 tokens, 7 per pass) and is not comparable across the
two methods.

| context | engine | T1 TTFT (cold) | decode T1 / T2 / T3 (tok/s) | decode mean | draft acceptance T1 / T2 / T3 | tokens T1 / T2 / T3 | needle |
|---|---|---|---|---|---|---|---|
| 2K | MTPLX M1 build | 11.9 s | 30.4 / 34.8 / 31.6 | **32.3** | 60% / 73% / 63% | 280 / 342 / 182 | 2/2 |
| 2K | MTPLX upstream | 11.4 s | 23.1 / 25.3 / 21.2 | **23.2** | 58% / 77% / 59% | 314 / 413 / 200 | 2/2 |
| 2K | MTPLX upstream + env | 11.6 s | 24.8 / 27.0 / 27.3 | **26.4** | 56% / 64% / 66% | 294 / 328 / 188 | 2/2 |
| 2K | Splash 1.1.0-m1 | 12.0 s | 29.8 / 36.1 / 32.2 | **32.7** | 31% / 41% / 35% | 294 / 398 / 178 | 2/2 |
| 32K | MTPLX M1 build | 4.2 min | 27.0 / 29.0 / 26.1 | **27.3** | 61% / 69% / 60% | 310 / 398 / 177 | 2/2 |
| 32K | MTPLX upstream | 4.0 min | 17.1 / 12.9 / 12.7 | **14.2** | 56% / 66% / 63% | 276 / 322 / 202 | 2/2 |
| 32K | MTPLX upstream + env | 4.1 min | 17.5 / 12.7 / 12.4 | **14.2** | 53% / 61% / 62% | 292 / 321 / 172 | 2/2 |
| 32K | Splash 1.1.0-m1 | 4.5 min | 22.3 / 30.3 / 24.9 | **25.8** | 28% / 41% / 33% | 382 / 382 / 194 | 2/2 |
| 64K | MTPLX M1 build | 9.8 min | 23.2 / 25.9 / 23.0 | **24.1** | 57% / 68% / 58% | 321 / 287 / 183 | 2/2 |
| 64K | MTPLX upstream | 9.7 min | 8.3 / 9.0 / 9.3 | **8.9** | 55% / 70% / 60% | 322 / 330 / 208 | 2/2 |
| 64K | MTPLX upstream + env | 10.2 min | 7.6 / 8.8 / 8.1 | **8.2** | 56% / 69% / 61% | 295 / 298 / 204 | 2/2 |
| 64K | Splash 1.1.0-m1 | 10.8 min | 20.8 / 24.1 / 20.9 | **22.0** | 30% / 37% / 31% | 303 / 384 / 172 | 2/2 |

- **The M1 build and Splash are even at 2K (32.3 against 32.7 tok/s); from 32K the M1
  build is ahead (+6% at 32K, +10% at 64K),** where greedy decoding had them tied.
- **Sampling barely moves the MTPLX engines** against their greedy speed (M1 build 32.3 vs
  31.8 at 2K, 24.1 vs 24.3 at 64K); Splash is 1% (2K), 5% (32K) and 9% (64K) slower than
  its greedy speed.
- **Upstream + env** is 14% faster than upstream at 2K, the same at 32K and 8% slower at
  64K, as with greedy decoding.
- The model ends turns on its own, so each engine generated a different number of tokens
  (turn 1: 280–382); every engine quoted the needle line in every run.

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
  can follow further than the 3-token MTP draft; the MTPLX M1 build commits 2.8–3.0 tokens per
  round on turn 2 and 2.7–2.9 on turns 1 and 3, with a ceiling of 4 (verify plus draft time
  per round, 2026-09-29 run).
- **Splash verifies twice as many positions per pass at a similar cost at short context:**
  105 ms per pass at 2K for 8 positions and a 7-token draft, against 87–90 ms per round on
  the M1 build for 4 positions and a 3-token MTP draft.
- **The pass cost grows faster with the prompt:** from 105 ms (2K) to 189–197 ms (128K)
  on Splash, against 87–90 to 137–140 ms per round on the M1 build. Each pass attends over the
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
- The three MTPLX columns ran together on 2026-09-29 (none needed a re-run). 128K ran once,
  in the order upstream → upstream + env (with command buffers, out of memory) → M1 build,
  which puts the M1 build on the warmest machine; upstream + env without command buffers
  ran its own 128K afterwards. Their 128K canaries were within 2.4% of each engine's best
  2K–64K canary. oMLX, Splash and TensorFold ran in earlier sessions with the same plans
  (details and canaries in [METHODOLOGY.md](METHODOLOGY.md)).
- Round-to-round decode difference in the 2026-09-29 run: median 0.7%. The largest is one
  turn of upstream + env at 64K (5.2 against 8.7 tok/s on byte-identical text; its verify
  time was 47 s against 27 s in the other round), which pulls that cell's turn-2 median
  to 6.9. All three MTPLX columns produced byte-identical text in both rounds in every
  turn.

## Reproduce

```bash
cp config/local.example.json config/local.json   # set your paths
python3 scripts/prepare_omlx.py                   # oMLX model view + base paths
./run_day.sh                                      # 2K-64K, two rounds (~3 h)
./run_night.sh                                    # 128K, one round (~3 h)
./run_fair.sh                                     # the three MTPLX columns + sampled scenario
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
