# Methodology

## What is measured

A **3-turn conversation** per engine and prompt length, over each engine's own
OpenAI-compatible HTTP server with streaming:

| turn | request |
|---|---|
| 1 | long synthetic context (telemetry records, one "needle" line at 50% depth) + question 1 — the whole prompt is cold |
| 2 | turn-1 history (the engine's own answer) + question 2 (~300 new tokens) |
| 3 | turn-2 history + question 3, which asks to quote the needle line |

Each turn generates at most 256 tokens. Prompt lengths: 2K, 8K, 32K, 64K and 128K
tokens (the context is trimmed to the target with the model's tokenizer).

All requests use `temperature 0`, `top_p 1`, thinking off
(`chat_template_kwargs.enable_thinking=false`). These are **speeds for a matched,
deterministic workload**, meant for comparing engines, not a prediction of speed with
sampling on; MTP acceptance, and so decode speed, depends on the content and the
sampler.

## Metrics

All timings are taken on the client from the stream, the same way for every engine.

| metric | definition |
|---|---|
| TTFT | request sent → first streamed content token |
| decode tok/s | (completion tokens − 1) / (last token time − first token time) |
| turn time | request sent → stream closed |
| 3-turn total | sum of the three turn times |
| cached tokens | `usage.prompt_tokens_details.cached_tokens` as reported by the engine |
| peak wired | max system-wide wired memory (`vm_stat`, 1 s samples) during a turn |
| peak RSS | max RSS of the engine's process tree during a turn |
| needle | whether turn 3's answer contains the needle's code (a sanity check that fast paths did not break the output) |
| MTP acceptance | engine-reported accepted / drafted tokens (MTPLX `/metrics`, oMLX server log), kept in the raw rows; the engines count drafted tokens differently (oMLX can stop a draft early), so the rates are not compared across engines |

Client-side TTFT and decode speed agree with MTPLX's own `/metrics` within 1%. oMLX
logs only whole-request speed (prefill included), so for oMLX the client numbers are
the only per-phase measurement.

## Fairness

- **Same model files for the MTPLX and oMLX engines** (Splash needs its own package; see
  below). oMLX reads the MTPLX checkpoint through a
  symlinked, text-only view (`scripts/prepare_omlx.py`): the MTP tensors are exposed
  under a file name mlx-lm loads and the MTP head's per-module quantization (4-bit,
  group 64) is written into `config.json`. No weight is converted or copied.
- **Same MTP depth** (3 draft tokens).
- **8-bit KV is each engine's own implementation**: MTPLX uses affine q8 paged KV;
  oMLX uses TurboQuant 8-bit (its default keeps the last KV layer unquantized). The
  qwen3.5-style model has KV only in its 16 full-attention layers; the other 48
  layers are linear attention (GDN) with a fixed-size state.
- **Engine defaults otherwise**, including the prefix cache: MTPLX keeps session
  state in RAM (SSD session cache off, the app default); oMLX stores its paged prefix
  cache on SSD in 4,096-token blocks. For this model family oMLX can only store a
  prefix at a block boundary (the GDN state has to be snapshotted there), so a
  follow-up turn re-reads everything after the last full block.
- **The two MTPLX engines get the same command line** (`engines/mtplx-*.json`): same
  model, profile, depth, KV mode, context window and session-cache setting. Only
  `PYTHONPATH` differs, which picks the code (the M1 build or upstream `1de2b1c`), and
  both run in the MTPLX app's Python runtime (MLX 0.32.2).
- **The M1 build changes defaults on M1, and upstream exposes some of them.** Part of
  the M1 build's lead is code upstream does not have (the M1 attention kernels, the
  session-bank fixes, prefill evaluated every four layers, the Japanese-aware FR-Spec
  list); part is defaults that upstream can reach with environment variables
  (context-copy off, fused projections, the FR-Spec draft head with upstream's
  64K code list, a 1 GiB MLX cache, two saved states per conversation, larger Metal
  command buffers). The `MTPLX upstream + env` column runs upstream with all of the
  latter (`engines/mtplx-upstream-tuned.json`), so the gap between it and the M1 build
  is the code alone.
- **Every cell starts the engine fresh**, with its on-disk cache wiped, then sends a
  short warmup request (excluded) and a thermal canary.
- **Order and heat.** 2K–64K ran twice, with the engine order reversed in the second
  round; the reported value is the median of the rounds. Before each conversation a
  fixed ~2K "canary" request measures decode speed; if it was more than 3% below
  that engine's best canary in the run, the machine rested 5 minutes and the cell
  was re-run once (the last attempt is used). 128K ran as one round in the order
  upstream → oMLX → M1 build, which puts the M1 build last, on the warmest machine; it was a
  separate run, so its canaries had no reference and no cell was re-run (they were
  1.7–4.8% below the engines' best canaries of the 2K–64K run).

## Re-run of the M1 build

- The M1 build column was re-measured on 2026-09-26 at `40b6113` (context-copy off by default
  on M1), with the same plans: 2K–64K in two rounds, then 128K once, running alone. The
  first measurement (`16751dc`) is kept in `rows.jsonl` under `mtplx-fork-16751dc`.
- It was re-measured again the same evening at `4fe8067` (M1 memory defaults: MLX buffer
  cache capped at 1 GiB, two saved states per conversation in the session bank, MMA
  prefill attention from the first chunk), with the same plans. The `40b6113` rows are
  kept under `mtplx-fork-40b6113`. Decode and cold prefill stayed within run-to-run noise;
  peak wired memory fell from 37.8 / 44.1 / 48.0 GB to 34.4 / 38.0 / 44.3 GB at
  32K / 64K / 128K.
- It was re-measured again on 2026-09-27 at `bba1b7b` (M1: small attention/GDN
  projections fused; the pruned FR-Spec draft head, with a Japanese-aware token list,
  now reaches the greedy draft path), with the same plans. The `4fe8067` rows are kept
  under `mtplx-fork-4fe8067`. Decode rose 1.7–6.3% (mean of the turns, 2K–128K) with
  the same acceptance, memory and cold prefill; turns 2–3 at 2K–8K generate different
  text.
- It was re-measured again on 2026-09-28 at `0cd2a73` (M1: up to 150 operations and
  1,000 MB per Metal command buffer, prefill evaluated every four layers), with the same
  plans. The `bba1b7b` rows are kept under `mtplx-fork-bba1b7b`. Decode rose 6.3–9.6%
  (mean of the turns, 2K–128K), cold prefill at 128K fell 4%, peak memory stayed the same
  or fell, and all 18 turns generated the same text as `bba1b7b`. A first attempt on
  2026-09-27 stopped when the Mac slept during the 64K prefill (Metal command-buffer error
  on wake); its rows were dropped and the whole plan was run again. In the re-run, the 8K
  cell of round 1 (just after a reboot) stayed 5% slow on its canary after one re-run and
  is kept, as the plan specifies.

## Same-session re-run (2026-09-29)

- The M1 build (at its release tag `v2.12.0-m1`, `6be2b28`), upstream `1de2b1c` and
  upstream + env ran in one session with the same plans (`plans/fair-day.json`: 2K–64K,
  two rounds, order reversed in the second; `plans/fair-night.json`: 128K once, in the
  order upstream → upstream + env → M1 build, which puts the M1 build on the warmest
  machine). Before, upstream had been measured on 2026-09-25/26 and the M1 build
  re-measured alone on later days.
- Earlier rows stay in `rows.jsonl`: `mtplx-upstream-0925` (upstream, first run) and
  `mtplx-fork-0cd2a73` (the M1 build's 2026-09-28 run), next to the older M1 build rounds.
- **Command buffers left out of upstream + env.** The first upstream + env arm
  (`mtplx-upstream-tuned`) also set `MLX_MAX_MB_PER_BUFFER=1000` and
  `MLX_MAX_OPS_PER_BUFFER=150`, the M1 build's command-buffer defaults. In the M1 build
  those come with a prefill that evaluates every four layers (`MTPLX_PREFILL_LAYER_EVAL_EVERY`,
  code upstream does not have) to hold its peak; without it, upstream's peak wired memory
  rose 12–15 GB, its 32K–64K cold prefill slowed 12–16%, and its 128K turn 1 failed with a
  Metal out-of-memory error, for the same decode (±2%). The published column
  (`mtplx-upstream-tuned-nobuf`) leaves both variables at MLX's default and ran the same
  plans afterwards (`plans/fair-nobuf*.json`); the first arm's rows stay in `rows.jsonl`.
- In the published sampled table, upstream + env is the same `-nobuf` arm
  (`plans/sampling-nobuf.json`, run after the others); the first arm's sampled rows stay
  in the raw data.
- The 128K canaries of this run were 0.2% (M1 build), 2.4% (upstream) below and 1.1%
  (upstream + env) above each engine's best 2K–64K canary; no cell needed a re-run.

## Sampled scenario

- Greedy decoding at 256 tokens is a matched workload, not how the model is normally
  run. The `sm-2k`, `sm-32k` and `sm-64k` scenarios repeat the same conversations at the
  model's own sampling settings (`generation_config.json`: temperature 1.0, top-p 0.95,
  top-k 20) with up to 512 tokens per turn, for the MTPLX engines and Splash
  1.1.0-m1 (`plans/sampling.json`, two rounds, order reversed). The canary stays greedy.
- Sampled text differs run to run, so turns 2 and 3 see different prompts and
  acceptance moves with the text; the tables report each turn's median over the
  rounds and the engine-reported draft acceptance. Results are in
  `results/2026-09-m1max-64gb-sampling/` (`scripts/summarize_sampling.py`).

## Splash M1 build

- Splash reads only its own packages (MLX affine 4-bit, group 64, or GGUF, prepared
  for its kernels, paired with a DFlash2 draft), so it runs `incoai/Qwen3.8-27B-Splash`
  (revision 9d27070, draft `incoai/Qwen3.8-27B-DFlash2` 015e795) instead of the MTPLX
  checkpoint. Speed differences therefore include the weight format and the speculative
  method, not only the engine.
- The published column is the 1.1.0-m1 release (2026-09-28), installed from the release
  tarball next to the older version after checking its SHA-256 and the engine and
  metallib hashes against the release notes. `--default-reasoning-effort none` plus
  `reasoning_effort: "none"` per request (Splash's switch); `--kv-format int8`
  (8-bit cells) or `bf16` (2K fp16 cells); `--max-context 256K`; everything else at its
  defaults (request timeout 10,000 s, bounded prefill).
- Its prefix cache is in memory only, so a fresh process per cell starts cold.
- It ran after the other engines, in two rounds of its own (2K–64K) and one 128K run.
  Per-turn draft and verify counts come from its `/status` counters, read before and
  after each request.
- Earlier Splash rows stay in `rows.jsonl`: `splash-m1-102` is the 1.0.2-m1 release
  (2026-09-26), which could not finish 128K (a Metal command-buffer error after 23
  minutes; a run under another GPU load, not counted; and a run on an idle machine cut
  off at that release's 30-minute request deadline); `splash-src` is a one-round source
  build of the port's head at the time (paperniuk/splash@5967821 plus one status
  counter). 1.1.0-m1 matched or beat both at every length and finished 128K, so only it
  is shown.

## TensorFold

- Installed with pip from the v0.3.5.1 tag into its own Python 3.12 environment
  (MLX 0.32.2). 0.3.5 and earlier stop at load on M1/M2 Max (a 512-thread threadgroup
  above the GPU's 448; TensorFold issue #48); 0.3.5.1 loads.
- It serves its own packages: `Vontra/Qwen3.8-27B-MLX-4bit` (revision 70ae7fa) and the
  `z-lab/Qwen3.8-27B-DFlash2` draft (revision 50307d4), which it quantizes to 4-bit at load.
- `--context 262144 --max-tokens 4096 --no-thinking --parallel 1 --snapshot-dir none`
  (no disk snapshots, so each cell starts cold); everything else at its defaults,
  including the in-memory prompt cache (an eighth of RAM, 8 GiB here). It has no 8-bit
  KV option, so its cells run bf16 KV and fill the 8-bit column, labelled "(bf16 KV)".
- Per-request rounds and accepted/drafted counts come from its server log line.
- It ran last, in two rounds of its own (2K–64K) and one 128K run.

## Known limitations

- One machine (M1 Max, 64 GB), one model, one synthetic workload (greedy, plus the
  sampled scenario above).
- Greedy decoding produces different text on different engines, so turn 2 and 3
  prompts differ between engines (by at most 6 tokens in this run).
- The oMLX run uses its default memory guard and scheduler settings; oMLX has other
  speed features (DFlash speculative decoding, ANE prefill, hot RAM cache) that are
  off here because they are off by default or need a different draft model.
