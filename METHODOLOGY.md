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
  cache on SSD in 4,096-token blocks. For this model family oMLX stores a prefix only
  where it snapshots the GDN state: in 0.6.4 at block boundaries, so a follow-up turn
  re-read everything after the last full block; 0.7.0 also snapshots the end of the
  first prompt, so turn 2 resumes there, while turn 3 still resumes at the last block
  boundary.
- **The two MTPLX engines get the same command line** (`engines/mtplx-*.json`): same
  model, profile, depth, KV mode, context window and session-cache setting. Only
  `PYTHONPATH` differs, which picks the code (the M1 fork or upstream `1de2b1c`), and
  both run in the MTPLX app's Python runtime (MLX 0.32.2).
- **The M1 fork changes defaults on M1, and upstream exposes some of them.** Part of
  the M1 fork's lead is code upstream does not have (the M1 attention kernels, the
  session-bank fixes, prefill evaluated every four layers, the Japanese-aware FR-Spec
  list); part is defaults that upstream can reach with environment variables
  (context-copy off, fused projections, the FR-Spec draft head with upstream's
  64K code list, a 1 GiB MLX cache, two saved states per conversation, larger Metal
  command buffers). The `MTPLX upstream + env` column runs upstream with all of the
  latter (`engines/mtplx-upstream-tuned.json`), so the gap between it and the M1 fork
  is the code alone.
- **Every cell starts the engine fresh**, with its on-disk cache wiped, then sends a
  short warmup request (excluded) and a thermal canary.
- **Order and heat.** 2K–64K ran twice, with the engine order reversed in the second
  round; the reported value is the median of the rounds. Before each conversation a
  fixed ~2K "canary" request measures decode speed; if it was more than 3% below
  that engine's best canary in the run, the machine rested 5 minutes and the cell
  was re-run once (the last attempt is used). 128K ran as one round in the order
  upstream → oMLX → M1 fork, which puts the M1 fork last, on the warmest machine; it was a
  separate run, so its canaries had no reference and no cell was re-run (they were
  1.7–4.8% below the engines' best canaries of the 2K–64K run).

## Re-run of the M1 fork

- The M1 fork column was re-measured on 2026-09-26 at `40b6113` (context-copy off by default
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

- The M1 fork (at its release tag `v2.12.0-m1`, `6be2b28`), upstream `1de2b1c` and
  upstream + env ran in one session with the same plans (`plans/fair-day.json`: 2K–64K,
  two rounds, order reversed in the second; `plans/fair-night.json`: 128K once, in the
  order upstream → upstream + env → M1 fork, which puts the M1 fork on the warmest
  machine). Before, upstream had been measured on 2026-09-25/26 and the M1 fork
  re-measured alone on later days.
- Earlier rows stay in `rows.jsonl`: `mtplx-upstream-0925` (upstream, first run) and
  `mtplx-fork-0cd2a73` (the M1 fork's 2026-09-28 run), next to the older M1 fork rounds.
- **Command buffers left out of upstream + env.** The first upstream + env arm
  (`mtplx-upstream-tuned`) also set `MLX_MAX_MB_PER_BUFFER=1000` and
  `MLX_MAX_OPS_PER_BUFFER=150`, the M1 fork's command-buffer defaults. In the M1 fork
  those come with a prefill that evaluates every four layers (`MTPLX_PREFILL_LAYER_EVAL_EVERY`,
  code upstream does not have) to hold its peak; without it, upstream's peak wired memory
  rose 12–15 GB, its 32K–64K cold prefill slowed 12–16%, and its 128K turn 1 failed with a
  Metal out-of-memory error, for the same decode (±2%). The published column
  (`mtplx-upstream-tuned-nobuf`) leaves both variables at MLX's default and ran the same
  plans afterwards (`plans/fair-nobuf*.json`); the first arm's rows stay in `rows.jsonl`.
- In the published sampled table, upstream + env is the same `-nobuf` arm
  (`plans/sampling-nobuf.json`, run after the others); the first arm's sampled rows stay
  in the raw data.
- The 128K canaries of this run were 0.2% (M1 fork), 2.4% (upstream) below and 1.1%
  (upstream + env) above each engine's best 2K–64K canary; no cell needed a re-run.

## Re-run of the M1 fork at v2.12.0-m1.2 (2026-09-30)

- The M1 fork's column is v2.12.0-m1.2 (larger Metal command buffers off by default on
  M1, plus an SSD prompt-cache fix this benchmark does not use), run alone with the same
  plans (`plans/fork-m12-*.json`, `run_fork_m12.sh`). Before the plans, a 2K probe cell had
  to show a canary of at least 29.9 tok/s (m1.2 on a cool, idle machine: 30.3–30.6); it
  read 30.6.
- A first attempt earlier that afternoon was stopped: canaries of 28.0–29.1 tok/s and a
  32K cold prefill 5% slower than the day before. Its rows are in
  `rows-fork-m12-aborted-20260930.jsonl`, not in `rows.jsonl`.
- The v2.12.0-m1 rows of 2026-09-29 (same session as upstream and upstream + env) stay as
  `mtplx-fork-m1`, in the main and the sampled results.

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

## Splash M1 fork

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

## Splash 1.3.0-m1.1 and the M1 fork v2.12.2-m1.1 / m1.2 (2026-10-13)

- Both re-measured together at their latest releases (`run_refresh_1013.sh`), interleaved,
  with the same plans as 2026-10-08 (2K–64K two rounds, 128K once, the sampled scenario,
  then 256K once each). Splash 1.3.0-m1.1 was installed beside 1.2.1-m1 (launcher
  `splash-m1-130`); the M1 fork ran from a worktree at tag `v2.12.2-m1.1` with its native
  extensions built by `scripts/build-m1-native-extensions.sh` (MLX 0.32.2). Earlier rows
  were renamed `splash-m1-121` and `mtplx-fork-m1-2122` (backup of the rows file:
  `rows.jsonl.bak-20261013`).
- Canaries: the fork 31.0–32.2 tok/s, Splash 22.7–23.8 (its 1.2.1-m1 read 28.7–29.6; the
  2K cells' decode stayed within 3%). Splash's round-1 64K cell and both sampled 64K
  cells read more than 3% below the reference and were re-run once (the re-runs are used).
- 256K: v2.12.2-m1.1 refused turn 1 with `insufficient_memory` (54.7 GiB projected against
  its 48 GiB limit); the row is kept. A diagnostic run with `--allow-swap`
  (`run_256k_allowswap.sh`, rows `mtplx-fork-allowswap`) finished without swap (swap use
  stayed at 0.9 GB, logged each minute) at a 53.2 GB peak. Two intermediate commits that
  were still refused at turn 1 or 2 are kept as `mtplx-fork-61af5233` and
  `mtplx-fork-b56e472d`. The published 256K row is commit `ca059bfb` = `v2.12.2-m1.2`'s code
  (`run_256k_m12b.sh`, renamed `mtplx-fork-m12`), run at its defaults.

## Splash 1.2.1-m1 and the M1 fork v2.12.2-m1 (2026-10-08)

- Both re-measured together at their latest releases (`run_refresh_1008.sh`), interleaved,
  with the same plans (2K–64K two rounds, 128K once, the sampled scenario, Splash 256K
  once). Splash 1.2.1-m1 was installed beside 1.1.0-m1 (launcher `splash-m1-121`); the
  M1 fork ran from a worktree at tag `v2.12.2-m1` with the MTPLX app's runtime (MLX 0.32.2).
  Earlier rows are kept as `splash-m1-110` and `mtplx-fork-m1.2`.
- 2K smoke runs first: fork canary 30.7, Splash 28.9 tok/s. In the run the fork's 8-bit
  canaries read 30.3–30.9 and Splash's 28.7–29.6; Splash's round-1 64K cell was 3.1% below
  its best and was re-run once (the re-run is used).
- At 128K v2.12.2-m1 refused turn 2 with `insufficient_memory` (its memory guard, from
  upstream 2.12.1/2.12.2, projected 0.5 GiB more than the 12.1 GiB the other apps left);
  turn 3 did not run. The row is kept. `scripts/summarize.py` fills the M1 fork's 128K
  cell from the m1.2 run (`FILL`) and marks it in the tables and charts; the fork's 256K
  plan was not run.

## oMLX 0.7.0 (2026-10-07)

- oMLX was re-measured at 0.7.0 (the 2026-09-30 release, installed beside 0.6.4) alone
  on 2026-10-07, with the same plans, model view and settings templates
  (`run_omlx070.sh`). The 0.6.4 rows are kept under `omlx-064`. A 2K probe of the M1
  build first read 31.1 tok/s on its canary (gate 29.9), so the machine was in its usual
  state after 5.8 days of uptime.
- 8-bit canaries stayed at 20.98–21.05 tok/s in every cell, including 128K and 256K;
  no cell was re-run. The 2K fp16-KV canary (separate reference) read 22.9.
- 0.7.0's rebuilt prefill memory guard is on by default (tier `balanced`). At 128K it
  paused the first turn once and read the rest in 1,024-token chunks (`Prefill
  throttled ... chunk 4096 -> 1024` in the server log); the 4,096 tokens read before the
  pause are reported as `cached_tokens` on that turn. TTFT is still the client's wall
  time from the request. 32K and 64K were not throttled.
- At 256K the guard rejected the first turn before reading it (HTTP 400: ~46.1 GB
  predicted peak against a 39.4 GB dynamic ceiling); that row is in `rows.jsonl` under
  `omlx` with the error. The 256K numbers come from one extra run with
  `memory_guard_tier: aggressive` (`omlx-aggressive`, `run_omlx070_256k_aggressive.sh`,
  templates `templates/omlx/*-q8-aggressive.json`), the only non-default oMLX setting in
  this benchmark.

## Known limitations

- One machine (M1 Max, 64 GB), one model, one synthetic workload (greedy, plus the
  sampled scenario above).
- Greedy decoding produces different text on different engines, so turn 2 and 3
  prompts differ between engines (by at most 6 tokens in this run).
- The oMLX runs use the default memory guard and scheduler settings (except the 256K
  run, tier `aggressive`); oMLX has other
  speed features (DFlash speculative decoding, ANE prefill, hot RAM cache) that are
  off here because they are off by default or need a different draft model.
