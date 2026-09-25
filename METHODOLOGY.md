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

- **Same model files for all engines.** oMLX reads the MTPLX checkpoint through a
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
- **Every cell starts the engine fresh**, with its on-disk cache wiped, then sends a
  short warmup request (excluded) and a thermal canary.
- **Order and heat.** 2K–64K ran twice, with the engine order reversed in the second
  round; the reported value is the median of the rounds. Before each conversation a
  fixed ~2K "canary" request measures decode speed; if it was more than 3% below
  that engine's best canary in the run, the machine rested 5 minutes and the cell
  was re-run once (the last attempt is used). 128K is planned as one round in the order
  upstream → oMLX → fork, which puts the fork last, on the warmest machine.

## Known limitations

- One machine (M1 Max, 64 GB), one model, one synthetic workload.
- Greedy decoding produces different text on different engines, so turn 2 and 3
  prompts differ between engines (by at most 6 tokens in this run).
- The oMLX run uses its default memory guard and scheduler settings; oMLX has other
  speed features (DFlash speculative decoding, ANE prefill, hot RAM cache) that are
  off here because they are off by default or need a different draft model.
