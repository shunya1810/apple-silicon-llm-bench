## Latest: newest releases, thinking on, same weights (2026-10-18/19)

All four engines at their newest release on 2026-10-18, on the same MacBook Pro (M1 Max,
64 GB), with **thinking on** and the **same body weights** for every engine. Each cell is a
3-turn conversation over a synthetic telemetry prompt (the same as the sections below), up
to 4,096 generated tokens per turn, thinking tokens included.

| engine | version | weights |
|---|---|---|
| **MTPLX M1 fork** | [`v2.12.2-m1.9`](https://github.com/shunya1810/MTPLX/releases/tag/v2.12.2-m1.9) (code `9c73c4ce`) | MTPLX pack below, MTP depth from the pack (27B: 3; A3B: 1), `--profile turbo`, q8 KV |
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
- **The M1 fork's turn 3 at 27B 32K / 128K / 246K** re-read part (3,061 tokens) or all
  (246K: 248,809 tokens, 51 min) of the prompt: turn 2's thinking hit the 4,096-token cap,
  and the save of turn 2's state had not finished when turn 3 arrived. At 246K the re-read
  also left the MTP drafts far less accurate for that turn (2.2 instead of 3.5 tokens per
  round; decode 12.8 tok/s). Known issue in the fork, fix in progress; the A3B does not show
  it.

### Qwen3.8-27B (dense)

{TABLES_27B}

### Qwen3.6-35B-A3B (MoE)

{TABLES_A3B}

### What the numbers say

{TAKEAWAYS}

Raw rows: [`results/2026-10-m1max-64gb-b2/`](results/2026-10-m1max-64gb-b2/), summary
`python scripts/summarize_b2.py results/2026-10-m1max-64gb-b2 --readme 27b` (or `a3b`),
plans `plans/b2-27b.json` / `plans/b2-a3b.json`, run script `run_b2.sh`.
