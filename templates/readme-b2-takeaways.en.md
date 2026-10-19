- **Short prompts: Splash.** At 2K Splash is the fastest engine on both models (27B decode
  57 vs 41 tok/s on turns 2–3, A3B 122–127 vs 111–116), and its DFlash2 draft commits 5–6.6
  tokens per verify round where MTP depth 3 commits at most 4 (the fork's 27B rounds are
  25–40% shorter, but carry 3.5 tokens). Splash also uses the least memory at every length
  (27B 24–33 GB, A3B 26–29 GB).
- **Long prompts: the M1 fork.** From 8K on, the fork reads prompts fastest on both models,
  and the gap grows with length: 27B 1.11× Splash at 8K, 1.28× at 128K; A3B 1.07× at 8K,
  1.51× at 128K, 1.57× at 246K. Its decode holds up deeper too: the 27B is level with Splash
  at 64K and ahead from 128K (turns 2–3: 27 vs 25 and 17.5 tok/s; 246K: 21 vs 14), the A3B level
  at 8K and ahead from 32K (32K: 97–99 vs 89–92 tok/s; 128K: 61–64 vs 49–53; 246K: 47–49 vs 34–36). Splash's 27B decode drops
  on later turns from 32K on (32K turn 3: 30 tok/s; 64K: 23; 128K: 17.5).
- **Against MTPLX upstream 2.12.2** (same weights, same flags), the fork reads prompts
  1.1–1.9× faster and decodes 1.3–4.4× faster (27B 128K: 25–28 vs 6 tok/s; A3B 246K: 47–49 vs
  12–13), with less memory from 32K on. Upstream could not finish the 27B at 246K.
- **oMLX 0.7.1rc1** sits between upstream and the fork on decode, but on the 27B it re-reads
  the whole prompt on every turn (turn 2 starts after 13 s at 2K, 43 s at 64K, 24 min at
  128K) and runs fp16 KV on M1. On the A3B it reuses the prompt up to 128K (turn 2: 2–20 s)
  and re-reads it at 246K.
- **oMLX's own oQ4e quantization** (extra row, different weights) runs about as fast as
  oMLX on the shared weights for the 27B (decode within ±10%), and slower for the A3B
  (prefill 0.5–0.75×, decode 0.8–1.0×): its 5–6-bit layers cost bandwidth. Its quality is
  not measured here.
