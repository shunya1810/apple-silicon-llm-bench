| context | engine | T1 TTFT (cold) | T2 TTFT | T3 TTFT | decode T1 / T2 / T3 (tok/s) | 3-turn total | cached T2 / T3 | peak wired | needle |
|---|---|---|---|---|---|---|---|---|---|
| 2K | MTPLX fork | 11.4 s | 0.62 s | 0.61 s | 28.2 / 26.1 / 26.2 | 38.2 s | 1,936 / 2,243 | 32.8 GB | 2/2 |
| 2K | MTPLX upstream | 11.5 s | 0.64 s | 0.64 s | 25.1 / 22.9 / 23.2 | 41.8 s | 1,936 / 2,243 | 34.0 GB | 2/2 |
| 2K | oMLX | 11.4 s | 13.5 s | 15.4 s | 20.4 / 23.5 / 21.0 | 1.2 min | 0 / 0 | 26.9 GB | 2/2 |
| 8K | MTPLX fork | 53.8 s | 0.68 s | 0.71 s | 26.5 / 26.7 / 25.1 | 1.4 min | 8,089 / 8,396 | 34.0 GB | 2/2 |
| 8K | MTPLX upstream | 54.2 s | 0.76 s | 0.77 s | 23.5 / 23.2 / 21.7 | 1.4 min | 8,089 / 8,396 | 35.5 GB | 2/2 |
| 8K | oMLX | 56.4 s | 30.0 s | 2.69 s | 20.2 / 19.8 / 18.3 | 2.1 min | 4,096 / 8,192 | 33.2 GB | 2/2 |
| 32K | MTPLX fork | 4.1 min | 0.97 s | 0.98 s | 22.4 / 21.3 / 20.9 | 4.7 min | 32,665 / 32,972 | 38.5 GB | 2/2 |
| 32K | MTPLX upstream | 4.1 min | 1.38 s | 1.37 s | 17.1 / 12.5 / 12.0 | 5.0 min | 32,665 / 32,972 | 40.8 GB | 2/2 |
| 32K | oMLX | 4.4 min | 44.6 s | 3.45 s | 14.4 / 14.0 / 14.4 | 6.0 min | 28,672 / 32,768 | 37.0 GB | 2/2 |
| 64K | MTPLX fork | 9.7 min | 1.37 s | 1.37 s | 21.7 / 21.1 / 19.8 | 10.3 min | 65,419 / 65,726 | 42.7 GB | 2/2 |
| 64K | MTPLX upstream | 9.9 min | 2.07 s | 2.12 s | 8.8 / 8.7 / 8.4 | 11.3 min | 65,419 / 65,726 | 48.4 GB | 2/2 |
| 64K | oMLX | 10.2 min | 1.0 min | 4.76 s | 10.9 / 10.9 / 10.4 | 12.4 min | 61,440 / 65,536 | 39.6 GB | 2/2 |

**2K: fp16 KV vs 8-bit KV** (decode tok/s, mean of turns 1–3)

| engine | fp16 KV | 8-bit KV | change |
|---|---|---|---|
| MTPLX fork | 24.9 | 26.8 | +7.6% |
| MTPLX upstream | 24.8 | 23.7 | -4.5% |
| oMLX | 23.1 | 21.6 | -6.4% |
