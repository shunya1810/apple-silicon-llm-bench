# Apple Silicon の LLM エンジン比較（長い文脈と複数ターン）

[English](README.md) | 日本語

Mac で長いプロンプトを読ませて3ターン会話するとき、推論エンジンによって速さはどれだけ変わるのか。
このリポジトリは、MacBook Pro（M1 Max、64 GB）の上で Qwen3.8-27B を複数の推論エンジンで動かし、同じ会話の速さを比べたベンチマークである。
どのエンジンも投機的デコードと 8-bit の KV キャッシュを使い（TensorFold は 8-bit の KV が無いので bf16）、各エンジンの OpenAI 互換のストリーミング API を同じ方法で計測した。

MTPLX の2つ（fork と upstream）と oMLX は、**同じモデルファイル**を MTP（深さ 3）で動かす。
Splash と TensorFold は専用のパッケージ（4bit、group size 64 の重みと DFlash2 の draft モデル）を読む。
そのため Splash と TensorFold の列は、エンジンの違いに加えて、量子化と投機的デコードの方式の違いも含む。

| エンジン | 版 | 備考 |
|---|---|---|
| **MTPLX fork** | [`shunya1810/MTPLX@0cd2a73`](https://github.com/shunya1810/MTPLX/tree/m1max-longctx) | M1 系向けの長い文脈の改善ブランチ（このベンチの作者が管理）。M1 で context-copy を止め（`40b6113`）、M1 のメモリの既定値を小さくし（`4fe8067`）、M1 で小さい射影の結合と draft head の縮小を既定にし（`bba1b7b`）、M1 で Metal の command buffer を大きくした（`0cd2a73`）後の版で、2026-09-26〜28 に測り直した（前の行は、生データに `mtplx-fork-16751dc`、`mtplx-fork-40b6113`、`mtplx-fork-4fe8067`、`mtplx-fork-bba1b7b` として残した） |
| **MTPLX upstream** | [`youssofal/MTPLX@1de2b1c`](https://github.com/youssofal/MTPLX/commit/1de2b1c049136ed117af0c6712baaadd81820b51) | 計測時点の `main`（fork の土台） |
| **oMLX** | [0.6.4](https://github.com/jundot/omlx/releases/tag/v0.6.4) | 計測時点の最新の正式版 |
| **Splash 1.1.0-m1** | [paperniuk/splash 1.1.0-m1](https://github.com/paperniuk/splash/releases/tag/1.1.0-m1) | [incoai/splash](https://github.com/incoai/splash) 1.1.0 を M1/M2 向けの kernel で動かすコミュニティ版。モデルは `incoai/Qwen3.8-27B-Splash`（4bit g64 と DFlash2 の draft）、KV は INT8。前に測った 1.0.2-m1 のリリースと、その時点の最新をビルドしたソース版を置き換えた（生データに `splash-m1-102`、`splash-src` として残した） |
| **TensorFold** | [0.3.5.1](https://github.com/ashhart/TensorFold/releases/tag/v0.3.5.1) | draft の受理を「本体のサンプルと一致したときだけ」にする Python/MLX のエンジン。モデルは `Vontra/Qwen3.8-27B-MLX-4bit`（4bit g64）と `z-lab/Qwen3.8-27B-DFlash2` の draft（読み込み時に 4bit）、KV は bf16（8-bit の設定は無い）。M1 Max で起動できる最初の版（2026-09-28） |

2K〜64K は各2回測り、中央値を載せた。128K は1回である。
Splash は、他の3つの後に別に2回測り、128K は1回測った。TensorFold は最後に同じ方法で測った。

## 主な結果（8-bit KV）

- **decode の速さ**：128K では MTPLX fork が **19.6 tok/s**、oMLX が 7.4、MTPLX upstream が 4.9 だった（3ターンの平均）。64K では 24.3、10.7、8.7 だった。
  fork と他の2つの差はプロンプトが長いほど開き、2K〜8K で +33〜56%、32K で 1.9〜2.0 倍、64K で 2.3〜2.8 倍、128K で 2.6〜4.0 倍になる。
- **2〜3ターン目の最初のトークンまでの時間（TTFT）**：128K では fork が **2.2 秒**、upstream が 3.5〜4.2 秒、oMLX が 1.6 分と 7.9 秒だった（64K では 1.4 秒、2.1 秒、1.0 分と 4.8 秒）。
  oMLX はプレフィックスキャッシュを 4,096 トークン単位で SSD に保存し、このモデル系統では、プロンプトが最後に越えたブロックの境界までしか保存できない。
  そのため次のターンでは、境界より後ろの部分（最大で約 4K トークン）を読み直す。
- **1ターン目（最初から読む prefill）**：64K までは、MTPLX と oMLX でほぼ同じ時間だった（64K で 9.8〜10.2 分、2K で 11.4〜11.7 秒）。
  prefill は同じ MLX の量子化行列積で律速されている（この GPU で、2K のとき約 7.9 TFLOPS）。
  128K では attention の占める割合が大きくなり、fork が 24.7 分、oMLX が 26.3 分、upstream が 30.5 分と差が出た。
- **128K の3ターン全体**：fork が 25.3 分、oMLX が 29.6 分、upstream が 33.0 分だった。
- **メモリ**：128K の wired メモリの最大値は、fork が 43.1 GB、oMLX が 48.5 GB、upstream が 52.7 GB だった（64K では 37.8、39.6、48.4 GB）。
  fork は `4fe8067` の既定値（MLX のバッファの cache の上限 1 GiB、1つの会話で保存する状態を2件、MMA の prefill attention を最初の chunk から使う）で、速さを変えずに 32K〜128K の最大値が 3.4〜6.4 GB 下がった（`40b6113` では 37.8 / 44.1 / 48.0 GB、`4fe8067` では 34.2 / 37.7 / 44.6 GB）。
- **fork の `bba1b7b` と `4fe8067` の比較**：decode が 2K で +5.4%、8K で +6.3%、32K で +3.3%、64K で +3.2%、128K で +1.7%。
  M1 の draft は 1ラウンドに 715 MB の draft head を3回読んでいて、頻度の高い語に絞った FR-Spec の head（コード用の 64K 語に日本語の語を加えた、語彙の 42%）がその経路に届いていなかった。あわせて、attention と GDN の小さい射影を1つにまとめた。
  受理率とメモリは同じ。2K〜8K の2〜3ターン目は、結合した経路で続きの prefill の丸めが変わるので、生成テキストが変わる。
- **fork の `0cd2a73` と `bba1b7b` の比較**：decode が 2K で +9.6%、8K で +6.3%、32K で +8.9%、64K で +7.5%、128K で +6.5%。
  1ラウンドが約 100 個の Metal command buffer に分かれていて、その継ぎ目ごとに GPU が止まっていた（1ラウンドの 13%）。M1 では 1つの command buffer に最大 150 個の処理と 1,000 MB を入れるようにし、prefill は 4 層ごとに評価して最大メモリが増えないようにした。
  128K の最初の prefill は 4% 速くなり、最大メモリは同じか少し下がった。18 ターンすべてで、生成テキストは `bba1b7b` と同じだった。
- **Splash 1.1.0-m1**
  - **decode**：2K / 8K / 32K / 64K / 128K で 33.1 / 32.5 / 27.1 / 24.1 / 18.7 tok/s（ターンの平均）。fork の 31.9 / 30.3 / 27.0 / 24.3 / 19.6 に対して、2K〜8K で 4〜7% 速く、32K〜64K で同じ、128K で 5% 遅い。
    2ターン目が最も速く（2K で 39.2、128K で 22.8 tok/s。fork は 32.2 と 20.8）、32K 以上の1ターン目と3ターン目は fork より遅い（128K で fork の 19.0 / 19.0 に対して 15.9 / 17.4）。
  - **prefill と TTFT**：最初から読む prefill は 32K 以上で遅い（64K で 10.9 分と 9.8 分、128K で 28.5 分と 24.7 分）。続きのターンでは約 300 トークンを読み直すので、TTFT は fork の 0.6〜2.2 秒に対して 2.6〜6.3 秒だった。128K の3ターン全体は、fork の 25.3 分に対して 29.4 分だった。
  - **メモリ**：最も少なく、どの長さでも wired が 22〜26 GB だった（fork は 30〜43 GB）。
- **Splash 1.1.0-m1 と前の Splash の比較**：1.0.2-m1 のリリースは、128K を最初から読む時間がリクエストの制限時間（30 分）を超えて完了できなかった。1.1.0 は制限時間の既定値が 10,000 秒になり、prefill を約 5 秒ずつの GPU のコマンドに分けるので、128K も最後まで完了した。
  decode は 1.0.2-m1 より 8K で +14%、32K で +12%、64K で +25%、前に測ったソース版より 64K で +9%、128K で +18% 速い（2K〜32K は同じ）。最初の prefill は 64K で 10.9 分（1.0.2-m1 は 12.8 分、ソース版は 12.3 分）、128K で 28.5 分（ソース版は 34.3 分）だった。前の行は生データに残した。
- **TensorFold 0.3.5.1（bf16 KV）**
  - **decode**：2K / 8K / 32K / 64K / 128K で 27.7 / 25.5 / 16.9 / 11.3 / 6.4 tok/s（ターンの平均）。2K〜8K で fork より 13〜16% 遅く、128K では fork の3分の1だった。
  - **prefill と TTFT**：最初から読む prefill は、64K までは5つの中で最も遅い（64K で 11.9 分）。続きのターンの TTFT は、64K までは 3.5〜5.3 秒だった。
    128K では、既定のプロンプトキャッシュ（RAM の8分の1、8 GiB）に会話が入らず、2ターン目と3ターン目で 131K トークンをすべて読み直した（それぞれ 29.3 分と 29.4 分）。3ターン全体では 89.6 分かかった。`--prompt-cache-gib` を大きくすれば避けられる可能性があるが、このベンチはすべてのエンジンを既定値で測っている。
  - **メモリ**：wired の最大値は 22.6〜37.2 GB だった。KV は bf16 で、8-bit の KV の設定は無い。
  - draft のトークンは、本体が自分でサンプルしたトークンと一致したときだけ受理する（draft の有無で出力が変わらない）。temperature 0 では問題ないが、サンプリングでは受理が減る。
- **出力の確認**：3ターン目では、文脈の中央に1行だけ埋め込んだ特別な行（needle）をそのまま引用させた。完了したすべての回で、すべてのエンジンが正しく答えた。

## decode の速さ

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-dark.svg?v=tensorfold">
  <img alt="プロンプト長ごとの decode tok/s（5エンジン）" src="charts/decode-light.svg?v=tensorfold">
</picture>

## 2〜3ターン目の TTFT

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/ttft-followup-dark.svg?v=tensorfold">
  <img alt="2〜3ターン目の TTFT（対数目盛り、5エンジン）" src="charts/ttft-followup-light.svg?v=tensorfold">
</picture>

## 3ターンの合計所要時間

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/conversation-dark.svg?v=tensorfold">
  <img alt="エンジンとプロンプト長ごとの3ターンの所要時間" src="charts/conversation-light.svg?v=tensorfold">
</picture>

## 最大メモリ使用量

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/memory-dark.svg?v=tensorfold">
  <img alt="プロンプト長ごとの wired メモリの最大値" src="charts/memory-light.svg?v=tensorfold">
</picture>

## 計測値の表

2K〜64K の各セルは2回の中央値、128K は1回の値である。
cache の列は、エンジンが返す `usage.prompt_tokens_details.cached_tokens` の値である。
wired メモリはシステム全体の値である。
needle の列は、3ターン目で needle を正しく引用した回数である。

| 文脈 | エンジン | T1 TTFT（cold） | T2 TTFT | T3 TTFT | decode T1 / T2 / T3（tok/s） | 3ターンの合計 | cache T2 / T3 | 最大 wired | needle |
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

**2K：fp16 KV と 8-bit KV**（decode tok/s、ターン1〜3の平均）

| エンジン | fp16 KV | 8-bit KV | 変化 |
|---|---|---|---|
| MTPLX fork | 29.9 | 31.9 | +6.6% |
| MTPLX upstream | 24.8 | 23.7 | -4.5% |
| oMLX | 23.1 | 21.6 | -6.4% |
| Splash 1.1.0-m1 | 33.5 | 33.1 | -1.3% |

2K の fp16 KV は、Splash では BF16 の KV（`--kv-format bf16`）を指す。

生データは [`results/2026-09-m1max-64gb/rows.jsonl`](results/2026-09-m1max-64gb/rows.jsonl)、ターンごとの CSV は [`summary.csv`](results/2026-09-m1max-64gb/summary.csv)、計測環境は [`environment.json`](results/2026-09-m1max-64gb/environment.json) にある。
各ターンの生成テキストは `results/2026-09-m1max-64gb/cells/*/` に置いた。

## Splash が2ターン目で速い理由

Splash は、エンジンが `/status` に出す draft と受理の数を、リクエストの前後で読める。
これで各ターンを verify の回数に分解できる。1回は、DFlash2 による 7 トークンの draft と、8 位置の verify からなる。
Splash 1.1.0-m1 の1回目、8-bit KV の値である。

| 文脈 | ターン | decode tok/s | verify の回数 | draft / 回 | 受理 / 回 | 確定トークン / 回 | 受理率 | ms / 回 |
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

- **draft がよく受理されるのは2ターン目だけ**：受理率は 37〜50% で、1ターン目と3ターン目は 26〜36% だった。そのため1回で確定するトークンが、2.8〜3.4 でなく 3.6〜4.5 になる。
  1回の時間はターンによらないので、この差がそのまま decode の速さの差になる。
  2ターン目の回答は、1ターン目で決まった比較の型を繰り返す。7 トークンの draft は、3トークンの MTP の draft より先まで当てられると考えられる。MTPLX fork は、2ターン目で1ラウンド 2.8〜3.0 トークン、1ターン目と3ターン目で 2.7〜2.9 トークンで、上限は 4 である。
- **短い文脈では、Splash は近い時間で2倍の位置を verify する**：2K で、8 位置の verify と 7 トークンの draft が1回 105 ms だった。fork は、4 位置の verify と3トークンの MTP の draft で1ラウンド 88〜91 ms である。
- **1回の時間は、プロンプトが長いほど大きく伸びる**：Splash は 2K の 105 ms から 128K の 189〜197 ms へ、fork は 88〜91 ms から 140〜142 ms へ伸びた。
  1回ごとに履歴全体への attention を、4 位置でなく 8 位置ぶん計算するためと考えられ、長い文脈の1ターン目と3ターン目で Splash が遅れる理由もこれだと推定している（kernel ごとの時間はまだ測っていない）。
- **前に測ったソース版との違い**：1.1.0-m1 は1回の時間が短い（64K で 163〜166 ms に対して 146〜157 ms、128K で 218〜225 ms に対して 189〜197 ms）。受理率はほぼ同じなので、長い文脈での速さの差はこの時間の差から来ている。

## 計測の方法

詳細は [METHODOLOGY.md](METHODOLOGY.md)（英語）にある。
要点は次のとおりである。

- **会話の内容**
  - 1ターン目：目標の長さの合成テレメトリのログ（50% の位置に needle の行を1行含む）と、質問を送る。
  - 2〜3ターン目：前の回答と新しい質問を足していく（合わせて約 300 トークン）。
  - 3ターン目では、needle の行の引用を求める。各ターンの生成は最大 256 トークンである。
- **生成の設定**：`temperature 0`、thinking off で測った。
  これは同じ決定的な課題の上でエンジンを比べるための速度である。サンプリングを有効にすると、投機的デコードの受理率が変わり、decode の速さも変わる。
- **モデルファイル**：MTPLX の2つと oMLX は同じモデルファイルを読む。
  oMLX は、MTPLX のチェックポイントを symlink で組んだテキスト専用のフォルダ経由で読む（[`scripts/prepare_omlx.py`](scripts/prepare_omlx.py)）。重みの変換やコピーはしていない。
  Splash は、専用のパッケージ `incoai/Qwen3.8-27B-Splash` を読む。
- **KV キャッシュ**：8-bit KV は各エンジン独自の実装である（MTPLX は affine q8 の paged KV、oMLX は TurboQuant 8-bit、Splash は INT8）。
  それ以外は、プレフィックスキャッシュも含めて各エンジンの既定値である（MTPLX は RAM、oMLX は SSD に 4,096 トークン単位、Splash は RAM）。
- **熱と順番**
  - セルごとにエンジンを起動し直し、エンジンのディスク上のキャッシュを消してから、ウォームアップと、熱の確認用のリクエスト（canary：約 2K の固定のプロンプトで decode の速さを測る）を送る。
  - 2回目は、エンジンの順番を逆にした。
  - canary がそのエンジンの最良値より 3% 以上遅いセルは、5 分休んでから測り直した（1回だけ起きた）。
  - 128K は1回だけ、upstream、oMLX、fork の順に測った（fork が最後で、最もマシンが温まった状態になる）。
    128K は別の計画として走らせたので canary の基準値が無く、測り直しは働かなかった。128K の canary は、2K〜64K の計測での最良値より、fork が 1.7%、upstream が 2.5%、oMLX が 4.8% 低かった。
  - fork は 2026-09-26〜28 に、`40b6113`、`4fe8067`、`bba1b7b`、`0cd2a73` で同じ計画を単独で測り直した（`bba1b7b` の 128K の canary は、2K〜64K の最良値より 0.2% 低かった。`0cd2a73` の1回目の 8K は、再起動の直後で再実行しても 5% 遅いままだったが、そのまま残した）。
- **回ごとのばらつき**：2回測ったセルの decode の差は、中央値で 1.3%、最大で 10% だった（oMLX の 64K の3ターン目。2回で生成テキストが異なった）。
  MTPLX の2つは、それぞれ 15 ターンすべてで2回の出力がバイト単位で一致し、oMLX は 15 ターン中 12 ターンで一致した。

## 再現の手順

```bash
cp config/local.example.json config/local.json   # パスを設定する
python3 scripts/prepare_omlx.py                   # oMLX 用のモデルフォルダと設定フォルダ
./run_day.sh                                      # 2K〜64K、2回（約 3 時間）
./run_night.sh                                    # 128K、1回（約 3 時間）
python3 scripts/summarize.py results/2026-09-m1max-64gb
```

MTPLX の2つは、`work/worktrees/` に置いた2つのコミットの git worktree から、MTPLX アプリの Python 環境で動かす（MTPLX の clone で `git worktree add --detach work/worktrees/mtplx-<commit> <commit>`）。
`runner.py` と `summarize.py` が必要とするのは、標準ライブラリと `tokenizers` だけである。

## エンジンの追加

エンジンは [`engines/`](engines/README.md) の JSON ファイル1つで表す。
中身は起動コマンド、ポート、KV のモード、エンジン自身の統計の読み方である。
追加したエンジンは、`plans/` の計画と `scripts/summarize.py` のエンジンの一覧に加える。

## ライセンス

このリポジトリのコードは MIT ライセンスである。
モデルと各エンジンは、それぞれのライセンスに従う。
