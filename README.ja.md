# Apple Silicon の LLM エンジン比較（長い文脈と複数ターン）

[English](README.md) | 日本語

Mac で長いプロンプトを読ませて3ターン会話するとき、推論エンジンによって速さはどれだけ変わるのか。
このリポジトリは、MacBook Pro（M1 Max、64 GB）の上で Qwen3.8-27B を複数の推論エンジンで動かし、同じ会話の速さを比べたベンチマークである。
どのエンジンも投機的デコードと 8-bit の KV キャッシュを使い（TensorFold は 8-bit の KV が無いので bf16）、各エンジンの OpenAI 互換のストリーミング API を同じ方法で計測した。
6つ目の列として、MTPLX upstream に M1 build の既定値を環境変数で与えたもの（upstream + env）を測り、M1 build の差のうちコードによる分と設定による分を分けた。

MTPLX の3つと oMLX は、**同じモデルファイル**を MTP（深さ 3）で動かす。
Splash と TensorFold は専用のパッケージ（4bit、group size 64 の重みと DFlash2 の draft モデル）を読む。
そのため Splash と TensorFold の列は、エンジンの違いに加えて、量子化と投機的デコードの方式の違いも含む。

> **開示**：MTPLX M1 build は、このベンチの作者が管理している。
> M1 build の列は公開したリリースのタグを upstream と同じコマンドラインで動かしたもので、以下の数値はすべて `rows.jsonl` から `scripts/summarize.py` で集計した。

| エンジン | 版 | 備考 |
|---|---|---|
| **MTPLX M1 build** | [`v2.12.0-m1`](https://github.com/shunya1810/MTPLX/releases/tag/v2.12.0-m1)（`6be2b28`、[`shunya1810/MTPLX`](https://github.com/shunya1810/MTPLX/tree/m1max-longctx)） | MTPLX 2.12.0 を M1 系の長い文脈向けに調整した非公式版。256K の確認は後継の [`v2.12.0-m1.1`](https://github.com/shunya1810/MTPLX/releases/tag/v2.12.0-m1.1)（command buffer の拡大を外した版。prompt の読み込みは upstream と同じ速さ、decode は 6〜8% 低い。[後述](#256k-の確認2エンジン)）で測った。前の回の結果は、生データに `mtplx-fork-16751dc`、`-40b6113`、`-4fe8067`、`-bba1b7b`、`-0cd2a73` として残した |
| **MTPLX upstream** | [`youssofal/MTPLX@1de2b1c`](https://github.com/youssofal/MTPLX/commit/1de2b1c049136ed117af0c6712baaadd81820b51) | 計測時点の `main`（2026-09-29 時点でも最新）。M1 build の土台。最初の計測（2026-09-25〜26）は生データに `mtplx-upstream-0925` として残した |
| **MTPLX upstream + env** | upstream `1de2b1c` | 同じコードに、M1 build の既定値のうち upstream が環境変数で持つものをすべて与えた：`MTPLX_CONTEXT_COPY=0`、`MTPLX_FUSE_PROJ=gdn,attn`、`MTPLX_FRSPEC_DRAFT=1 MTPLX_FRSPEC_LEGACY=1 MTPLX_FRSPEC_VOCAB=builtin:qwen38-code-64k`、`MTPLX_MLX_CACHE_LIMIT=1G`、`MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES=2`（[エンジンの定義](engines/mtplx-upstream-tuned-nobuf.json)）。M1 build の Metal command buffer の拡大は入れていない（後述） |
| **oMLX** | [0.6.4](https://github.com/jundot/omlx/releases/tag/v0.6.4) | 計測時点の最新の正式版 |
| **Splash 1.1.0-m1** | [paperniuk/splash 1.1.0-m1](https://github.com/paperniuk/splash/releases/tag/1.1.0-m1) | [incoai/splash](https://github.com/incoai/splash) 1.1.0 を M1/M2 向けの kernel で動かすコミュニティ版。モデルは `incoai/Qwen3.8-27B-Splash`（4bit g64 と DFlash2 の draft）、KV は INT8。前に測った 1.0.2-m1 のリリースと、その時点の最新をビルドしたソース版を置き換えた（生データに `splash-m1-102`、`splash-src` として残した） |
| **TensorFold** | [0.3.5.1](https://github.com/ashhart/TensorFold/releases/tag/v0.3.5.1) | draft の受理を「本体のサンプルと一致したときだけ」にする Python/MLX のエンジン。モデルは `Vontra/Qwen3.8-27B-MLX-4bit`（4bit g64）と `z-lab/Qwen3.8-27B-DFlash2` の draft（読み込み時に 4bit）、KV は bf16（8-bit の設定は無い）。M1 Max で起動できる最初の版（2026-09-28） |

2K〜64K は各2回測り、中央値を載せた。128K は1回である。
MTPLX の3つは 2026-09-29 に同じセッションで交互に測った。oMLX は 2026-09-25〜26、Splash と TensorFold はその後にそれぞれ別のセッションで、同じ方法で測った。

## 主な結果（8-bit KV）

decode は3ターンの平均（tok/s）で、2K / 8K / 32K / 64K / 128K の順である。

| エンジン | 2K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| MTPLX M1 build | 31.8 | 31.2 | 27.0 | **24.3** | **19.6** |
| MTPLX upstream | 23.6 | 22.5 | 14.0 | 8.7 | 5.0 |
| MTPLX upstream + env | 27.2 | 25.8 | 15.3 | 7.8 | 4.6 |
| oMLX | 21.6 | 19.4 | 14.3 | 10.7 | 7.4 |
| Splash 1.1.0-m1 | **33.1** | **32.5** | **27.1** | 24.1 | 18.7 |
| TensorFold（bf16 KV） | 27.7 | 25.5 | 17.0 | 11.3 | 6.4 |

- **短いプロンプトは Splash が速く、長いプロンプトは M1 build と同じくらい**：Splash 1.1.0-m1 は 2K〜8K で M1 build より 4% 速い。
  32K〜64K では両者の差は 1% 以内で、128K では M1 build が 5% 速い。
  どちらも oMLX の 1.9 倍（32K）から 2.5〜2.6 倍（128K）である。
- **サンプリングでは順番が少し変わる**：モデル本来の設定（temperature 1.0、top-p 0.95、top-k 20、最大 512 トークン）では、2K / 32K / 64K で M1 build が 32.3 / 27.3 / 24.1 tok/s、Splash が 32.7 / 25.8 / 22.0 だった。2K では同じで、32K で 6%、64K で 10% M1 build が速い。upstream は 23.2 / 14.2 / 8.9 だった（[詳細](#サンプリングtemperature-10)）。
- **M1 build の設定を upstream に与えた効果**：upstream + env は upstream より 2K〜8K で 15%、32K で 9% 速い。
  一方で 64K では 10%、128K では 8% 遅い（context-copy の draft を切ると、upstream の重い長文脈のラウンドが増え、得より損が大きい）。
  M1 build と upstream + env の差（2K で 17%、32K で 1.8 倍、64K で 3.1 倍、128K で 4.3 倍）は、したがって M1 build のコードによる。
  中身は、verify・draft・prefill の M1 向け attention の kernel と、セッションバンクの修正である（[変更内容](https://github.com/shunya1810/MTPLX/blob/m1max-longctx/docs/m1max-longctx/CHANGES.ja.md)）。
- **command buffer の設定は upstream には効かない**：M1 build は MLX の Metal command buffer を 150 処理・1,000 MB に広げ、prefill 側をコードで変えて組み合わせている。
  upstream に `MLX_MAX_OPS_PER_BUFFER` と `MLX_MAX_MB_PER_BUFFER` だけを与えると、最大メモリが（与えない upstream + env より）12〜15 GB 増え、32K〜64K の最初の prefill が 12〜16% 遅くなり、128K では GPU のメモリが足りずに失敗した。decode は変わらなかった。その行は生データに `mtplx-upstream-tuned` として残した。
- **2〜3ターン目の最初のトークンまでの時間（TTFT）**：128K では M1 build が **2.2 秒**、upstream が 3.5 秒、upstream + env が 4.2 秒、Splash が 6.2 秒（約 300 トークンを読み直す）、TensorFold が 29 分（後述）、oMLX が 1.6 分と 7.9 秒だった。
  oMLX はプレフィックスキャッシュを 4,096 トークン単位で SSD に保存し、このモデル系統では、プロンプトが最後に越えたブロックの境界までしか保存できない。
  そのため次のターンでは、境界より後ろの部分（最大で約 4K トークン）を読み直す。
- **1ターン目（最初から読む prefill）**：64K までは各エンジンとも近い（64K で 9.7〜11.9 分、TensorFold が最も遅い）。
  M1 build は、乱れの少ない回で 2K〜32K の prefill が upstream より 1〜5% 遅い（command buffer の拡大と、4 層ごとの評価による）。
  64K の中央値 10.4 分は、マシンに別の負荷があった回の 11.1 分と、もう1回の 9.7 分の中央値である。
  128K では attention の割合が大きく、M1 build が 24.6 分、oMLX が 26.3 分、upstream が 28.1 分、Splash が 28.5 分、TensorFold が 29.2 分、upstream + env が 29.5 分だった。
- **128K の3ターン全体**：M1 build が 25.3 分、Splash が 29.4 分、oMLX が 29.6 分、upstream が 30.6 分、upstream + env が 32.2 分、TensorFold が 89.6 分だった。
- **メモリ**：Splash が最も少なく、どの長さでも wired が 22〜26 GB だった。
  最大値は M1 build が 30〜46 GB、upstream が 34〜52 GB、upstream + env が 30〜53 GB、oMLX が 27〜49 GB だった（システム全体の wired の最大値。Splash は重みの持ち方が違うので、比べるときは注意が要る）。
- **TensorFold 0.3.5.1（bf16 KV）**：2K〜8K で M1 build より 13〜18% 遅く、128K では M1 build の3分の1だった。
  128K では、既定のプロンプトキャッシュ（RAM の8分の1、8 GiB）に会話が入らず、2ターン目と3ターン目で 131K トークンをすべて読み直し、3ターン全体で 89.6 分かかった。`--prompt-cache-gib` を大きくすれば避けられる可能性があるが、このベンチはすべてのエンジンを既定値で測っている。
  draft のトークンは、本体が自分でサンプルしたトークンと一致したときだけ受理する。temperature 0 では問題ないが、サンプリングでは受理が減る。
- **256K（2エンジン、各1回）**：M1 build（m1.1）は prompt の読み込みが 67.9 分（Splash は 87.3 分）、decode が 14.5 tok/s（同 12.9）、続きのターンの TTFT が 3.6 秒（同 9.3〜10.1 秒）だった。メモリは Splash が 30.4 GB、M1 build が 51.8 GB だった（[詳細](#256k-の確認2エンジン)）。
- **測っていないもの**：GPU の温度、消費電力、1トークンあたりのエネルギー。M1 では MTPLX 系が prefill 中に Splash より熱くなるという報告があるが、このベンチでは確かめていない。
- **出力の確認**：3ターン目では、文脈の中央に1行だけ埋め込んだ特別な行（needle）をそのまま引用させた。完了したすべての回で、すべてのエンジンが正しく答えた。

## decode の速さ

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-dark.svg?v=fair0929">
  <img alt="プロンプト長ごとの decode tok/s（5エンジン）" src="charts/decode-light.svg?v=fair0929">
</picture>

## 2〜3ターン目の TTFT

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/ttft-followup-dark.svg?v=fair0929">
  <img alt="2〜3ターン目の TTFT（対数目盛り、5エンジン）" src="charts/ttft-followup-light.svg?v=fair0929">
</picture>

## 3ターンの合計所要時間

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/conversation-dark.svg?v=fair0929">
  <img alt="エンジンとプロンプト長ごとの3ターンの所要時間" src="charts/conversation-light.svg?v=fair0929">
</picture>

## 最大メモリ使用量

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/memory-dark.svg?v=fair0929">
  <img alt="プロンプト長ごとの wired メモリの最大値" src="charts/memory-light.svg?v=fair0929">
</picture>

## 計測値の表

2K〜64K の各セルは2回の中央値、128K は1回の値である。
cache の列は、エンジンが返す `usage.prompt_tokens_details.cached_tokens` の値である。
wired メモリはシステム全体の値である。
needle の列は、3ターン目で needle を正しく引用した回数である。

| 文脈 | エンジン | T1 TTFT（cold） | T2 TTFT | T3 TTFT | decode T1 / T2 / T3（tok/s） | 3ターンの合計 | cache T2 / T3 | 最大 wired | needle |
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

**2K：fp16 KV と 8-bit KV**（decode tok/s、ターン1〜3の平均）

| エンジン | fp16 KV | 8-bit KV | 変化 |
|---|---|---|---|
| MTPLX M1 build | 29.2 | 31.8 | +8.6% |
| MTPLX upstream | 24.8 | 23.6 | -5.0% |
| MTPLX upstream + env | 28.1 | 27.2 | -3.3% |
| oMLX | 23.1 | 21.6 | -6.4% |
| Splash 1.1.0-m1 | 33.5 | 33.1 | -1.3% |

生データは [`results/2026-09-m1max-64gb/rows.jsonl`](results/2026-09-m1max-64gb/rows.jsonl)、ターンごとの CSV は [`summary.csv`](results/2026-09-m1max-64gb/summary.csv)、計測環境は [`environment.json`](results/2026-09-m1max-64gb/environment.json) にある。
各ターンの生成テキストは `results/2026-09-m1max-64gb/cells/*/` に置いた。

## 256K の確認（2エンジン）

128K で上位だった2つだけを、258,048 トークンの文脈で1回ずつ測った（3ターン目で約 258K トークンになり、両エンジンの 262,144 トークンの窓に収まる）。
2026-09-29 に Splash 1.1.0-m1、MTPLX M1 build **v2.12.0-m1.1**（表の下の注を参照）の順に測ったので、M1 build のほうがマシンが温まった状態になる。
ほかのエンジンは 128K で decode が 4.6〜7.4 tok/s で、256K の1ターン目だけで1時間を大きく超える見込みなので測っていない。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/spot256k-dark.svg?v=spot256">
  <img alt="256K の確認：Splash 1.1.0-m1 と MTPLX M1 build m1.1 の最初の prefill、続きのターンの TTFT、decode、最大メモリ" src="charts/spot256k-light.svg?v=spot256">
</picture>

| | Splash 1.1.0-m1 | MTPLX M1 build m1.1 |
|---|---|---|
| T1 TTFT（cold） | 87.3 分 | **67.9 分** |
| T2 / T3 TTFT | 9.3 秒 / 10.1 秒 | **3.6 秒 / 3.6 秒** |
| decode T1 / T2 / T3（tok/s） | 11.4 / 14.9 / 12.5 | 14.5 / 14.9 / 14.2 |
| decode（ターンの平均） | 12.9 | **14.5** |
| 3ターンの合計 | 88.5 分 | **68.9 分** |
| 最大 wired | **30.4 GB** | 51.8 GB |
| needle | 1/1 | 1/1 |

- それぞれ1回の計測である。計測前の 2K の canary は、Splash が 29.2 tok/s（その日の最良値より 3.7% 低い）、M1 build が 30.0 tok/s（m1.1 の最良値より 2% 低い）で、Splash のときのほうがマシンが少し遅い状態だった。
- **v2.12.0-m1.1**（2026-09-29 公開）は、v2.12.0-m1 が M1 で有効にしていた Metal の command buffer の拡大を既定で外した版である。拡大すると decode は +6〜7% になるが、最初の prefill がアイドル時で 2〜5%、使用中や電力が足りないときは 13〜21% 遅くなっていた。
  m1.1 の prompt の読み込みは upstream と同じ速さで（8K で 54.5 秒。続けて測った upstream は 54.3 秒、m1 は 67.1 秒）、decode は m1 より 6〜8% 低く、生成テキストは同じである。上の 2K〜128K の表は v2.12.0-m1 の値である。

## サンプリング（temperature 1.0）

temperature 0 で 256 トークンという条件は、比べるために揃えた課題で、普段の使い方ではない。
そこで 2K、32K、64K の同じ会話を、モデル本来のサンプリングの設定（`generation_config.json`：temperature 1.0、top-p 0.95、top-k 20）、各ターン最大 512 トークン、8-bit KV で、順番を逆にして2回測った（2026-09-29、Splash も同じセッション）。
サンプリングでは回ごとに生成テキストが変わるので、2〜3ターン目のプロンプトもエンジンと回で異なる。
受理率はエンジン自身の数え方（MTPLX は MTP の draft、Splash は DFlash2 の draft で1回 7 トークン）なので、方式をまたいで比べられない。

| コンテキスト | エンジン | T1 TTFT（コールド） | decode T1 / T2 / T3 (tok/s) | decode 平均 | ドラフト受理率 T1 / T2 / T3 | 生成トークン T1 / T2 / T3 | needle |
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

- **2K では M1 build と Splash は同じ（32.3 と 32.7 tok/s）で、32K 以上では M1 build が速い（32K で +6%、64K で +10%）**。temperature 0 では両者は同じくらいだった。
- **MTPLX 系はサンプリングでも速さがほとんど変わらない**（M1 build は 2K で 32.3 と 31.8、64K で 24.1 と 24.3）。Splash は temperature 0 より 2K で 1%、32K で 5%、64K で 9% 遅い。
- **upstream + env** は upstream より 2K で 14% 速く、32K で同じ、64K で 8% 遅い。temperature 0 と同じ傾向である。
- ターンの終わりはモデルが決めるので、生成トークン数はエンジンごとに異なる（1ターン目で 280〜382）。すべての回で、すべてのエンジンが needle を正しく引用した。

生データ：[`results/2026-09-m1max-64gb-sampling/rows.jsonl`](results/2026-09-m1max-64gb-sampling/rows.jsonl)

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
  2ターン目の回答は、1ターン目で決まった比較の型を繰り返す。7 トークンの draft は、3トークンの MTP の draft より先まで当てられると考えられる。MTPLX M1 build は、2ターン目で1ラウンド 2.8〜3.0 トークン、1ターン目と3ターン目で 2.7〜2.9 トークンで、上限は 4 である。
- **短い文脈では、Splash は近い時間で2倍の位置を verify する**：2K で、8 位置の verify と 7 トークンの draft が1回 105 ms だった。M1 build は、4 位置の verify と3トークンの MTP の draft で1ラウンド 87〜90 ms である（2026-09-29 の計測の verify と draft の時間）。
- **1回の時間は、プロンプトが長いほど大きく伸びる**：Splash は 2K の 105 ms から 128K の 189〜197 ms へ、M1 build は 87〜90 ms から 137〜140 ms へ伸びた。
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
  これは同じ決定的な課題の上でエンジンを比べるための速度である。2K〜64K は、上のサンプリングの条件でも測った。
- **モデルファイル**：MTPLX の2つと oMLX は同じモデルファイルを読む。
  oMLX は、MTPLX のチェックポイントを symlink で組んだテキスト専用のフォルダ経由で読む（[`scripts/prepare_omlx.py`](scripts/prepare_omlx.py)）。重みの変換やコピーはしていない。
  Splash は、専用のパッケージ `incoai/Qwen3.8-27B-Splash` を読む。
- **KV キャッシュ**：8-bit KV は各エンジン独自の実装である（MTPLX は affine q8 の paged KV、oMLX は TurboQuant 8-bit、Splash は INT8）。
  それ以外は、プレフィックスキャッシュも含めて各エンジンの既定値である（MTPLX は RAM、oMLX は SSD に 4,096 トークン単位、Splash は RAM）。
- **熱と順番**
  - セルごとにエンジンを起動し直し、エンジンのディスク上のキャッシュを消してから、ウォームアップと、熱の確認用のリクエスト（canary：約 2K の固定のプロンプトで decode の速さを測る）を送る。
  - 2回目は、エンジンの順番を逆にした。
  - canary がそのエンジンの最良値より 3% 以上遅いセルは、5 分休んでから測り直した。
  - MTPLX の3つは 2026-09-29 に一緒に測った（測り直しは起きなかった）。128K は1回だけ、upstream、upstream + env（command buffer あり、メモリ不足で失敗）、M1 build の順に測った（M1 build が最後で、最もマシンが温まった状態になる）。command buffer なしの upstream + env の 128K は、その後に測った。128K の canary は、各エンジンの 2K〜64K の最良値から 2.4% 以内だった。
  - oMLX、Splash、TensorFold は、それより前のセッションで同じ計画で測った（詳細と canary は [METHODOLOGY.md](METHODOLOGY.md)）。
- **回ごとのばらつき**：2026-09-29 の計測で、2回測ったセルの decode の差は中央値で 0.7% だった。
  最大は upstream + env の 64K の1ターン（生成テキストはバイト単位で同じだが 5.2 と 8.7 tok/s。verify の時間が 47 秒と 27 秒だった）で、このセルの2ターン目の中央値は 6.9 になった。
  MTPLX の3つは、すべてのターンで2回の出力がバイト単位で一致した。

## 再現の手順

```bash
cp config/local.example.json config/local.json   # パスを設定する
python3 scripts/prepare_omlx.py                   # oMLX 用のモデルフォルダと設定フォルダ
./run_day.sh                                      # 2K〜64K、2回（約 3 時間）
./run_night.sh                                    # 128K、1回（約 3 時間）
./run_fair.sh                                     # MTPLX の3つとサンプリングの条件
python3 scripts/summarize.py results/2026-09-m1max-64gb
python3 scripts/summarize_sampling.py results/2026-09-m1max-64gb-sampling
```

MTPLX は、`work/worktrees/` に置いた git worktree（`mtplx-1de2b1c`、`mtplx-v2.12.0-m1`）から、MTPLX アプリの Python 環境で動かす（MTPLX の clone で `git worktree add --detach work/worktrees/mtplx-<commit> <commit>`）。
`runner.py` と `summarize.py` が必要とするのは、標準ライブラリと `tokenizers` だけである。

## エンジンの追加

エンジンは [`engines/`](engines/README.md) の JSON ファイル1つで表す。
中身は起動コマンド、ポート、KV のモード、エンジン自身の統計の読み方である。
追加したエンジンは、`plans/` の計画と `scripts/summarize.py` のエンジンの一覧に加える。

## ライセンス

このリポジトリのコードは MIT ライセンスである。
モデルと各エンジンは、それぞれのライセンスに従う。
