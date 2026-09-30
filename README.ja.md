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
| **MTPLX M1 build** | [`v2.12.0-m1.2`](https://github.com/shunya1810/MTPLX/releases/tag/v2.12.0-m1.2)（`9984c63`、[`shunya1810/MTPLX`](https://github.com/shunya1810/MTPLX/tree/m1max-longctx)） | MTPLX 2.12.0 を M1 系の長い文脈向けに調整した非公式版。2026-09-30 に単独で測った（後述）。最初のリリース `v2.12.0-m1` は 2026-09-29 に他の MTPLX の列と一緒に測り、生データに `mtplx-fork-m1` として残した（decode は 3〜9% 速く、2K〜32K の prompt の読み込みは 2〜9% 遅かった。[理由](#いまの版が-decode-より-prompt-の読み込みを優先する理由)）。256K の確認は `v2.12.0-m1.1` で測った。m1.2 との違いは、このベンチが使っていない SSD のプロンプトキャッシュの修正だけである。前の回：`mtplx-fork-16751dc`、`-40b6113`、`-4fe8067`、`-bba1b7b`、`-0cd2a73` |
| **MTPLX upstream** | [`youssofal/MTPLX@1de2b1c`](https://github.com/youssofal/MTPLX/commit/1de2b1c049136ed117af0c6712baaadd81820b51) | 計測時点の `main`（2026-09-29 時点でも最新）。M1 build の土台。最初の計測（2026-09-25〜26）は生データに `mtplx-upstream-0925` として残した |
| **MTPLX upstream + env** | upstream `1de2b1c` | 同じコードに、M1 build の既定値のうち upstream が環境変数で持つものをすべて与えた：`MTPLX_CONTEXT_COPY=0`、`MTPLX_FUSE_PROJ=gdn,attn`、`MTPLX_FRSPEC_DRAFT=1 MTPLX_FRSPEC_LEGACY=1 MTPLX_FRSPEC_VOCAB=builtin:qwen38-code-64k`、`MTPLX_MLX_CACHE_LIMIT=1G`、`MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES=2`（[エンジンの定義](engines/mtplx-upstream-tuned-nobuf.json)）。M1 build の Metal command buffer の拡大は入れていない（後述） |
| **oMLX** | [0.6.4](https://github.com/jundot/omlx/releases/tag/v0.6.4) | 計測時点の最新の正式版 |
| **Splash 1.1.0-m1** | [paperniuk/splash 1.1.0-m1](https://github.com/paperniuk/splash/releases/tag/1.1.0-m1) | [incoai/splash](https://github.com/incoai/splash) 1.1.0 を M1/M2 向けの kernel で動かすコミュニティ版。モデルは `incoai/Qwen3.8-27B-Splash`（4bit g64 と DFlash2 の draft）、KV は INT8。前に測った 1.0.2-m1 のリリースと、その時点の最新をビルドしたソース版を置き換えた（生データに `splash-m1-102`、`splash-src` として残した） |
| **TensorFold** | [0.3.5.1](https://github.com/ashhart/TensorFold/releases/tag/v0.3.5.1) | draft の受理を「本体のサンプルと一致したときだけ」にする Python/MLX のエンジン。モデルは `Vontra/Qwen3.8-27B-MLX-4bit`（4bit g64）と `z-lab/Qwen3.8-27B-DFlash2` の draft（読み込み時に 4bit）、KV は bf16（8-bit の設定は無い）。M1 Max で起動できる最初の版（2026-09-28） |

2K〜64K は各2回測り、中央値を載せた。128K は1回である。
MTPLX upstream と upstream + env は 2026-09-29 に同じセッションで交互に測った。M1 build（m1.2）は 2026-09-30 に単独で、2K の確認でマシンが同じ状態（canary 30.6 tok/s）であることを確かめてから測った。oMLX は 2026-09-25〜26、Splash と TensorFold はそれぞれ別のセッションで、同じ方法で測った。

## 主な結果（8-bit KV）

**3ターンの会話全体の時間**（1ターン目は prompt を最初から読み、2〜3ターン目で約 300 トークンずつ足す。各ターンの生成は最大 256 トークン）：

| エンジン | 2K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| MTPLX M1 build | **36 秒** | **78 秒** | **4.6 分** | **10.1 分** | **24.9 分** |
| MTPLX upstream | 42 秒 | 86 秒 | 5.0 分 | 11.3 分 | 30.6 分 |
| MTPLX upstream + env | 38 秒 | 84 秒 | 4.9 分 | 11.3 分 | 32.2 分 |
| oMLX | 71 秒 | 125 秒 | 6.0 分 | 12.4 分 | 29.6 分 |
| Splash 1.1.0-m1 | 38 秒 | 84 秒 | 5.1 分 | 11.5 分 | 29.4 分 |
| TensorFold（bf16 KV） | 47 秒 | 106 秒 | 6.1 分 | 13.1 分 | 89.6 分 |

**decode**（3ターンの平均、tok/s）：

| エンジン | 2K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| MTPLX M1 build | 29.3 | 28.8 | 25.5 | 23.4 | **19.0** |
| MTPLX upstream | 23.6 | 22.5 | 14.0 | 8.7 | 5.0 |
| MTPLX upstream + env | 27.2 | 25.8 | 15.3 | 7.8 | 4.6 |
| oMLX | 21.6 | 19.4 | 14.3 | 10.7 | 7.4 |
| Splash 1.1.0-m1 | **33.1** | **32.5** | **27.1** | **24.1** | 18.7 |
| TensorFold（bf16 KV） | 27.7 | 25.5 | 17.0 | 11.3 | 6.4 |

- **どちらが速いかは、使い方の形で決まる**：decode は 64K まで Splash が最も速い（2K〜8K で M1 build より 13%、32K〜64K で 3〜6% 速く、128K では 2% 遅い）。
  M1 build は prompt の読み込みと続きのターンの開始が速いので、このベンチの形（長い入力に短く答える）では、会話全体の時間がすべての長さで最も短い。Splash より 2K で 5%、32K で 9%、64K で 13%、128K で 15% 短い。
  読む量より生成する量がずっと多いターン（チャットや、ファイル全体のコード生成）では、Splash の decode の速さが効く。
- **サンプリング**（temperature 1.0、top-p 0.95、top-k 20、最大 512 トークン）：2K / 32K / 64K で M1 build が 27.5 / 25.2 / 23.8 tok/s、Splash が 32.7 / 25.8 / 22.0 だった。2K で Splash が 19%、32K で 2% 速く、64K では M1 build が 8% 速い。upstream は 23.2 / 14.2 / 8.9 だった（[詳細](#サンプリングtemperature-10)）。
- **M1 build の設定を upstream に与えた効果**：upstream + env は upstream より 2K〜8K で 15%、32K で 9% 速い。
  一方で 64K では 10%、128K では 8% 遅い（context-copy の draft を切ると、upstream の重い長文脈のラウンドが増え、得より損が大きい）。
  M1 build と upstream + env の差（2K で 8%、32K で 1.7 倍、64K で 3.0 倍、128K で 4.1 倍）は、したがって M1 build のコードによる。
  中身は、verify・draft・prefill の M1 向け attention の kernel と、セッションバンクの修正である（[変更内容](https://github.com/shunya1810/MTPLX/blob/m1max-longctx/docs/m1max-longctx/CHANGES.ja.md)）。
- **1ターン目（最初から読む prefill）**：M1 build は 2K〜8K を upstream と同じ速さ（11.4 秒、53 秒）、32K を 2.5% 遅く（4.1 分）読み、64K と 128K では全エンジンで最も速い。64K は 9.5 分（upstream + env 9.7、upstream 9.9、oMLX 10.2、Splash 10.9、TensorFold 11.9）、128K は 24.2 分（oMLX 26.3、upstream 28.1、Splash 28.5、TensorFold 29.2、upstream + env 29.5）だった。
- **2〜3ターン目の最初のトークンまでの時間（TTFT）**：128K では M1 build が **2.1 秒**、upstream が 3.5 秒、upstream + env が 4.2 秒、Splash が 6.2 秒（約 300 トークンを読み直す）、TensorFold が 29 分（後述）、oMLX が 1.6 分と 7.9 秒だった。
  oMLX はプレフィックスキャッシュを 4,096 トークン単位で SSD に保存し、このモデル系統では、プロンプトが最後に越えたブロックの境界までしか保存できない。
  そのため次のターンでは、境界より後ろの部分（最大で約 4K トークン）を読み直す。
- **メモリ**：Splash が最も少なく、どの長さでも wired が 22〜26 GB だった。
  最大値は M1 build が 30〜44 GB、upstream が 34〜52 GB、upstream + env が 30〜53 GB、oMLX が 27〜49 GB だった（システム全体の wired の最大値。Splash は重みの持ち方が違うので、比べるときは注意が要る）。
- **command buffer の設定は upstream には効かない**：MLX の Metal command buffer を 150 処理・1,000 MB に広げる設定（最初の M1 build の既定値）を upstream に与えると、最大メモリが 12〜15 GB 増え、32K〜64K の最初の prefill が 12〜16% 遅くなり、128K では GPU のメモリが足りずに失敗した。decode は変わらなかった。その行は生データに `mtplx-upstream-tuned` として残した。
- **TensorFold 0.3.5.1（bf16 KV）**：decode は 2K〜8K で M1 build より 5〜12% 遅く、128K では M1 build の3分の1だった。
  128K では、既定のプロンプトキャッシュ（RAM の8分の1、8 GiB）に会話が入らず、2ターン目と3ターン目で 131K トークンをすべて読み直し、3ターン全体で 89.6 分かかった。`--prompt-cache-gib` を大きくすれば避けられる可能性があるが、このベンチはすべてのエンジンを既定値で測っている。
  draft のトークンは、本体が自分でサンプルしたトークンと一致したときだけ受理する。temperature 0 では問題ないが、サンプリングでは受理が減る。
- **256K（2エンジン、各1回）**：M1 build は prompt の読み込みが 67.9 分（Splash は 87.3 分）、decode が 14.5 tok/s（同 12.9）、続きのターンの TTFT が 3.6 秒（同 9.3〜10.1 秒）だった。メモリは Splash が 30.4 GB、M1 build が 51.8 GB だった（[詳細](#256k-の確認2エンジン)）。
- **短いプロンプト、推論あり**（Splash の投稿のベンチマーク）：Splash の decode が 29% 速く、1トークンあたりのエネルギーは約3分の2だった。44K の prompt の読み込みは M1 build が 20% 速かった（[詳細](#短いプロンプトsplash-の投稿と同じ条件)）。
- **出力の確認**：3ターン目では、文脈の中央に1行だけ埋め込んだ特別な行（needle）をそのまま引用させた。完了したすべての回で、すべてのエンジンが正しく答えた。

### いまの版が decode より prompt の読み込みを優先する理由

最初のリリース（`v2.12.0-m1`）は、M1 で MLX の Metal command buffer の上限を上げていた。
このベンチでは decode が 3〜9% 速くなる一方、2K〜32K の最初の prompt の読み込みがアイドル時で 2〜9% 遅くなっていた（別の試験では、マシンの使用中や電力不足のときに 13〜21% 遅くなった）。片方だけを得る設定は見つからなかった。
長い文脈のターンでは時間の大半が prompt の読み込みなので、`v2.12.0-m1.1` から MLX の既定値に戻した。
目安として、1ターンで新しく N トークン読み、M トークン生成するとき、M が N の約8分の1より少なければ（使用中なら N の約6割より少なければ）m1.2 の方が速い。
生成が中心の使い方では、`MTPLX_MLX_COMMAND_BUFFER_MB=1000` で m1 の挙動に戻せる。
m1 の行（`mtplx-fork-m1`）は、2K / 8K / 32K / 64K / 128K で decode が 31.8 / 31.2 / 27.0 / 24.3 / 19.6 tok/s、会話全体が 35 秒 / 78 秒 / 4.7 分 / 10.9 分 / 25.3 分だった。

## decode の速さ

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-dark.svg?v=m12">
  <img alt="プロンプト長ごとの decode tok/s（5エンジン）" src="charts/decode-light.svg?v=m12">
</picture>


256K の確認をした2エンジンについて、256K まで伸ばしたもの（M1 build の 256K は v2.12.0-m1.1 で、SSD キャッシュの修正を除けば m1.2 と同じ）：

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-256k-dark.svg?v=m12">
  <img alt="2K から 256K までの decode tok/s（MTPLX M1 build、MTPLX upstream、Splash 1.1.0-m1）" src="charts/decode-256k-light.svg?v=m12">
</picture>

## 2〜3ターン目の TTFT

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/ttft-followup-dark.svg?v=m12">
  <img alt="2〜3ターン目の TTFT（対数目盛り、5エンジン）" src="charts/ttft-followup-light.svg?v=m12">
</picture>

## 3ターンの合計所要時間

最後のパネルは 256K の確認である（各1回。M1 build は v2.12.0-m1.1）。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/conversation-dark.svg?v=m12">
  <img alt="エンジンとプロンプト長ごとの3ターンの所要時間" src="charts/conversation-light.svg?v=m12">
</picture>

## 最大メモリ使用量

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/memory-dark.svg?v=m12">
  <img alt="プロンプト長ごとの wired メモリの最大値" src="charts/memory-light.svg?v=m12">
</picture>

## 計測値の表

2K〜64K の各セルは2回の中央値、128K は1回の値である。
cache の列は、エンジンが返す `usage.prompt_tokens_details.cached_tokens` の値である。
wired メモリはシステム全体の値である。
needle の列は、3ターン目で needle を正しく引用した回数である。

| 文脈 | エンジン | T1 TTFT（cold） | T2 TTFT | T3 TTFT | decode T1 / T2 / T3（tok/s） | 3ターンの合計 | cache T2 / T3 | 最大 wired | needle |
|---|---|---|---|---|---|---|---|---|---|
| 2K | MTPLX M1 build | 11.4 s | 0.62 s | 0.61 s | 29.2 / 29.5 / 29.3 | 36.0 s | 1,936 / 2,243 | 30.1 GB | 2/2 |
| 2K | MTPLX upstream | 11.4 s | 0.63 s | 0.64 s | 25.0 / 22.8 / 23.1 | 41.8 s | 1,936 / 2,243 | 34.0 GB | 2/2 |
| 2K | MTPLX upstream + env | 11.4 s | 0.64 s | 0.63 s | 24.5 / 31.6 / 25.4 | 37.5 s | 1,928 / 2,234 | 29.8 GB | 2/2 |
| 2K | oMLX | 11.4 s | 13.5 s | 15.4 s | 20.4 / 23.5 / 21.0 | 1.2 min | 0 / 0 | 26.9 GB | 2/2 |
| 2K | Splash 1.1.0-m1 | 12.0 s | 2.56 s | 2.76 s | 31.3 / 39.2 / 28.8 | 37.8 s | 1,664 / 1,952 | 21.9 GB | 2/2 |
| 2K | TensorFold (bf16 KV) | 15.7 s | 3.47 s | 3.48 s | 26.1 / 28.9 / 28.0 | 47.0 s | 1,673 / 1,980 | 22.9 GB | 2/2 |
| 8K | MTPLX M1 build | 53.3 s | 0.68 s | 0.69 s | 28.4 / 30.8 / 27.3 | 1.3 min | 8,089 / 8,396 | 30.6 GB | 2/2 |
| 8K | MTPLX upstream | 53.2 s | 0.74 s | 0.77 s | 22.9 / 23.0 / 21.5 | 1.4 min | 8,089 / 8,396 | 35.5 GB | 2/2 |
| 8K | MTPLX upstream + env | 55.5 s | 0.78 s | 0.79 s | 25.1 / 27.2 / 25.2 | 1.4 min | 8,089 / 8,396 | 31.3 GB | 2/2 |
| 8K | oMLX | 56.4 s | 30.0 s | 2.69 s | 20.2 / 19.8 / 18.3 | 2.1 min | 4,096 / 8,192 | 33.2 GB | 2/2 |
| 8K | Splash 1.1.0-m1 | 56.8 s | 2.91 s | 2.79 s | 31.1 / 37.4 / 29.1 | 1.4 min | 7,808 / 8,128 | 22.1 GB | 2/2 |
| 8K | TensorFold (bf16 KV) | 1.2 min | 3.68 s | 3.69 s | 24.6 / 26.0 / 26.0 | 1.8 min | 7,826 / 8,132 | 22.6 GB | 2/2 |
| 32K | MTPLX M1 build | 4.1 min | 0.96 s | 0.96 s | 25.9 / 25.3 / 25.3 | 4.6 min | 32,665 / 32,972 | 33.8 GB | 2/2 |
| 32K | MTPLX upstream | 4.0 min | 1.32 s | 1.34 s | 17.3 / 12.7 / 12.0 | 5.0 min | 32,665 / 32,972 | 42.5 GB | 2/2 |
| 32K | MTPLX upstream + env | 4.0 min | 1.31 s | 1.34 s | 19.8 / 14.5 / 11.6 | 4.9 min | 32,665 / 32,972 | 38.1 GB | 2/2 |
| 32K | oMLX | 4.4 min | 44.6 s | 3.45 s | 14.4 / 14.0 / 14.4 | 6.0 min | 28,672 / 32,768 | 37.0 GB | 2/2 |
| 32K | Splash 1.1.0-m1 | 4.5 min | 3.57 s | 3.57 s | 23.3 / 29.1 / 28.7 | 5.1 min | 32,384 / 32,704 | 22.9 GB | 2/2 |
| 32K | TensorFold (bf16 KV) | 5.3 min | 4.31 s | 4.31 s | 15.4 / 18.9 / 16.5 | 6.1 min | 32,402 / 32,709 | 27.1 GB | 2/2 |
| 64K | MTPLX M1 build | 9.5 min | 1.34 s | 1.33 s | 23.0 / 24.3 / 23.0 | 10.1 min | 65,419 / 65,726 | 37.8 GB | 2/2 |
| 64K | MTPLX upstream | 9.9 min | 2.07 s | 2.09 s | 8.8 / 8.8 / 8.4 | 11.3 min | 65,419 / 65,726 | 48.6 GB | 2/2 |
| 64K | MTPLX upstream + env | 9.7 min | 2.09 s | 2.12 s | 8.2 / 6.9 / 8.2 | 11.3 min | 65,419 / 65,726 | 44.5 GB | 2/2 |
| 64K | oMLX | 10.2 min | 1.0 min | 4.76 s | 10.9 / 10.9 / 10.4 | 12.4 min | 61,440 / 65,536 | 39.6 GB | 2/2 |
| 64K | Splash 1.1.0-m1 | 10.9 min | 4.28 s | 4.90 s | 23.2 / 29.9 / 19.1 | 11.5 min | 65,152 / 65,440 | 24.1 GB | 2/2 |
| 64K | TensorFold (bf16 KV) | 11.9 min | 5.25 s | 5.19 s | 10.8 / 13.1 / 10.0 | 13.1 min | 65,156 / 65,463 | 33.2 GB | 2/2 |
| 128K | MTPLX M1 build | 24.2 min | 2.11 s | 2.10 s | 18.4 / 20.3 / 18.5 | 24.9 min | 130,969 / 131,276 | 43.5 GB | 1/1 |
| 128K | MTPLX upstream | 28.1 min | 3.53 s | 3.50 s | 4.7 / 5.1 / 5.0 | 30.6 min | 130,969 / 131,276 | 52.4 GB | 1/1 |
| 128K | MTPLX upstream + env | 29.5 min | 4.19 s | 4.16 s | 4.4 / 4.9 / 4.5 | 32.2 min | 130,969 / 131,276 | 52.7 GB | 1/1 |
| 128K | oMLX | 26.3 min | 1.6 min | 7.92 s | 7.6 / 7.8 / 6.9 | 29.6 min | 126,976 / 131,072 | 48.5 GB | 1/1 |
| 128K | Splash 1.1.0-m1 | 28.5 min | 6.27 s | 6.20 s | 15.9 / 22.8 / 17.4 | 29.4 min | 130,688 / 131,008 | 26.2 GB | 1/1 |
| 128K | TensorFold (bf16 KV) | 29.2 min | 29.3 min | 29.4 min | 6.3 / 6.5 / 6.3 | 89.6 min | 0 / 0 | 37.2 GB | 1/1 |

**2K：fp16 KV と 8-bit KV**（decode tok/s、ターン1〜3の平均）

| エンジン | fp16 KV | 8-bit KV | 変化 |
|---|---|---|---|
| MTPLX M1 build | 28.0 | 29.3 | +5.0% |
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
  <source media="(prefers-color-scheme: dark)" srcset="charts/spot256k-dark.svg?v=m12">
  <img alt="256K の確認：Splash 1.1.0-m1 と MTPLX M1 build m1.1 の最初の prefill、続きのターンの TTFT、decode、最大メモリ" src="charts/spot256k-light.svg?v=m12">
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
- ここでの M1 build は **v2.12.0-m1.1** である。上の表の m1.2 との違いは、このベンチが使っていない SSD のプロンプトキャッシュの修正だけである。

## 短いプロンプト（Splash の投稿と同じ条件）

「Splash on M1, part 2」の投稿のベンチマーク5を、MTPLX M1 build（v2.12.0-m1.2）と Splash 1.1.0-m1 だけで測り直した。
npanj の splash-plus の5つのプロンプトを5分間ループし（250 トークン、temperature 0、reasoning xhigh。両エンジンで入力トークン数が同じことを確認した）、3分休んだあと、44K のプロンプトを1回コールドで読ませた。
2026-09-30 に ABBA の順で各2回、macmon で1秒ごとに記録した（[`scripts/erp_repro.py`](scripts/erp_repro.py)、[`run_erp.sh`](run_erp.sh)、生データは [`results/2026-09-m1max-64gb-erp/`](results/2026-09-m1max-64gb-erp/)）。値は2回の平均である。

| | MTPLX M1 build m1.2 | Splash 1.1.0-m1 |
|---|---|---|
| decode（5分間のループ、1リクエストの平均） | 31.5 tok/s | **40.7 tok/s** |
| ループ中の GPU 温度（平均 / 最大） | 88.4 / 94.5 °C | 90.1 / 92.8 °C |
| ループ中に 80 °C を超えていた時間 | 94% | 93% |
| ループ中のパッケージ電力 | 47.1 W | 42.0 W |
| 生成1トークンあたりのエネルギー | 1.74 J | **1.17 J** |
| ループ中のファン（2基の平均） | 3,400 rpm | 1,900 rpm |
| 44K のコールドプロンプト | **364 秒（123 tok/s）** | 438 秒（103 tok/s） |
| その prefill 中の GPU 温度（平均 / 最大） | 85.2 / 97.7 °C | 90.8 / 94.1 °C |
| 44K の後のエンジンの footprint | 29.0 GB（最大 31.1） | 注を参照 |

- Splash の投稿と同じ傾向になった。短いプロンプトでは Splash の decode が 29% 速く、1トークンあたりのエネルギーは約3分の2で、長いプロンプトの読み込みは MLX 系のエンジンが約 20% 速い。どのループでも GPU のクロックは最初から最後まで約 1,295 MHz で、速度低下はなかった（100 W のアダプタ、バッテリー 93〜95%）。
- 温度の絶対値は、その投稿とは比べられない。ここでは macOS の既定のファン制御のままで、両エンジンとも 90 °C 近くになった。その投稿は 80 °C でファンが全開になる曲線を使っていて、80 °C 未満に収まったのは Splash だけだった。prefill 中の GPU センサーの最大値は MTPLX build の方が高く（97〜98 °C）、その投稿が MTPLX について書いたことと一致する。
- 1トークンあたりのエネルギーは、ループ全体（短い prefill も含む）のパッケージ電力を生成トークン数で割った値である。
- Splash は重みをディスクから直接マップするので、`footprint` には約 4.9 GB しか現れない。上の長い文脈の計測での wired メモリは 22〜26 GB だった。

## サンプリング（temperature 1.0）

temperature 0 で 256 トークンという条件は、比べるために揃えた課題で、普段の使い方ではない。
そこで 2K、32K、64K の同じ会話を、モデル本来のサンプリングの設定（`generation_config.json`：temperature 1.0、top-p 0.95、top-k 20）、各ターン最大 512 トークン、8-bit KV で、順番を逆にして2回測った（2026-09-29、Splash も同じセッション）。
サンプリングでは回ごとに生成テキストが変わるので、2〜3ターン目のプロンプトもエンジンと回で異なる。
受理率はエンジン自身の数え方（MTPLX は MTP の draft、Splash は DFlash2 の draft で1回 7 トークン）なので、方式をまたいで比べられない。

| コンテキスト | エンジン | T1 TTFT（コールド） | decode T1 / T2 / T3 (tok/s) | decode 平均 | ドラフト受理率 T1 / T2 / T3 | 生成トークン T1 / T2 / T3 | needle |
|---|---|---|---|---|---|---|---|
| 2K | MTPLX M1 build | 11.3 s | 26.7 / 30.0 / 25.9 | **27.5** | 58% / 70% / 55% | 269 / 426 / 178 | 2/2 |
| 2K | MTPLX upstream | 11.4 s | 23.1 / 25.3 / 21.2 | **23.2** | 58% / 77% / 59% | 314 / 413 / 200 | 2/2 |
| 2K | MTPLX upstream + env | 11.6 s | 24.8 / 27.0 / 27.3 | **26.4** | 56% / 64% / 66% | 294 / 328 / 188 | 2/2 |
| 2K | Splash 1.1.0-m1 | 12.0 s | 29.8 / 36.1 / 32.2 | **32.7** | 31% / 41% / 35% | 294 / 398 / 178 | 2/2 |
| 32K | MTPLX M1 build | 4.1 min | 25.5 / 26.2 / 23.8 | **25.2** | 60% / 63% / 58% | 242 / 292 / 161 | 2/2 |
| 32K | MTPLX upstream | 4.0 min | 17.1 / 12.9 / 12.7 | **14.2** | 56% / 66% / 63% | 276 / 322 / 202 | 2/2 |
| 32K | MTPLX upstream + env | 4.1 min | 17.5 / 12.7 / 12.4 | **14.2** | 53% / 61% / 62% | 292 / 321 / 172 | 2/2 |
| 32K | Splash 1.1.0-m1 | 4.5 min | 22.3 / 30.3 / 24.9 | **25.8** | 28% / 41% / 33% | 382 / 382 / 194 | 2/2 |
| 64K | MTPLX M1 build | 9.5 min | 24.2 / 24.3 / 22.8 | **23.8** | 65% / 66% / 62% | 282 / 298 / 172 | 2/2 |
| 64K | MTPLX upstream | 9.7 min | 8.3 / 9.0 / 9.3 | **8.9** | 55% / 70% / 60% | 322 / 330 / 208 | 2/2 |
| 64K | MTPLX upstream + env | 10.2 min | 7.6 / 8.8 / 8.1 | **8.2** | 56% / 69% / 61% | 295 / 298 / 204 | 2/2 |
| 64K | Splash 1.1.0-m1 | 10.8 min | 20.8 / 24.1 / 20.9 | **22.0** | 30% / 37% / 31% | 303 / 384 / 172 | 2/2 |

- **2K（32.7 と 27.5 tok/s）と 32K（25.8 と 25.2）では Splash が速く、64K（22.0 と 23.8）では M1 build が速い**。command buffer を広げていた最初の M1 build（m1）は、ここで 32.3 / 27.3 / 24.1 だった。
- **サンプリングで速さが大きく変わるのは Splash の方**：Splash は temperature 0 より 2K で 1%、32K で 5%、64K で 9% 遅い。M1 build は 2K で 6%、32K〜64K で 1〜2% 遅い。
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
- **短い文脈では、Splash は近い時間で2倍の位置を verify する**：2K で、8 位置の verify と 7 トークンの draft が1回 105 ms だった。M1 build は、4 位置の verify と3トークンの MTP の draft で1ラウンド 93〜98 ms である（m1.2 の verify と draft の時間）。
- **1回の時間は、プロンプトが長いほど大きく伸びる**：Splash は 2K の 105 ms から 128K の 189〜197 ms へ、M1 build は 93〜98 ms から 141〜144 ms へ伸びた。
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
  - MTPLX upstream と upstream + env は 2026-09-29 に一緒に測った（測り直しは起きなかった）。128K は1回だけで、command buffer ありの upstream + env はメモリ不足で失敗し、公開した upstream + env（command buffer なし）の 128K はその後に測った。128K の canary は、各エンジンの 2K〜64K の最良値から 2.4% 以内だった。
  - M1 build（v2.12.0-m1.2）は 2026-09-30 に同じ計画で単独で測った。先に 2K の確認を行い、canary が 29.9 tok/s 以上であること（実際は 30.6）を条件にした。同じ日の午後の最初の試みは、マシンが約 5% 遅い状態（canary 28.0〜29.1）だったので止め、その行は `rows-fork-m12-aborted-20260930.jsonl` に移した。使った回では、2回目の 2K と 8K の canary がその回の最良値より 3.1% 低く、1回ずつ測り直した（測り直しも 3% 低く、測り直した値を使った）。128K の canary は最良値より 2.6% 低かった。最初のリリース（m1）は 2026-09-29 に他の MTPLX の列と一緒に測り、`mtplx-fork-m1` として残した。
  - oMLX、Splash、TensorFold は、それより前のセッションで同じ計画で測った（詳細と canary は [METHODOLOGY.md](METHODOLOGY.md)）。
- **回ごとのばらつき**：2026-09-29 の計測で、2回測ったセルの decode の差は中央値で 0.7% だった。
  最大は upstream + env の 64K の1ターン（生成テキストはバイト単位で同じだが 5.2 と 8.7 tok/s。verify の時間が 47 秒と 27 秒だった）で、このセルの2ターン目の中央値は 6.9 になった。
  MTPLX の列は、すべてのターンで回ごとの出力がバイト単位で一致した。

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
