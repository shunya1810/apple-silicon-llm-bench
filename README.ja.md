# Apple Silicon の LLM エンジン比較（長い文脈と複数ターン）

[English](README.md) | 日本語

Mac で長いプロンプトを読ませて3ターン会話するとき、推論エンジンによって速さはどれだけ変わるのか。
このリポジトリは、MacBook Pro（M1 Max、64 GB）の上で Qwen3.8-27B を複数の推論エンジンで動かし、同じ会話の速さを比べたベンチマークである。
どのエンジンも投機的デコードと 8-bit の KV キャッシュを使い、各エンジンの OpenAI 互換のストリーミング API を同じ方法で計測した。

MTPLX の2つ（fork と upstream）と oMLX は、**同じモデルファイル**を MTP（深さ 3）で動かす。
Splash は専用のパッケージ（mlx-community の 4bit、group size 64 の重みと DFlash2 の draft モデル）しか読めない。
そのため Splash の列は、エンジンの違いに加えて、量子化と投機的デコードの方式の違いも含む。

| エンジン | 版 | 備考 |
|---|---|---|
| **MTPLX fork** | [`shunya1810/MTPLX@4fe8067`](https://github.com/shunya1810/MTPLX/tree/m1max-longctx) | M1 系向けの長い文脈の改善ブランチ（このベンチの作者が管理）。M1 で context-copy を止め（`40b6113`）、M1 のメモリの既定値を小さくした（`4fe8067`）後の版で、2026-09-26 に測り直した（前の行は、生データに `mtplx-fork-16751dc` と `mtplx-fork-40b6113` として残した） |
| **MTPLX upstream** | [`youssofal/MTPLX@1de2b1c`](https://github.com/youssofal/MTPLX/commit/1de2b1c049136ed117af0c6712baaadd81820b51) | 計測時点の `main`（fork の土台） |
| **oMLX** | [0.6.4](https://github.com/jundot/omlx/releases/tag/v0.6.4) | 計測時点の最新の正式版 |
| **Splash M1 build** | [paperniuk/splash 1.0.2-m1](https://github.com/paperniuk/splash/releases/tag/1.0.2-m1) | [incoai/splash](https://github.com/incoai/splash) を M1/M2 向けの kernel で動かすコミュニティ版。モデルは `incoai/Qwen3.8-27B-Splash`（4bit g64 と DFlash2 の draft）、KV は INT8 |
| **Splash source (HEAD)** | [paperniuk/splash@5967821](https://github.com/paperniuk/splash/tree/apple7-m1-kernels) + [a4f7e96](https://github.com/shunya1810/splash/tree/m1-analysis-metrics) | M1 版の未リリースの最新を分析用に手元でビルドしたもの（1回。表と下の節のみ） |

2K〜64K は各2回測り、中央値を載せた。128K は1回である。
Splash のリリース版は、他の3つの後に別に2回測った（128K は完了しなかった。後述）。
Splash のソース版は、分析のために1回だけ測った。

## 主な結果（8-bit KV）

- **decode の速さ**：128K では MTPLX fork が **18.1 tok/s**、oMLX が 7.4、MTPLX upstream が 4.9 だった（3ターンの平均）。64K では 21.9、10.7、8.7 だった。
  fork と他の2つの差はプロンプトが長いほど開き、2K〜8K で +16〜38%、32K で約 1.7 倍、64K で 2.0〜2.5 倍、128K で 2.4〜3.7 倍になる。
- **2〜3ターン目の最初のトークンまでの時間（TTFT）**：128K では fork が **2.1 秒**、upstream が 3.5〜4.2 秒、oMLX が 1.6 分と 7.9 秒だった（64K では 1.4 秒、2.1 秒、1.0 分と 4.8 秒）。
  oMLX はプレフィックスキャッシュを 4,096 トークン単位で SSD に保存し、このモデル系統では、プロンプトが最後に越えたブロックの境界までしか保存できない。
  そのため次のターンでは、境界より後ろの部分（最大で約 4K トークン）を読み直す。
- **1ターン目（最初から読む prefill）**：64K までは、MTPLX と oMLX でほぼ同じ時間だった（64K で 9.9〜10.2 分、2K で 11.4〜11.7 秒）。
  prefill は同じ MLX の量子化行列積で律速されている（この GPU で、2K のとき約 7.9 TFLOPS）。
  128K では attention の占める割合が大きくなり、fork が 25.1 分、oMLX が 26.3 分、upstream が 30.5 分と差が出た。
- **128K の3ターン全体**：fork が 25.9 分、oMLX が 29.6 分、upstream が 33.0 分だった。
- **メモリ**：128K の wired メモリの最大値は、fork が 44.3 GB、oMLX が 48.5 GB、upstream が 52.7 GB だった（64K では 38.0、39.6、48.4 GB）。
  fork は `4fe8067` の既定値（MLX のバッファの cache の上限 1 GiB、1つの会話で保存する状態を2件、MMA の prefill attention を最初の chunk から使う）で、速さを変えずに 32K〜128K の最大値が 3.4〜6.1 GB 下がった（`40b6113` では 37.8 / 44.1 / 48.0 GB）。
- **Splash M1 build（リリース版、2K〜64K）**
  - **decode**：2ターン目で最も速い（2K で fork の 29.0 に対して 41.6 tok/s、8K で 29.2 に対して 31.9）。1ターン目と3ターン目は、32K までは fork に近く（−6%〜+13%）、64K では遅い（fork の 21.8 / 21.4 に対して 18.7 / 16.5）。
  - **prefill と TTFT**：最初から読む prefill は遅い（32K で 5.0 分と 4.2 分、64K で 12.8 分と 9.9 分）。続きのターンでは約 300 トークンを読み直すので、TTFT は fork の 0.6〜1.4 秒に対して 2.7〜6.3 秒だった。
  - **メモリ**：最も少なく、どの長さでも wired が 22〜25 GB だった（fork は 31〜44 GB）。
  - **128K**：会話を始められなかった。この GPU では 128K を最初から読むのに、Splash のリクエストの制限時間（30 分）より長くかかる（64K で 12.8 分）。
    3回試した。1回目は prefill を 23 分続けたところで、Metal のコマンドがエラーで打ち切られた（`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`）。
    2回目は、別の GPU の負荷でマシンの速度が3分の1に落ちた状態で走ったので、数に入れていない。
    3回目は、canary が正常な状態（26.8 tok/s）で走り、ちょうど 30.0 分で Splash に打ち切られた（`request timed out`）。このリリースには、制限時間を変える設定が無い。
- **Splash source (HEAD)**：M1 版の未リリースの最新をソースからビルドしたもの。制限時間が 10,000 秒に上がり、Apple7/8 用の新しい attention の kernel が入っている。
  128K を最後まで完了し（1ターン目 34.3 分、decode 14.4 / 18.5 / 14.6 tok/s）、8K 以上ではリリース版より decode が 10〜15% 速い（64K で 19.3 に対して 22.0 tok/s、ターンの平均）。
  計測は1回だけなので、グラフではビルドの要らないリリース版と並べて示している。
- **出力の確認**：3ターン目では、文脈の中央に1行だけ埋め込んだ特別な行（needle）をそのまま引用させた。完了したすべての回で、すべてのエンジンが正しく答えた。

## decode の速さ

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-dark.svg">
  <img alt="プロンプト長ごとの decode tok/s（5エンジン）" src="charts/decode-light.svg">
</picture>

## 2〜3ターン目の TTFT

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/ttft-followup-dark.svg">
  <img alt="2〜3ターン目の TTFT（対数目盛り、5エンジン）" src="charts/ttft-followup-light.svg">
</picture>

## 3ターンの合計所要時間

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/conversation-dark.svg">
  <img alt="エンジンとプロンプト長ごとの3ターンの所要時間" src="charts/conversation-light.svg">
</picture>

## 最大メモリ使用量

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/memory-dark.svg">
  <img alt="プロンプト長ごとの wired メモリの最大値" src="charts/memory-light.svg">
</picture>

## 計測値の表

2K〜64K の各セルは2回の中央値、128K と Splash のソース版は1回の値である。
cache の列は、エンジンが返す `usage.prompt_tokens_details.cached_tokens` の値である。
wired メモリはシステム全体の値である。
needle の列は、3ターン目で needle を正しく引用した回数である。

| 文脈 | エンジン | T1 TTFT（cold） | T2 TTFT | T3 TTFT | decode T1 / T2 / T3（tok/s） | 3ターンの合計 | cache T2 / T3 | 最大 wired | needle |
|---|---|---|---|---|---|---|---|---|---|
| 2K | MTPLX fork | 11.6 s | 0.62 s | 0.62 s | 27.5 / 29.0 / 26.3 | 37.6 s | 1,936 / 2,243 | 30.8 GB | 2/2 |
| 2K | MTPLX upstream | 11.5 s | 0.64 s | 0.64 s | 25.1 / 22.9 / 23.2 | 41.8 s | 1,936 / 2,243 | 34.0 GB | 2/2 |
| 2K | oMLX | 11.4 s | 13.5 s | 15.4 s | 20.4 / 23.5 / 21.0 | 1.2 min | 0 / 0 | 26.9 GB | 2/2 |
| 2K | Splash M1 build | 12.1 s | 2.87 s | 2.68 s | 27.6 / 41.6 / 29.6 | 39.0 s | 1,664 / 1,984 | 21.7 GB | 2/2 |
| 2K | Splash source (HEAD) | 12.2 s | 2.81 s | 2.62 s | 28.6 / 35.9 / 32.3 | 39.2 s | 1,664 / 1,984 | 21.1 GB | 1/1 |
| 8K | MTPLX fork | 55.1 s | 0.69 s | 0.73 s | 26.7 / 29.2 / 24.3 | 1.4 min | 8,089 / 8,396 | 31.4 GB | 2/2 |
| 8K | MTPLX upstream | 54.2 s | 0.76 s | 0.77 s | 23.5 / 23.2 / 21.7 | 1.4 min | 8,089 / 8,396 | 35.5 GB | 2/2 |
| 8K | oMLX | 56.4 s | 30.0 s | 2.69 s | 20.2 / 19.8 / 18.3 | 2.1 min | 4,096 / 8,192 | 33.2 GB | 2/2 |
| 8K | Splash M1 build | 59.6 s | 3.21 s | 3.05 s | 26.6 / 31.9 / 27.1 | 1.5 min | 7,808 / 8,128 | 21.6 GB | 2/2 |
| 8K | Splash source (HEAD) | 58.6 s | 3.06 s | 2.91 s | 28.5 / 40.1 / 29.2 | 1.4 min | 7,808 / 8,128 | 21.4 GB | 1/1 |
| 32K | MTPLX fork | 4.2 min | 0.97 s | 0.98 s | 24.6 / 24.0 / 23.4 | 4.8 min | 32,665 / 32,972 | 34.4 GB | 2/2 |
| 32K | MTPLX upstream | 4.1 min | 1.38 s | 1.37 s | 17.1 / 12.5 / 12.0 | 5.0 min | 32,665 / 32,972 | 40.8 GB | 2/2 |
| 32K | oMLX | 4.4 min | 44.6 s | 3.45 s | 14.4 / 14.0 / 14.4 | 6.0 min | 28,672 / 32,768 | 37.0 GB | 2/2 |
| 32K | Splash M1 build | 5.0 min | 4.43 s | 4.18 s | 23.2 / 27.2 / 22.1 | 5.7 min | 32,384 / 32,704 | 23.1 GB | 2/2 |
| 32K | Splash source (HEAD) | 4.9 min | 3.97 s | 3.96 s | 25.3 / 29.1 / 27.1 | 5.5 min | 32,384 / 32,704 | 22.2 GB | 1/1 |
| 64K | MTPLX fork | 9.9 min | 1.36 s | 1.37 s | 21.8 / 22.4 / 21.4 | 10.4 min | 65,419 / 65,726 | 38.0 GB | 2/2 |
| 64K | MTPLX upstream | 9.9 min | 2.07 s | 2.12 s | 8.8 / 8.7 / 8.4 | 11.3 min | 65,419 / 65,726 | 48.4 GB | 2/2 |
| 64K | oMLX | 10.2 min | 1.0 min | 4.76 s | 10.9 / 10.9 / 10.4 | 12.4 min | 61,440 / 65,536 | 39.6 GB | 2/2 |
| 64K | Splash M1 build | 12.8 min | 5.68 s | 6.34 s | 18.7 / 22.5 / 16.5 | 13.6 min | 65,152 / 65,440 | 23.4 GB | 2/2 |
| 64K | Splash source (HEAD) | 12.3 min | 5.01 s | 5.78 s | 20.7 / 27.1 / 18.3 | 13.0 min | 65,152 / 65,440 | 23.3 GB | 1/1 |
| 128K | MTPLX fork | 25.1 min | 2.14 s | 2.13 s | 17.3 / 18.9 / 18.1 | 25.9 min | 130,969 / 131,276 | 44.3 GB | 1/1 |
| 128K | MTPLX upstream | 30.5 min | 4.16 s | 3.49 s | 4.6 / 5.1 / 5.0 | 33.0 min | 130,969 / 131,276 | 52.7 GB | 1/1 |
| 128K | oMLX | 26.3 min | 1.6 min | 7.92 s | 7.6 / 7.8 / 6.9 | 29.6 min | 126,976 / 131,072 | 48.5 GB | 1/1 |
| 128K | Splash M1 build | — | — | — | — / — / — | — | — / — | 24.7 GB | — |
| 128K | Splash source (HEAD) | 34.3 min | 8.02 s | 7.81 s | 14.4 / 18.5 / 14.6 | 35.3 min | 130,688 / 131,008 | 25.5 GB | 1/1 |

**2K：fp16 KV と 8-bit KV**（decode tok/s、ターン1〜3の平均）

| エンジン | fp16 KV | 8-bit KV | 変化 |
|---|---|---|---|
| MTPLX fork | 26.3 | 27.6 | +4.7% |
| MTPLX upstream | 24.8 | 23.7 | -4.5% |
| oMLX | 23.1 | 21.6 | -6.4% |
| Splash M1 build | 33.8 | 33.0 | -2.5% |
| Splash source (HEAD) | — | 32.3 | — |

2K の fp16 KV は、Splash では BF16 の KV（`--kv-format bf16`）を指す。

生データは [`results/2026-09-m1max-64gb/rows.jsonl`](results/2026-09-m1max-64gb/rows.jsonl)、ターンごとの CSV は [`summary.csv`](results/2026-09-m1max-64gb/summary.csv)、計測環境は [`environment.json`](results/2026-09-m1max-64gb/environment.json) にある。
各ターンの生成テキストは `results/2026-09-m1max-64gb/cells/*/` に置いた。

## Splash が2ターン目で速い理由

Splash の M1 版の現在の最新（[paperniuk/splash@5967821](https://github.com/paperniuk/splash/tree/apple7-m1-kernels)）を手元でビルドし、エンジンが `/status` に出す draft と受理の数を、リクエストの前後で読んだ（verify の回数のカウンターを1つ[足した](https://github.com/shunya1810/splash/tree/m1-analysis-metrics)）。
これで各ターンを verify の回数に分解できる。1回は、DFlash2 による 7 トークンの draft と、8 位置の verify からなる。
1回の計測、8-bit KV の値である。

| 文脈 | ターン | decode tok/s | verify の回数 | draft / 回 | 受理 / 回 | 確定トークン / 回 | 受理率 | ms / 回 |
|---|---|---|---|---|---|---|---|---|
| 2K | 1 | 28.6 | 84 | 7.0 | 2.04 | 3.05 | 29% | 107 |
| 2K | 2 | 35.9 | 67 | 7.0 | 2.82 | 3.82 | 40% | 107 |
| 2K | 3 | 32.3 | 52 | 7.0 | 2.44 | 3.46 | 35% | 107 |
| 8K | 1 | 28.5 | 81 | 7.0 | 2.16 | 3.16 | 31% | 111 |
| 8K | 2 | 40.1 | 57 | 7.0 | 3.49 | 4.49 | 50% | 113 |
| 8K | 3 | 29.2 | 56 | 7.0 | 2.27 | 3.29 | 32% | 113 |
| 32K | 1 | 25.3 | 76 | 7.0 | 2.36 | 3.37 | 34% | 133 |
| 32K | 2 | 29.1 | 66 | 7.0 | 2.86 | 3.88 | 41% | 133 |
| 32K | 3 | 27.1 | 48 | 7.0 | 2.62 | 3.65 | 38% | 136 |
| 64K | 1 | 20.7 | 76 | 7.0 | 2.37 | 3.37 | 34% | 163 |
| 64K | 2 | 27.1 | 58 | 7.0 | 3.40 | 4.41 | 49% | 164 |
| 64K | 3 | 18.3 | 59 | 7.0 | 2.00 | 3.02 | 29% | 166 |
| 128K | 1 | 14.4 | 81 | 7.0 | 2.16 | 3.16 | 31% | 220 |
| 128K | 2 | 18.5 | 62 | 7.0 | 3.13 | 4.13 | 45% | 225 |
| 128K | 3 | 14.6 | 54 | 7.0 | 2.15 | 3.17 | 31% | 218 |

- **draft がよく受理されるのは2ターン目だけ**：受理率は 40〜50% で、1ターン目と3ターン目は 29〜38% だった。そのため1回で確定するトークンが、3.0〜3.7 でなく 3.8〜4.5 になる。
  1回の時間はターンによらないので、この差がそのまま decode の速さの差になる。
  2ターン目の回答は、1ターン目で決まった比較の型を繰り返す。7 トークンの draft は、3トークンの MTP の draft より先まで当てられると考えられる。MTPLX fork は、2ターン目で1ラウンド 2.9〜3.0 トークン、1ターン目と3ターン目で 2.6〜2.9 トークンで、上限は 4 である。
- **短い文脈では、Splash は同じくらいの時間で2倍の位置を verify する**：2K で、8 位置の verify と 7 トークンの draft が1回 107 ms だった。fork は、4 位置の verify と3トークンの MTP の draft で1ラウンド約 100 ms である。
- **1回の時間は、プロンプトが長いほど大きく伸びる**：Splash は 2K の 107 ms から 128K の 220 ms へ、fork は 100 ms から 157 ms へ伸びた。
  1回ごとに履歴全体への attention を、4 位置でなく 8 位置ぶん計算するためと考えられ、64K 以上の1ターン目と3ターン目で Splash が遅れる理由もこれだと推定している（kernel ごとの時間はまだ測っていない）。
- **ソース版で変わったこと**：サーバーの `--request-timeout` の既定値が 10,000 秒に上がり（1.0.2-m1 のリリースは 1,800 秒）、128K を最後まで完了した（1ターン目 34.3 分、続きのターンの TTFT は 8.0 秒と 7.8 秒）。
  Apple7/8 用の attention の新しいコミットも入っていて、64K の decode はリリースの 18.7 / 22.5 / 16.5 に対して 20.7 / 27.1 / 18.3 tok/s だった。

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
  - fork は 2026-09-26 に、`40b6113` と `4fe8067` で同じ計画を単独で測り直した（`4fe8067` の 128K の canary は、2K〜64K の最良値より 0.1% 低かった）。
- **回ごとのばらつき**：2回測ったセルの decode の差は、中央値で 0.9%、最大で 10% だった（oMLX の 64K の3ターン目。2回で生成テキストが異なった）。
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
