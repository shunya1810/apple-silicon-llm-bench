# Apple Silicon の LLM エンジン比較（長い文脈と複数ターン）

[English](README.md) | 日本語

Mac で長いプロンプトを読ませて3ターン会話するとき、推論エンジンによって速さはどれだけ変わるのか。
このリポジトリは、MacBook Pro（M1 Max、64 GB）の上で、4つのエンジンに Qwen3.8-27B を動かして測った結果である。
どのエンジンも投機的デコードと 8-bit の KV キャッシュを使い、各エンジンの OpenAI 互換のストリーミング API を同じ方法で計測した。
MTPLX の2つと oMLX は**同じモデルファイル**を MTP（深さ 3）で動かす。
Splash は専用のパッケージ（mlx-community の 4bit、group size 64 の重みと DFlash2 の draft モデル）しか読めないので、Splash の列は量子化と投機的デコードの方式の違いも含む。

| エンジン | 版 | 備考 |
|---|---|---|
| **MTPLX fork** | [`shunya1810/MTPLX@16751dc`](https://github.com/shunya1810/MTPLX/tree/m1max-longctx) | M1 系向けの長い文脈の改善ブランチ（このベンチの作者が管理） |
| **MTPLX upstream** | [`youssofal/MTPLX@1de2b1c`](https://github.com/youssofal/MTPLX/commit/1de2b1c049136ed117af0c6712baaadd81820b51) | 計測時点の `main`（fork の土台） |
| **oMLX** | [0.6.4](https://github.com/jundot/omlx/releases/tag/v0.6.4) | 計測時点の最新の正式版 |
| **Splash M1 build** | [paperniuk/splash 1.0.2-m1](https://github.com/paperniuk/splash/releases/tag/1.0.2-m1) | [incoai/splash](https://github.com/incoai/splash) を M1/M2 向けの kernel で動かすコミュニティ版。モデルは `incoai/Qwen3.8-27B-Splash`（4bit g64 と DFlash2 の draft）、KV は INT8 |

2K〜64K は各2回（中央値を示す）、128K は1回測った。
Splash は他の3つの後に、別に2回測った。Splash の 128K は完了しなかった（後述）。

## 要点（8-bit KV）

- **decode**：128K では MTPLX fork が **16.7 tok/s**、oMLX が 7.4、MTPLX upstream が 4.9 だった（3ターンの平均）。64K では 20.9、10.7、8.7 だった。
  fork の差はプロンプトが長いほど開き、2K〜8K で +13〜35%、32K で約 1.5 倍、64K で 2.0〜2.4 倍、128K で 2.3〜3.4 倍になる。
- **続きのターン**：128K の2〜3ターン目の最初のトークンまでの時間（TTFT）は、fork が **2.2 秒**、upstream が 3.5〜4.2 秒、oMLX が 1.6 分と 7.9 秒だった（64K では 1.4 秒、2.1 秒、1.0 分と 4.8 秒）。
  oMLX はプレフィックスキャッシュを 4,096 トークン単位で SSD に保存し、このモデル系統では、プロンプトが最後に越えたブロックの境界までしか保存できない。
  そのため、次のターンでは残りの部分（最大で約 4K トークン）を読み直す。
- **1ターン目**：64K までは、プロンプトを最初から読む時間が3つとも同じだった（64K で 9.7〜10.2 分、2K で 11.4〜11.5 秒）。
  prefill は同じ MLX の量子化行列積で律速されている（この GPU で、2K のとき約 7.9 TFLOPS）。
  128K では attention の占める割合が大きくなり、fork が 24.9 分、oMLX が 26.3 分、upstream が 30.5 分と差が出た。
- **128K の会話全体**：fork が 25.7 分、oMLX が 29.6 分、upstream が 33.0 分だった。
- **メモリ**：128K の wired メモリの最大値は fork が 47.8 GB、oMLX が 48.5 GB、upstream が 52.7 GB だった（64K では 42.7、39.6、48.4 GB）。
- **Splash M1 build（2K〜64K）**：32K までは decode が最も速く、特に2ターン目で差が大きい（2K で 41.6 tok/s、8K で 31.9、32K で 27.2）。64K では、1ターン目と3ターン目で MTPLX fork の方が速い（21.7 / 21.1 / 19.8 に対して 18.7 / 22.5 / 16.5）。
  最初から読む prefill は遅く（32K で 5.0 分と 4.1 分、64K で 12.8 分と 9.7 分）、続きのターンでは約 300 トークンを読み直す（TTFT は fork の 0.6〜1.4 秒に対して 2.7〜6.3 秒）。
  メモリは最も少なく、どの長さでも wired が 22〜23 GB だった（fork は 33〜43 GB）。
  **128K では会話を始められなかった。** この GPU では 128K を最初から読むのに、Splash のリクエストの制限時間（30 分）より長くかかる（64K で 12.8 分）。
  3回試した。1回目は prefill を 23 分続けたところで Metal のコマンドがエラーで打ち切られた（`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`）。
  2回目は、別の GPU の負荷でマシンの速度が3分の1に落ちた状態で走ったので、数に入れていない。
  3回目は、マシンが空いている状態（canary 26.8 tok/s）で、ちょうど 30.0 分で Splash に打ち切られた（`request timed out`）。
  この制限時間を変える設定は、文書には見当たらない。
- **出力の確認**：3ターン目で needle の行を引用させる質問には、完了したすべての回で、すべてのエンジンが正しく答えた。

## decode の速さ

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-dark.svg">
  <img alt="プロンプト長ごとの decode tok/s（3エンジン）" src="charts/decode-light.svg">
</picture>

## 続きのターン

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/ttft-followup-dark.svg">
  <img alt="2〜3ターン目の TTFT（対数目盛り）" src="charts/ttft-followup-light.svg">
</picture>

## 会話全体

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/conversation-dark.svg">
  <img alt="エンジンとプロンプト長ごとの3ターンの所要時間" src="charts/conversation-light.svg">
</picture>

## メモリ

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/memory-dark.svg">
  <img alt="プロンプト長ごとの wired メモリの最大値" src="charts/memory-light.svg">
</picture>

## 表

2K〜64K の各セルは2回の中央値、128K は1回の値である。
cache の列は、エンジンが返す `usage.prompt_tokens_details.cached_tokens` の値である。
wired メモリはシステム全体の値である。

| 文脈 | エンジン | T1 TTFT（cold） | T2 TTFT | T3 TTFT | decode T1 / T2 / T3（tok/s） | 3ターンの合計 | cache T2 / T3 | 最大 wired | needle |
|---|---|---|---|---|---|---|---|---|---|
| 2K | MTPLX fork | 11.4 s | 0.62 s | 0.61 s | 28.2 / 26.1 / 26.2 | 38.2 s | 1,936 / 2,243 | 32.8 GB | 2/2 |
| 2K | MTPLX upstream | 11.5 s | 0.64 s | 0.64 s | 25.1 / 22.9 / 23.2 | 41.8 s | 1,936 / 2,243 | 34.0 GB | 2/2 |
| 2K | oMLX | 11.4 s | 13.5 s | 15.4 s | 20.4 / 23.5 / 21.0 | 1.2 min | 0 / 0 | 26.9 GB | 2/2 |
| 2K | Splash M1 build | 12.1 s | 2.87 s | 2.68 s | 27.6 / 41.6 / 29.6 | 39.0 s | 1,664 / 1,984 | 21.7 GB | 2/2 |
| 8K | MTPLX fork | 53.8 s | 0.68 s | 0.71 s | 26.5 / 26.7 / 25.1 | 1.4 min | 8,089 / 8,396 | 34.0 GB | 2/2 |
| 8K | MTPLX upstream | 54.2 s | 0.76 s | 0.77 s | 23.5 / 23.2 / 21.7 | 1.4 min | 8,089 / 8,396 | 35.5 GB | 2/2 |
| 8K | oMLX | 56.4 s | 30.0 s | 2.69 s | 20.2 / 19.8 / 18.3 | 2.1 min | 4,096 / 8,192 | 33.2 GB | 2/2 |
| 8K | Splash M1 build | 59.6 s | 3.21 s | 3.05 s | 26.6 / 31.9 / 27.1 | 1.5 min | 7,808 / 8,128 | 21.6 GB | 2/2 |
| 32K | MTPLX fork | 4.1 min | 0.97 s | 0.98 s | 22.4 / 21.3 / 20.9 | 4.7 min | 32,665 / 32,972 | 38.5 GB | 2/2 |
| 32K | MTPLX upstream | 4.1 min | 1.38 s | 1.37 s | 17.1 / 12.5 / 12.0 | 5.0 min | 32,665 / 32,972 | 40.8 GB | 2/2 |
| 32K | oMLX | 4.4 min | 44.6 s | 3.45 s | 14.4 / 14.0 / 14.4 | 6.0 min | 28,672 / 32,768 | 37.0 GB | 2/2 |
| 32K | Splash M1 build | 5.0 min | 4.43 s | 4.18 s | 23.2 / 27.2 / 22.1 | 5.7 min | 32,384 / 32,704 | 23.1 GB | 2/2 |
| 64K | MTPLX fork | 9.7 min | 1.37 s | 1.37 s | 21.7 / 21.1 / 19.8 | 10.3 min | 65,419 / 65,726 | 42.7 GB | 2/2 |
| 64K | MTPLX upstream | 9.9 min | 2.07 s | 2.12 s | 8.8 / 8.7 / 8.4 | 11.3 min | 65,419 / 65,726 | 48.4 GB | 2/2 |
| 64K | oMLX | 10.2 min | 1.0 min | 4.76 s | 10.9 / 10.9 / 10.4 | 12.4 min | 61,440 / 65,536 | 39.6 GB | 2/2 |
| 64K | Splash M1 build | 12.8 min | 5.68 s | 6.34 s | 18.7 / 22.5 / 16.5 | 13.6 min | 65,152 / 65,440 | 23.4 GB | 2/2 |
| 128K | MTPLX fork | 24.9 min | 2.24 s | 2.25 s | 17.8 / 17.2 / 15.0 | 25.7 min | 130,969 / 131,276 | 47.8 GB | 1/1 |
| 128K | MTPLX upstream | 30.5 min | 4.16 s | 3.49 s | 4.6 / 5.1 / 5.0 | 33.0 min | 130,969 / 131,276 | 52.7 GB | 1/1 |
| 128K | oMLX | 26.3 min | 1.6 min | 7.92 s | 7.6 / 7.8 / 6.9 | 29.6 min | 126,976 / 131,072 | 48.5 GB | 1/1 |
| 128K | Splash M1 build | — | — | — | — / — / — | — | — / — | 23.8 GB | — |

**2K：fp16 KV と 8-bit KV**（decode tok/s、ターン1〜3の平均）

| エンジン | fp16 KV | 8-bit KV | 変化 |
|---|---|---|---|
| MTPLX fork | 24.9 | 26.8 | +7.6% |
| MTPLX upstream | 24.8 | 23.7 | -4.5% |
| oMLX | 23.1 | 21.6 | -6.4% |
| Splash M1 build | 33.8 | 33.0 | -2.5% |

生データは [`results/2026-09-m1max-64gb/rows.jsonl`](results/2026-09-m1max-64gb/rows.jsonl)、ターンごとの CSV は [`summary.csv`](results/2026-09-m1max-64gb/summary.csv)、計測環境は [`environment.json`](results/2026-09-m1max-64gb/environment.json) にある。
各ターンの生成テキストは `results/2026-09-m1max-64gb/cells/*/` に置いた。

## 計測の方法

詳細は [METHODOLOGY.md](METHODOLOGY.md)（英語）にある。
要点は次のとおりである。

- 1ターン目には、目標の長さの合成テレメトリのログ（50% の位置に needle の行を1行含む）と質問を送る。
  2〜3ターン目は、前の回答と新しい質問（約 300 トークン）を足していく。
  3ターン目では needle の行の引用を求める。
  各ターンの生成は最大 256 トークンである。
- `temperature 0`、thinking off で測った。
  これは同じ決定的な課題の上でエンジンを比べるための速度であり、サンプリングを有効にすると MTP の受理率が変わり、decode の速さも変わる。
- どのエンジンも同じモデルファイルを読む。
  oMLX は、MTPLX のチェックポイントを symlink で組んだテキスト専用のフォルダ経由で読む（[`scripts/prepare_omlx.py`](scripts/prepare_omlx.py)）。
  重みの変換やコピーはしていない。
- 8-bit KV は各エンジン独自の実装である（MTPLX は affine q8 の paged KV、oMLX は TurboQuant 8-bit）。
  それ以外はプレフィックスキャッシュも含めて各エンジンの既定値である（MTPLX は RAM、oMLX は SSD に 4,096 トークン単位）。
- セルごとにエンジンを起動し直し、キャッシュを消してから、ウォームアップと熱の確認用のリクエスト（canary）を送る。
  2回目はエンジンの順番を逆にした。
  canary がそのエンジンの最良値より 3% 以上遅いセルは、5 分休んでから測り直した（1回だけ起きた）。
  128K は1回だけ、upstream、oMLX、fork の順に測った（fork が最後で、最もマシンが温まった状態になる）。
  128K の canary は、2K〜64K の計測での各エンジンの最良値より、fork が 1.7%、upstream が 2.5%、oMLX が 4.8% 低く、測り直しはしていない。
- 2回の計測の decode の差は、中央値で 1.0%、最大で 10% だった（oMLX の 64K の3ターン目。2回で生成テキストが異なった）。
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
