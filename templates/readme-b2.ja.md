## 最新: 各エンジンの最新版、thinking あり、同じ重み（2026-10-18〜19）

2026-10-18 時点の最新版の 4 つのエンジンを、同じ MacBook Pro（M1 Max、64 GB）の上で、**thinking あり**、**全エンジンで本体の重みを同じ**にして比べた。
各セルは、合成したテレメトリのプロンプト（下の節と同じ）を読ませる 3 ターンの会話で、各ターンの生成は思考を含めて最大 4,096 トークンである。

| エンジン | バージョン | 重み |
|---|---|---|
| **MTPLX M1 fork** | [`v2.12.2-m1.10`](https://github.com/shunya1810/MTPLX/releases/tag/v2.12.2-m1.10)（コードは `25927ca3`） | 下の MTPLX 用パック。MTP の深さはパックの既定（27B は 3、A3B は 1）、`--profile turbo`、8bit の KV |
| **MTPLX upstream** | [`v2.12.2`](https://github.com/youssofal/MTPLX/releases/tag/v2.12.2)（`9882703f`、計測時の `main`） | fork と同じパックと引数 |
| **oMLX** | [0.7.1rc1](https://github.com/jundot/omlx/releases/tag/v0.7.1rc1)（wheel、mlx 0.32.4） | MTPLX と同じパックを、テキスト専用のシンボリックリンクのビューで読む（MTP ヘッドは mlx-lm の名前に変えた）。MTP の draft は 3 トークン。**KV は fp16**（下を参照） |
| **Splash** | [paperniuk/splash 1.3.2-m1.1](https://github.com/paperniuk/splash/releases/tag/1.3.2-m1.1) | `mlx-community/Qwen3.8-27B-4bit` と `mlx-community/Qwen3.6-35B-A3B-4bit` をそのまま読み、Splash が選ぶ DFlash2 の draft を使う。KV は INT8 |
| oMLX + oQ4e（補足の行） | oMLX 0.7.1rc1 | `Jundot/Qwen3.8-27B-oQ4e-fp16-mtp` と `Jundot/Qwen3.6-35B-A3B-oQ4e-mtp`。oMLX 独自の 4/5/6bit の混合量子化で、**ほかの行と重みが違う** |

**同じ重み。** Splash は mlx-community の 4bit のチェックポイントをそのまま読む。
MTPLX と oMLX の行は、同じ 4bit の量子化コードに Qwen 純正の MTP ヘッドを足したものを読む（MTPLX には MTP ヘッドが必要だが、mlx-community のチェックポイントには入っていない）。
- **27B**: `mlx-community/Qwen3.8-27B-4bit`（全層 4bit、group 64）の bf16 の float（scale、bias、norm）を fp16 にしたもの（MTPLX の forge が M1 向けにするのと同じ処理）に、`Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` の MTP ヘッド（INT4、group 64）を足した。量子化コードは mlx-community のものとバイト単位で同じである。
- **A3B**: `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16` をそのまま使う。本体は、同じく fp16 にした `mlx-community/Qwen3.6-35B-A3B-4bit` とテンソル単位で同じである（2,090 テンソルすべてを比べた。量子化コード 512 個、float 1,578 個）。サイズの差は、MoE の MTP ヘッドとビジョン部分による。

**条件。** thinking あり、`reasoning_effort` は全エンジンで medium（指定しないと Splash はテンプレート既定の xhigh の指示文を入れ、プロンプトが 42 トークン違う）、temperature 1.0、top_p 0.95、top_k 20、seed 7（モデルの `generation_config`）。
2K で、全エンジンが 1 ターン目を同じ 1,678 トークンにすることを確かめた。
2〜3 ターン目では、前のターンの回答を思考なしで返す（`reasoning_content` を返したときの扱いがエンジンごとに違うため。回答だけなら、どのエンジンも同じ次のプロンプトを組み立てる）。
各セルはエンジンを起動し直してから測り、各セル 1 回である。
256K の行は、4,096 トークンのターンを 3 回生成しても 262,144 トークンの窓に収まるよう、プロンプトを 246,000 トークンにした。

**読むときの注意。**
- このテレメトリの文章は当たりやすい。M1 fork の MTP は、ここでは 1 回の verify で 3.5〜3.7 トークン進むが、作者の実際のコーディングエージェントのセッションでは 3.1 トークンである。そのため decode は、どの投機デコードのエンジンでも（Splash の DFlash2 も含めて）エージェントで使うときより高めに出る。1〜2 tok/s の差は、計測ごとのばらつきの範囲である。
- temperature 1.0 では、エンジンごとに思考の長さが違う。そのため会話全体の時間は比べず、3 ターン目の needle の正解は、主に思考が 4,096 トークン以内に終わったかどうかを表す。
- **oMLX は KV を fp16 で動かした。** 0.7.1rc1 の TurboQuant 8bit の KV は M1 で落ちる（量子化した `sdpa_vector_2pass` のカーネルが 1 スレッドグループに 960 スレッドを要求し、M1 の上限は 896）。そのため oMLX は、8bit の行より KV に多くのメモリを使う。
- **27B の oMLX は、毎ターン、プロンプト全体を読み直す**（履歴が変わるところで、ハイブリッドモデルの再帰状態を prefix cache から復元できない）。そこで、oMLX の 27B・246K と、oQ4e の行の 27B・128K と 246K、A3B・246K は、1 ターン目だけを測った（残りのターンは 1 回ごとに 1 時間ほどかかる）。A3B では 128K まで、プロンプトの一部を再利用する。
- **MTPLX upstream は、27B の 246K を処理できなかった。** 1 ターン目の prefill の途中で、メモリガードが止めた（メモリの逼迫が続いたため。wired のピークは 58.4 GB）。
- **M1 fork の行は v2.12.2-m1.10 の計測である（2026-10-20 に測り直した）。ほかのエンジンの行は 2026-10-18〜19 の計測である。** v2.12.2-m1.9 では、fork の 27B の 3 ターン目が、32K、128K、246K でプロンプトの一部（3,061 トークン）または全部（246K: 248,809 トークン、51 分）を読み直していた。2 ターン目の思考が 4,096 トークンの上限で打ち切られたうえ、3 ターン目が届いた時点で、2 ターン目の状態の保存が終わっていなかったためである。246K では、そのターンの MTP の当たりも大きく落ちていた（1 ラウンド 3.5 → 2.2 トークン、decode 12.8 tok/s）。v2.12.2-m1.10 でどちらも直り、3 ターン目は 32K で 1.6 秒、128K で 4.1 秒、246K で 3.3 秒で応答を始め、246K の decode は 18.2 tok/s（1 ラウンド 3.0 トークン）である。fork の A3B の 246K の 1 ターン目は、今回の計測で v2.12.2-m1.9 より 11% 遅かった（973 秒と 871 秒）。同じ Mac の状態で続けて測ると、2 つの版の 246K の読み込みは 291 tok/s と 289 tok/s で同じだったので、この差は計測ごとのばらつきである。

### Qwen3.8-27B（dense）

{TABLES_27B}

### Qwen3.6-35B-A3B（MoE）

{TABLES_A3B}

### 数字から言えること

{TAKEAWAYS}

元のデータ: [`results/2026-10-m1max-64gb-b2/`](results/2026-10-m1max-64gb-b2/)、集計 `python scripts/summarize_b2.py results/2026-10-m1max-64gb-b2 --readme 27b`（または `a3b`）、計画 `plans/b2-27b.json` と `plans/b2-a3b.json`、実行スクリプト `run_b2.sh`。
