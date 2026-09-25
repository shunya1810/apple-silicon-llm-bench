# Engine adapters

Each engine is one JSON file. `runner.py` starts the command, waits for the health
URL, drives the OpenAI-compatible `/v1/chat/completions` endpoint with streaming, and
stops the process group afterwards. `${VAR}` is expanded from `config/local.json`,
the environment, and `ROOT` (the repository root).

| field | meaning |
|---|---|
| `id`, `label` | short id used in results; human label |
| `port` | port the engine listens on (each engine has its own) |
| `cmd` | command line; the KV variant's `cmd_extra` is appended |
| `env` | extra environment variables |
| `health_path` | GET path that answers once the server is up |
| `model_id` | `model` field sent in requests |
| `tokenizer_dir` | directory with `tokenizer.json`, used to build prompts of an exact token length |
| `kv_variants` | `{name: {kv_impl, cmd_extra, wipe_before_start}}`; `--kv name` picks one (default `default_kv`) |
| `wipe_before_start` | directories deleted before start (on-disk prefix caches), so every cell starts cold |
| `stats` | how to read engine-side numbers: `mtplx-metrics` (GET /metrics) or `omlx-log` (server log); omit if none |
| `version_cmds` | shell commands whose output is stored with each row (commit, version) |
| `model_artifact`, `mtp_impl` | free text stored with each row, shown in the README tables |

## Adding an engine

1. Copy the closest file, give it a new `id` and `port`.
2. Point `cmd` at the engine's OpenAI-compatible server and load the same model
   artifact if the engine can read it (say so in `model_artifact` if it cannot).
3. Map the KV modes you want to compare to `kv_variants`, and list any on-disk cache
   directory under `wipe_before_start`.
4. Add the id to a plan in `plans/` and, for charts, to `ENGINES`/`NAMES` in
   `scripts/summarize.py` (colors are assigned in that fixed order).

Requests use `temperature 0`, `top_p 1`, `chat_template_kwargs.enable_thinking=false`
and `stream_options.include_usage`. An engine that ignores `chat_template_kwargs`
needs its own way to turn thinking off: fields in the engine file's `request_extra`
object are merged into every request body.
