# Agent Harness — Learn What an AI Harness Is by Building One in Rust

A minimal, security-hardened AI coding harness in Rust. It wraps a language model with the parts a model cannot do for itself: sandboxed file and command tools, output limits, a verification gate, and a step-by-step log.

This repo is the hands-on lab for the written guide: [Learn What an AI Harness Is by Building One in Rust](https://www.devguides.com/jordan/learn-what-an-ai-harness-is-by-building-one-in-rust). Read the guide for the full explanation. Use this README to get running.

**No API key is needed to start.** Experiments 1–3 run fully offline with a built-in scripted (replay) model. Connect a real model only in optional Experiment 4.

## What is an AI harness?

Think of the language model as a function that only returns text or tool requests. The **harness** is the deterministic Rust program around it. It owns:

- the turn loop and stop limits (max turns, token budget)
- workspace sandboxing (which files and commands are allowed)
- observation bounding (trimming huge tool output safely)
- tool dispatch that recovers from errors instead of crashing
- provider translation (OpenAI / Gemini / Claude wire formats)
- a `--verify` gate (for example `cargo test`) plus a `trace.jsonl` log

In this lab the harness fixes a deliberately broken crate called `duration-lab`. You will watch it stop too early without verification, then self-correct with `--verify`.

## Step 0: Install Rust

You need a terminal (Terminal on macOS, your distro terminal on Linux, or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows — recommended so paths and line continuations match this README).

### 1. Install the Rust toolchain

On macOS or Linux (including WSL), run:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --component rustfmt,clippy
```

This installs `cargo` (Rust's build tool and dependency manager — like `npm`/`pip` plus a compiler driver), `rustc` (the compiler), `clippy` (linter), and `rustfmt` (formatter). No extra system packages are required by this project.

If your shell says `cargo: command not found`, load Cargo into the current terminal and retry:

```bash
. "$HOME/.cargo/env"
```

On native Windows (without WSL), install [rustup](https://www.rust-lang.org/tools/install) (`rustup-init.exe`, default options) instead, then open a new PowerShell window. Note that backslash (`\`) line continuations and `VAR="..." command` syntax in this README are for bash/zsh; in PowerShell join commands onto one line (see [Troubleshooting](#troubleshooting)).

### 2. Verify the install

```bash
rustc --version
cargo --version
```

Any recent stable Rust works. This project uses Rust edition 2021.

Optional: [`jq`](https://jqlang.org/) makes the `trace.jsonl` log easier to read (`brew install jq` on macOS, `sudo apt install jq` on Debian/Ubuntu). Plain `cat` works without it.

## Step 1: Open the project and build it

Open a terminal **in this folder** (the folder containing `Cargo.toml`). If you downloaded or cloned the project, `cd` into it first, for example:

```bash
cd agent-harness
ls
```

You should see `Cargo.toml`, `src/`, and `tests/`.

Now build:

```bash
cargo build
```

**What just happened?** Cargo read `Cargo.toml` and automatically downloaded and compiled the dependencies (`clap`, `serde`, `serde_json`, `thiserror`, `ureq`; `tempfile` is test-only). There is no separate `npm install`-style step — Cargo does it for you. The first build takes a minute; later builds are fast. Output goes to `target/debug/harness`. You can safely ignore (and never commit) the `target/` folder.

Sanity-check the CLI:

```bash
cargo run --quiet -- --help
cargo run --quiet -- run --help
```

> **How to read these commands:** `cargo run -- ...` means "build, then run the `harness` program with everything after `--`". The first `--` separates Cargo's own flags from your program's flags. `--quiet` just hides Cargo's build chatter. Copy-paste one block at a time and wait for it to finish.

## Step 2: Run the project's own tests (sanity check)

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```

Expected: all tests pass (7 unit tests + 5 integration tests) and Clippy reports no warnings. If this fails, fix the environment before starting the lab — see [Troubleshooting](#troubleshooting).

## Step 3: Create the broken lab crate

```bash
cargo run --quiet -- init-lab
```

This creates `./lab-workspace/` containing a tiny Rust crate `duration-lab` with one function, `parse_duration_ms`, which should turn strings like `"250ms"`, `"5s"`, and `"2m"` into milliseconds. It has **two planted bugs** and 4 unit tests.

## Step 4: See the bugs yourself

```bash
(cd ./lab-workspace && cargo test --quiet)
```

Expected: tests fail. The two bugs are:

1. Input is never trimmed — `"  12s \n"` and `"   "` are mishandled.
2. `'s'` is stripped before `"ms"` — `"250ms"` becomes `"250m"` and fails to parse.

Keep these in mind. The scripted agent fixes Bug 1 first and tries to stop, leaving Bug 2 behind.

## Step 5: Experiment 1 — run without verification (watch it stop too early)

Run the harness with the offline scripted model. `--verbose` prints each turn so you can follow along:

```bash
cargo run --quiet -- run \
  --workspace ./lab-workspace \
  --provider scripted \
  --verbose
```

Expected ending:

```text
Stop reason:       Completed
Turns used:        3
Tool calls:        2
Policy denials:    0
Verification runs: 0
```

The agent fixed only the `.trim()` bug, then stopped. Prove it — one test still fails:

```bash
(cd ./lab-workspace && cargo test --quiet)
```

**Takeaway:** without a verification gate, "the model said it's done" is not proof the work is done.

## Step 6: Experiment 2 — turn on `--verify` (watch it self-heal)

Reset the lab to its broken starting state, delete the old log, and run again with `--verify`. The harness will now refuse to finish until `cargo test` passes:

```bash
cargo run --quiet -- init-lab --force
rm -f ./lab-workspace/trace.jsonl

cargo run --quiet -- run \
  --workspace ./lab-workspace \
  --provider scripted \
  --verify "cargo test --quiet" \
  --verbose
```

Expected: on Turn 3 you see `verification gate FAILED ... — continuing loop`, the agent fixes the `"ms"` bug on Turn 4, and the run ends with:

```text
Stop reason:       Completed
Turns used:        5
Tool calls:        3
Verification runs: 2
```

Confirm all 4 tests now pass:

```bash
(cd ./lab-workspace && cargo test --quiet)
```

## Step 7: Read the trajectory log (`trace.jsonl`)

Every step is appended as one JSON object per line to `trace.jsonl` in the workspace (default: `./lab-workspace/trace.jsonl`). It is **append-only**, so re-running adds more lines — delete it as in Step 6 when you want a clean timeline, or pass `--trace-file <path>`.

```bash
cat ./lab-workspace/trace.jsonl
```

If you installed `jq`, get a compact timeline instead:

```bash
jq -c '{event, turn, tool_name, passed, stop_reason}' ./lab-workspace/trace.jsonl
```

Look for `verification_checked` with `"passed":false` on Turn 3, then `"passed":true` on Turn 5, ending with `run_finished` / `"completed"`.

## Step 8: Experiment 3 — trip the sandbox

Switch the scripted scenario to `security`. The fake model now tries to read `../../etc/hosts` and run `curl` — the sandbox should block both **without crashing** the loop:

```bash
cargo run --quiet -- run \
  --workspace ./lab-workspace \
  --provider scripted \
  --scenario security \
  --trace-file ./lab-workspace/security-trace.jsonl \
  --verbose
```

Expected ending: `Stop reason: Completed` with `Policy denials: 2`. Paths outside `--workspace` and commands outside the allowlist (`cargo,rustc,rustfmt,git,ls` by default) are rejected, and the model recovers on the next turn.

### Reset the lab at any time

```bash
cargo run --quiet -- init-lab --force
rm -f ./lab-workspace/trace.jsonl ./lab-workspace/security-trace.jsonl
```

This overwrites `./lab-workspace/Cargo.toml` and `src/lib.rs` with the original broken code. Without `--force`, `init-lab` refuses to overwrite an existing lab.

## Optional — Experiment 4: connect a real model

Need a fresh lab first (`init-lab --force` as above). Pick **one** provider below. Always keep `--verify` on with live models.

### OpenAI

Create a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys), then:

```bash
export OPENAI_API_KEY="sk-..."

cargo run --quiet -- run \
  --workspace ./lab-workspace \
  --provider openai --model gpt-5.4 \
  --verify "cargo test --quiet" --verbose
```

Defaults if omitted: base URL `https://api.openai.com/v1`, model `gpt-5.4`. The key is required for `api.openai.com`.

### Google Gemini

Create a key in [Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
export GEMINI_API_KEY="AIza..."

cargo run --quiet -- run \
  --workspace ./lab-workspace \
  --provider gemini --model gemini-2.5-pro \
  --verify "cargo test --quiet" --verbose
```

Defaults: base URL `https://generativelanguage.googleapis.com/v1beta/openai`, model `gemini-2.5-pro`.

### Anthropic Claude

Create a key in the [Anthropic Console](https://console.anthropic.com/settings/keys), then:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

cargo run --quiet -- run \
  --workspace ./lab-workspace \
  --provider anthropic --model claude-sonnet-4-6 \
  --verify "cargo test --quiet" --verbose
```

Defaults: base URL `https://api.anthropic.com/v1`, model `claude-sonnet-4-6`.

### Local Ollama (no cloud key)

Install [Ollama](https://ollama.com/), then in a **separate** terminal start the server and pull a model (one-time download):

```bash
ollama serve
ollama pull qwen2.5-coder:14b
```

Back in this project, reuse the OpenAI-compatible provider with a local base URL:

```bash
cargo run --quiet -- run \
  --workspace ./lab-workspace \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder:14b \
  --verify "cargo test --quiet" --verbose
```

> Live runs make network calls, cost tokens, and are nondeterministic — turn counts will differ from the scripted lab. Never commit API keys. For PowerShell, set keys with `$env:OPENAI_API_KEY="sk-..."` (similarly `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`) instead of `export`.

## Use the harness on your own project

The same pattern works on any folder. Point `--workspace` at a **throwaway copy** of your code, describe the goal, and require a passing check:

```bash
cargo run --quiet -- run \
  --workspace /path/to/your-project-copy \
  --goal "Fix the failing tests" \
  --provider openai \
  --verify "cargo test --quiet" \
  --verbose
```

Two gotchas for custom projects:

- The verify command runs inside the sandbox, so its first word must be allowlisted. If your check is not (for example `npm test`), extend the list: `--allow-cmd cargo,npm,node`. Raise `--timeout-secs` for slow suites.
- The agent edits files directly. Review the diff afterwards (`git diff` in that folder) and never point it at your only copy.

## How to read the output

Every `run` ends with a summary. Example:

```text
Stop reason:       Completed
Turns used:        5
Tool calls:        3
Policy denials:    0
Verification runs: 2
Tokens (in/out):   1234 / 200
Trajectory log:    /path/to/lab-workspace/trace.jsonl

Final answer:
Both bugs are now resolved: ...
```

| Field | Meaning |
|---|---|
| `Stop reason` | `Completed` (success), `MaxTurnsReached`, or `TokenBudgetExceeded` (see exit codes below) |
| `Turns used` / `Tool calls` | How many model turns and tool executions happened |
| `Policy denials` | How many file/command requests the sandbox blocked |
| `Verification runs` | How many times the `--verify` command ran (`0` if you omitted `--verify`) |
| `Tokens (in/out)` | Cumulative token usage reported by the provider |
| `Trajectory log` | Path to the `trace.jsonl` audit log |

Exit codes: `0` = `Completed`; `2` = stopped by `MaxTurnsReached` or `TokenBudgetExceeded`; `1` = config/IO/provider error (the message is printed as `Error: ...`).

Inside a run the model has exactly four tools: `read_file`, `write_file`, `edit_file`, and `run_command` (allowlisted binaries only). Tool output is capped at `--max-observation-bytes` (default `4000`, keeping a 60% head + 40% tail on safe UTF-8 boundaries), and long conversations are compacted past 32,000 bytes while keeping the 4 most recent messages intact.

## Project map — read the code in this order

New to Rust or harnesses? Open the files top to bottom. Each file does one job.

| # | File | What it does (one line) |
|---|---|---|
| — | `Cargo.toml` | Package `agent-harness` (binary `harness`) and its dependencies |
| — | `src/lib.rs` | Lists the 7 modules below so tests can import them |
| 1 | `src/types.rs` | Shared messages, tool calls, token counts, stop reasons, errors |
| 2 | `src/sandbox.rs` | Keeps every path inside the workspace; runs only allowlisted commands with a timeout |
| 3 | `src/bounding.rs` | Truncates huge tool output safely (UTF-8 aware); compacts old history |
| 4 | `src/tools.rs` | The 4 tools (`read_file`, `write_file`, `edit_file`, `run_command`); errors become retryable feedback |
| 5 | `src/provider.rs` | Talks to OpenAI-compatible APIs (OpenAI / Gemini / Ollama), Anthropic, and the offline scripted replay |
| 6 | `src/trace.rs` | Appends one JSON event per line to `trace.jsonl` |
| 7 | `src/runner.rs` | The turn loop: ask model → run tools → enforce budgets → `--verify` gate before stopping |
| 8 | `src/main.rs` | CLI (`init-lab`, `run`) and the embedded broken `duration-lab` fixture |
| — | `tests/harness_integration.rs` | End-to-end proofs: unverified vs. verified runs, sandbox denials, timeouts, budgets, HTTP provider |

Generated folders you can ignore: `target/` (build output) and `lab-workspace/` (your editable lab crate).

## Command reference

Run `cargo run --quiet -- --help` any time to see this in your terminal.

### `init-lab` — create or reset the broken crate

```bash
cargo run --quiet -- init-lab [DIR] [--force]
```

- `[DIR]` defaults to `./lab-workspace`.
- `--force` overwrites `Cargo.toml` and `src/lib.rs` if they already exist.

### `run` — run the agent loop

```bash
cargo run --quiet -- run [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--workspace <dir>` | `.` | Folder all tools are confined to (must already exist) |
| `--goal <text>` | `Fix the bugs in src/lib.rs so every unit test passes.` | Task for the agent |
| `--provider <name>` | `scripted` | `scripted` (offline), `openai`, `gemini`, or `anthropic` |
| `--scenario <name>` | `bugfix` | Scripted replay to use: `bugfix` or `security` (ignored if `--script` is set) |
| `--script <file.json>` | — | Custom scripted turns: a JSON array of model responses |
| `--model <id>` | per provider (see Experiment 4) | Model name for live providers |
| `--base-url <url>` | per provider | Override API endpoint (for example Ollama) |
| `--verify "<cmd>"` | off | Command that must exit `0` before the harness accepts completion |
| `--max-turns <n>` | `12` | Stop after this many model turns |
| `--max-tokens <n>` | `100000` | Stop when total input + output tokens exceed this |
| `--max-observation-bytes <n>` | `4000` | Max bytes returned from any single tool call |
| `--timeout-secs <n>` | `30` | Kill any single command after this many seconds |
| `--allow-cmd <a,b,c>` | `cargo,rustc,rustfmt,git,ls` | Binaries `run_command` (and `--verify`) may execute |
| `--trace-file <path>` | `<workspace>/trace.jsonl` | Where to append the JSONL trajectory log |
| `-v`, `--verbose` | off | Print turn-by-turn activity |
| `-h`, `--help` | — | Show help |

## Troubleshooting

**`cargo: command not found`** — run `. "$HOME/.cargo/env"` (macOS/Linux/WSL) or open a new terminal after installing rustup. On native Windows, reopen PowerShell.

**`clippy` not found / `cargo-clippy` error** — add it once with `rustup component add clippy`.

**`'.../src/lib.rs' already exists. Pass --force`** — the lab is already created. Re-run with `cargo run --quiet -- init-lab --force` to reset it.

**Workspace / canonicalize error** — `--workspace` must point to a folder that already exists. Create it first (or run `init-lab` to create `./lab-workspace`).

**Environment variable `..._API_KEY is not set`** — you selected a live provider without exporting its key (see Experiment 4), or switch back to `--provider scripted` for offline mode.

**Exit code `2` / `MaxTurnsReached` or `TokenBudgetExceeded`** — the agent ran out of turns or tokens. Raise `--max-turns` or `--max-tokens` and retry on a reset lab.

**`scripted provider ran out of turns`** — the replay script ended before the agent finished (common with strict `--verify` loops or a too-short `--script` file). Reset the lab and use the built-in scenario, or add more scripted turns.

**Ollama connection refused** — make sure `ollama serve` is running in another terminal and the model was pulled (`ollama pull qwen2.5-coder:14b`).

**Windows PowerShell notes** — join multi-line `\` commands onto one line (or end lines with a backtick `` ` ``). Set keys with `$env:OPENAI_API_KEY="sk-..."`. The lab subshell `(cd ./lab-workspace && cargo test --quiet)` is bash syntax; in PowerShell run `cargo test --quiet --manifest-path ./lab-workspace/Cargo.toml` instead. When in doubt, use WSL and copy this README verbatim.

### Why did the harness behave that way?

| Symptom | What the harness does about it |
|---|---|
| `edit_file` keeps failing | It requires exactly one exact match. Read the file first and copy the text (including indentation) precisely; `0` or `>1` matches are rejected with guidance. |
| A command hangs forever | Commands run directly (never via a shell), output is drained concurrently, and the child is killed at `--timeout-secs` (default 30s). |
| Huge compiler output | Each observation is bounded to `--max-observation-bytes` on UTF-8 boundaries (60% head + 40% tail), so errors at the top and summaries at the bottom survive without panicking. |
| Anthropic `HTTP 400` after multi-tool turns | Consecutive tool results are merged into a single `user` message, as the Anthropic API requires. |

## Next steps

- Re-read the [full DevGuides walkthrough](https://www.devguides.com/jordan/learn-what-an-ai-harness-is-by-building-one-in-rust) alongside `src/runner.rs` and `src/sandbox.rs`.
- Change `--max-observation-bytes` or `--allow-cmd` and watch `trace.jsonl` change.
- Write your own `--script script.json` replay and run it against a fresh lab.

## License

MIT — see `Cargo.toml`.
