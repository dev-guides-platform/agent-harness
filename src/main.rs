use agent_harness::provider::{
    AnthropicProvider, ModelProvider, OpenAiCompatibleProvider, ScriptedProvider,
};
use agent_harness::runner::{HarnessConfig, HarnessRunner};
use agent_harness::sandbox::SandboxPolicy;
use agent_harness::tools::ToolRegistry;
use agent_harness::types::HarnessError;
use clap::{Parser, Subcommand, ValueEnum};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

const LAB_CARGO_TOML: &str = r#"[package]
name = "duration-lab"
version = "0.1.0"
edition = "2021"
"#;

const LAB_LIB_RS: &str = r#"#[derive(Debug, PartialEq, Eq)]
pub enum DurationError {
    Empty,
    InvalidNumber(String),
    UnknownUnit(String),
}

/// Parses human-friendly duration strings such as `"250ms"`, `"5s"`, or `"2m"`
/// into milliseconds.
pub fn parse_duration_ms(input: &str) -> Result<u64, DurationError> {
    if input.is_empty() {
        return Err(DurationError::Empty);
    }
    if let Some(num) = input.strip_suffix('s') {
        let secs = parse_u64(num)?;
        return Ok(secs * 1_000);
    }
    if let Some(num) = input.strip_suffix("ms") {
        return parse_u64(num);
    }
    if let Some(num) = input.strip_suffix('m') {
        let mins = parse_u64(num)?;
        return Ok(mins * 60_000);
    }
    Err(DurationError::UnknownUnit(input.to_string()))
}

fn parse_u64(raw: &str) -> Result<u64, DurationError> {
    raw.parse::<u64>()
        .map_err(|_| DurationError::InvalidNumber(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_seconds_and_minutes() {
        assert_eq!(parse_duration_ms("5s"), Ok(5_000));
        assert_eq!(parse_duration_ms("2m"), Ok(120_000));
    }

    #[test]
    fn trims_surrounding_whitespace() {
        assert_eq!(parse_duration_ms("  12s \n"), Ok(12_000));
    }

    #[test]
    fn parses_milliseconds() {
        assert_eq!(parse_duration_ms("250ms"), Ok(250));
    }

    #[test]
    fn rejects_empty_and_unknown_units() {
        assert_eq!(parse_duration_ms("   "), Err(DurationError::Empty));
        assert_eq!(
            parse_duration_ms("10h"),
            Err(DurationError::UnknownUnit("10h".to_string()))
        );
    }
}
"#;

#[derive(Parser, Debug)]
#[command(
    name = "harness",
    version,
    about = "Minimal security-hardened AI coding harness in Rust"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Scaffold or reset the broken `duration-lab` Rust crate for hands-on experiments
    InitLab {
        /// Destination directory for the sample buggy crate
        #[arg(default_value = "./lab-workspace")]
        dir: PathBuf,

        /// Overwrite `Cargo.toml` and `src/lib.rs` if the directory already exists
        #[arg(long)]
        force: bool,
    },

    /// Run the agent harness loop inside a target workspace
    Run(Box<RunArgs>),
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// Target workspace directory (all file and command tools are confined here)
    #[arg(long, default_value = ".")]
    workspace: PathBuf,

    /// Engineering task for the agent to accomplish
    #[arg(
        long,
        default_value = "Fix the bugs in `src/lib.rs` so every unit test passes."
    )]
    goal: String,

    /// Model backend (`scripted` runs offline with zero API keys)
    #[arg(long, value_enum, default_value_t = ProviderKind::Scripted)]
    provider: ProviderKind,

    /// Built-in scripted scenario when `--provider scripted` is used without `--script`
    #[arg(long, value_enum, default_value_t = ScriptScenario::Bugfix)]
    scenario: ScriptScenario,

    /// Optional JSON file of scripted `ModelResponse` turns
    #[arg(long)]
    script: Option<PathBuf>,

    /// Model identifier for live providers (OpenAI, Gemini, Anthropic, Ollama)
    #[arg(long)]
    model: Option<String>,

    /// Override provider API base URL (e.g. `http://localhost:11434/v1` for Ollama)
    #[arg(long)]
    base_url: Option<String>,

    /// Verification command that must exit 0 before the harness allows completion
    #[arg(long)]
    verify: Option<String>,

    /// Maximum model turns before halting
    #[arg(long, default_value_t = 12)]
    max_turns: usize,

    /// Maximum cumulative input + output tokens before halting
    #[arg(long, default_value_t = 100_000)]
    max_tokens: u64,

    /// Maximum bytes returned from any single tool execution
    #[arg(long, default_value_t = 4_000)]
    max_observation_bytes: usize,

    /// Per-command wall-clock timeout in seconds
    #[arg(long, default_value_t = 30)]
    timeout_secs: u64,

    /// Comma-separated allowlist of executable names for `run_command`
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "cargo,rustc,rustfmt,git,ls"
    )]
    allow_cmd: Vec<String>,

    /// Path to the append-only JSONL trajectory file
    #[arg(long)]
    trace_file: Option<PathBuf>,

    /// Print turn-by-turn harness activity to stderr
    #[arg(long, short)]
    verbose: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum ProviderKind {
    Scripted,
    Openai,
    Gemini,
    Anthropic,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum ScriptScenario {
    Bugfix,
    Security,
}

fn init_lab(dir: &Path, force: bool) -> Result<(), HarnessError> {
    let lib_path = dir.join("src").join("lib.rs");
    if lib_path.exists() && !force {
        return Err(HarnessError::Config(format!(
            "'{}' already exists. Pass `--force` to reset it to the broken starting state.",
            lib_path.display()
        )));
    }
    fs::create_dir_all(dir.join("src"))?;
    fs::write(dir.join("Cargo.toml"), LAB_CARGO_TOML)?;
    fs::write(&lib_path, LAB_LIB_RS)?;
    println!(
        "Initialized broken `duration-lab` crate at {}",
        dir.display()
    );
    Ok(())
}

fn require_env(var: &str) -> Result<String, HarnessError> {
    env::var(var)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| {
            HarnessError::Config(format!(
                "environment variable `{var}` is not set (or use `--provider scripted` for offline mode)"
            ))
        })
}

fn execute() -> Result<ExitCode, HarnessError> {
    let cli = Cli::parse();

    match cli.command {
        Commands::InitLab { dir, force } => {
            init_lab(&dir, force)?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Run(args) => {
            let RunArgs {
                workspace,
                goal,
                provider,
                scenario,
                script,
                model,
                base_url,
                verify,
                max_turns,
                max_tokens,
                max_observation_bytes,
                timeout_secs,
                allow_cmd,
                trace_file,
                verbose,
            } = *args;
            let sandbox =
                SandboxPolicy::new(&workspace, allow_cmd, Duration::from_secs(timeout_secs))?;
            let trace_path =
                trace_file.unwrap_or_else(|| sandbox.workspace_root().join("trace.jsonl"));

            let verify_command = verify.and_then(|raw| {
                let parts: Vec<String> = raw.split_whitespace().map(ToString::to_string).collect();
                if parts.is_empty() {
                    None
                } else {
                    Some(parts)
                }
            });

            let config = HarnessConfig {
                max_turns,
                max_total_tokens: max_tokens,
                max_context_bytes: 32_000,
                max_observation_bytes,
                verify_command,
                trace_path,
                verbose,
            };

            let tools = ToolRegistry::new(sandbox, max_observation_bytes);

            let mut provider_box: Box<dyn ModelProvider> = match provider {
                ProviderKind::Scripted => {
                    if let Some(script_path) = script {
                        Box::new(ScriptedProvider::from_json_file(script_path)?)
                    } else {
                        match scenario {
                            ScriptScenario::Bugfix => {
                                Box::new(ScriptedProvider::builtin_lab_demo())
                            }
                            ScriptScenario::Security => {
                                Box::new(ScriptedProvider::builtin_security_demo())
                            }
                        }
                    }
                }
                ProviderKind::Openai => {
                    let api_key = env::var("OPENAI_API_KEY").unwrap_or_default();
                    let url = base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string());
                    if url.starts_with("https://api.openai.com") && api_key.trim().is_empty() {
                        return Err(HarnessError::Config(
                            "OPENAI_API_KEY is required when calling api.openai.com".to_string(),
                        ));
                    }
                    let chosen_model = model.unwrap_or_else(|| "gpt-5.4".to_string());
                    Box::new(OpenAiCompatibleProvider::new(
                        url,
                        api_key,
                        chosen_model,
                        4096,
                    ))
                }
                ProviderKind::Gemini => {
                    let api_key = require_env("GEMINI_API_KEY")?;
                    let url = base_url.unwrap_or_else(|| {
                        "https://generativelanguage.googleapis.com/v1beta/openai".to_string()
                    });
                    let chosen_model = model.unwrap_or_else(|| "gemini-3.8-flash".to_string());
                    Box::new(OpenAiCompatibleProvider::new(
                        url,
                        api_key,
                        chosen_model,
                        4096,
                    ))
                }
                ProviderKind::Anthropic => {
                    let api_key = require_env("ANTHROPIC_API_KEY")?;
                    let url =
                        base_url.unwrap_or_else(|| "https://api.anthropic.com/v1".to_string());
                    let chosen_model = model.unwrap_or_else(|| "claude-sonnet-4-6".to_string());
                    Box::new(AnthropicProvider::new(url, api_key, chosen_model, 4096))
                }
            };

            let mut runner = HarnessRunner::new(provider_box.as_mut(), tools, config);
            let report = runner.run(&goal)?;

            println!("Stop reason:       {:?}", report.stop_reason);
            println!("Turns used:        {}", report.turns_used);
            println!("Tool calls:        {}", report.tool_calls_made);
            println!("Policy denials:    {}", report.policy_denials);
            println!("Verification runs: {}", report.verification_runs);
            println!(
                "Tokens (in/out):   {} / {}",
                report.total_usage.input_tokens, report.total_usage.output_tokens
            );
            println!("Trajectory log:    {}", report.trace_path.display());
            println!("\nFinal answer:\n{}", report.final_answer);

            if report.stop_reason == agent_harness::types::StopReason::Completed {
                Ok(ExitCode::SUCCESS)
            } else {
                Ok(ExitCode::from(2))
            }
        }
    }
}

fn main() -> ExitCode {
    match execute() {
        Ok(code) => code,
        Err(err) => {
            eprintln!("Error: {err}");
            ExitCode::FAILURE
        }
    }
}
