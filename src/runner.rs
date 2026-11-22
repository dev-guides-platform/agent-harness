use crate::bounding::{bound_observation, compact_history};
use crate::provider::ModelProvider;
use crate::tools::ToolRegistry;
use crate::trace::{TraceEvent, TraceLogger};
use crate::types::{HarnessError, Message, RunReport, StopReason, TokenUsage};
use std::path::PathBuf;

pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a disciplined software engineering agent operating inside a sandboxed workspace.
Rules:
1. Read relevant files with `read_file` before editing them.
2. Make minimal, targeted changes with `edit_file` or `write_file`.
3. Verify your work with `run_command` (for example `cargo test`) before declaring completion.
4. Respect workspace boundaries: never access paths outside the workspace or unallowlisted commands.";

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub max_turns: usize,
    pub max_total_tokens: u64,
    pub max_context_bytes: usize,
    pub max_observation_bytes: usize,
    pub verify_command: Option<Vec<String>>,
    pub trace_path: PathBuf,
    pub verbose: bool,
}

impl HarnessConfig {
    pub fn new(trace_path: PathBuf) -> Self {
        Self {
            max_turns: 12,
            max_total_tokens: 100_000,
            max_context_bytes: 32_000,
            max_observation_bytes: 4_000,
            verify_command: None,
            trace_path,
            verbose: false,
        }
    }
}

#[derive(Default)]
struct RunCounters {
    tool_calls_made: usize,
    policy_denials: usize,
    verification_runs: usize,
    compactions: usize,
    total_usage: TokenUsage,
}

pub struct HarnessRunner<'a> {
    provider: &'a mut dyn ModelProvider,
    tools: ToolRegistry,
    config: HarnessConfig,
}

impl<'a> HarnessRunner<'a> {
    pub fn new(
        provider: &'a mut dyn ModelProvider,
        tools: ToolRegistry,
        config: HarnessConfig,
    ) -> Self {
        Self {
            provider,
            tools,
            config,
        }
    }

    fn finish(
        &self,
        trace: &mut TraceLogger,
        stop_reason: StopReason,
        turns_used: usize,
        final_answer: String,
        counters: &RunCounters,
    ) -> Result<RunReport, HarnessError> {
        trace.record(&TraceEvent::RunFinished {
            timestamp_ms: TraceLogger::now_ms(),
            stop_reason,
            turns_used,
            tool_calls_made: counters.tool_calls_made,
            policy_denials: counters.policy_denials,
            verification_runs: counters.verification_runs,
            total_usage: counters.total_usage,
        })?;
        Ok(RunReport {
            stop_reason,
            final_answer,
            turns_used,
            tool_calls_made: counters.tool_calls_made,
            policy_denials: counters.policy_denials,
            verification_runs: counters.verification_runs,
            compactions: counters.compactions,
            total_usage: counters.total_usage,
            trace_path: self.config.trace_path.clone(),
        })
    }

    pub fn run(&mut self, goal: &str) -> Result<RunReport, HarnessError> {
        let mut trace = TraceLogger::new(&self.config.trace_path)?;
        let verify_display = self.config.verify_command.as_ref().map(|p| p.join(" "));
        let workspace_display = self.tools.sandbox().workspace_root().display().to_string();

        trace.record(&TraceEvent::RunStarted {
            timestamp_ms: TraceLogger::now_ms(),
            workspace: &workspace_display,
            goal,
            verify_command: verify_display.as_deref(),
            max_turns: self.config.max_turns,
        })?;

        let tool_specs = self.tools.specs();
        let mut messages = vec![
            Message::system(DEFAULT_SYSTEM_PROMPT),
            Message::user(goal.to_string()),
        ];
        let mut counters = RunCounters::default();
        let mut last_answer = String::new();

        for turn in 1..=self.config.max_turns {
            let compacted = compact_history(&mut messages, self.config.max_context_bytes, 4);
            if compacted > 0 {
                counters.compactions += compacted;
                trace.record(&TraceEvent::ContextCompacted {
                    timestamp_ms: TraceLogger::now_ms(),
                    turn,
                    messages_compacted: compacted,
                })?;
            }

            let response = self.provider.complete(&messages, &tool_specs)?;
            counters.total_usage.accumulate(response.usage);

            let call_names: Vec<&str> = response
                .tool_calls
                .iter()
                .map(|c| c.name.as_str())
                .collect();
            trace.record(&TraceEvent::ModelTurn {
                timestamp_ms: TraceLogger::now_ms(),
                turn,
                assistant_text: &response.content,
                tool_call_names: call_names,
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
            })?;

            if self.config.verbose && !response.content.trim().is_empty() {
                eprintln!("[turn {turn}] model: {}", response.content.trim());
            }

            if counters.total_usage.total() > self.config.max_total_tokens {
                return self.finish(
                    &mut trace,
                    StopReason::TokenBudgetExceeded,
                    turn,
                    response.content,
                    &counters,
                );
            }

            last_answer = response.content.clone();
            let requested_calls = response.tool_calls.clone();
            messages.push(Message::assistant(response.content, response.tool_calls));

            if !requested_calls.is_empty() {
                for call in &requested_calls {
                    counters.tool_calls_made += 1;
                    if self.config.verbose {
                        eprintln!("[turn {turn}] tool -> {}({})", call.name, call.arguments);
                    }
                    let outcome = self.tools.execute(call);
                    if outcome.policy_denied {
                        counters.policy_denials += 1;
                    }
                    if self.config.verbose && outcome.is_error {
                        let first = outcome.output.lines().next().unwrap_or("");
                        eprintln!("[turn {turn}]   ! {first}");
                    }
                    trace.record(&TraceEvent::ToolExecuted {
                        timestamp_ms: TraceLogger::now_ms(),
                        turn,
                        tool_call_id: &call.id,
                        tool_name: &call.name,
                        arguments: &call.arguments,
                        is_error: outcome.is_error,
                        truncated: outcome.truncated,
                        policy_denied: outcome.policy_denied,
                    })?;
                    messages.push(Message::tool_result(
                        &call.id,
                        outcome.output,
                        outcome.is_error,
                    ));
                }
                continue;
            }

            // No tool calls requested — run the optional Verification Gate before stopping.
            if let Some(verify_parts) = &self.config.verify_command {
                if let Some((program, args)) = verify_parts.split_first() {
                    counters.verification_runs += 1;
                    let cmd_str = verify_parts.join(" ");
                    let check = self.tools.sandbox().run_command(program, args)?;
                    let passed = !check.timed_out && check.exit_code == Some(0);

                    trace.record(&TraceEvent::VerificationChecked {
                        timestamp_ms: TraceLogger::now_ms(),
                        turn,
                        command: &cmd_str,
                        passed,
                        exit_code: check.exit_code,
                    })?;

                    if !passed {
                        let combined = format!(
                            "stdout:\n{}\nstderr:\n{}",
                            check.stdout.trim_end(),
                            check.stderr.trim_end()
                        );
                        let (bounded, _) =
                            bound_observation(&combined, self.config.max_observation_bytes);
                        if self.config.verbose {
                            eprintln!(
                                "[turn {turn}] verification gate FAILED (`{cmd_str}`, exit {:?}) — continuing loop",
                                check.exit_code
                            );
                        }
                        messages.push(Message::user(format!(
                            "[Verification Gate Failed]\n\
                             Your turn ended without passing `{cmd_str}` (exit code: {:?}).\n\
                             Inspect the failure below, fix the root cause, and try again:\n\n{bounded}",
                            check.exit_code
                        )));
                        continue;
                    }
                    if self.config.verbose {
                        eprintln!("[turn {turn}] verification gate PASSED (`{cmd_str}`)");
                    }
                }
            }

            return self.finish(
                &mut trace,
                StopReason::Completed,
                turn,
                last_answer,
                &counters,
            );
        }

        self.finish(
            &mut trace,
            StopReason::MaxTurnsReached,
            self.config.max_turns,
            last_answer,
            &counters,
        )
    }
}
