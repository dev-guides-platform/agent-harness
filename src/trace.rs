use crate::types::{HarnessError, StopReason, TokenUsage};
use serde::Serialize;
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TraceEvent<'a> {
    RunStarted {
        timestamp_ms: u128,
        workspace: &'a str,
        goal: &'a str,
        verify_command: Option<&'a str>,
        max_turns: usize,
    },
    ContextCompacted {
        timestamp_ms: u128,
        turn: usize,
        messages_compacted: usize,
    },
    ModelTurn {
        timestamp_ms: u128,
        turn: usize,
        assistant_text: &'a str,
        tool_call_names: Vec<&'a str>,
        input_tokens: u64,
        output_tokens: u64,
    },
    ToolExecuted {
        timestamp_ms: u128,
        turn: usize,
        tool_call_id: &'a str,
        tool_name: &'a str,
        arguments: &'a Value,
        is_error: bool,
        truncated: bool,
        policy_denied: bool,
    },
    VerificationChecked {
        timestamp_ms: u128,
        turn: usize,
        command: &'a str,
        passed: bool,
        exit_code: Option<i32>,
    },
    RunFinished {
        timestamp_ms: u128,
        stop_reason: StopReason,
        turns_used: usize,
        tool_calls_made: usize,
        policy_denials: usize,
        verification_runs: usize,
        total_usage: TokenUsage,
    },
}

pub struct TraceLogger {
    path: PathBuf,
    file: File,
}

impl TraceLogger {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, HarnessError> {
        let path_buf = path.as_ref().to_path_buf();
        if let Some(parent) = path_buf.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path_buf)?;
        Ok(Self {
            path: path_buf,
            file,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn now_ms() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    }

    pub fn record(&mut self, event: &TraceEvent<'_>) -> Result<(), HarnessError> {
        let line = serde_json::to_string(event)?;
        writeln!(self.file, "{line}")?;
        self.file.flush()?;
        Ok(())
    }
}
