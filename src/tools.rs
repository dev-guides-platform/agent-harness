use crate::bounding::bound_observation;
use crate::sandbox::SandboxPolicy;
use crate::types::{HarnessError, ToolCall, ToolSpec};
use serde::Deserialize;
use serde_json::json;
use std::fs;

#[derive(Debug, Clone)]
pub struct ToolExecution {
    pub output: String,
    pub is_error: bool,
    pub truncated: bool,
    pub policy_denied: bool,
}

pub struct ToolRegistry {
    sandbox: SandboxPolicy,
    max_observation_bytes: usize,
}

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
}

#[derive(Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
}

#[derive(Deserialize)]
struct EditFileArgs {
    path: String,
    old_text: String,
    new_text: String,
}

#[derive(Deserialize)]
struct RunCommandArgs {
    program: String,
    #[serde(default)]
    args: Vec<String>,
}

impl ToolRegistry {
    pub fn new(sandbox: SandboxPolicy, max_observation_bytes: usize) -> Self {
        Self {
            sandbox,
            max_observation_bytes,
        }
    }

    pub fn sandbox(&self) -> &SandboxPolicy {
        &self.sandbox
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        let allowed: Vec<&str> = self
            .sandbox
            .allowed_commands()
            .iter()
            .map(String::as_str)
            .collect();

        vec![
            ToolSpec {
                name: "read_file".to_string(),
                description: "Read UTF-8 text from a workspace file with line numbers.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Workspace-relative file path." },
                        "start_line": { "type": "integer", "description": "Optional 1-based start line." },
                        "end_line": { "type": "integer", "description": "Optional 1-based inclusive end line." }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
            },
            ToolSpec {
                name: "write_file".to_string(),
                description: "Create or overwrite a file inside the workspace.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Workspace-relative file path." },
                        "content": { "type": "string", "description": "Complete UTF-8 file contents." }
                    },
                    "required": ["path", "content"],
                    "additionalProperties": false
                }),
            },
            ToolSpec {
                name: "edit_file".to_string(),
                description:
                    "Replace an exact, unique block of text in an existing workspace file."
                        .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Workspace-relative file path." },
                        "old_text": { "type": "string", "description": "Exact existing snippet (must match once)." },
                        "new_text": { "type": "string", "description": "Replacement snippet." }
                    },
                    "required": ["path", "old_text", "new_text"],
                    "additionalProperties": false
                }),
            },
            ToolSpec {
                name: "run_command".to_string(),
                description: format!(
                    "Execute an allowlisted command inside the workspace. Allowed binaries: {}.",
                    allowed.join(", ")
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "program": { "type": "string", "description": "Allowlisted binary name (e.g. 'cargo')." },
                        "args": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Arguments passed directly to the binary."
                        }
                    },
                    "required": ["program"],
                    "additionalProperties": false
                }),
            },
        ]
    }

    /// Executes a model's tool call safely. Tool-level failures and sandbox
    /// blocks are converted into `is_error: true` observations so the model can
    /// self-correct on the next turn instead of crashing the harness.
    pub fn execute(&self, call: &ToolCall) -> ToolExecution {
        let raw_result = match call.name.as_str() {
            "read_file" => self.handle_read_file(&call.arguments),
            "write_file" => self.handle_write_file(&call.arguments),
            "edit_file" => self.handle_edit_file(&call.arguments),
            "run_command" => self.handle_run_command(&call.arguments),
            unknown => Err(HarnessError::Config(format!(
                "unknown tool '{unknown}'; available tools: read_file, write_file, edit_file, run_command"
            ))),
        };

        match raw_result {
            Ok((text, is_error)) => {
                let (bounded, truncated) = bound_observation(&text, self.max_observation_bytes);
                ToolExecution {
                    output: bounded,
                    is_error,
                    truncated,
                    policy_denied: false,
                }
            }
            Err(err) => {
                let policy_denied = matches!(err, HarnessError::PolicyViolation(_));
                let formatted = format!("Tool error: {err}");
                let (bounded, truncated) =
                    bound_observation(&formatted, self.max_observation_bytes);
                ToolExecution {
                    output: bounded,
                    is_error: true,
                    truncated,
                    policy_denied,
                }
            }
        }
    }

    fn handle_read_file(&self, args: &serde_json::Value) -> Result<(String, bool), HarnessError> {
        let parsed: ReadFileArgs = serde_json::from_value(args.clone())?;
        let path = self.sandbox.resolve_path(&parsed.path)?;
        let content = fs::read_to_string(&path)?;

        let lines: Vec<&str> = content.lines().collect();
        let start = parsed.start_line.unwrap_or(1).max(1);
        let end = parsed.end_line.unwrap_or(lines.len()).min(lines.len());

        if lines.is_empty() {
            return Ok(("(empty file)".to_string(), false));
        }
        if start > end {
            return Ok((
                format!(
                    "Requested line range {start}..={end} is empty (file has {} lines).",
                    lines.len()
                ),
                true,
            ));
        }

        let mut numbered = String::new();
        for (idx, line) in lines[(start - 1)..end].iter().enumerate() {
            let line_no = start + idx;
            numbered.push_str(&format!("{line_no:>4} | {line}\n"));
        }
        Ok((numbered, false))
    }

    fn handle_write_file(&self, args: &serde_json::Value) -> Result<(String, bool), HarnessError> {
        let parsed: WriteFileArgs = serde_json::from_value(args.clone())?;
        let path = self.sandbox.resolve_path(&parsed.path)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &parsed.content)?;
        Ok((
            format!("Wrote {} bytes to '{}'.", parsed.content.len(), parsed.path),
            false,
        ))
    }

    fn handle_edit_file(&self, args: &serde_json::Value) -> Result<(String, bool), HarnessError> {
        let parsed: EditFileArgs = serde_json::from_value(args.clone())?;
        if parsed.old_text.is_empty() {
            return Ok((
                "edit_file requires a non-empty 'old_text' target.".to_string(),
                true,
            ));
        }
        let path = self.sandbox.resolve_path(&parsed.path)?;
        let current = fs::read_to_string(&path)?;

        let match_count = current.matches(&parsed.old_text).count();
        if match_count == 0 {
            return Ok((
                format!(
                    "old_text was not found in '{}'. Call read_file first to copy exact lines and indentation.",
                    parsed.path
                ),
                true,
            ));
        }
        if match_count > 1 {
            return Ok((
                format!(
                    "old_text matched {match_count} locations in '{}'. Include more surrounding context lines so the match is unique.",
                    parsed.path
                ),
                true,
            ));
        }

        let updated = current.replacen(&parsed.old_text, &parsed.new_text, 1);
        fs::write(&path, &updated)?;
        Ok((format!("Updated '{}'.", parsed.path), false))
    }

    fn handle_run_command(&self, args: &serde_json::Value) -> Result<(String, bool), HarnessError> {
        let parsed: RunCommandArgs = serde_json::from_value(args.clone())?;
        let out = self.sandbox.run_command(&parsed.program, &parsed.args)?;

        if out.timed_out {
            return Ok((
                format!(
                    "Command '{} {}' timed out after {} ms and was killed.\nstdout:\n{}\nstderr:\n{}",
                    parsed.program,
                    parsed.args.join(" "),
                    out.duration_ms,
                    out.stdout,
                    out.stderr
                ),
                true,
            ));
        }

        let code = out.exit_code.unwrap_or(-1);
        let is_error = code != 0;
        let combined = format!(
            "exit_code: {code} ({} ms)\nstdout:\n{}\nstderr:\n{}",
            out.duration_ms,
            if out.stdout.trim().is_empty() {
                "(empty)"
            } else {
                out.stdout.trim_end()
            },
            if out.stderr.trim().is_empty() {
                "(empty)"
            } else {
                out.stderr.trim_end()
            }
        );
        Ok((combined, is_error))
    }
}
