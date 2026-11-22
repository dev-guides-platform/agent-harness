use crate::types::HarnessError;
use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// The only variables a spawned command inherits. Everything else in the
/// harness's environment, including the API key it read at startup, is dropped.
const INHERITED_ENV: &[&str] = &[
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "RUSTUP_TOOLCHAIN",
];

#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    workspace_root: PathBuf,
    allowed_commands: BTreeSet<String>,
    command_timeout: Duration,
}

impl SandboxPolicy {
    pub fn new(
        workspace_root: impl AsRef<Path>,
        allowed_commands: impl IntoIterator<Item = impl Into<String>>,
        command_timeout: Duration,
    ) -> Result<Self, HarnessError> {
        let canonical_root = fs::canonicalize(workspace_root.as_ref()).map_err(|e| {
            HarnessError::Config(format!(
                "failed to canonicalize workspace root '{}': {e}",
                workspace_root.as_ref().display()
            ))
        })?;

        let allowed = allowed_commands.into_iter().map(Into::into).collect();
        Ok(Self {
            workspace_root: canonical_root,
            allowed_commands: allowed,
            command_timeout,
        })
    }

    pub fn default_for_workspace(workspace_root: impl AsRef<Path>) -> Result<Self, HarnessError> {
        Self::new(
            workspace_root,
            ["cargo", "rustc", "rustfmt", "git", "ls"],
            Duration::from_secs(30),
        )
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub fn allowed_commands(&self) -> &BTreeSet<String> {
        &self.allowed_commands
    }

    /// Resolves a candidate path (including files or parent directories that do
    /// not yet exist) and verifies that its canonical destination stays strictly
    /// inside `workspace_root`.
    pub fn resolve_path(&self, raw_path: &str) -> Result<PathBuf, HarnessError> {
        let trimmed = raw_path.trim();
        if trimmed.is_empty() {
            return Err(HarnessError::PolicyViolation(
                "empty file path is not allowed".to_string(),
            ));
        }
        if trimmed.contains('\0') {
            return Err(HarnessError::PolicyViolation(
                "path contains a NUL byte".to_string(),
            ));
        }

        let input = Path::new(trimmed);
        let joined = if input.is_absolute() {
            input.to_path_buf()
        } else {
            self.workspace_root.join(input)
        };

        // Normalize lexical `.` and `..` components first without touching disk,
        // rejecting attempts to walk above root.
        let mut normalized = PathBuf::new();
        for comp in joined.components() {
            match comp {
                Component::Prefix(p) => normalized.push(p.as_os_str()),
                Component::RootDir => normalized.push(comp.as_os_str()),
                Component::CurDir => {}
                Component::Normal(part) => normalized.push(part),
                Component::ParentDir => {
                    if !normalized.pop() {
                        return Err(HarnessError::PolicyViolation(format!(
                            "path '{raw_path}' escapes workspace root"
                        )));
                    }
                }
            }
        }

        // Walk upward to the deepest existing ancestor so we can canonicalize
        // symlinks even when writing a brand-new file or subdirectory.
        let mut existing = normalized.as_path();
        let mut tail_parts: Vec<&std::ffi::OsStr> = Vec::new();
        while !existing.exists() {
            let Some(name) = existing.file_name() else {
                return Err(HarnessError::PolicyViolation(format!(
                    "cannot resolve ancestor for path '{raw_path}'"
                )));
            };
            tail_parts.push(name);
            let Some(parent) = existing.parent() else {
                return Err(HarnessError::PolicyViolation(format!(
                    "cannot resolve parent directory for path '{raw_path}'"
                )));
            };
            existing = parent;
        }

        let mut canonical = fs::canonicalize(existing).map_err(|e| {
            HarnessError::PolicyViolation(format!(
                "failed to canonicalize '{}' while checking sandbox boundary: {e}",
                existing.display()
            ))
        })?;

        for part in tail_parts.into_iter().rev() {
            canonical.push(part);
        }

        if !canonical.starts_with(&self.workspace_root) {
            return Err(HarnessError::PolicyViolation(format!(
                "path '{raw_path}' resolves to '{}', outside workspace '{}'",
                canonical.display(),
                self.workspace_root.display()
            )));
        }

        Ok(canonical)
    }

    /// Executes an allowlisted binary directly (never via `sh -c`) inside
    /// `workspace_root` with a minimal environment, draining pipes concurrently
    /// and killing the child if it exceeds `command_timeout`.
    pub fn run_command(
        &self,
        program: &str,
        args: &[String],
    ) -> Result<CommandOutput, HarnessError> {
        let binary = program.trim();
        if binary.is_empty() || binary.contains('/') || binary.contains('\\') {
            return Err(HarnessError::PolicyViolation(format!(
                "command '{program}' must be a bare allowlisted binary name, not a path"
            )));
        }

        if !self.allowed_commands.contains(binary) {
            let allowed_list: Vec<&str> =
                self.allowed_commands.iter().map(String::as_str).collect();
            return Err(HarnessError::PolicyViolation(format!(
                "command '{binary}' is not in the sandbox allowlist ({})",
                allowed_list.join(", ")
            )));
        }

        let start = Instant::now();
        let mut child = Command::new(binary)
            .args(args)
            .current_dir(&self.workspace_root)
            .env_clear()
            .envs(
                std::env::vars_os()
                    .filter(|(key, _)| INHERITED_ENV.iter().any(|name| key == *name)),
            )
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                HarnessError::Io(std::io::Error::new(
                    e.kind(),
                    format!("failed to spawn '{binary}': {e}"),
                ))
            })?;

        let mut stdout_pipe = child.stdout.take().expect("stdout piped");
        let mut stderr_pipe = child.stderr.take().expect("stderr piped");

        let stdout_handle = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout_pipe.read_to_end(&mut buf);
            String::from_utf8_lossy(&buf).into_owned()
        });

        let stderr_handle = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr_pipe.read_to_end(&mut buf);
            String::from_utf8_lossy(&buf).into_owned()
        });

        let mut timed_out = false;
        let exit_code = loop {
            if let Some(status) = child.try_wait()? {
                break status.code();
            }
            if start.elapsed() >= self.command_timeout {
                timed_out = true;
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            thread::sleep(Duration::from_millis(10));
        };

        let stdout = stdout_handle.join().unwrap_or_default();
        let stderr = stderr_handle.join().unwrap_or_default();

        Ok(CommandOutput {
            exit_code,
            stdout,
            stderr,
            timed_out,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn blocks_parent_traversal_and_absolute_escape() {
        let dir = TempDir::new().expect("tempdir");
        let policy = SandboxPolicy::default_for_workspace(dir.path()).expect("policy");

        assert!(policy.resolve_path("../outside.txt").is_err());
        assert!(policy.resolve_path("src/../../outside.txt").is_err());
        assert!(policy.resolve_path("/etc/hosts").is_err());

        let ok = policy
            .resolve_path("src/nested/new_file.rs")
            .expect("valid relative path");
        assert!(ok.starts_with(policy.workspace_root()));
    }

    #[cfg(unix)]
    #[test]
    fn blocks_symlink_pointing_outside_workspace() {
        let workspace = TempDir::new().expect("workspace");
        let outside = TempDir::new().expect("outside");
        let secret = outside.path().join("secret.txt");
        fs::write(&secret, "top-secret").expect("write secret");

        let link = workspace.path().join("escape_link.txt");
        std::os::unix::fs::symlink(&secret, &link).expect("create symlink");

        let policy = SandboxPolicy::default_for_workspace(workspace.path()).expect("policy");
        let err = policy
            .resolve_path("escape_link.txt")
            .expect_err("symlink escape must fail");
        assert!(err.to_string().contains("outside workspace"));
    }

    #[test]
    fn blocks_unlisted_binaries_and_path_executables() {
        let dir = TempDir::new().expect("tempdir");
        let policy = SandboxPolicy::default_for_workspace(dir.path()).expect("policy");

        let denied = policy
            .run_command("curl", &["https://example.com".to_string()])
            .expect_err("curl must be denied");
        assert!(denied.to_string().contains("not in the sandbox allowlist"));

        let slash = policy
            .run_command("/bin/ls", &[])
            .expect_err("absolute binary path must be denied");
        assert!(slash.to_string().contains("bare allowlisted binary name"));
    }
}
