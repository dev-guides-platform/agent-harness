use agent_harness::provider::{OpenAiCompatibleProvider, ScriptedProvider};
use agent_harness::runner::{HarnessConfig, HarnessRunner};
use agent_harness::sandbox::SandboxPolicy;
use agent_harness::tools::ToolRegistry;
use agent_harness::types::{ModelResponse, StopReason, TokenUsage, ToolCall};
use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

fn harness_bin() -> &'static str {
    env!("CARGO_BIN_EXE_harness")
}

#[test]
fn unverified_run_stops_early_while_verified_run_self_heals() {
    let dir = TempDir::new().expect("tempdir");
    let lab_dir = dir.path().join("lab");

    // Step 1: Initialize the broken crate with the real `harness` binary.
    let init = Command::new(harness_bin())
        .args(["init-lab", lab_dir.to_str().unwrap()])
        .output()
        .expect("run init-lab");
    assert!(init.status.success());

    // Step 2: Run WITHOUT `--verify`. The model stops on Turn 3 after fixing only `.trim()`.
    let unverified = Command::new(harness_bin())
        .args([
            "run",
            "--workspace",
            lab_dir.to_str().unwrap(),
            "--provider",
            "scripted",
        ])
        .output()
        .expect("run unverified");
    assert!(unverified.status.success());
    let stdout_unverified = String::from_utf8_lossy(&unverified.stdout);
    assert!(stdout_unverified.contains("Turns used:        3"));

    // Confirm `cargo test` inside the workspace STILL fails on `250ms`!
    let test_after_unverified = Command::new("cargo")
        .args(["test", "--quiet"])
        .current_dir(&lab_dir)
        .output()
        .expect("cargo test");
    assert!(
        !test_after_unverified.status.success(),
        "unverified run left millisecond bug unfixed"
    );

    // Step 3: Reset the broken crate and run WITH `--verify "cargo test --quiet"`.
    let reset = Command::new(harness_bin())
        .args(["init-lab", lab_dir.to_str().unwrap(), "--force"])
        .output()
        .expect("reset lab");
    assert!(reset.status.success());

    let verified = Command::new(harness_bin())
        .args([
            "run",
            "--workspace",
            lab_dir.to_str().unwrap(),
            "--provider",
            "scripted",
            "--verify",
            "cargo test --quiet",
        ])
        .output()
        .expect("run verified");
    assert!(
        verified.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&verified.stderr)
    );
    let stdout_verified = String::from_utf8_lossy(&verified.stdout);
    assert!(stdout_verified.contains("Turns used:        5"));
    assert!(stdout_verified.contains("Verification runs: 2"));

    // Confirm `cargo test` now passes all 4 unit tests!
    let test_after_verified = Command::new("cargo")
        .args(["test", "--quiet"])
        .current_dir(&lab_dir)
        .output()
        .expect("cargo test after verified");
    assert!(test_after_verified.status.success());
}

#[test]
fn sandbox_denials_are_logged_and_returned_without_crashing() {
    let dir = TempDir::new().expect("tempdir");
    fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
    )
    .expect("write Cargo.toml");

    let sandbox = SandboxPolicy::default_for_workspace(dir.path()).expect("sandbox");
    let trace_path = dir.path().join("trace.jsonl");
    let config = HarnessConfig::new(trace_path.clone());
    let tools = ToolRegistry::new(sandbox, 4_000);
    let mut provider = ScriptedProvider::builtin_security_demo();

    let mut runner = HarnessRunner::new(&mut provider, tools, config);
    let report = runner.run("Audit workspace").expect("runner succeeds");

    assert_eq!(report.stop_reason, StopReason::Completed);
    assert_eq!(report.policy_denials, 2);
    assert_eq!(report.tool_calls_made, 3);

    let trace_raw = fs::read_to_string(&trace_path).expect("read trace");
    let events: Vec<Value> = trace_raw
        .lines()
        .map(|l| serde_json::from_str(l).expect("valid JSONL line"))
        .collect();
    let denied_events = events
        .iter()
        .filter(|e| e["event"] == "tool_executed" && e["policy_denied"] == true)
        .count();
    assert_eq!(denied_events, 2);
}

#[test]
fn enforces_command_timeout_and_kills_hanging_process() {
    let dir = TempDir::new().expect("tempdir");
    let policy =
        SandboxPolicy::new(dir.path(), ["sleep"], Duration::from_millis(120)).expect("policy");

    let out = policy
        .run_command("sleep", &["5".to_string()])
        .expect("run sleep");
    assert!(out.timed_out);
    assert!(out.duration_ms < 2_000);
}

#[test]
fn halts_cleanly_on_max_turns_and_token_budget_circuit_breakers() {
    let dir = TempDir::new().expect("tempdir");

    // 1. Token budget breaker
    let sandbox = SandboxPolicy::default_for_workspace(dir.path()).expect("sandbox");
    let mut cfg = HarnessConfig::new(dir.path().join("budget.jsonl"));
    cfg.max_total_tokens = 500;
    let mut provider = ScriptedProvider::from_responses(vec![ModelResponse {
        content: "Expensive turn".to_string(),
        tool_calls: vec![],
        usage: TokenUsage {
            input_tokens: 400,
            output_tokens: 200,
        },
    }]);
    let mut runner = HarnessRunner::new(&mut provider, ToolRegistry::new(sandbox, 2000), cfg);
    let report = runner.run("test budget").expect("run");
    assert_eq!(report.stop_reason, StopReason::TokenBudgetExceeded);

    // 2. Max turns breaker
    let sandbox2 = SandboxPolicy::default_for_workspace(dir.path()).expect("sandbox2");
    let mut cfg2 = HarnessConfig::new(dir.path().join("turns.jsonl"));
    cfg2.max_turns = 2;
    let mut provider2 = ScriptedProvider::from_responses(vec![
        ModelResponse {
            content: "Loop 1".to_string(),
            tool_calls: vec![ToolCall {
                id: "c1".to_string(),
                name: "read_file".to_string(),
                arguments: json!({"path": "missing.txt"}),
            }],
            usage: TokenUsage::default(),
        },
        ModelResponse {
            content: "Loop 2".to_string(),
            tool_calls: vec![ToolCall {
                id: "c2".to_string(),
                name: "read_file".to_string(),
                arguments: json!({"path": "missing.txt"}),
            }],
            usage: TokenUsage::default(),
        },
    ]);
    let mut runner2 = HarnessRunner::new(&mut provider2, ToolRegistry::new(sandbox2, 2000), cfg2);
    let report2 = runner2.run("test max turns").expect("run2");
    assert_eq!(report2.stop_reason, StopReason::MaxTurnsReached);
}

#[test]
fn openai_compatible_provider_completes_against_local_http_server() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");

    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut received = Vec::new();
        let mut chunk = [0u8; 2048];
        loop {
            let n = stream.read(&mut chunk).expect("read request");
            if n == 0 {
                break;
            }
            received.extend_from_slice(&chunk[..n]);
            if let Some(hdr_end) = received.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&received[..hdr_end]);
                let content_len = headers
                    .lines()
                    .find_map(|line| {
                        let (k, v) = line.split_once(':')?;
                        if k.eq_ignore_ascii_case("content-length") {
                            v.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                if received.len() >= hdr_end + 4 + content_len {
                    break;
                }
            }
        }
        let req_text = String::from_utf8_lossy(&received);
        assert!(req_text.contains("POST /v1/chat/completions"));
        assert!(req_text.contains("Bearer test-key"));

        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "All set from local mock HTTP server."
                }
            }],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 9
            }
        })
        .to_string();

        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(resp.as_bytes()).expect("write response");
        stream.flush().expect("flush response");
        let _ = stream.shutdown(std::net::Shutdown::Write);
    });

    let dir = TempDir::new().expect("tempdir");
    let sandbox = SandboxPolicy::default_for_workspace(dir.path()).expect("sandbox");
    let cfg = HarnessConfig::new(dir.path().join("http_trace.jsonl"));
    let mut provider = OpenAiCompatibleProvider::new(
        format!("http://{addr}/v1"),
        "test-key",
        "local-test-model",
        512,
    );

    let mut runner = HarnessRunner::new(&mut provider, ToolRegistry::new(sandbox, 2000), cfg);
    let report = runner.run("Say hello").expect("HTTP run succeeds");

    server.join().expect("server thread");
    assert_eq!(report.stop_reason, StopReason::Completed);
    assert_eq!(report.final_answer, "All set from local mock HTTP server.");
    assert_eq!(report.total_usage.total(), 51);
}
