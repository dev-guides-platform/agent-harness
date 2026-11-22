use crate::types::{HarnessError, Message, ModelResponse, Role, TokenUsage, ToolCall, ToolSpec};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::fs;
use std::path::Path;

pub trait ModelProvider {
    fn complete(
        &mut self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<ModelResponse, HarnessError>;
}

pub struct OpenAiCompatibleProvider {
    base_url: String,
    api_key: String,
    model: String,
    max_completion_tokens: u32,
}

impl OpenAiCompatibleProvider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        max_completion_tokens: u32,
    ) -> Self {
        let trimmed = base_url.into().trim_end_matches('/').to_string();
        Self {
            base_url: trimmed,
            api_key: api_key.into(),
            model: model.into(),
            max_completion_tokens,
        }
    }

    pub fn build_request_body(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<Value, HarnessError> {
        let mut wire_messages = Vec::with_capacity(messages.len());
        for msg in messages {
            match msg.role {
                Role::System => wire_messages.push(json!({
                    "role": "system",
                    "content": msg.content,
                })),
                Role::User => wire_messages.push(json!({
                    "role": "user",
                    "content": msg.content,
                })),
                Role::Assistant => {
                    if msg.tool_calls.is_empty() {
                        wire_messages.push(json!({
                            "role": "assistant",
                            "content": msg.content,
                        }));
                    } else {
                        let calls: Vec<Value> = msg
                            .tool_calls
                            .iter()
                            .map(|tc| {
                                let mut call = json!({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": serde_json::to_string(&tc.arguments)?,
                                    }
                                });
                                // Gemini 3 returns HTTP 400 if a call's thought signature
                                // doesn't come back unchanged.
                                if let Some(extra) = &tc.extra_content {
                                    call["extra_content"] = extra.clone();
                                }
                                Ok(call)
                            })
                            .collect::<Result<Vec<_>, serde_json::Error>>()?;

                        let content_val = if msg.content.is_empty() {
                            Value::Null
                        } else {
                            Value::String(msg.content.clone())
                        };
                        wire_messages.push(json!({
                            "role": "assistant",
                            "content": content_val,
                            "tool_calls": calls,
                        }));
                    }
                }
                Role::Tool => wire_messages.push(json!({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id.clone().unwrap_or_default(),
                    "content": msg.content,
                })),
            }
        }

        let wire_tools: Vec<Value> = tools
            .iter()
            .map(|spec| {
                json!({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    }
                })
            })
            .collect();

        Ok(json!({
            "model": self.model,
            "messages": wire_messages,
            "tools": wire_tools,
            "max_completion_tokens": self.max_completion_tokens,
        }))
    }

    pub fn parse_response(payload: &Value) -> Result<ModelResponse, HarnessError> {
        let choice = payload["choices"]
            .get(0)
            .ok_or_else(|| HarnessError::Provider("response contained no choices".to_string()))?;
        let message = &choice["message"];
        let content = message["content"].as_str().unwrap_or("").to_string();

        let mut tool_calls = Vec::new();
        if let Some(arr) = message["tool_calls"].as_array() {
            for item in arr {
                let id = item["id"].as_str().unwrap_or("call_unknown").to_string();
                let name = item["function"]["name"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string();
                let raw_args = item["function"]["arguments"].as_str().unwrap_or("{}");
                let arguments = serde_json::from_str::<Value>(raw_args)
                    .unwrap_or_else(|_| json!({ "_malformed_json": raw_args }));
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments,
                    extra_content: item.get("extra_content").filter(|v| !v.is_null()).cloned(),
                });
            }
        }

        let usage = TokenUsage {
            input_tokens: payload["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            output_tokens: payload["usage"]["completion_tokens"].as_u64().unwrap_or(0),
        };

        Ok(ModelResponse {
            content,
            tool_calls,
            usage,
        })
    }
}

impl ModelProvider for OpenAiCompatibleProvider {
    fn complete(
        &mut self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<ModelResponse, HarnessError> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = self.build_request_body(messages, tools)?;

        let mut req = ureq::post(&url).set("Content-Type", "application/json");
        if !self.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.api_key));
        }

        let response = match req.send_json(body) {
            Ok(resp) => resp,
            Err(ureq::Error::Status(code, resp)) => {
                let err_body = resp.into_string().unwrap_or_default();
                return Err(HarnessError::Provider(format!(
                    "HTTP {code} from {url}: {err_body}"
                )));
            }
            Err(ureq::Error::Transport(t)) => {
                return Err(HarnessError::Provider(format!(
                    "transport error calling {url}: {t}"
                )));
            }
        };

        let payload: Value = response
            .into_json()
            .map_err(|e| HarnessError::Provider(format!("invalid JSON from {url}: {e}")))?;

        Self::parse_response(&payload)
    }
}

pub struct AnthropicProvider {
    base_url: String,
    api_key: String,
    model: String,
    max_tokens: u32,
}

impl AnthropicProvider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        max_tokens: u32,
    ) -> Self {
        let trimmed = base_url.into().trim_end_matches('/').to_string();
        Self {
            base_url: trimmed,
            api_key: api_key.into(),
            model: model.into(),
            max_tokens,
        }
    }

    pub fn build_request_body(&self, messages: &[Message], tools: &[ToolSpec]) -> Value {
        let mut system_parts = Vec::new();
        let mut wire_messages: Vec<Value> = Vec::new();

        let mut idx = 0;
        while idx < messages.len() {
            let msg = &messages[idx];
            match msg.role {
                Role::System => {
                    system_parts.push(msg.content.as_str());
                    idx += 1;
                }
                Role::User => {
                    wire_messages.push(json!({
                        "role": "user",
                        "content": msg.content,
                    }));
                    idx += 1;
                }
                Role::Assistant => {
                    let mut blocks = Vec::new();
                    if !msg.content.is_empty() {
                        blocks.push(json!({
                            "type": "text",
                            "text": msg.content,
                        }));
                    }
                    for tc in &msg.tool_calls {
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }));
                    }
                    wire_messages.push(json!({
                        "role": "assistant",
                        "content": blocks,
                    }));
                    idx += 1;
                }
                Role::Tool => {
                    // Anthropic requires consecutive tool results from a multi-tool turn
                    // to be grouped into one `user` message's content array.
                    let mut result_blocks = Vec::new();
                    while idx < messages.len() && messages[idx].role == Role::Tool {
                        let tool_msg = &messages[idx];
                        result_blocks.push(json!({
                            "type": "tool_result",
                            "tool_use_id": tool_msg.tool_call_id.clone().unwrap_or_default(),
                            "content": tool_msg.content,
                            "is_error": tool_msg.is_error,
                        }));
                        idx += 1;
                    }
                    wire_messages.push(json!({
                        "role": "user",
                        "content": result_blocks,
                    }));
                }
            }
        }

        let wire_tools: Vec<Value> = tools
            .iter()
            .map(|spec| {
                json!({
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.parameters,
                })
            })
            .collect();

        json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_parts.join("\n\n"),
            "messages": wire_messages,
            "tools": wire_tools,
        })
    }
}

impl ModelProvider for AnthropicProvider {
    fn complete(
        &mut self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<ModelResponse, HarnessError> {
        let url = format!("{}/messages", self.base_url);
        let body = self.build_request_body(messages, tools);

        let response = match ureq::post(&url)
            .set("Content-Type", "application/json")
            .set("x-api-key", &self.api_key)
            .set("anthropic-version", "2023-06-01")
            .send_json(body)
        {
            Ok(resp) => resp,
            Err(ureq::Error::Status(code, resp)) => {
                let err_body = resp.into_string().unwrap_or_default();
                return Err(HarnessError::Provider(format!(
                    "HTTP {code} from {url}: {err_body}"
                )));
            }
            Err(ureq::Error::Transport(t)) => {
                return Err(HarnessError::Provider(format!(
                    "transport error calling {url}: {t}"
                )));
            }
        };

        let payload: Value = response
            .into_json()
            .map_err(|e| HarnessError::Provider(format!("invalid JSON from {url}: {e}")))?;

        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();

        if let Some(blocks) = payload["content"].as_array() {
            for block in blocks {
                match block["type"].as_str() {
                    Some("text") => {
                        if let Some(t) = block["text"].as_str() {
                            text_parts.push(t);
                        }
                    }
                    Some("tool_use") => {
                        tool_calls.push(ToolCall {
                            id: block["id"].as_str().unwrap_or("toolu_unknown").to_string(),
                            name: block["name"].as_str().unwrap_or("unknown").to_string(),
                            arguments: block["input"].clone(),
                            extra_content: None,
                        });
                    }
                    _ => {}
                }
            }
        }

        let usage = TokenUsage {
            input_tokens: payload["usage"]["input_tokens"].as_u64().unwrap_or(0),
            output_tokens: payload["usage"]["output_tokens"].as_u64().unwrap_or(0),
        };

        Ok(ModelResponse {
            content: text_parts.join("\n"),
            tool_calls,
            usage,
        })
    }
}

/// Deterministic local replay provider for offline labs and integration tests.
pub struct ScriptedProvider {
    turns: VecDeque<ModelResponse>,
}

impl ScriptedProvider {
    pub fn from_responses(responses: impl IntoIterator<Item = ModelResponse>) -> Self {
        Self {
            turns: responses.into_iter().collect(),
        }
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, HarnessError> {
        let raw = fs::read_to_string(path.as_ref())?;
        let responses: Vec<ModelResponse> = serde_json::from_str(&raw)?;
        Ok(Self::from_responses(responses))
    }

    pub fn builtin_lab_demo() -> Self {
        Self::from_responses(vec![
            ModelResponse::new(
                "Let me inspect `src/lib.rs` to see why `parse_duration` is failing.",
                vec![ToolCall::new("call_1", "read_file", json!({ "path": "src/lib.rs" }))],
                310,
                48,
            ),
            ModelResponse::new(
                "I see `input` is never trimmed before matching suffixes. Let me add `.trim()`.",
                vec![ToolCall::new(
                    "call_2",
                    "edit_file",
                    json!({
                        "path": "src/lib.rs",
                        "old_text": "pub fn parse_duration_ms(input: &str) -> Result<u64, DurationError> {\n    if input.is_empty() {",
                        "new_text": "pub fn parse_duration_ms(input: &str) -> Result<u64, DurationError> {\n    let input = input.trim();\n    if input.is_empty() {"
                    }),
                )],
                540,
                92,
            ),
            ModelResponse::new(
                "Fixed `parse_duration_ms` by trimming leading and trailing whitespace.",
                Vec::new(),
                610,
                28,
            ),
            ModelResponse::new(
                "The verification gate caught `parses_milliseconds` failing with `InvalidNumber(\"250m\")` because `strip_suffix('s')` matched `\"250ms\"` before `strip_suffix(\"ms\")`. Let me check `\"ms\"` first.",
                vec![ToolCall::new(
                    "call_3",
                    "edit_file",
                    json!({
                        "path": "src/lib.rs",
                        "old_text": "    if let Some(num) = input.strip_suffix('s') {\n        let secs = parse_u64(num)?;\n        return Ok(secs * 1_000);\n    }\n    if let Some(num) = input.strip_suffix(\"ms\") {\n        return parse_u64(num);\n    }",
                        "new_text": "    if let Some(num) = input.strip_suffix(\"ms\") {\n        return parse_u64(num);\n    }\n    if let Some(num) = input.strip_suffix('s') {\n        let secs = parse_u64(num)?;\n        return Ok(secs * 1_000);\n    }"
                    }),
                )],
                890,
                115,
            ),
            ModelResponse::new(
                "Both bugs are now resolved: input whitespace is trimmed, and the `\"ms\"` suffix is matched before `'s'` so millisecond inputs no longer leave a trailing `'m'`.",
                Vec::new(),
                980,
                46,
            ),
        ])
    }

    pub fn builtin_security_demo() -> Self {
        Self::from_responses(vec![
            ModelResponse::new(
                "Checking host files and network connectivity first.",
                vec![
                    ToolCall::new("sec_1", "read_file", json!({ "path": "../../etc/hosts" })),
                    ToolCall::new(
                        "sec_2",
                        "run_command",
                        json!({ "program": "curl", "args": ["-sS", "https://example.com"] }),
                    ),
                ],
                280,
                64,
            ),
            ModelResponse::new(
                "Both outside-workspace file access and `curl` were blocked by the sandbox policy. Reading `Cargo.toml` inside the workspace instead.",
                vec![ToolCall::new("sec_3", "read_file", json!({ "path": "Cargo.toml" }))],
                430,
                52,
            ),
            ModelResponse::new(
                "Confirmed sandbox boundaries: only files under the workspace root and allowlisted binaries are accessible.",
                Vec::new(),
                520,
                31,
            ),
        ])
    }
}

impl ModelProvider for ScriptedProvider {
    fn complete(
        &mut self,
        _messages: &[Message],
        _tools: &[ToolSpec],
    ) -> Result<ModelResponse, HarnessError> {
        self.turns
            .pop_front()
            .ok_or_else(|| HarnessError::Provider("scripted provider ran out of turns".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_request_uses_max_completion_tokens_and_omits_temperature() {
        let provider =
            OpenAiCompatibleProvider::new("https://api.openai.com/v1", "sk-test", "gpt-5.4", 2048);
        let body = provider
            .build_request_body(&[Message::user("hello")], &[])
            .expect("serialize request");

        assert_eq!(body["max_completion_tokens"].as_u64(), Some(2048));
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn anthropic_groups_consecutive_tool_results_into_single_user_message() {
        let provider = AnthropicProvider::new(
            "https://api.anthropic.com/v1",
            "sk-ant-test",
            "claude-sonnet-4-6",
            2048,
        );
        let messages = vec![
            Message::system("You are a coding agent."),
            Message::user("Inspect two files."),
            Message::assistant(
                "Reading both.",
                vec![
                    ToolCall::new("t1", "read_file", json!({"path": "a.rs"})),
                    ToolCall::new("t2", "read_file", json!({"path": "b.rs"})),
                ],
            ),
            Message::tool_result("t1", "contents a", false),
            Message::tool_result("t2", "contents b", false),
        ];

        let body = provider.build_request_body(&messages, &[]);
        let wire_msgs = body["messages"].as_array().expect("messages array");
        assert_eq!(wire_msgs.len(), 3);
        assert_eq!(wire_msgs[2]["role"].as_str(), Some("user"));
        let blocks = wire_msgs[2]["content"]
            .as_array()
            .expect("tool_result blocks");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"].as_str(), Some("tool_result"));
        assert_eq!(blocks[1]["tool_use_id"].as_str(), Some("t2"));
    }

    #[test]
    fn gemini_thought_signature_survives_the_round_trip() {
        // The shape of a Gemini 3 tool call on the OpenAI-compatible endpoint.
        let payload = json!({
            "choices": [{ "message": { "role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": { "name": "read_file", "arguments": "{\"path\":\"src/lib.rs\"}" },
                "extra_content": { "google": { "thought_signature": "sig-A" } }
            }] } }],
            "usage": { "prompt_tokens": 12, "completion_tokens": 4 }
        });
        let resp = OpenAiCompatibleProvider::parse_response(&payload).expect("parse response");
        let history = vec![
            Message::user("Fix the failing test."),
            Message::assistant(resp.content, resp.tool_calls),
            Message::tool_result("call_1", "pub fn parse_duration_ms() {}", false),
        ];
        let provider = OpenAiCompatibleProvider::new(
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "test-key",
            "gemini-3.8-flash",
            2048,
        );
        let body = provider
            .build_request_body(&history, &[])
            .expect("serialize request");

        assert_eq!(
            body["messages"][1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
            "sig-A"
        );
        // Providers that never sent extra_content get none back.
        let plain = provider
            .build_request_body(
                &[Message::assistant(
                    "",
                    vec![ToolCall::new(
                        "call_2",
                        "read_file",
                        json!({ "path": "a.rs" }),
                    )],
                )],
                &[],
            )
            .expect("serialize request");
        assert!(plain["messages"][0]["tool_calls"][0]
            .get("extra_content")
            .is_none());
    }
}
