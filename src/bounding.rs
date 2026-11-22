use crate::types::{Message, Role};

/// Bounds a tool observation to at most `max_bytes` of UTF-8 content by keeping
/// both the head (60%, for compiler headers and first errors) and the tail
/// (40%, for test runner summaries and exit codes), splitting strictly on
/// valid UTF-8 character boundaries.
pub fn bound_observation(text: &str, max_bytes: usize) -> (String, bool) {
    if text.len() <= max_bytes {
        return (text.to_string(), false);
    }

    let head_budget = (max_bytes * 3) / 5;
    let tail_budget = max_bytes.saturating_sub(head_budget);

    let mut head_end = head_budget.min(text.len());
    while head_end > 0 && !text.is_char_boundary(head_end) {
        head_end -= 1;
    }

    let mut tail_start = text.len().saturating_sub(tail_budget);
    while tail_start < text.len() && !text.is_char_boundary(tail_start) {
        tail_start += 1;
    }

    if head_end >= tail_start {
        return (text.to_string(), false);
    }

    let omitted = tail_start - head_end;
    let bounded = format!(
        "{}\n\n[... harness truncated {omitted} bytes to stay within observation budget ...]\n\n{}",
        &text[..head_end],
        &text[tail_start..]
    );
    (bounded, true)
}

/// Compacts older `Role::Tool` outputs in-place when total message content
/// exceeds `max_total_bytes`, preserving the system prompt, the user's goal,
/// and the `keep_recent` most recent messages untouched.
pub fn compact_history(
    messages: &mut [Message],
    max_total_bytes: usize,
    keep_recent: usize,
) -> usize {
    let mut total_bytes: usize = messages.iter().map(|m| m.content.len()).sum();
    if total_bytes <= max_total_bytes {
        return 0;
    }

    let protected_tail_start = messages.len().saturating_sub(keep_recent);
    let mut compacted_count = 0;

    for msg in messages.iter_mut().take(protected_tail_start) {
        if total_bytes <= max_total_bytes {
            break;
        }
        if msg.role != Role::Tool {
            continue;
        }
        if msg.content.starts_with("[compacted by harness:") {
            continue;
        }
        let original_len = msg.content.len();
        if original_len <= 240 {
            continue;
        }

        let mut preview_end = 120.min(original_len);
        while preview_end > 0 && !msg.content.is_char_boundary(preview_end) {
            preview_end -= 1;
        }
        let first_line = msg.content[..preview_end]
            .lines()
            .next()
            .unwrap_or("")
            .trim();
        let replacement = format!(
            "[compacted by harness: {original_len} bytes from earlier turn; preview: \"{first_line}\"]"
        );
        if replacement.len() < original_len {
            total_bytes = total_bytes - original_len + replacement.len();
            msg.content = replacement;
            compacted_count += 1;
        }
    }

    compacted_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_multibyte_utf8_without_panicking_and_keeps_head_and_tail() {
        let arrows = "error[E0308]: ──→ mismatched types\n".repeat(40);
        let input = format!("HEADER_START\n{arrows}SUMMARY_END: 3 passed; 1 failed");

        let (bounded, truncated) = bound_observation(&input, 220);
        assert!(truncated);
        assert!(bounded.starts_with("HEADER_START"));
        assert!(bounded.ends_with("SUMMARY_END: 3 passed; 1 failed"));
        assert!(bounded.contains("harness truncated"));
    }

    #[test]
    fn compacts_old_tool_outputs_while_preserving_recent_turns() {
        let mut history = vec![
            Message::system("system rules"),
            Message::user("fix failing tests"),
            Message::assistant("reading", vec![]),
            Message::tool_result("call_1", "A".repeat(1200), false),
            Message::assistant("testing", vec![]),
            Message::tool_result("call_2", "B".repeat(1200), false),
            Message::assistant("editing", vec![]),
            Message::tool_result("call_3", "RECENT_OUTPUT".repeat(40), false),
        ];

        let count = compact_history(&mut history, 1500, 2);
        assert_eq!(count, 2);
        assert!(history[3].content.starts_with("[compacted by harness:"));
        assert!(history[5].content.starts_with("[compacted by harness:"));
        assert!(history[7].content.starts_with("RECENT_OUTPUT"));
    }
}
