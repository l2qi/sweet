// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Wire-format DTOs for Anthropic's native `/v1/messages` endpoint.

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sweet_core::{FinishReason, Message, Role, ThinkingContent, ToolCall};

use crate::error::ProviderError;

// ---------------------------------------------------------------------------
// Request DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub(crate) struct MessagesRequest<'a> {
    pub model: &'a str,
    pub max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool<'a>>>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<WireThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
}

/// Anthropic's `output_config` block. Carries the reasoning `effort` level
/// (`low`/`medium`/`high`/...).
#[derive(Debug, Serialize)]
pub(crate) struct OutputConfig<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<&'a str>,
    /// Structured-output schema: `{type:"json_schema", schema}`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireMessage {
    pub role: String,
    pub content: WireContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum WireContent {
    Text(String),
    Blocks(Vec<WireContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum WireContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: WireToolResultContent,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "image")]
    Image { source: WireImageSource },
    #[serde(rename = "document")]
    Document { source: WireDocumentSource },
}

/// Anthropic tool_result content can be a plain string or a content block array.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum WireToolResultContent {
    Text(String),
    Blocks(Vec<WireContentBlock>),
}

#[derive(Debug, Serialize)]
pub(crate) struct WireImageSource {
    r#type: String,
    media_type: String,
    data: String,
}

/// Wire payload for a document content block sent to Anthropic's API.
/// Mirrors `WireImageSource` but uses the `"document"` block type.
#[derive(Debug, Serialize)]
pub(crate) struct WireDocumentSource {
    r#type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum WireThinking {
    #[serde(rename = "enabled")]
    Enabled { budget_tokens: usize },
    #[serde(rename = "adaptive")]
    Adaptive,
    #[serde(rename = "disabled")]
    Disabled,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireTool<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub input_schema: Value,
}

// ---------------------------------------------------------------------------
// Response DTOs (non-streaming)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(crate) struct MessagesResponse {
    #[serde(rename = "id")]
    pub _id: String,
    #[serde(rename = "type")]
    pub _msg_type: String,
    #[serde(rename = "role")]
    pub _role: String,
    pub content: Vec<ContentBlock>,
    #[serde(rename = "model")]
    pub _model: String,
    pub stop_reason: Option<String>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub(crate) enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(rename = "thinking")]
        thinking: String,
        #[serde(rename = "signature")]
        signature: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize, Clone)]
pub(crate) struct Usage {
    pub input_tokens: Option<usize>,
    pub output_tokens: Option<usize>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<usize>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<usize>,
}

impl Usage {
    /// Total input tokens including cache writes and reads. Anthropic reports
    /// `input_tokens` as the *uncached* portion only, so prompt caching would
    /// otherwise undercount the real context size.
    pub(crate) fn total_input(&self) -> Option<usize> {
        self.input_tokens.map(|i| {
            i + self.cache_creation_input_tokens.unwrap_or(0)
                + self.cache_read_input_tokens.unwrap_or(0)
        })
    }
}

// ---------------------------------------------------------------------------
// Streaming DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: StreamMessageMeta },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: StreamDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: Option<MessageDeltaInner>,
        usage: Option<Usage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "error")]
    Error { error: StreamErrorDetail },
    #[serde(other)]
    Ping,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamMessageMeta {
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MessageDeltaInner {
    pub stop_reason: Option<String>,
    #[serde(rename = "stop_sequence")]
    pub _stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum StreamDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
    #[serde(other)]
    Other,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert a slice of sweet-core [`Message`]s into Anthropic's request shape.
///
/// System messages are extracted and joined into the returned `Option<String>`.
/// All remaining messages are mapped into Anthropic `messages`. Thinking
/// blocks on assistant messages are emitted on the wire only when their
/// `ThinkingContent.signature` is `Some` - Anthropic requires a valid
/// signature to verify thinking provenance on multi-turn requests.
pub(crate) fn convert_messages(messages: &[Message]) -> (Option<String>, Vec<WireMessage>) {
    let system_parts: Vec<String> = messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.text_content())
        .collect();
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    };

    let mut out = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        if msg.role == Role::System {
            i += 1;
            continue;
        }

        if msg.role == Role::Tool {
            // Group consecutive tool-result messages into a single user
            // message with multiple tool_result blocks.
            let mut blocks = Vec::new();
            while i < messages.len() && messages[i].role == Role::Tool {
                let tool_msg = &messages[i];
                // A tool result that carries an image (e.g. a screenshot) is
                // sent as a content-block array - Anthropic accepts text and
                // image blocks inside a tool_result. Plain text stays a string.
                let content = if tool_msg.has_images() {
                    WireToolResultContent::Blocks(
                        tool_msg
                            .content
                            .iter()
                            .filter_map(|block| match block {
                                sweet_core::ContentBlock::Text { text } => {
                                    Some(WireContentBlock::Text { text: text.clone() })
                                }
                                sweet_core::ContentBlock::Image { data, media_type } => {
                                    let b64 = base64::prelude::BASE64_STANDARD.encode(data);
                                    Some(WireContentBlock::Image {
                                        source: WireImageSource {
                                            r#type: "base64".to_string(),
                                            media_type: media_type.clone(),
                                            data: b64,
                                        },
                                    })
                                }
                                // Documents are not valid inside a tool_result;
                                // drop them (text blocks carry any context).
                                sweet_core::ContentBlock::File { .. } => None,
                            })
                            .collect(),
                    )
                } else {
                    WireToolResultContent::Text(tool_msg.text_content())
                };
                blocks.push(WireContentBlock::ToolResult {
                    tool_use_id: tool_msg.tool_call_id.clone().unwrap_or_default(),
                    content,
                });
                i += 1;
            }
            out.push(WireMessage {
                role: "user".into(),
                content: WireContent::Blocks(blocks),
            });
            continue;
        }

        if msg.role == Role::User {
            let has_attachments = msg.has_attachments();
            if has_attachments {
                let blocks: Vec<WireContentBlock> = msg
                    .content
                    .iter()
                    .map(|block| match block {
                        sweet_core::ContentBlock::Text { text } => {
                            WireContentBlock::Text { text: text.clone() }
                        }
                        sweet_core::ContentBlock::Image { data, media_type } => {
                            let b64 = base64::prelude::BASE64_STANDARD.encode(data);
                            WireContentBlock::Image {
                                source: WireImageSource {
                                    r#type: "base64".to_string(),
                                    media_type: media_type.clone(),
                                    data: b64,
                                },
                            }
                        }
                        sweet_core::ContentBlock::File {
                            data, media_type, ..
                        } => {
                            let b64 = base64::prelude::BASE64_STANDARD.encode(data);
                            WireContentBlock::Document {
                                source: WireDocumentSource {
                                    r#type: "base64".to_string(),
                                    media_type: media_type.clone(),
                                    data: b64,
                                },
                            }
                        }
                    })
                    .collect();
                out.push(WireMessage {
                    role: "user".into(),
                    content: WireContent::Blocks(blocks),
                });
            } else {
                out.push(WireMessage {
                    role: "user".into(),
                    content: WireContent::Text(msg.text_content()),
                });
            }
            i += 1;
            continue;
        }

        // Role::Assistant
        let mut blocks = Vec::new();

        for tc in &msg.thinking_content {
            if let Some(data) = &tc.redacted_data {
                // A redacted (encrypted) thinking block round-trips as-is.
                blocks.push(WireContentBlock::RedactedThinking { data: data.clone() });
            } else if let Some(sig) = &tc.signature {
                blocks.push(WireContentBlock::Thinking {
                    thinking: tc.text.clone(),
                    signature: sig.clone(),
                });
            } else {
                // Anthropic rejects thinking blocks without a valid signature,
                // so dropping is the only safe option. This typically means the
                // block originated from a different provider (OpenAI/Gemini) or
                // was hand-constructed.
                tracing::warn!(
                    target: "sweet_llm::anthropic",
                    text_len = tc.text.len(),
                    "dropping thinking block with no signature; \
                     not sent to Anthropic"
                );
            }
        }

        let text = msg.text_content();
        if !text.is_empty() {
            blocks.push(WireContentBlock::Text { text });
        }
        for tc in &msg.tool_calls {
            blocks.push(WireContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.name.clone(),
                input: tc.arguments.clone(),
            });
        }
        if blocks.is_empty() {
            blocks.push(WireContentBlock::Text {
                text: String::new(),
            });
        }
        out.push(WireMessage {
            role: "assistant".into(),
            content: WireContent::Blocks(blocks),
        });
        i += 1;
    }

    (system, out)
}

/// Parse a non-streaming [`MessagesResponse`] into a [`Message`].
pub(crate) fn parse_response(resp: MessagesResponse) -> Result<Message, ProviderError> {
    let token_count = resp
        .usage
        .as_ref()
        .and_then(|u| u.total_input().zip(u.output_tokens).map(|(i, o)| i + o));
    let context_tokens = resp.usage.as_ref().and_then(|u| u.total_input());
    let finish_reason = resp.stop_reason.as_deref().map(map_stop_reason);
    let mut msg = message_from_blocks(resp.content, token_count, context_tokens)?;
    msg.finish_reason = finish_reason;
    Ok(msg)
}

/// Insert `cache_control` breakpoints (5-minute ephemeral) into a serialized
/// request body so Anthropic caches the stable prefix. Three breakpoints,
/// matching Claude Code's scheme: the tool definitions, the system prompt, and
/// the last message. The last-message breakpoint caches the whole
/// prompt-plus-history prefix; the tool/system ones keep the static preamble
/// cached as the conversation grows. (Anthropic allows up to four.)
pub(crate) fn apply_prompt_caching(body: &mut Value) {
    let cc = serde_json::json!({ "type": "ephemeral" });
    let Some(obj) = body.as_object_mut() else {
        return;
    };

    // System: a plain string becomes a single cached text block; an existing
    // block array gets the breakpoint on its last block.
    match obj.get_mut("system") {
        Some(Value::String(text)) => {
            let text = std::mem::take(text);
            obj.insert(
                "system".to_string(),
                serde_json::json!([{ "type": "text", "text": text, "cache_control": cc }]),
            );
        }
        Some(Value::Array(blocks)) => set_cache_control_on_last(blocks, &cc),
        _ => {}
    }

    // Tools: breakpoint on the last tool definition.
    if let Some(Value::Array(tools)) = obj.get_mut("tools") {
        set_cache_control_on_last(tools, &cc);
    }

    // Messages: breakpoint on the last content block of the last message, but
    // only when that message is a user/tool message - its blocks are cacheable.
    // An assistant reply can end in a `thinking` block, which Anthropic rejects
    // `cache_control` on with a 400; guarding (rather than assuming the caller
    // never ends on an assistant message) keeps the request valid.
    if let Some(Value::Array(messages)) = obj.get_mut("messages") {
        if let Some(Value::Object(last)) = messages.last_mut() {
            let is_cacheable_role = matches!(
                last.get("role"),
                Some(Value::String(r)) if r == "user" || r == "tool"
            );
            if !is_cacheable_role {
                return;
            }
            match last.get_mut("content") {
                Some(Value::String(text)) => {
                    let text = std::mem::take(text);
                    last.insert(
                        "content".to_string(),
                        serde_json::json!([{ "type": "text", "text": text, "cache_control": cc }]),
                    );
                }
                Some(Value::Array(blocks)) => set_cache_control_on_last(blocks, &cc),
                _ => {}
            }
        }
    }
}

/// Attach `cache_control` to the last object in `items` (no-op if empty or the
/// last element isn't a JSON object).
fn set_cache_control_on_last(items: &mut [Value], cc: &Value) {
    if let Some(Value::Object(last)) = items.last_mut() {
        last.insert("cache_control".to_string(), cc.clone());
    }
}

/// Map an Anthropic `stop_reason` to the cross-provider [`FinishReason`].
/// `refusal` (including Fable 5 / Opus 4.8 HTTP-200 refusals) maps to
/// [`FinishReason::Refusal`]; unknown values are preserved.
pub(crate) fn map_stop_reason(raw: &str) -> FinishReason {
    match raw {
        "end_turn" | "stop_sequence" | "pause_turn" => FinishReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "refusal" => FinishReason::Refusal,
        other => FinishReason::Other(other.to_string()),
    }
}

/// Build a [`Message`] from accumulated content blocks (used by streaming).
pub(crate) fn message_from_content_blocks(
    blocks: Vec<ContentBlock>,
    usage: Option<Usage>,
) -> Result<Message, ProviderError> {
    let token_count = usage
        .as_ref()
        .and_then(|u| u.total_input().zip(u.output_tokens).map(|(i, o)| i + o));
    let context_tokens = usage.as_ref().and_then(|u| u.total_input());
    message_from_blocks(blocks, token_count, context_tokens)
}

fn message_from_blocks(
    blocks: Vec<ContentBlock>,
    token_count: Option<usize>,
    context_tokens: Option<usize>,
) -> Result<Message, ProviderError> {
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut thinking_content = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&text);
            }
            ContentBlock::ToolUse { id, name, input }
            | ContentBlock::ServerToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: input,
                });
            }
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                thinking_content.push(ThinkingContent {
                    text: thinking,
                    signature: Some(signature),
                    redacted_data: None,
                });
            }
            ContentBlock::RedactedThinking { data } => {
                thinking_content.push(ThinkingContent {
                    text: String::new(),
                    signature: None,
                    redacted_data: Some(data),
                });
            }
            ContentBlock::Unknown => {}
        }
    }

    Ok(Message {
        role: Role::Assistant,
        content: vec![sweet_core::ContentBlock::text(content)],
        thinking_content,
        tool_calls,
        tool_call_id: None,
        token_count,
        context_tokens,
        compacted: false,
        finish_reason: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_prompt_caching_sets_three_breakpoints() {
        let mut body = serde_json::json!({
            "system": "you are helpful",
            "tools": [{ "name": "a" }, { "name": "b" }],
            "messages": [
                { "role": "user", "content": "first" },
                { "role": "user", "content": "latest" }
            ]
        });
        apply_prompt_caching(&mut body);

        // System: string lifted into a cached text block.
        assert_eq!(body["system"][0]["type"], "text");
        assert_eq!(body["system"][0]["text"], "you are helpful");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");

        // Tools: only the last tool gets the breakpoint.
        assert!(body["tools"][0].get("cache_control").is_none());
        assert_eq!(body["tools"][1]["cache_control"]["type"], "ephemeral");

        // Messages: only the last message's last block gets the breakpoint.
        assert!(body["messages"][0]["content"].is_string());
        assert_eq!(body["messages"][1]["content"][0]["type"], "text");
        assert_eq!(body["messages"][1]["content"][0]["text"], "latest");
        assert_eq!(
            body["messages"][1]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn apply_prompt_caching_caches_last_block_of_array_content() {
        let mut body = serde_json::json!({
            "messages": [
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "t", "content": "r" }
                ] }
            ]
        });
        apply_prompt_caching(&mut body);
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn apply_prompt_caching_skips_assistant_last_message() {
        // An assistant reply ending the messages could land `cache_control` on
        // a non-cacheable block (e.g. thinking), which Anthropic rejects with a
        // 400. The breakpoint is skipped when the last message isn't user/tool.
        let mut body = serde_json::json!({
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": [
                    { "type": "text", "text": "hello" }
                ] }
            ]
        });
        apply_prompt_caching(&mut body);
        assert!(body["messages"][1]["content"][0]
            .get("cache_control")
            .is_none());
    }

    #[test]
    fn usage_total_input_sums_cache_tokens() {
        let usage = Usage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            cache_creation_input_tokens: Some(200),
            cache_read_input_tokens: Some(700),
        };
        assert_eq!(usage.total_input(), Some(1000));
    }

    #[test]
    fn redacted_thinking_replays_to_wire() {
        let mut msg = Message::assistant("answer");
        msg.thinking_content = vec![ThinkingContent {
            text: String::new(),
            signature: None,
            redacted_data: Some("ENC".to_string()),
        }];
        let (_system, wire) = convert_messages(std::slice::from_ref(&msg));
        let json = serde_json::to_value(&wire).unwrap();
        let blocks = json[0]["content"].as_array().unwrap();
        assert!(blocks
            .iter()
            .any(|b| b["type"] == "redacted_thinking" && b["data"] == "ENC"));
    }

    #[test]
    fn redacted_thinking_parsed_from_response() {
        let resp: MessagesResponse = serde_json::from_value(serde_json::json!({
            "id": "m", "type": "message", "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "ENC"},
                {"type": "text", "text": "hi"}
            ],
            "model": "claude", "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .unwrap();
        let msg = parse_response(resp).unwrap();
        assert_eq!(msg.text_content(), "hi");
        assert_eq!(msg.thinking_content.len(), 1);
        assert_eq!(
            msg.thinking_content[0].redacted_data.as_deref(),
            Some("ENC")
        );
    }
}
