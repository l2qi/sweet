// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Wire-format DTOs for Anthropic's native `/v1/messages` endpoint.

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sweet_core::{Message, Role, ThinkingContent, ToolCall};

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
    #[allow(dead_code)]
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
    #[serde(rename = "stop_reason")]
    pub _stop_reason: Option<String>,
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
    RedactedThinking {
        #[serde(rename = "data")]
        _data: String,
    },
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
        #[serde(rename = "delta")]
        _delta: Option<MessageDeltaInner>,
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
    #[serde(rename = "stop_reason")]
    pub _stop_reason: Option<String>,
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
/// `ThinkingContent.signature` is `Some` — Anthropic requires a valid
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
                blocks.push(WireContentBlock::ToolResult {
                    tool_use_id: tool_msg.tool_call_id.clone().unwrap_or_default(),
                    content: WireToolResultContent::Text(tool_msg.text_content()),
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
            match &tc.signature {
                Some(sig) => {
                    blocks.push(WireContentBlock::Thinking {
                        thinking: tc.text.clone(),
                        signature: sig.clone(),
                    });
                }
                None => {
                    // Anthropic rejects thinking blocks without a valid
                    // signature, so dropping is the only safe option. This
                    // typically means the block originated from a different
                    // provider (OpenAI/Gemini) or was hand-constructed.
                    tracing::warn!(
                        target: "sweet_llm::anthropic",
                        text_len = tc.text.len(),
                        "dropping thinking block with no signature; \
                         not sent to Anthropic"
                    );
                }
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
        .and_then(|u| u.input_tokens.zip(u.output_tokens).map(|(i, o)| i + o));
    let context_tokens = resp.usage.as_ref().and_then(|u| u.input_tokens);
    message_from_blocks(resp.content, token_count, context_tokens)
}

/// Build a [`Message`] from accumulated content blocks (used by streaming).
pub(crate) fn message_from_content_blocks(
    blocks: Vec<ContentBlock>,
    usage: Option<Usage>,
) -> Result<Message, ProviderError> {
    let token_count = usage
        .as_ref()
        .and_then(|u| u.input_tokens.zip(u.output_tokens).map(|(i, o)| i + o));
    let context_tokens = usage.as_ref().and_then(|u| u.input_tokens);
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
                });
            }
            ContentBlock::RedactedThinking { .. } | ContentBlock::Unknown => {}
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
    })
}
