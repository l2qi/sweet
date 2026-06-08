// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Wire-format DTOs for OpenAI's `/v1/chat/completions` endpoint.

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sweet_core::{Message, Role, ToolCall};

use super::reasoning::ReasoningContent;
use crate::error::ProviderError;

#[derive(Debug, Serialize)]
pub(crate) struct ChatRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool<'a>>>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ThinkingConfig {
    pub r#type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct StreamOptions {
    pub include_usage: bool,
}

/// A single part in OpenAI's multimodal content array (user messages only).
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireContentPart {
    Text { text: String },
    ImageUrl { image_url: WireImageUrl },
    File { file: WireFile },
}

#[derive(Debug, Serialize)]
pub(crate) struct WireImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Wire payload for an inline file attachment (e.g. PDF) sent to the
/// OpenAI Chat Completions API as `{"type":"file","file":{...}}`.
/// Both fields are required by the API for base64 file data.
#[derive(Debug, Serialize)]
pub(crate) struct WireFile {
    /// Data URI: `data:{media_type};base64,{b64}`.
    pub file_data: String,
    /// Original file name. OpenAI rejects base64 `file` parts without it.
    pub filename: String,
}

/// Content for a wire message. Serialized as either a plain string or a
/// multimodal parts array. Only user messages with images need the array
/// form; all other roles (and text-only user messages) use the plain string
/// to match the OpenAI API's expected wire shape.
///
/// This enum is only ever **serialized** (never deserialized), so
/// `#[serde(untagged)]` has no deserialization ambiguity concerns.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum WireContent {
    Text(String),
    Parts(Vec<WireContentPart>),
}

#[derive(Debug, Serialize)]
pub(crate) struct WireMessage<'a> {
    pub role: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<WireContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<&'a str>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<WireToolCall<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<&'a str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireTool<'a> {
    pub r#type: &'static str,
    pub function: WireToolFunction<'a>,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireToolFunction<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub parameters: Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct WireToolCall<'a> {
    pub id: &'a str,
    pub r#type: &'static str,
    pub function: WireToolCallFunction<'a>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct WireToolCallFunction<'a> {
    pub name: &'a str,
    // OpenAI's wire format carries tool-call arguments as a JSON-encoded string
    // (not a JSON value), so serialize the Value here at conversion time.
    pub arguments: String,
}

impl<'a> WireMessage<'a> {
    /// Build a wire message from a [`Message`].
    ///
    /// `include_reasoning` controls whether the message's `reasoning_content`
    /// is echoed on the wire. Plain OpenAI servers do not understand the
    /// field and some strict proxies reject unknown `messages[]` keys, so
    /// callers should only set this when targeting a thinking-aware backend.
    pub(crate) fn new(m: &'a Message, include_reasoning: bool) -> Self {
        let role = match m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        let reasoning_content = match m.reasoning_content() {
            // Always echo reasoning_content back when the model provided
            // it, even if the provider wasn't explicitly configured for
            // thinking (DeepSeek/Kimi auto-enable it for thinking models).
            Some(text) => Some(text),
            // When targeting a thinking-aware backend, ensure the field
            // is present on assistant messages even if empty.
            None if include_reasoning && role == "assistant" => Some(""),
            None => None,
        };
        Self {
            role,
            content: if role == "user" && m.has_attachments() {
                // User messages with images or file attachments: multimodal parts array.
                Some(WireContent::Parts(
                    m.content
                        .iter()
                        .filter_map(|block| match block {
                            sweet_core::ContentBlock::Text { text } if !text.is_empty() => {
                                Some(WireContentPart::Text { text: text.clone() })
                            }
                            sweet_core::ContentBlock::Image { data, media_type } => {
                                let b64 = base64::prelude::BASE64_STANDARD.encode(data);
                                Some(WireContentPart::ImageUrl {
                                    image_url: WireImageUrl {
                                        url: format!("data:{};base64,{}", media_type, b64),
                                        detail: None,
                                    },
                                })
                            }
                            sweet_core::ContentBlock::File {
                                data,
                                media_type,
                                filename,
                            } => {
                                let b64 = base64::prelude::BASE64_STANDARD.encode(data);
                                Some(WireContentPart::File {
                                    file: WireFile {
                                        file_data: format!("data:{};base64,{}", media_type, b64),
                                        filename: filename.clone(),
                                    },
                                })
                            }
                            _ => None,
                        })
                        .collect(),
                ))
            } else {
                // System, tool, assistant, and text-only user messages:
                // plain string content (OpenAI wire format requirement).
                let text = m.text_content();
                if text.is_empty() && !m.tool_calls.is_empty() {
                    // Assistant messages with only tool_calls and no text
                    // omit the content field entirely.
                    None
                } else {
                    Some(WireContent::Text(text))
                }
            },
            reasoning_content,
            tool_calls: m
                .tool_calls
                .iter()
                .map(|tc| WireToolCall {
                    id: &tc.id,
                    r#type: "function",
                    function: WireToolCallFunction {
                        name: &tc.name,
                        arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
                    },
                })
                .collect(),
            tool_call_id: m.tool_call_id.as_deref(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ChatResponse {
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct Choice {
    pub message: ResponseMessage,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ResponseMessage {
    pub role: String,
    // OpenAI/OpenRouter send `"content": null` on assistant messages that
    // carry tool_calls instead of text, so accept both missing and null.
    #[serde(default, deserialize_with = "deserialize_string_or_null")]
    pub content: String,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ResponseToolCall>,
}

fn deserialize_string_or_null<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ResponseToolCall {
    pub id: String,
    /// OpenAI always sends `"type": "function"`; the field is deserialized
    /// but never read by our parser.
    #[serde(default, rename = "type")]
    pub _type: String,
    pub function: ResponseToolCallFunction,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ResponseToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// One server-sent event chunk from a streaming `/chat/completions` response.
#[derive(Debug, Deserialize, Default)]
pub(crate) struct StreamChunk {
    #[serde(default)]
    pub choices: Vec<StreamChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamChoice {
    #[serde(default)]
    pub delta: StreamDelta,
    /// Present in every streaming chunk but not consumed by our parser.
    #[serde(default, rename = "finish_reason")]
    pub _finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct StreamDelta {
    #[serde(default, deserialize_with = "deserialize_string_or_null")]
    pub content: String,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<StreamToolCallDelta>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub function: Option<StreamToolCallFunctionDelta>,
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct StreamToolCallFunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

impl TryFrom<ResponseMessage> for Message {
    type Error = ProviderError;

    fn try_from(m: ResponseMessage) -> Result<Self, Self::Error> {
        let role = match m.role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => return Err(ProviderError::UnknownRole(other.to_string())),
        };

        let tool_calls: Vec<ToolCall> = m
            .tool_calls
            .into_iter()
            .map(|tc| {
                let args: Value =
                    serde_json::from_str(&tc.function.arguments).map_err(ProviderError::Decode)?;
                Ok(ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: args,
                })
            })
            .collect::<Result<Vec<_>, ProviderError>>()?;

        // OpenAI assistant responses only carry text in `content`; collapse
        // into a single `Text` block. If a future provider returns image
        // content here this would need to be extended.
        let mut msg = Message {
            role,
            content: vec![sweet_core::ContentBlock::text(m.content)],
            thinking_content: Vec::new(),
            tool_calls,
            tool_call_id: None,
            token_count: None,
            context_tokens: None,
            compacted: false,
        };
        if let Some(rc) = m.reasoning_content {
            msg.set_reasoning_content(rc);
        }
        Ok(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_message_serializes_role_strings() {
        let cases = [
            (Message::system("s"), "system"),
            (Message::user("u"), "user"),
            (Message::assistant("a"), "assistant"),
        ];
        for (msg, expected_role) in cases {
            let wire = WireMessage::new(&msg, false);
            let json = serde_json::to_value(&wire).unwrap();
            assert_eq!(json["role"], expected_role);
            // Text-only messages serialize content as a plain string for
            // all roles. The multimodal parts array is only used when a
            // user message carries images (see `image_user_message_*` tests
            // in `tests/openai.rs`).
            assert_eq!(json["content"], msg.text_content());
        }
    }

    #[test]
    fn response_message_decodes_known_roles() {
        let raw = r#"{"role":"assistant","content":"hi"}"#;
        let m: ResponseMessage = serde_json::from_str(raw).unwrap();
        let msg = Message::try_from(m).unwrap();
        assert_eq!(msg, Message::assistant("hi"));
    }

    #[test]
    fn response_message_accepts_tool_role() {
        let raw = r#"{"role":"tool","content":"result"}"#;
        let m: ResponseMessage = serde_json::from_str(raw).unwrap();
        let msg = Message::try_from(m).unwrap();
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.text_content(), "result");
    }

    #[test]
    fn response_message_rejects_unknown_role() {
        let m = ResponseMessage {
            role: "unknown".into(),
            content: String::new(),
            reasoning_content: None,
            tool_calls: Vec::new(),
        };
        let err = Message::try_from(m).unwrap_err();
        assert!(matches!(err, ProviderError::UnknownRole(s) if s == "unknown"));
    }

    #[test]
    fn response_message_defaults_empty_content() {
        let raw = r#"{"role":"assistant"}"#;
        let m: ResponseMessage = serde_json::from_str(raw).unwrap();
        assert_eq!(m.content, "");
    }

    #[test]
    fn response_message_accepts_null_content() {
        // OpenAI/OpenRouter emit `"content": null` on tool-calling assistant
        // messages.
        let raw = r#"{"role":"assistant","content":null,"tool_calls":[]}"#;
        let m: ResponseMessage = serde_json::from_str(raw).unwrap();
        assert_eq!(m.content, "");
    }

    #[test]
    fn wire_message_serializes_tool_call_arguments_as_json_string() {
        let msg = Message::with_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "echo".into(),
            arguments: serde_json::json!({"msg": "hi"}),
        }]);
        let wire = WireMessage::new(&msg, false);
        let json = serde_json::to_value(&wire).unwrap();
        // OpenAI requires arguments to be a JSON-encoded string, not a Value.
        let arguments = json["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .expect("arguments must be a string");
        let parsed: serde_json::Value = serde_json::from_str(arguments).unwrap();
        assert_eq!(parsed, serde_json::json!({"msg": "hi"}));
    }

    #[test]
    fn response_message_parses_tool_calls() {
        let raw = r#"{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": "{\"msg\":\"hello\"}"
                }
            }]
        }"#;
        let m: ResponseMessage = serde_json::from_str(raw).unwrap();
        let msg = Message::try_from(m).unwrap();
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].id, "call_1");
        assert_eq!(msg.tool_calls[0].name, "echo");
        assert_eq!(
            msg.tool_calls[0].arguments,
            serde_json::json!({"msg": "hello"})
        );
    }

    #[test]
    fn wire_message_serializes_reasoning_content_when_include_true_and_present() {
        let mut msg = Message::assistant("hello");
        msg.set_reasoning_content("thinking");
        let wire = WireMessage::new(&msg, true);
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["reasoning_content"], "thinking");
    }

    #[test]
    fn wire_message_serializes_reasoning_content_when_present_even_if_include_false() {
        // reasoning_content is always echoed back when the model provided it,
        // regardless of include_reasoning (DeepSeek/Kimi auto-enable thinking).
        let mut msg = Message::assistant("hello");
        msg.set_reasoning_content("thinking");
        let wire = WireMessage::new(&msg, false);
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["reasoning_content"], "thinking");
    }

    #[test]
    fn wire_message_serializes_empty_reasoning_content_for_assistant_when_include_true() {
        // When targeting a thinking-aware backend, assistant messages must
        // always carry reasoning_content even if empty (required by DeepSeek
        // / Kimi for tool-call turns).
        let msg = Message::assistant("hello");
        let wire = WireMessage::new(&msg, true);
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["reasoning_content"], "");
    }

    #[test]
    fn wire_message_omits_reasoning_content_for_user_when_include_true() {
        // Non-assistant messages never carry reasoning_content.
        let msg = Message::user("hello");
        let wire = WireMessage::new(&msg, true);
        let json = serde_json::to_value(&wire).unwrap();
        assert!(!json.as_object().unwrap().contains_key("reasoning_content"));
    }

    #[test]
    fn response_message_deserializes_reasoning_content() {
        let raw = r#"{"role":"assistant","content":"hello","reasoning_content":"thinking"}"#;
        let m: ResponseMessage = serde_json::from_str(raw).unwrap();
        assert_eq!(m.reasoning_content, Some("thinking".into()));
        let msg = Message::try_from(m).unwrap();
        assert_eq!(msg.reasoning_content(), Some("thinking"));
    }

    #[test]
    fn response_message_defaults_reasoning_content_to_none() {
        let raw = r#"{"role":"assistant","content":"hello"}"#;
        let m: ResponseMessage = serde_json::from_str(raw).unwrap();
        assert_eq!(m.reasoning_content, None);
    }

    #[test]
    fn stream_delta_deserializes_reasoning_content() {
        let raw = r#"{"content":"hello","reasoning_content":"thinking"}"#;
        let delta: StreamDelta = serde_json::from_str(raw).unwrap();
        assert_eq!(delta.reasoning_content, Some("thinking".into()));
    }

    #[test]
    fn chat_request_serializes_reasoning_effort_and_thinking() {
        let msg = Message::user("hi");
        let body = ChatRequest {
            model: "gpt-4",
            messages: vec![WireMessage::new(&msg, false)],
            tools: None,
            stream: false,
            stream_options: None,
            reasoning_effort: Some("high"),
            thinking: Some(ThinkingConfig {
                r#type: "enabled",
                keep: Some("all"),
            }),
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["reasoning_effort"], "high");
        assert_eq!(json["thinking"]["type"], "enabled");
        assert_eq!(json["thinking"]["keep"], "all");
    }

    #[test]
    fn chat_request_omits_reasoning_effort_and_thinking_when_none() {
        let msg = Message::user("hi");
        let body = ChatRequest {
            model: "gpt-4",
            messages: vec![WireMessage::new(&msg, false)],
            tools: None,
            stream: false,
            stream_options: None,
            reasoning_effort: None,
            thinking: None,
        };
        let json = serde_json::to_value(&body).unwrap();
        assert!(!json.as_object().unwrap().contains_key("reasoning_effort"));
        assert!(!json.as_object().unwrap().contains_key("thinking"));
    }
}
