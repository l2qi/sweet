// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// Wire-format name (matches the serde rename used on the enum).
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// A single tool call requested by the assistant.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// One chain-of-thought block emitted by a thinking-mode model.
///
/// Most providers produce at most one per response; Anthropic can interleave
/// multiple thinking blocks with tool use, so `Message` stores a `Vec` rather
/// than `Option`. `signature` is an opaque token used by Anthropic to verify
/// thinking provenance on multi-turn requests; other providers leave it
/// `None`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ThinkingContent {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub signature: Option<String>,
}

impl ThinkingContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            signature: None,
        }
    }
}

/// A single content block within a message.
///
/// Most messages contain only `Text` blocks. `Image` blocks carry binary
/// image data (e.g. from `@photo.png` or clipboard paste) and are only
/// sent to models that support vision. `File` blocks carry non-image binary
/// data (e.g. PDFs) and are sent to models that support document input.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        /// Raw image bytes.
        #[serde(with = "serde_base64")]
        data: Vec<u8>,
        /// MIME type, e.g. `"image/png"`.
        media_type: String,
    },
    File {
        /// Raw file bytes.
        #[serde(with = "serde_base64")]
        data: Vec<u8>,
        /// MIME type, e.g. `"application/pdf"`.
        media_type: String,
        /// Original file name, e.g. `"report.pdf"`. Required by the OpenAI
        /// Chat Completions `file` content part; other providers ignore it.
        filename: String,
    },
}

impl ContentBlock {
    /// Convenience constructor for a text block.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    /// Returns the text content if this is a `Text` block, `None` otherwise.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            ContentBlock::Image { .. } | ContentBlock::File { .. } => None,
        }
    }

    /// Returns `true` if this block carries binary data (image or file).
    pub fn is_attachment(&self) -> bool {
        matches!(self, ContentBlock::Image { .. } | ContentBlock::File { .. })
    }
}

/// Format a byte length as a compact human-readable size (`KB` below 1 MiB,
/// `MB` at or above it).
fn human_size(len: usize) -> String {
    if len >= 1_048_576 {
        format!("{:.1} MB", len as f64 / 1_048_576.0)
    } else {
        format!("{:.0} KB", len as f64 / 1024.0)
    }
}

impl std::fmt::Display for ContentBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContentBlock::Text { text } => write!(f, "{text}"),
            ContentBlock::Image { data, media_type } => {
                write!(f, "[image: {media_type}, {}]", human_size(data.len()))
            }
            ContentBlock::File {
                data,
                media_type,
                filename,
            } => {
                write!(
                    f,
                    "[file: {filename}, {media_type}, {}]",
                    human_size(data.len())
                )
            }
        }
    }
}

/// serde helper for `Vec<u8>` ↔ base64 string.
mod serde_base64 {
    use base64::prelude::*;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(data: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&BASE64_STANDARD.encode(data))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        BASE64_STANDARD.decode(&s).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: Role,
    /// Content blocks. Most messages contain a single `ContentBlock::Text`.
    /// Vision-capable models also accept `ContentBlock::Image` blocks.
    pub content: Vec<ContentBlock>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub thinking_content: Vec<ThinkingContent>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
    /// Actual token count reported by the provider's `usage` field.
    /// `None` if the provider does not report token usage.
    pub token_count: Option<usize>,
    /// `usage.prompt_tokens` from the provider response — the actual input
    /// context size at the time of this call. Used for context-window
    /// tracking and compaction threshold decisions.
    pub context_tokens: Option<usize>,
    /// `true` on user/assistant messages injected by session compaction.
    /// Providers never set this; it marks messages the wire layer should
    /// include but that are not part of the original conversation.
    #[serde(skip_serializing_if = "std::ops::Not::not", default)]
    pub compacted: bool,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentBlock::text(content)],
            thinking_content: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            token_count: None,
            context_tokens: None,
            compacted: false,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Construct a user message from pre-built content blocks (e.g. text + image).
    pub fn user_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content: blocks,
            thinking_content: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            token_count: None,
            context_tokens: None,
            compacted: false,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// Construct a `Role::Tool` message carrying the result of a tool call.
    pub fn tool_result(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentBlock::text(content)],
            thinking_content: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: Some(id.into()),
            token_count: None,
            context_tokens: None,
            compacted: false,
        }
    }

    /// Construct an assistant message that carries tool calls instead of (or
    /// alongside) content.
    pub fn with_tool_calls(calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: Vec::new(),
            thinking_content: Vec::new(),
            tool_calls: calls,
            tool_call_id: None,
            token_count: None,
            context_tokens: None,
            compacted: false,
        }
    }

    /// Concatenate all `ContentBlock::Text` entries into a single string.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| b.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns `true` if any content block is an image.
    pub fn has_images(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::Image { .. }))
    }

    /// Returns `true` if any content block is a non-image file attachment.
    pub fn has_files(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::File { .. }))
    }

    /// Returns `true` if any content block carries binary data (image or file).
    pub fn has_attachments(&self) -> bool {
        self.content.iter().any(|b| b.is_attachment())
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for block in &self.content {
            write!(f, "{block}")?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_set_role_and_content() {
        let s = Message::system("sys");
        assert_eq!(s.role, Role::System);
        assert_eq!(s.text_content(), "sys");

        let u = Message::user("hi");
        assert_eq!(u.role, Role::User);
        assert_eq!(u.text_content(), "hi");

        let a = Message::assistant("there");
        assert_eq!(a.role, Role::Assistant);
        assert_eq!(a.text_content(), "there");
    }

    #[test]
    fn new_accepts_owned_or_borrowed_content() {
        let owned = Message::new(Role::User, String::from("a"));
        let borrowed = Message::new(Role::User, "a");
        assert_eq!(owned, borrowed);
    }

    #[test]
    fn user_blocks_accepts_mixed_content() {
        let msg = Message::user_blocks(vec![
            ContentBlock::text("describe this: "),
            ContentBlock::Image {
                data: vec![1, 2, 3],
                media_type: "image/png".to_string(),
            },
        ]);
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text_content(), "describe this: ");
        assert!(msg.has_images());
    }

    #[test]
    fn has_images_false_for_text_only() {
        assert!(!Message::user("hello").has_images());
    }

    #[test]
    fn has_files_true_for_file_block() {
        let msg = Message::user_blocks(vec![ContentBlock::File {
            data: vec![1, 2, 3],
            media_type: "application/pdf".to_string(),
            filename: "doc.pdf".to_string(),
        }]);
        assert!(!msg.has_images());
        assert!(msg.has_files());
        assert!(msg.has_attachments());
    }

    #[test]
    fn has_files_false_for_text_only() {
        assert!(!Message::user("hello").has_files());
    }

    #[test]
    fn has_attachments_true_for_mixed() {
        let msg = Message::user_blocks(vec![
            ContentBlock::text("see this"),
            ContentBlock::Image {
                data: vec![],
                media_type: "image/png".to_string(),
            },
        ]);
        assert!(msg.has_attachments());
        let msg = Message::user_blocks(vec![ContentBlock::File {
            data: vec![],
            media_type: "application/pdf".to_string(),
            filename: "doc.pdf".to_string(),
        }]);
        assert!(msg.has_attachments());
        assert!(!Message::user("text").has_attachments());
    }

    #[test]
    fn content_block_display_image() {
        let block = ContentBlock::Image {
            data: vec![0u8; 2_097_152], // 2 MB
            media_type: "image/png".to_string(),
        };
        let s = format!("{block}");
        assert!(s.contains("image/png"));
        assert!(s.contains("2.0 MB"));
    }

    #[test]
    fn content_block_display_file() {
        let block = ContentBlock::File {
            data: vec![0u8; 512_000],
            media_type: "application/pdf".to_string(),
            filename: "report.pdf".to_string(),
        };
        let s = format!("{block}");
        assert!(s.contains("report.pdf"));
        assert!(s.contains("application/pdf"));
        assert!(s.contains("500 KB"));
    }

    #[test]
    fn content_block_is_attachment() {
        assert!(ContentBlock::Image {
            data: vec![],
            media_type: "image/png".into()
        }
        .is_attachment());
        assert!(ContentBlock::File {
            data: vec![],
            media_type: "application/pdf".into(),
            filename: "doc.pdf".into()
        }
        .is_attachment());
        assert!(!ContentBlock::text("hi").is_attachment());
    }

    /// Guard against drift between `Role::as_str` and the serde rename. If
    /// someone edits the enum's `rename_all` attribute, this round-trip catches
    /// it; a hand-written equality check would silently desync.
    #[test]
    fn role_as_str_matches_serde_form() {
        for role in [Role::System, Role::User, Role::Assistant, Role::Tool] {
            let serialized = serde_json::to_value(role).unwrap();
            assert_eq!(serialized, serde_json::Value::String(role.as_str().into()));
        }
    }
}
