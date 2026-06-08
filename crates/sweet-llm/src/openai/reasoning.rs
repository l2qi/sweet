// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Extension trait that surfaces OpenAI's single-block `reasoning_content`
//! view of a [`Message`]'s thinking content.
//!
//! OpenAI-protocol providers (DeepSeek, Kimi, Cerebras, etc.) carry chain-of-
//! thought as a single `reasoning_content` string per turn, with no signature.
//! This trait lets call sites work in those terms while the canonical
//! representation on `Message` remains `Vec<ThinkingContent>`.

use sweet_core::{Message, ThinkingContent};

pub trait ReasoningContent {
    /// First thinking block's text, if any. Multi-block consumers should
    /// iterate `Message::thinking_content` directly.
    fn reasoning_content(&self) -> Option<&str>;

    /// Replace `thinking_content` with a single block carrying `text` and no
    /// signature. Empty `text` is preserved as an empty-text block — Kimi
    /// distinguishes "explicit empty reasoning_content" from "no field
    /// present" on multi-turn requests, and clearing the Vec would lose that
    /// signal.
    fn set_reasoning_content(&mut self, text: impl Into<String>);
}

impl ReasoningContent for Message {
    fn reasoning_content(&self) -> Option<&str> {
        self.thinking_content.first().map(|t| t.text.as_str())
    }

    fn set_reasoning_content(&mut self, text: impl Into<String>) {
        self.thinking_content = vec![ThinkingContent::new(text)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_returns_none_for_empty_vec() {
        let msg = Message::assistant("hi");
        assert_eq!(msg.reasoning_content(), None);
    }

    #[test]
    fn get_returns_first_block_text() {
        let mut msg = Message::assistant("hi");
        msg.set_reasoning_content("thinking");
        assert_eq!(msg.reasoning_content(), Some("thinking"));
    }

    #[test]
    fn set_empty_preserves_empty_block() {
        let mut msg = Message::assistant("hi");
        msg.set_reasoning_content("");
        assert_eq!(msg.thinking_content.len(), 1);
        assert_eq!(msg.reasoning_content(), Some(""));
    }

    #[test]
    fn set_overwrites_existing_blocks() {
        let mut msg = Message::assistant("hi");
        msg.set_reasoning_content("first");
        msg.set_reasoning_content("second");
        assert_eq!(msg.thinking_content.len(), 1);
        assert_eq!(msg.reasoning_content(), Some("second"));
    }
}
