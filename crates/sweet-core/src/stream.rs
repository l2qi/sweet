// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Streaming sink used by [`crate::Model::complete_stream`] to emit incremental
//! events as the assistant reply is produced.
//!
//! The sink is intentionally narrow: providers report content deltas as they
//! arrive and report each tool call once its arguments have been fully
//! received. Everything else (tool execution, hooks, session writes) is the
//! agent's job and is reported through `AgentIo` (in the `sweet-agent` crate).

use async_trait::async_trait;

use crate::error::Result;
use crate::message::ToolCall;

#[async_trait]
pub trait StreamSink: Send {
    /// Incremental delta of the assistant's text content.
    async fn on_content_delta(&mut self, delta: &str) -> Result<()>;

    /// Incremental delta of the assistant's thinking text (chain-of-thought).
    async fn on_thinking_delta(&mut self, _delta: &str) -> Result<()> {
        Ok(())
    }

    /// A tool call has been fully assembled from the stream.
    async fn on_tool_call(&mut self, call: &ToolCall) -> Result<()>;
}

/// A sink that drops every event. Used by callers that want the final
/// `Message` without paying for streaming UI.
pub struct NoopSink;

#[async_trait]
impl StreamSink for NoopSink {
    async fn on_content_delta(&mut self, _delta: &str) -> Result<()> {
        Ok(())
    }

    async fn on_tool_call(&mut self, _call: &ToolCall) -> Result<()> {
        Ok(())
    }
}
