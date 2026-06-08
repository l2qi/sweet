// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::message::Message;
use crate::stream::StreamSink;
use crate::tool::ToolSpec;

/// An inference backend that can respond to a conversation.
///
/// Kept deliberately small: the only required method is `complete`, which takes
/// a borrowed slice of messages and returns the assistant reply. Streaming and
/// tool-calling extensions can be added as additional methods with default
/// implementations so existing impls keep compiling.
#[async_trait]
pub trait Model: Send + Sync {
    async fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message>;

    /// Streaming variant of [`Model::complete`].
    ///
    /// Providers that support server-side streaming should override this and
    /// emit content deltas plus completed tool calls to `sink` as they arrive.
    /// The default implementation calls [`Model::complete`] and emits the full
    /// reply as a single delta — fine for tests and providers without
    /// streaming support.
    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> Result<Message> {
        let reply = self.complete(messages, tools).await?;
        if !reply.text_content().is_empty() {
            sink.on_content_delta(&reply.text_content()).await?;
        }
        for call in &reply.tool_calls {
            sink.on_tool_call(call).await?;
        }
        Ok(reply)
    }

    /// Returns the model's maximum context window in tokens, if known.
    /// Providers that don't expose this should leave the default; callers must
    /// handle `None`.
    fn context_window(&self) -> Option<usize> {
        None
    }
}

#[async_trait]
impl<M: Model + ?Sized> Model for Box<M> {
    async fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message> {
        (**self).complete(messages, tools).await
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> Result<Message> {
        (**self).complete_stream(messages, tools, sink).await
    }

    fn context_window(&self) -> Option<usize> {
        (**self).context_window()
    }
}

#[async_trait]
impl<M: Model + ?Sized> Model for Arc<M> {
    async fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message> {
        (**self).complete(messages, tools).await
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> Result<Message> {
        (**self).complete_stream(messages, tools, sink).await
    }

    fn context_window(&self) -> Option<usize> {
        (**self).context_window()
    }
}

#[async_trait]
impl<M: Model + ?Sized> Model for &M {
    async fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message> {
        (**self).complete(messages, tools).await
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> Result<Message> {
        (**self).complete_stream(messages, tools, sink).await
    }

    fn context_window(&self) -> Option<usize> {
        (**self).context_window()
    }
}
