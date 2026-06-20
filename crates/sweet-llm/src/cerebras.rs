// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Cerebras Inference provider - OpenAI-compatible transport with Cerebras's
//! own reasoning controls.

use sweet_core::stream::StreamSink;
use sweet_core::{async_trait, Message, Model, Result, ToolSpec};

use crate::openai::{OpenAIProvider, ReasoningHistoryKey};
use crate::{ReasoningConfig, SamplingConfig, StructuredOutput, ToolChoice};

/// Inference provider for [Cerebras Inference](https://inference.cerebras.ai).
///
/// Cerebras speaks the OpenAI `/v1/chat/completions` protocol, so this composes
/// an [`OpenAIProvider`] ("inheritance" via a wrapper) and forwards every
/// request to it. What differs is **reasoning control**: Cerebras rejects the
/// OpenAI-compatible `thinking` object
/// (`HTTP 400 ... property 'thinking' is unsupported`). Its only documented
/// reasoning knob is `reasoning_effort`, whose accepted values are
/// model-specific (`gpt-oss-120b`: `low`/`medium`/`high`; `zai-glm-4.7`: only
/// `none`, to *disable*) - and its reasoning models reason **by default**.
///
/// So this provider is effort-only: [`with_reasoning`](Self::with_reasoning)
/// maps to `reasoning_effort` (or, for `Toggle(false)`, the documented
/// `"none"`) and **never** enables the `thinking` object that tripped the 400.
/// With no reasoning configured it sends no reasoning parameter at all and the
/// model still reasons on its own. All Cerebras-specific policy lives here;
/// `OpenAIProvider` stays a generic OpenAI-compatible transport.
#[derive(Debug, Clone)]
pub struct CerebrasProvider {
    inner: OpenAIProvider,
}

impl CerebrasProvider {
    /// Construct a provider with an explicit API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            // Cerebras expects replayed assistant reasoning under `reasoning`,
            // not the OpenAI-compatible `reasoning_content` (it renames the
            // field server-side and rejects the wrong key in strict modes).
            inner: OpenAIProvider::new(api_key)
                .with_reasoning_history_key(ReasoningHistoryKey::Reasoning),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.inner = self.inner.with_base_url(url);
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.inner = self.inner.with_model(model);
        self
    }

    pub fn with_context_window(mut self, tokens: usize) -> Self {
        self.inner = self.inner.with_context_window(tokens);
        self
    }

    /// Set cross-provider sampling parameters (forwarded to the inner
    /// OpenAI-compatible transport).
    pub fn with_sampling(mut self, sampling: SamplingConfig) -> Self {
        self.inner = self.inner.with_sampling(sampling);
        self
    }

    /// Constrain how the model chooses among tools (forwarded to the inner
    /// transport).
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.inner = self.inner.with_tool_choice(choice);
        self
    }

    /// Constrain the response to JSON matching a schema (forwarded to the inner
    /// transport; Cerebras supports OpenAI-style `response_format`).
    pub fn with_structured_output(mut self, output: StructuredOutput) -> Self {
        self.inner = self.inner.with_structured_output(output);
        self
    }

    /// Configure reasoning, Cerebras-style: effort only, never the `thinking`
    /// object Cerebras rejects.
    ///
    /// - [`ReasoningConfig::Effort`] -> `reasoning_effort`.
    /// - `Toggle(false)` -> `reasoning_effort: "none"` (the documented way to
    ///   *disable* reasoning on models that support it).
    /// - `Toggle(true)` and [`ReasoningConfig::Budget`] are no-ops: Cerebras
    ///   reasons by default and has no token-budget knob.
    pub fn with_reasoning(mut self, config: ReasoningConfig) -> Self {
        let effort = match config {
            ReasoningConfig::Effort(e) => Some(e),
            ReasoningConfig::Toggle(false) => Some("none".to_string()),
            ReasoningConfig::Toggle(true) | ReasoningConfig::Budget(_) => None,
        };
        if let Some(effort) = effort {
            self.inner = self.inner.with_reasoning(ReasoningConfig::Effort(effort));
        }
        self
    }
}

#[async_trait]
impl Model for CerebrasProvider {
    async fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message> {
        self.inner.complete(messages, tools).await
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> Result<Message> {
        self.inner.complete_stream(messages, tools, sink).await
    }

    fn context_window(&self) -> Option<usize> {
        self.inner.context_window()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_never_enables_thinking_or_reasoning_echo() {
        // With nothing configured, Cerebras reasons by default; this provider
        // must send no reasoning parameter and never echo `reasoning_content`.
        let p = CerebrasProvider::new("k").with_model("zai-glm-4.7");
        assert!(!p.inner.echo_reasoning());
        assert!(!p.inner.has_thinking());
    }

    #[test]
    fn effort_sets_reasoning_effort_but_never_thinking() {
        // `reasoning_effort` flips reasoning echo on, but the `thinking` object
        // Cerebras rejects must stay absent.
        let p = CerebrasProvider::new("k")
            .with_model("gpt-oss-120b")
            .with_reasoning(ReasoningConfig::Effort("high".to_string()));
        assert!(p.inner.echo_reasoning());
        assert!(!p.inner.has_thinking());
    }

    #[test]
    fn toggle_off_maps_to_effort_none_not_thinking() {
        let p = CerebrasProvider::new("k")
            .with_model("zai-glm-4.7")
            .with_reasoning(ReasoningConfig::Toggle(false));
        assert!(p.inner.echo_reasoning());
        assert!(!p.inner.has_thinking());
    }

    #[test]
    fn toggle_on_and_budget_are_noops() {
        let p = CerebrasProvider::new("k")
            .with_reasoning(ReasoningConfig::Toggle(true))
            .with_reasoning(ReasoningConfig::Budget(2048));
        assert!(!p.inner.echo_reasoning());
        assert!(!p.inner.has_thinking());
    }

    #[test]
    fn context_window_delegates_to_inner() {
        let p = CerebrasProvider::new("k").with_context_window(131_072);
        assert_eq!(p.context_window(), Some(131_072));
    }
}
