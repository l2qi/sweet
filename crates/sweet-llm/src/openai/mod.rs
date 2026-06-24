// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use async_trait::async_trait;
use base64::prelude::{Engine as _, BASE64_STANDARD};
use futures_util::StreamExt;
use serde_json::Value;
use sweet_core::message::{ContentBlock, Role, ToolCall};
use sweet_core::stream::StreamSink;
use sweet_core::{Message, Model, Result, ToolSpec, SWEET_VERSION};

use crate::error::ProviderError;
use crate::schema::sanitize_schema;
use crate::util::{elapsed_ms, json_string, provider_error_from_core};
use crate::{ReasoningConfig, SamplingConfig, StructuredOutput, ToolChoice};

mod embeddings;
pub use embeddings::{OpenAIEmbedder, DEFAULT_EMBEDDING_MODEL};

mod reasoning;
pub use reasoning::ReasoningContent;

mod wire;
use wire::{
    ChatRequest, StreamChunk, StreamOptions, WireContent, WireContentPart, WireImageUrl,
    WireMessage, WireTool, WireToolFunction,
};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_API_KEY_ENV: &str = "OPENAI_API_KEY";
pub const DEFAULT_MODEL: &str = "gpt-4o-mini";

/// Inference provider for OpenAI's `/v1/chat/completions` API.
///
/// Compatible with any OpenAI-protocol endpoint (Cerebras, local llama
/// servers, etc.) - point [`OpenAIProvider::with_base_url`] at the right URL.
#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    context_window: Option<usize>,
    user_agent: String,
    reasoning_effort: Option<String>,
    thinking: Option<ThinkingState>,
    reasoning_history_key: ReasoningHistoryKey,
    sampling: SamplingConfig,
    tool_choice: Option<ToolChoice>,
    structured_output: Option<StructuredOutput>,
}

/// Which wire field carries replayed assistant reasoning history on a request.
///
/// OpenAI-compatible backends disagree on whether prior reasoning may be
/// replayed and under which field; the right choice is per-model and (for
/// callers driven by the models.dev catalog) comes from its `interleaved` field:
/// - [`ReasoningContent`](Self::ReasoningContent) - replay under `reasoning_content`
///   (DeepSeek, Kimi, Qwen-on-some).
/// - [`Reasoning`](Self::Reasoning) - replay under `reasoning`.
/// - [`ReasoningDetails`](Self::ReasoningDetails) - replay OpenRouter's structured
///   `reasoning_details` array verbatim.
/// - [`Omit`](Self::Omit) - replay nothing (Cerebras, Qwen, and any model that
///   rejects a replayed reasoning property). This is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningHistoryKey {
    ReasoningContent,
    Reasoning,
    ReasoningDetails,
    Omit,
}

/// Internal state for the OpenAI-compatible `thinking` object. Built via
/// [`OpenAIProvider::with_reasoning`] / [`OpenAIProvider::with_preserved_reasoning_history`]
/// and rendered to the wire by [`OpenAIProvider::thinking_config`].
#[derive(Debug, Clone, Copy)]
struct ThinkingState {
    /// `thinking.type` - `enabled` vs `disabled`.
    enabled: bool,
    /// `thinking.keep = "all"` - Kimi-specific history preservation.
    preserve_history: bool,
    /// `thinking.budget_tokens` - token budget for the reasoning pass.
    budget_tokens: Option<u32>,
}

impl OpenAIProvider {
    /// Construct a provider with an explicit API key, using built-in defaults
    /// for everything else.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            context_window: None,
            user_agent: format!("sweet/{}", SWEET_VERSION),
            reasoning_effort: None,
            thinking: None,
            // Replay nothing by default - most OpenAI-compatible models reject a
            // replayed reasoning property. Replay is opt-in per model via
            // `with_reasoning_history_key` (catalog-driven in downstream crates).
            reasoning_history_key: ReasoningHistoryKey::Omit,
            sampling: SamplingConfig::default(),
            tool_choice: None,
            structured_output: None,
        }
    }

    /// Constrain how the model chooses among tools. Only sent when the request
    /// carries tools.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Constrain the response to JSON matching a schema (`response_format`).
    pub fn with_structured_output(mut self, output: StructuredOutput) -> Self {
        self.structured_output = Some(output);
        self
    }

    /// The `tool_choice` wire value, or `None` when unset or no tools present.
    fn tool_choice_value(&self, has_tools: bool) -> Option<Value> {
        if !has_tools {
            return None;
        }
        Some(match self.tool_choice.as_ref()? {
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::None => serde_json::json!("none"),
            ToolChoice::Required => serde_json::json!("required"),
            ToolChoice::Tool(name) => {
                serde_json::json!({ "type": "function", "function": { "name": name } })
            }
        })
    }

    /// The `response_format` wire value for a structured-output request.
    fn response_format_value(&self) -> Option<Value> {
        let so = self.structured_output.as_ref()?;
        let mut schema = so.schema.clone();
        sanitize_schema(&mut schema);
        Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": so.name_or_default(),
                "schema": schema,
                "strict": so.strict,
            }
        }))
    }

    /// Set cross-provider sampling parameters. Unsupported keys (`top_k`) are
    /// dropped with a warning; the rest map to the chat-completions body.
    pub fn with_sampling(mut self, sampling: SamplingConfig) -> Self {
        if sampling.top_k.is_some() {
            tracing::warn!(
                target: "sweet_llm::openai",
                "top_k is unsupported by the OpenAI chat-completions API; ignoring"
            );
        }
        self.sampling = sampling;
        self
    }

    /// Choose which wire field replays assistant reasoning history (see
    /// [`ReasoningHistoryKey`]). Defaults to [`ReasoningHistoryKey::Omit`] - no
    /// replay. Set this per model from the catalog's `interleaved` field.
    pub fn with_reasoning_history_key(mut self, key: ReasoningHistoryKey) -> Self {
        self.reasoning_history_key = key;
        self
    }

    /// Construct a provider by reading the API key from the standard
    /// `OPENAI_API_KEY` environment variable.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var(DEFAULT_API_KEY_ENV).map_err(|_| ProviderError::MissingApiKey {
            var: DEFAULT_API_KEY_ENV,
        })?;
        Ok(Self::new(key))
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    pub fn with_context_window(mut self, tokens: usize) -> Self {
        self.context_window = Some(tokens);
        self
    }

    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    pub fn prepend_user_agent(mut self, prefix: impl Into<String>) -> Self {
        self.user_agent = format!("{} {}", prefix.into(), self.user_agent);
        self
    }

    /// Configure reasoning for this provider.
    ///
    /// Maps the cross-provider [`ReasoningConfig`] to the OpenAI-compatible
    /// wire:
    /// - [`Toggle(b)`](ReasoningConfig::Toggle) -> `thinking: {type: enabled|disabled}`.
    /// - [`Effort(e)`](ReasoningConfig::Effort) -> `reasoning_effort: e`.
    /// - [`Budget(n)`](ReasoningConfig::Budget) -> `thinking: {type: enabled, budget_tokens: n}`.
    ///
    /// The dialects are mutually exclusive, so setting one clears the others.
    /// When unset, no reasoning field is sent and prior `reasoning_content` is
    /// suppressed from outgoing messages.
    pub fn with_reasoning(mut self, config: ReasoningConfig) -> Self {
        match config {
            ReasoningConfig::Toggle(enabled) => {
                self.thinking = Some(ThinkingState {
                    enabled,
                    preserve_history: false,
                    budget_tokens: None,
                });
                self.reasoning_effort = None;
            }
            ReasoningConfig::Budget(budget_tokens) => {
                self.thinking = Some(ThinkingState {
                    enabled: true,
                    preserve_history: false,
                    budget_tokens: Some(budget_tokens),
                });
                self.reasoning_effort = None;
            }
            ReasoningConfig::Effort(effort) => {
                self.reasoning_effort = Some(effort);
                self.thinking = None;
            }
        }
        self
    }

    /// Ask a thinking-aware backend to re-feed prior turns' `reasoning_content`
    /// (`thinking.keep = "all"`). Kimi-specific (`kimi-k2.6`); ignored by
    /// providers that do not support it. Enables the `thinking` object when one
    /// is not already configured.
    pub fn with_preserved_reasoning_history(mut self, preserve: bool) -> Self {
        match self.thinking.as_mut() {
            Some(state) => state.preserve_history = preserve,
            None => {
                self.thinking = Some(ThinkingState {
                    enabled: true,
                    preserve_history: preserve,
                    budget_tokens: None,
                })
            }
        }
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// The wire field that replays assistant reasoning history (see
    /// [`Self::with_reasoning_history_key`]).
    pub fn reasoning_history_key(&self) -> ReasoningHistoryKey {
        self.reasoning_history_key
    }

    /// Build the per-request `thinking` config, or `None` when the user has
    /// not configured one.
    fn thinking_config(&self) -> Option<wire::ThinkingConfig> {
        self.thinking.map(|m| wire::ThinkingConfig {
            r#type: if m.enabled { "enabled" } else { "disabled" },
            keep: if m.preserve_history {
                Some("all")
            } else {
                None
            },
            budget_tokens: m.budget_tokens,
        })
    }

    /// Whether to echo prior turns' `reasoning_content` on the wire. Only
    /// true when the user has opted into a thinking-aware backend by setting
    /// one of the reasoning parameters.
    pub(crate) fn echo_reasoning(&self) -> bool {
        self.thinking.is_some() || self.reasoning_effort.is_some()
    }

    /// Whether a `thinking` object will be sent. Used by [`CerebrasProvider`](crate::CerebrasProvider)
    /// tests to assert it never enables the object Cerebras rejects.
    #[cfg(test)]
    pub(crate) fn has_thinking(&self) -> bool {
        self.thinking.is_some()
    }

    /// The `User-Agent` header value. Used by [`CerebrasProvider`](crate::CerebrasProvider)
    /// tests to assert the override forwards to the inner transport.
    #[cfg(test)]
    pub(crate) fn user_agent(&self) -> &str {
        &self.user_agent
    }

    fn wire_messages<'a>(&self, messages: &'a [Message]) -> Vec<WireMessage<'a>> {
        let include_reasoning = self.echo_reasoning();
        let mut out: Vec<WireMessage<'a>> = Vec::with_capacity(messages.len());
        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];
            if msg.role != Role::Tool {
                out.push(WireMessage::new_with_key(
                    msg,
                    include_reasoning,
                    self.reasoning_history_key,
                ));
                i += 1;
                continue;
            }

            // A run of consecutive tool results. The Chat Completions protocol
            // requires `tool` message content to be a string, so each tool
            // message stays text (WireMessage::new drops any image blocks).
            // Images a tool returned (e.g. a screenshot) are instead surfaced as
            // a single follow-up `user` message after the whole run - the only
            // place this protocol accepts images - so vision models can see
            // them. The user message must come *after* every tool message in the
            // group: no other role may interleave a turn's tool results.
            let mut image_parts: Vec<WireContentPart> = Vec::new();
            while i < messages.len() && messages[i].role == Role::Tool {
                let tool_msg = &messages[i];
                for block in &tool_msg.content {
                    if let ContentBlock::Image { data, media_type } = block {
                        let b64 = BASE64_STANDARD.encode(data);
                        image_parts.push(WireContentPart::ImageUrl {
                            image_url: WireImageUrl {
                                url: format!("data:{media_type};base64,{b64}"),
                                detail: None,
                            },
                        });
                    }
                }
                out.push(WireMessage::new_with_key(
                    tool_msg,
                    include_reasoning,
                    self.reasoning_history_key,
                ));
                i += 1;
            }

            if !image_parts.is_empty() {
                let mut parts = Vec::with_capacity(image_parts.len() + 1);
                parts.push(WireContentPart::Text {
                    text: "Image(s) returned by the preceding tool call(s):".to_string(),
                });
                parts.extend(image_parts);
                out.push(WireMessage {
                    role: "user",
                    content: Some(WireContent::Parts(parts)),
                    reasoning_content: None,
                    reasoning: None,
                    reasoning_details: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                });
            }
        }
        out
    }

    async fn complete_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> std::result::Result<Message, ProviderError> {
        let tools_wire = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| {
                        let mut params = t.parameters_schema.clone();
                        sanitize_schema(&mut params);
                        WireTool {
                            r#type: "function",
                            function: WireToolFunction {
                                name: &t.name,
                                description: &t.description,
                                parameters: params,
                            },
                        }
                    })
                    .collect(),
            )
        };

        let body = ChatRequest {
            model: &self.model,
            messages: self.wire_messages(messages),
            tools: tools_wire,
            stream: false,
            stream_options: None,
            reasoning_effort: self.reasoning_effort.as_deref(),
            thinking: self.thinking_config(),
            temperature: self.sampling.temperature,
            top_p: self.sampling.top_p,
            frequency_penalty: self.sampling.frequency_penalty,
            presence_penalty: self.sampling.presence_penalty,
            seed: self.sampling.seed,
            max_tokens: self.sampling.max_tokens,
            stop: (!self.sampling.stop.is_empty()).then_some(self.sampling.stop.as_slice()),
            tool_choice: self.tool_choice_value(!tools.is_empty()),
            response_format: self.response_format_value(),
        };
        let mut body_value = serde_json::to_value(&body)?;
        crate::sampling::merge_extra(&mut body_value, &self.sampling.extra);

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let request_body = json_string(&body_value);
        tracing::debug!(
            target: "sweet_llm::observability",
            event = "openai.complete.start",
            base_url = %self.base_url,
            endpoint = %url,
            model = %self.model,
            message_count = messages.len(),
            tool_count = tools.len(),
            request_body = %request_body,
            "openai complete start"
        );

        let started = Instant::now();
        let mut req = self
            .http
            .post(&url)
            .header("User-Agent", &self.user_agent)
            .json(&body_value);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = match req.send().await {
            Ok(resp) => resp,
            Err(err) => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_llm::observability",
                    event = "openai.complete",
                    base_url = %self.base_url,
                    endpoint = %url,
                    model = %self.model,
                    message_count = messages.len(),
                    tool_count = tools.len(),
                    duration_ms,
                    status = "error",
                    error = %err,
                    "openai complete network error"
                );
                return Err(err.into());
            }
        };

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let duration_ms = elapsed_ms(started);
            tracing::debug!(
                target: "sweet_llm::observability",
                event = "openai.complete",
                base_url = %self.base_url,
                endpoint = %url,
                model = %self.model,
                message_count = messages.len(),
                tool_count = tools.len(),
                duration_ms,
                status = "error",
                http_status = status.as_u16(),
                response_body = %body,
                "openai complete http error"
            );
            return Err(ProviderError::Http { status, body });
        }

        let response_body = match resp.text().await {
            Ok(body) => body,
            Err(err) => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_llm::observability",
                    event = "openai.complete",
                    base_url = %self.base_url,
                    endpoint = %url,
                    model = %self.model,
                    message_count = messages.len(),
                    tool_count = tools.len(),
                    duration_ms,
                    status = "error",
                    http_status = status.as_u16(),
                    error = %err,
                    "openai complete response body read error"
                );
                return Err(err.into());
            }
        };
        let parsed: wire::ChatResponse = match serde_json::from_str(&response_body) {
            Ok(parsed) => parsed,
            Err(err) => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_llm::observability",
                    event = "openai.complete",
                    base_url = %self.base_url,
                    endpoint = %url,
                    model = %self.model,
                    message_count = messages.len(),
                    tool_count = tools.len(),
                    duration_ms,
                    status = "error",
                    http_status = status.as_u16(),
                    response_body = %response_body,
                    error = %err,
                    "openai complete decode error"
                );
                return Err(err.into());
            }
        };
        let choice = match parsed.choices.into_iter().next() {
            Some(choice) => choice,
            None => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_llm::observability",
                    event = "openai.complete",
                    base_url = %self.base_url,
                    endpoint = %url,
                    model = %self.model,
                    message_count = messages.len(),
                    tool_count = tools.len(),
                    duration_ms,
                    status = "error",
                    http_status = status.as_u16(),
                    response_body = %response_body,
                    error = %ProviderError::EmptyResponse,
                    "openai complete empty response"
                );
                return Err(ProviderError::EmptyResponse);
            }
        };
        let response_message = choice.message;
        let response_message_json = json_string(&response_message);
        let mut reply = match Message::try_from(response_message) {
            Ok(reply) => reply,
            Err(err) => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_llm::observability",
                    event = "openai.complete",
                    base_url = %self.base_url,
                    endpoint = %url,
                    model = %self.model,
                    message_count = messages.len(),
                    tool_count = tools.len(),
                    duration_ms,
                    status = "error",
                    http_status = status.as_u16(),
                    response_body = %response_body,
                    response_message = %response_message_json,
                    error = %err,
                    "openai complete response conversion error"
                );
                return Err(err);
            }
        };

        if let Some(usage) = parsed.usage {
            reply.token_count = Some(usage.total_tokens);
            reply.context_tokens = Some(usage.prompt_tokens);
        }
        reply.finish_reason = choice.finish_reason.as_deref().map(wire::map_finish_reason);

        let duration_ms = elapsed_ms(started);
        tracing::debug!(
            target: "sweet_llm::observability",
            event = "openai.complete",
            base_url = %self.base_url,
            endpoint = %url,
            model = %self.model,
            message_count = messages.len(),
            tool_count = tools.len(),
            duration_ms,
            status = "ok",
            http_status = status.as_u16(),
            response_body = %response_body,
            assistant = %json_string(&reply),
            "openai complete"
        );
        Ok(reply)
    }

    async fn complete_stream_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> std::result::Result<Message, ProviderError> {
        let tools_wire = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| {
                        let mut params = t.parameters_schema.clone();
                        sanitize_schema(&mut params);
                        WireTool {
                            r#type: "function",
                            function: WireToolFunction {
                                name: &t.name,
                                description: &t.description,
                                parameters: params,
                            },
                        }
                    })
                    .collect(),
            )
        };

        let body = ChatRequest {
            model: &self.model,
            messages: self.wire_messages(messages),
            tools: tools_wire,
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            reasoning_effort: self.reasoning_effort.as_deref(),
            thinking: self.thinking_config(),
            temperature: self.sampling.temperature,
            top_p: self.sampling.top_p,
            frequency_penalty: self.sampling.frequency_penalty,
            presence_penalty: self.sampling.presence_penalty,
            seed: self.sampling.seed,
            max_tokens: self.sampling.max_tokens,
            stop: (!self.sampling.stop.is_empty()).then_some(self.sampling.stop.as_slice()),
            tool_choice: self.tool_choice_value(!tools.is_empty()),
            response_format: self.response_format_value(),
        };
        let mut body_value = serde_json::to_value(&body)?;
        crate::sampling::merge_extra(&mut body_value, &self.sampling.extra);

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        tracing::debug!(
            target: "sweet_llm::observability",
            event = "openai.complete_stream.start",
            base_url = %self.base_url,
            endpoint = %url,
            model = %self.model,
            message_count = messages.len(),
            tool_count = tools.len(),
            "openai stream complete start"
        );

        let started = Instant::now();
        let mut req = self
            .http
            .post(&url)
            .header("User-Agent", &self.user_agent)
            .json(&body_value);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send().await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let duration_ms = elapsed_ms(started);
            tracing::debug!(
                target: "sweet_llm::observability",
                event = "openai.complete_stream",
                base_url = %self.base_url,
                endpoint = %url,
                model = %self.model,
                duration_ms,
                status = "error",
                http_status = status.as_u16(),
                response_body = %body,
                "openai stream complete http error"
            );
            return Err(ProviderError::Http { status, body });
        }

        let mut stream = resp.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();
        let mut content = String::new();
        let mut reasoning_content = String::new();
        // Tracked separately from `reasoning_content.is_empty()` so that an
        // explicit `reasoning_content: ""` from Kimi round-trips as a single
        // empty-text block (matches the non-streaming `TryFrom` path).
        let mut saw_reasoning_field = false;
        // OpenRouter streams structured `reasoning_details` as fragments of one
        // logical block (metadata first, then incremental text) keyed by `index`.
        // Reassemble per index so the final blocks match the non-streaming
        // response and replay verbatim - naive concatenation would emit N partial
        // entries and 400 on the next request. Takes precedence over the string
        // view below.
        let mut reasoning_accums: Vec<ReasoningDetailAccum> = Vec::new();
        let mut tool_call_accums: Vec<ToolCallAccum> = Vec::new();
        let mut total_tokens: Option<usize> = None;
        let mut prompt_tokens: Option<usize> = None;
        let mut finish_reason: Option<String> = None;
        let mut done = false;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buffer.extend_from_slice(&bytes);
            while let Some(result) = crate::sse::drain_event(&mut buffer) {
                let event_text = result?;
                for data in crate::sse::data_lines(&event_text) {
                    if data == "[DONE]" {
                        done = true;
                        break;
                    }
                    let chunk: StreamChunk = serde_json::from_str(data)?;
                    if let Some(usage) = chunk.usage {
                        total_tokens = Some(usage.total_tokens);
                        prompt_tokens = Some(usage.prompt_tokens);
                    }
                    for choice in chunk.choices {
                        if let Some(ref fr) = choice.finish_reason {
                            finish_reason = Some(fr.clone());
                        }
                        let reasoning_delta = choice
                            .delta
                            .reasoning_content
                            .as_deref()
                            .or(choice.delta.reasoning.as_deref());
                        // Whether a non-empty reasoning string actually streamed
                        // this delta (an explicit `reasoning: ""` does not count).
                        let streamed_reasoning_text =
                            reasoning_delta.is_some_and(|s| !s.is_empty());
                        if let Some(rc) = reasoning_delta {
                            saw_reasoning_field = true;
                            if !rc.is_empty() {
                                reasoning_content.push_str(rc);
                                sink.on_thinking_delta(rc)
                                    .await
                                    .map_err(provider_error_from_core)?;
                            }
                        }
                        for block in choice.delta.reasoning_details.into_iter().flatten() {
                            // Surface this fragment's text live, unless a reasoning
                            // string already streamed it this delta (avoids
                            // double-rendering when a provider sends both encodings).
                            if !streamed_reasoning_text {
                                if let Some(t) = block
                                    .get("text")
                                    .and_then(|v| v.as_str())
                                    .filter(|t| !t.is_empty())
                                {
                                    sink.on_thinking_delta(t)
                                        .await
                                        .map_err(provider_error_from_core)?;
                                }
                            }
                            // Merge into the entry for this block's `index` so a
                            // fragmented block reassembles into one (verbatim replay).
                            let idx = block
                                .get("index")
                                .and_then(|v| v.as_u64())
                                .map(|n| n as usize)
                                .unwrap_or(reasoning_accums.len());
                            if reasoning_accums.len() <= idx {
                                reasoning_accums
                                    .resize_with(idx + 1, ReasoningDetailAccum::default);
                            }
                            reasoning_accums[idx].merge(block);
                        }
                        if !choice.delta.content.is_empty() {
                            content.push_str(&choice.delta.content);
                            sink.on_content_delta(&choice.delta.content)
                                .await
                                .map_err(provider_error_from_core)?;
                        }
                        for tc_delta in choice.delta.tool_calls {
                            if tool_call_accums.len() <= tc_delta.index {
                                tool_call_accums
                                    .resize_with(tc_delta.index + 1, ToolCallAccum::default);
                            }
                            let accum = &mut tool_call_accums[tc_delta.index];
                            if let Some(id) = tc_delta.id {
                                accum.id = id;
                            }
                            if let Some(fn_delta) = tc_delta.function {
                                if let Some(name) = fn_delta.name {
                                    accum.name.push_str(&name);
                                }
                                if let Some(args) = fn_delta.arguments {
                                    accum.arguments.push_str(&args);
                                }
                            }
                        }
                    }
                }
                if done {
                    break;
                }
            }
            if done {
                break;
            }
        }

        let mut tool_calls = Vec::with_capacity(tool_call_accums.len());
        for accum in tool_call_accums {
            let arguments: serde_json::Value = if accum.arguments.is_empty() {
                serde_json::Value::Object(serde_json::Map::new())
            } else {
                serde_json::from_str(&accum.arguments)?
            };
            let call = ToolCall {
                id: accum.id,
                name: accum.name,
                arguments,
            };
            sink.on_tool_call(&call)
                .await
                .map_err(provider_error_from_core)?;
            tool_calls.push(call);
        }

        let mut reply = Message {
            role: Role::Assistant,
            content: vec![sweet_core::ContentBlock::text(content)],
            thinking_content: Vec::new(),
            tool_calls,
            tool_call_id: None,
            token_count: total_tokens,
            context_tokens: prompt_tokens,
            compacted: false,
            finish_reason: None,
        };
        // Precedence mirrors the non-streaming path: structured blocks
        // (reassembled, preserved verbatim) win over the single-string view.
        // Skip empty slots left by any gap in the `index` sequence.
        let reasoning_blocks: Vec<_> = reasoning_accums
            .into_iter()
            .filter(|a| !a.is_empty())
            .map(|a| wire::thinking_from_detail(a.into_value()))
            .collect();
        if !reasoning_blocks.is_empty() {
            reply.thinking_content = reasoning_blocks;
        } else if saw_reasoning_field {
            reply.set_reasoning_content(reasoning_content);
        }
        reply.finish_reason = finish_reason.as_deref().map(wire::map_finish_reason);

        let duration_ms = elapsed_ms(started);
        tracing::debug!(
            target: "sweet_llm::observability",
            event = "openai.complete_stream",
            base_url = %self.base_url,
            endpoint = %url,
            model = %self.model,
            duration_ms,
            status = "ok",
            http_status = status.as_u16(),
            assistant = %json_string(&reply),
            "openai stream complete"
        );
        Ok(reply)
    }
}

#[async_trait]
impl Model for OpenAIProvider {
    async fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message> {
        Ok(self.complete_inner(messages, tools).await?)
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> Result<Message> {
        Ok(self.complete_stream_inner(messages, tools, sink).await?)
    }

    fn context_window(&self) -> Option<usize> {
        self.context_window
    }
}

#[derive(Default)]
struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
}

/// Reassembles the streamed fragments of one OpenRouter `reasoning_details`
/// block (grouped by `index`) into a single block, mirroring [`ToolCallAccum`].
/// Content fields (`text`/`summary`/`data`) arrive incrementally and concatenate;
/// identity fields (`type`/`index`/`format`/`id`/`signature`) are stable and keep
/// their latest value.
#[derive(Default)]
struct ReasoningDetailAccum {
    block: serde_json::Map<String, serde_json::Value>,
}

impl ReasoningDetailAccum {
    fn is_empty(&self) -> bool {
        self.block.is_empty()
    }

    fn into_value(self) -> serde_json::Value {
        serde_json::Value::Object(self.block)
    }

    fn merge(&mut self, incoming: serde_json::Value) {
        let map = match incoming {
            serde_json::Value::Object(map) => map,
            other => {
                // A well-formed `reasoning_details` element is always an object;
                // log and skip anything else rather than failing the stream.
                tracing::debug!(
                    target: "sweet_llm::observability",
                    fragment = %other,
                    "ignoring non-object reasoning_details fragment"
                );
                return;
            }
        };
        for (k, v) in map {
            let concat = matches!(k.as_str(), "text" | "summary" | "data")
                && v.is_string()
                && matches!(self.block.get(&k), Some(serde_json::Value::String(_)));
            if concat {
                if let Some(serde_json::Value::String(existing)) = self.block.get_mut(&k) {
                    existing.push_str(v.as_str().unwrap_or_default());
                }
            } else {
                self.block.insert(k, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_user_agent_is_sweet_version() {
        let p = OpenAIProvider::new("k");
        assert_eq!(p.user_agent, format!("sweet/{}", SWEET_VERSION));
    }

    #[test]
    fn with_user_agent_overwrites() {
        let p = OpenAIProvider::new("k").with_user_agent("custom/1.0");
        assert_eq!(p.user_agent, "custom/1.0");
    }

    #[test]
    fn prepend_user_agent_prepends_with_space() {
        let p = OpenAIProvider::new("k").prepend_user_agent("app/1.0");
        assert_eq!(p.user_agent, format!("app/1.0 sweet/{}", SWEET_VERSION));
    }

    #[test]
    fn thinking_config_is_none_when_unset() {
        let p = OpenAIProvider::new("k");
        assert!(p.thinking_config().is_none());
    }

    #[test]
    fn toggle_on_emits_type_enabled_without_keep() {
        let cfg = OpenAIProvider::new("k")
            .with_reasoning(ReasoningConfig::Toggle(true))
            .thinking_config()
            .unwrap();
        assert_eq!(cfg.r#type, "enabled");
        assert_eq!(cfg.keep, None);
        assert_eq!(cfg.budget_tokens, None);
    }

    #[test]
    fn toggle_off_emits_type_disabled() {
        let cfg = OpenAIProvider::new("k")
            .with_reasoning(ReasoningConfig::Toggle(false))
            .thinking_config()
            .unwrap();
        assert_eq!(cfg.r#type, "disabled");
        assert_eq!(cfg.keep, None);
    }

    #[test]
    fn budget_emits_enabled_with_budget_tokens() {
        let cfg = OpenAIProvider::new("k")
            .with_reasoning(ReasoningConfig::Budget(2048))
            .thinking_config()
            .unwrap();
        assert_eq!(cfg.r#type, "enabled");
        assert_eq!(cfg.budget_tokens, Some(2048));
    }

    #[test]
    fn effort_sets_reasoning_effort_not_thinking() {
        let p =
            OpenAIProvider::new("k").with_reasoning(ReasoningConfig::Effort("high".to_string()));
        assert_eq!(p.reasoning_effort.as_deref(), Some("high"));
        assert!(p.thinking_config().is_none());
    }

    #[test]
    fn setting_one_dialect_clears_the_other() {
        // Effort after a toggle drops the thinking object, and vice-versa.
        let p = OpenAIProvider::new("k")
            .with_reasoning(ReasoningConfig::Toggle(true))
            .with_reasoning(ReasoningConfig::Effort("low".to_string()));
        assert!(p.thinking_config().is_none());
        assert_eq!(p.reasoning_effort.as_deref(), Some("low"));

        let p = OpenAIProvider::new("k")
            .with_reasoning(ReasoningConfig::Effort("low".to_string()))
            .with_reasoning(ReasoningConfig::Toggle(true));
        assert!(p.thinking_config().is_some());
        assert_eq!(p.reasoning_effort, None);
    }

    #[test]
    fn preserved_history_emits_enabled_with_keep_all() {
        let cfg = OpenAIProvider::new("k")
            .with_reasoning(ReasoningConfig::Toggle(true))
            .with_preserved_reasoning_history(true)
            .thinking_config()
            .unwrap();
        assert_eq!(cfg.r#type, "enabled");
        assert_eq!(cfg.keep, Some("all"));
    }

    #[test]
    fn preserved_history_without_prior_thinking_enables_it() {
        let cfg = OpenAIProvider::new("k")
            .with_preserved_reasoning_history(true)
            .thinking_config()
            .unwrap();
        assert_eq!(cfg.r#type, "enabled");
        assert_eq!(cfg.keep, Some("all"));
    }

    #[test]
    fn echo_reasoning_off_by_default() {
        let p = OpenAIProvider::new("k");
        assert!(!p.echo_reasoning());
    }

    #[test]
    fn echo_reasoning_on_when_thinking_or_effort_set() {
        assert!(OpenAIProvider::new("k")
            .with_reasoning(ReasoningConfig::Toggle(true))
            .echo_reasoning());
        assert!(OpenAIProvider::new("k")
            .with_reasoning(ReasoningConfig::Toggle(false))
            .echo_reasoning());
        assert!(OpenAIProvider::new("k")
            .with_reasoning(ReasoningConfig::Budget(1024))
            .echo_reasoning());
        assert!(OpenAIProvider::new("k")
            .with_reasoning(ReasoningConfig::Effort("high".to_string()))
            .echo_reasoning());
    }

    #[test]
    fn tool_choice_maps_to_wire() {
        let p = OpenAIProvider::new("k").with_tool_choice(ToolChoice::Required);
        assert_eq!(
            p.tool_choice_value(true),
            Some(serde_json::json!("required"))
        );
        // Omitted when the request carries no tools.
        assert_eq!(p.tool_choice_value(false), None);

        let p = OpenAIProvider::new("k").with_tool_choice(ToolChoice::Tool("x".into()));
        assert_eq!(
            p.tool_choice_value(true),
            Some(serde_json::json!({ "type": "function", "function": { "name": "x" } }))
        );
    }

    #[test]
    fn structured_output_maps_to_response_format() {
        let p = OpenAIProvider::new("k").with_structured_output(StructuredOutput::new(
            serde_json::json!({ "type": "object" }),
        ));
        let rf = p.response_format_value().unwrap();
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["name"], "response");
        assert_eq!(rf["json_schema"]["strict"], true);
        assert_eq!(rf["json_schema"]["schema"]["type"], "object");
    }
}
