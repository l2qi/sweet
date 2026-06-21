// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::Value;
use sweet_core::message::ToolCall;
use sweet_core::stream::StreamSink;
use sweet_core::{Message, Model, Result, ToolSpec, SWEET_VERSION};

use crate::error::ProviderError;
use crate::schema::sanitize_schema;
use crate::util::{elapsed_ms, provider_error_from_core};
use crate::{ReasoningConfig, SamplingConfig, StructuredOutput, ToolChoice};

mod wire;
use wire::{
    convert_messages, message_from_content_blocks, parse_response, ContentBlock, MessagesRequest,
    MessagesResponse, OutputConfig, StreamDelta, StreamEvent, WireThinking, WireTool,
};

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
pub const DEFAULT_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";
pub const DEFAULT_MAX_TOKENS: usize = 4096;
pub const API_VERSION: &str = "2023-06-01";
/// Anthropic requires extended-thinking budgets of at least 1024 tokens.
const MIN_THINKING_BUDGET: usize = 1024;

/// Inference provider for Anthropic's native `/v1/messages` API.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    max_tokens: usize,
    user_agent: String,
    reasoning: Option<ReasoningConfig>,
    sampling: SamplingConfig,
    prompt_caching: bool,
    tool_choice: Option<ToolChoice>,
    structured_output: Option<StructuredOutput>,
    auth_token: Option<String>,
    extra_betas: Vec<String>,
}

/// Resolved sampling fields for one request (after Anthropic's clamps).
struct SamplingFields<'a> {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    stop_sequences: Option<&'a [String]>,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            user_agent: format!("sweet/{}", SWEET_VERSION),
            reasoning: None,
            sampling: SamplingConfig::default(),
            prompt_caching: true,
            tool_choice: None,
            structured_output: None,
            auth_token: None,
            extra_betas: Vec::new(),
        }
    }

    /// Authenticate with `Authorization: Bearer <token>` (e.g. an OAuth access
    /// token) instead of `x-api-key`. Takes precedence over the API key.
    pub fn with_auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Add an `anthropic-beta` header value sent with every request.
    pub fn with_beta(mut self, beta: impl Into<String>) -> Self {
        self.extra_betas.push(beta.into());
        self
    }

    /// Constrain how the model chooses among tools. Only sent when the request
    /// carries tools.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Constrain the response to JSON matching a schema via native
    /// `output_config.format` (adds the `structured-outputs` beta header).
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
            ToolChoice::Auto => serde_json::json!({ "type": "auto" }),
            ToolChoice::None => serde_json::json!({ "type": "none" }),
            ToolChoice::Required => serde_json::json!({ "type": "any" }),
            ToolChoice::Tool(name) => serde_json::json!({ "type": "tool", "name": name }),
        })
    }

    /// `anthropic-beta` header values to send for this request: any explicitly
    /// added betas, the structured-output beta when structured output is
    /// requested, and the PDF beta when the prompt carries a file attachment.
    fn betas(&self, messages: &[Message]) -> Vec<String> {
        let mut betas = self.extra_betas.clone();
        if self.structured_output.is_some() {
            betas.push("structured-outputs-2025-11-13".to_string());
        }
        if messages.iter().any(|m| m.has_files()) {
            betas.push("pdfs-2024-09-25".to_string());
        }
        betas
    }

    /// Enable or disable automatic prompt caching (on by default). When on, the
    /// provider inserts `cache_control` breakpoints on the tools, system prompt,
    /// and last message so Anthropic caches the stable prefix across turns.
    pub fn with_prompt_caching(mut self, enabled: bool) -> Self {
        self.prompt_caching = enabled;
        self
    }

    /// Set cross-provider sampling parameters. `frequency_penalty`,
    /// `presence_penalty`, and `seed` are unsupported by Anthropic and dropped
    /// with a warning; `temperature`/`top_p`/`top_k` are additionally suppressed
    /// when extended thinking is on (the API rejects them together).
    pub fn with_sampling(mut self, sampling: SamplingConfig) -> Self {
        if sampling.frequency_penalty.is_some()
            || sampling.presence_penalty.is_some()
            || sampling.seed.is_some()
        {
            tracing::warn!(
                target: "sweet_llm::anthropic",
                "frequency_penalty/presence_penalty/seed are unsupported by Anthropic; ignoring"
            );
        }
        self.sampling = sampling;
        self
    }

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

    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
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

    /// Configure reasoning for this provider. The provider never sniffs the
    /// model name; the caller picks the variant the configured model accepts.
    ///
    /// - [`Toggle(true)`](ReasoningConfig::Toggle) -> `thinking: {type: adaptive}`,
    ///   the modern Claude 4.6+ shape. Models that reject adaptive thinking
    ///   should be sent an explicit [`Budget`](ReasoningConfig::Budget) instead.
    /// - [`Toggle(false)`](ReasoningConfig::Toggle) -> `thinking: {type: disabled}`.
    /// - [`Budget(n)`](ReasoningConfig::Budget) -> `thinking: {type: enabled, budget_tokens: n}`
    ///   (sent verbatim, clamped only to Anthropic's `1024 <= b < max_tokens` rule).
    /// - [`Effort(e)`](ReasoningConfig::Effort) -> `output_config: {effort: e}` (no `thinking` object).
    pub fn with_reasoning(mut self, config: ReasoningConfig) -> Self {
        self.reasoning = Some(config);
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn wire_thinking(&self) -> Option<WireThinking> {
        let reasoning = self.reasoning.as_ref()?;
        Some(match reasoning {
            ReasoningConfig::Toggle(false) => WireThinking::Disabled,
            // Effort rides on `output_config`, not `thinking`.
            ReasoningConfig::Effort(_) => return None,
            // `Toggle(true)` ("reasoning on") maps to adaptive thinking, the
            // modern Claude 4.6+ shape. Models that reject `{type: adaptive}`
            // must be sent an explicit budget by the caller; the provider never
            // rewrites based on the model name.
            ReasoningConfig::Toggle(true) => WireThinking::Adaptive,
            // An explicit budget is emitted verbatim, clamped only to
            // Anthropic's hard `1024 <= b < max_tokens` rule. When the output
            // cap is too small for a valid budget (`thinking_budget() == 0`),
            // thinking is dropped rather than sending a request Anthropic would
            // reject with a 400.
            ReasoningConfig::Budget(_) => {
                let budget = self.thinking_budget();
                if budget == 0 {
                    return None;
                }
                WireThinking::Enabled {
                    budget_tokens: budget,
                }
            }
        })
    }

    /// The `output_config` block: the reasoning `effort` level (effort dialect)
    /// and/or the structured-output `format` schema. `None` when neither is set.
    fn wire_output_config(&self) -> Option<OutputConfig<'_>> {
        let effort = match self.reasoning.as_ref() {
            Some(ReasoningConfig::Effort(e)) => Some(e.as_str()),
            _ => None,
        };
        let format = self.structured_output.as_ref().map(|so| {
            let mut schema = so.schema.clone();
            sanitize_schema(&mut schema);
            serde_json::json!({ "type": "json_schema", "schema": schema })
        });
        if effort.is_none() && format.is_none() {
            return None;
        }
        Some(OutputConfig { effort, format })
    }

    /// The thinking-token budget that would be sent, clamped to Anthropic's
    /// `1024 <= budget_tokens < max_tokens` rule. The upper bound is the
    /// *request* cap ([`request_max_tokens`](Self::request_max_tokens)), not the
    /// model ceiling, so a `sampling.max_tokens` override bounds the budget too
    /// (otherwise `budget_tokens` could exceed `max_tokens` and the API 400s).
    /// Returns `0` unless reasoning is an explicit [`Budget`](ReasoningConfig::Budget);
    /// also `0` when the output cap leaves no room for a budget >= the 1024
    /// minimum (there is no valid `budget_tokens` in that case, so
    /// [`wire_thinking`](Self::wire_thinking) drops thinking rather than send
    /// an invalid request).
    fn thinking_budget(&self) -> usize {
        let requested = match self.reasoning.as_ref() {
            Some(ReasoningConfig::Budget(n)) => *n as usize,
            _ => return 0,
        };
        let upper = self.request_max_tokens().saturating_sub(1);
        // If the output cap leaves no room for a budget >= the 1024 minimum,
        // there is no valid `budget_tokens` value: drop thinking (return 0)
        // rather than send one Anthropic would reject with a 400.
        if upper < MIN_THINKING_BUDGET {
            return 0;
        }
        requested.clamp(MIN_THINKING_BUDGET, upper)
    }

    /// The output cap to send: an explicit sampling override, else the model
    /// ceiling configured via [`with_max_tokens`](Self::with_max_tokens).
    fn request_max_tokens(&self) -> usize {
        self.sampling.max_tokens.unwrap_or(self.max_tokens)
    }

    /// Sampling fields for the request, applying Anthropic's constraints:
    /// `temperature`/`top_p`/`top_k` are dropped while thinking is on, and
    /// `temperature` and `top_p` are mutually exclusive.
    fn sampling_fields(&self) -> SamplingFields<'_> {
        let stop_sequences =
            (!self.sampling.stop.is_empty()).then_some(self.sampling.stop.as_slice());
        let thinking_on = matches!(
            self.wire_thinking(),
            Some(WireThinking::Adaptive | WireThinking::Enabled { .. })
        );
        if thinking_on {
            return SamplingFields {
                temperature: None,
                top_p: None,
                top_k: None,
                stop_sequences,
            };
        }
        let temperature = self.sampling.temperature;
        let top_p = if temperature.is_some() {
            None
        } else {
            self.sampling.top_p
        };
        SamplingFields {
            temperature,
            top_p,
            top_k: self.sampling.top_k,
            stop_sequences,
        }
    }

    async fn complete_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> std::result::Result<Message, ProviderError> {
        let (system, anthropic_messages) = convert_messages(messages);

        let tools_wire = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| WireTool {
                        name: &t.name,
                        description: &t.description,
                        input_schema: t.parameters_schema.clone(),
                    })
                    .collect(),
            )
        };

        let SamplingFields {
            temperature,
            top_p,
            top_k,
            stop_sequences,
        } = self.sampling_fields();
        let body = MessagesRequest {
            model: &self.model,
            max_tokens: self.request_max_tokens(),
            system,
            messages: anthropic_messages,
            tools: tools_wire,
            stream: false,
            thinking: self.wire_thinking(),
            output_config: self.wire_output_config(),
            temperature,
            top_p,
            top_k,
            stop_sequences,
            tool_choice: self.tool_choice_value(!tools.is_empty()),
        };
        let mut body_value = serde_json::to_value(&body)?;
        crate::sampling::merge_extra(&mut body_value, &self.sampling.extra);
        if self.prompt_caching {
            wire::apply_prompt_caching(&mut body_value);
        }

        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        let started = Instant::now();
        let mut req = self
            .http
            .post(&url)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .header("User-Agent", &self.user_agent)
            .json(&body_value);
        let betas = self.betas(messages);
        if !betas.is_empty() {
            req = req.header("anthropic-beta", betas.join(","));
        }
        if let Some(token) = &self.auth_token {
            req = req.header("authorization", format!("Bearer {token}"));
        } else if !self.api_key.is_empty() {
            req = req.header("x-api-key", &self.api_key);
        }
        let resp = req.send().await?;

        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Http {
                status,
                body: body_text,
            });
        }

        let response_body = resp.text().await?;
        let parsed: MessagesResponse = serde_json::from_str(&response_body)?;
        let reply = parse_response(parsed)?;

        tracing::debug!(
            target: "sweet_llm::observability",
            event = "anthropic.complete",
            duration_ms = elapsed_ms(started),
            status = "ok",
            model = %self.model,
            "anthropic complete"
        );

        Ok(reply)
    }

    async fn complete_stream_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> std::result::Result<Message, ProviderError> {
        let (system, anthropic_messages) = convert_messages(messages);

        let tools_wire = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| WireTool {
                        name: &t.name,
                        description: &t.description,
                        input_schema: t.parameters_schema.clone(),
                    })
                    .collect(),
            )
        };

        let SamplingFields {
            temperature,
            top_p,
            top_k,
            stop_sequences,
        } = self.sampling_fields();
        let body = MessagesRequest {
            model: &self.model,
            max_tokens: self.request_max_tokens(),
            system,
            messages: anthropic_messages,
            tools: tools_wire,
            stream: true,
            thinking: self.wire_thinking(),
            output_config: self.wire_output_config(),
            temperature,
            top_p,
            top_k,
            stop_sequences,
            tool_choice: self.tool_choice_value(!tools.is_empty()),
        };
        let mut body_value = serde_json::to_value(&body)?;
        crate::sampling::merge_extra(&mut body_value, &self.sampling.extra);
        if self.prompt_caching {
            wire::apply_prompt_caching(&mut body_value);
        }

        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        let started = Instant::now();
        let mut req = self
            .http
            .post(&url)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .header("User-Agent", &self.user_agent)
            .json(&body_value);
        let betas = self.betas(messages);
        if !betas.is_empty() {
            req = req.header("anthropic-beta", betas.join(","));
        }
        if let Some(token) = &self.auth_token {
            req = req.header("authorization", format!("Bearer {token}"));
        } else if !self.api_key.is_empty() {
            req = req.header("x-api-key", &self.api_key);
        }
        let resp = req.send().await?;

        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Http {
                status,
                body: body_text,
            });
        }

        let mut stream = resp.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();

        let mut block_states: Vec<BlockState> = Vec::new();
        let mut input_tokens: Option<usize> = None;
        let mut cache_creation_input_tokens: Option<usize> = None;
        let mut cache_read_input_tokens: Option<usize> = None;
        let mut output_tokens: Option<usize> = None;
        let mut stop_reason: Option<String> = None;
        let mut done = false;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buffer.extend_from_slice(&bytes);
            while let Some(result) = crate::sse::drain_event(&mut buffer) {
                let event_text = result?;
                let Some(data) = crate::sse::data_lines(&event_text).last() else {
                    continue;
                };

                let event: StreamEvent = serde_json::from_str(data)?;
                match event {
                    StreamEvent::MessageStart { message } => {
                        if let Some(usage) = message.usage {
                            // `input_tokens` is the *uncached* count; the cache
                            // read/write totals are reported separately and must
                            // be folded in so context tracking matches the
                            // non-streaming path (see `Usage::total_input`).
                            input_tokens = usage.input_tokens;
                            cache_creation_input_tokens = usage.cache_creation_input_tokens;
                            cache_read_input_tokens = usage.cache_read_input_tokens;
                        }
                    }
                    StreamEvent::ContentBlockStart {
                        index,
                        content_block,
                    } => {
                        if block_states.len() <= index {
                            block_states.resize_with(index + 1, || BlockState::Text(String::new()));
                        }
                        match content_block {
                            ContentBlock::Text { .. } => {
                                block_states[index] = BlockState::Text(String::new());
                            }
                            ContentBlock::ToolUse { id, name, .. }
                            | ContentBlock::ServerToolUse { id, name, .. } => {
                                block_states[index] = BlockState::ToolUse {
                                    id,
                                    name,
                                    partial_json: String::new(),
                                };
                            }
                            ContentBlock::Thinking { .. } => {
                                block_states[index] = BlockState::Thinking {
                                    text: String::new(),
                                    signature: String::new(),
                                };
                            }
                            ContentBlock::RedactedThinking { .. } | ContentBlock::Unknown => {
                                block_states[index] = BlockState::Unknown;
                            }
                        }
                    }
                    StreamEvent::ContentBlockDelta { index, delta } => {
                        if index >= block_states.len() {
                            continue;
                        }
                        match delta {
                            StreamDelta::TextDelta { text } => {
                                if let BlockState::Text(ref mut acc) = block_states[index] {
                                    acc.push_str(&text);
                                    sink.on_content_delta(&text)
                                        .await
                                        .map_err(provider_error_from_core)?;
                                }
                            }
                            StreamDelta::InputJsonDelta { partial_json } => {
                                if let BlockState::ToolUse {
                                    partial_json: ref mut acc,
                                    ..
                                } = block_states[index]
                                {
                                    acc.push_str(&partial_json);
                                }
                            }
                            StreamDelta::ThinkingDelta { thinking } => {
                                if let BlockState::Thinking { text, .. } = &mut block_states[index]
                                {
                                    text.push_str(&thinking);
                                    sink.on_thinking_delta(&thinking)
                                        .await
                                        .map_err(provider_error_from_core)?;
                                }
                            }
                            StreamDelta::SignatureDelta { signature } => {
                                if let BlockState::Thinking { signature: sig, .. } =
                                    &mut block_states[index]
                                {
                                    sig.push_str(&signature);
                                }
                            }
                            StreamDelta::Other => {}
                        }
                    }
                    StreamEvent::ContentBlockStop { index } => {
                        if index >= block_states.len() {
                            continue;
                        }
                        match &block_states[index] {
                            BlockState::ToolUse {
                                id,
                                name,
                                partial_json,
                            } => {
                                let input: Value = if partial_json.is_empty() {
                                    Value::Object(serde_json::Map::new())
                                } else {
                                    serde_json::from_str(partial_json)?
                                };
                                let call = ToolCall {
                                    id: id.clone(),
                                    name: name.clone(),
                                    arguments: input,
                                };
                                sink.on_tool_call(&call)
                                    .await
                                    .map_err(provider_error_from_core)?;
                            }
                            BlockState::Text(_) | BlockState::Thinking { .. } => {}
                            BlockState::Unknown => {}
                        }
                    }
                    StreamEvent::MessageDelta { delta, usage } => {
                        if let Some(sr) = delta.and_then(|d| d.stop_reason) {
                            stop_reason = Some(sr);
                        }
                        if let Some(usage) = usage {
                            output_tokens = usage.output_tokens;
                        }
                    }
                    StreamEvent::MessageStop => {
                        done = true;
                    }
                    StreamEvent::Error { error } => {
                        return Err(ProviderError::Http {
                            status: reqwest::StatusCode::from_u16(529)
                                .unwrap_or(reqwest::StatusCode::SERVICE_UNAVAILABLE),
                            body: format!(
                                "Anthropic streaming error ({}): {}",
                                error.error_type, error.message
                            ),
                        });
                    }
                    StreamEvent::Ping => {}
                }

                if done {
                    break;
                }
            }
            if done {
                break;
            }
        }

        let mut final_blocks = Vec::with_capacity(block_states.len());
        for state in block_states {
            match state {
                BlockState::Text(text) => {
                    final_blocks.push(ContentBlock::Text { text });
                }
                BlockState::ToolUse {
                    id,
                    name,
                    partial_json,
                } => {
                    let input: Value = if partial_json.is_empty() {
                        Value::Object(serde_json::Map::new())
                    } else {
                        serde_json::from_str(&partial_json)?
                    };
                    final_blocks.push(ContentBlock::ToolUse { id, name, input });
                }
                BlockState::Thinking { text, signature } => {
                    final_blocks.push(ContentBlock::Thinking {
                        thinking: text,
                        signature,
                    });
                }
                BlockState::Unknown => {}
            }
        }

        let usage = input_tokens
            .zip(output_tokens)
            .map(|(input, output)| wire::Usage {
                input_tokens: Some(input),
                output_tokens: Some(output),
                cache_creation_input_tokens,
                cache_read_input_tokens,
            });

        let mut reply = message_from_content_blocks(final_blocks, usage)?;
        reply.finish_reason = stop_reason.as_deref().map(wire::map_stop_reason);

        tracing::debug!(
            target: "sweet_llm::observability",
            event = "anthropic.complete_stream",
            duration_ms = elapsed_ms(started),
            status = "ok",
            model = %self.model,
            "anthropic stream complete"
        );

        Ok(reply)
    }
}

#[async_trait]
impl Model for AnthropicProvider {
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
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

enum BlockState {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        partial_json: String,
    },
    Thinking {
        text: String,
        signature: String,
    },
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_user_agent_is_sweet_version() {
        let p = AnthropicProvider::new("k");
        assert_eq!(p.user_agent, format!("sweet/{}", SWEET_VERSION));
    }

    #[test]
    fn with_user_agent_overwrites() {
        let p = AnthropicProvider::new("k").with_user_agent("custom/1.0");
        assert_eq!(p.user_agent, "custom/1.0");
    }

    #[test]
    fn prepend_user_agent_prepends_with_space() {
        let p = AnthropicProvider::new("k").prepend_user_agent("app/1.0");
        assert_eq!(p.user_agent, format!("app/1.0 sweet/{}", SWEET_VERSION));
    }

    #[test]
    fn toggle_on_uses_adaptive_thinking() {
        // `Toggle(true)` maps to `{type: adaptive}`, the modern Claude shape; no
        // model name is sniffed. Models without adaptive take an explicit Budget.
        let p = AnthropicProvider::new("k").with_reasoning(ReasoningConfig::Toggle(true));
        assert!(matches!(p.wire_thinking(), Some(WireThinking::Adaptive)));
        assert!(p.wire_output_config().is_none());
    }

    #[test]
    fn toggle_off_disables_thinking() {
        let p = AnthropicProvider::new("k").with_reasoning(ReasoningConfig::Toggle(false));
        assert!(matches!(p.wire_thinking(), Some(WireThinking::Disabled)));
        assert!(p.wire_output_config().is_none());
    }

    #[test]
    fn budget_enables_thinking_with_budget_tokens() {
        let p = AnthropicProvider::new("k")
            .with_max_tokens(64_000)
            .with_reasoning(ReasoningConfig::Budget(4096));
        assert!(matches!(
            p.wire_thinking(),
            Some(WireThinking::Enabled {
                budget_tokens: 4096
            })
        ));
        assert!(p.wire_output_config().is_none());
    }

    #[test]
    fn effort_sets_output_config_not_thinking() {
        let p = AnthropicProvider::new("k").with_reasoning(ReasoningConfig::Effort("high".into()));
        assert_eq!(p.wire_output_config().and_then(|o| o.effort), Some("high"));
        assert!(p.wire_thinking().is_none());
    }

    #[test]
    fn no_reasoning_by_default() {
        let p = AnthropicProvider::new("k");
        assert!(p.wire_thinking().is_none());
        assert!(p.wire_output_config().is_none());
    }

    #[test]
    fn budget_below_minimum_is_clamped() {
        let p = AnthropicProvider::new("k").with_reasoning(ReasoningConfig::Budget(500));
        assert!(matches!(
            p.wire_thinking(),
            Some(WireThinking::Enabled {
                budget_tokens: MIN_THINKING_BUDGET
            })
        ));
    }

    #[test]
    fn with_max_tokens_sets_output_ceiling() {
        // `max_tokens` is the model's output ceiling (from the catalog); the
        // thinking budget is carved out of it, not added on top.
        let p = AnthropicProvider::new("k")
            .with_max_tokens(64_000)
            .with_reasoning(ReasoningConfig::Budget(8_000));
        assert_eq!(p.max_tokens(), 64_000);
        assert_eq!(p.thinking_budget(), 8_000);
        assert!(p.thinking_budget() < p.max_tokens());
    }

    #[test]
    fn budget_sent_verbatim() {
        // An explicit budget is never rewritten by the model name; the caller
        // owns correctness. `Budget` always produces `{type: enabled, budget_tokens}`.
        let p = AnthropicProvider::new("k")
            .with_max_tokens(64_000)
            .with_reasoning(ReasoningConfig::Budget(8_000));
        assert!(matches!(
            p.wire_thinking(),
            Some(WireThinking::Enabled {
                budget_tokens: 8_000
            })
        ));
    }

    #[test]
    fn budget_is_clamped_below_max_tokens() {
        // Default max_tokens is 4096; an over-large budget clamps to < max_tokens.
        let p = AnthropicProvider::new("k").with_reasoning(ReasoningConfig::Budget(50_000));
        assert!(p.thinking_budget() < p.max_tokens());
        assert_eq!(p.thinking_budget(), p.max_tokens() - 1);
    }

    #[test]
    fn budget_clamped_below_sampling_max_tokens_override() {
        // The budget is clamped against the *request* cap, not the model
        // ceiling. When that cap is too small for a budget >= the 1024 minimum
        // (here `max_tokens` = 1000 leaves only `upper` = 999), there is no
        // valid `budget_tokens`: thinking is dropped rather than sending one
        // Anthropic would reject with a 400.
        let p = AnthropicProvider::new("k")
            .with_max_tokens(64_000)
            .with_sampling(SamplingConfig {
                max_tokens: Some(1000),
                ..Default::default()
            })
            .with_reasoning(ReasoningConfig::Budget(8_000));
        assert_eq!(p.thinking_budget(), 0);
        assert!(p.wire_thinking().is_none());
    }

    #[test]
    fn stop_reason_maps_refusal_and_length() {
        use super::wire::map_stop_reason;
        use sweet_core::FinishReason;
        assert_eq!(map_stop_reason("end_turn"), FinishReason::Stop);
        assert_eq!(map_stop_reason("max_tokens"), FinishReason::Length);
        assert_eq!(map_stop_reason("tool_use"), FinishReason::ToolCalls);
        assert_eq!(map_stop_reason("refusal"), FinishReason::Refusal);
        assert_eq!(
            map_stop_reason("???"),
            FinishReason::Other("???".to_string())
        );
    }

    #[test]
    fn sampling_dropped_when_thinking_on() {
        // `Toggle(true)` enables (adaptive) thinking, so temperature/top_p/top_k
        // are suppressed (Anthropic rejects them together).
        let p = AnthropicProvider::new("k")
            .with_reasoning(ReasoningConfig::Toggle(true))
            .with_sampling(SamplingConfig {
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: Some(40),
                ..Default::default()
            });
        let SamplingFields {
            temperature: t,
            top_p,
            top_k,
            ..
        } = p.sampling_fields();
        assert!(t.is_none() && top_p.is_none() && top_k.is_none());
    }

    #[test]
    fn temperature_and_top_p_mutually_exclusive() {
        let p = AnthropicProvider::new("k").with_sampling(SamplingConfig {
            temperature: Some(0.7),
            top_p: Some(0.9),
            ..Default::default()
        });
        let SamplingFields {
            temperature: t,
            top_p,
            ..
        } = p.sampling_fields();
        assert_eq!(t, Some(0.7));
        assert!(top_p.is_none());
    }

    #[test]
    fn sampling_max_tokens_overrides_ceiling() {
        let p = AnthropicProvider::new("k")
            .with_max_tokens(64_000)
            .with_sampling(SamplingConfig {
                max_tokens: Some(1000),
                ..Default::default()
            });
        assert_eq!(p.request_max_tokens(), 1000);
    }

    #[test]
    fn tool_choice_maps_to_anthropic_wire() {
        let p = AnthropicProvider::new("k").with_tool_choice(ToolChoice::Required);
        assert_eq!(
            p.tool_choice_value(true),
            Some(serde_json::json!({ "type": "any" }))
        );
        assert!(p.tool_choice_value(false).is_none());

        let p = AnthropicProvider::new("k").with_tool_choice(ToolChoice::Tool("x".into()));
        assert_eq!(
            p.tool_choice_value(true),
            Some(serde_json::json!({ "type": "tool", "name": "x" }))
        );
    }

    #[test]
    fn structured_output_sets_output_config_format_and_beta() {
        let p = AnthropicProvider::new("k").with_structured_output(StructuredOutput::new(
            serde_json::json!({ "type": "object" }),
        ));
        let oc = p.wire_output_config().expect("output_config present");
        let format = oc.format.expect("format present");
        assert_eq!(format["type"], "json_schema");
        assert_eq!(format["schema"]["type"], "object");
        assert_eq!(
            p.betas(&[]),
            vec!["structured-outputs-2025-11-13".to_string()]
        );
    }

    #[test]
    fn betas_include_custom_and_pdf() {
        let p = AnthropicProvider::new("k").with_beta("custom-beta");
        // No attachments: just the custom beta.
        assert_eq!(
            p.betas(&[Message::user("hi")]),
            vec!["custom-beta".to_string()]
        );
        // A file attachment adds the PDF beta.
        let with_file = Message::user_blocks(vec![sweet_core::ContentBlock::File {
            data: vec![1, 2, 3],
            media_type: "application/pdf".into(),
            filename: "d.pdf".into(),
        }]);
        let betas = p.betas(std::slice::from_ref(&with_file));
        assert!(betas.contains(&"custom-beta".to_string()));
        assert!(betas.contains(&"pdfs-2024-09-25".to_string()));
    }

    #[test]
    fn auth_token_takes_precedence_over_api_key() {
        let p = AnthropicProvider::new("api-key").with_auth_token("bearer-tok");
        assert_eq!(p.auth_token.as_deref(), Some("bearer-tok"));
    }
}
