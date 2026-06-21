// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Native Google Gemini provider via the Generative Language API.
//!
//! Endpoint: `POST /v1beta/models/{model}:generateContent`  
//! Streaming: `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`
//!
//! This module speaks the native Gemini protocol (rather than the
//! OpenAI-compatible endpoint) so it can correctly handle the
//! `thoughtSignature` fields that Gemini 3 models require for multi-turn tool
//! use.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;

use sweet_core::{
    Message, Model, Result as CoreResult, Role, StreamSink, ThinkingContent, ToolCall, ToolSpec,
    SWEET_VERSION,
};

use crate::error::ProviderError;
use crate::schema::sanitize_schema;
use crate::util::provider_error_from_core;
use crate::{ReasoningConfig, SamplingConfig, StructuredOutput, ToolChoice};

mod embeddings;
pub use embeddings::{GeminiEmbedder, DEFAULT_EMBEDDING_MODEL, DEFAULT_OUTPUT_DIMENSIONALITY};

mod wire;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
pub const DEFAULT_API_KEY_ENV: &str = "GEMINI_API_KEY";
pub const DEFAULT_MODEL: &str = "gemini-3-flash-preview";

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GeminiProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    /// Per-model output cap (models.dev `limit.output`). `None` omits
    /// `maxOutputTokens` so the model uses its own default.
    max_tokens: Option<usize>,
    user_agent: String,
    /// Map `tool_call_id -> thoughtSignature` so that when the same tool call
    /// is re-injected into a later request (e.g. after the model calls it) we
    /// can echo the signature back exactly as Gemini requires.
    thought_signatures: Arc<Mutex<HashMap<String, String>>>,
    /// Map `tool_call_id -> function_name` so that `functionResponse` parts in
    /// history carry the correct `name` field (required by Gemini).
    tool_names: Arc<Mutex<HashMap<String, String>>>,
    /// Reasoning configuration, rendered to `generationConfig.thinkingConfig`.
    reasoning: Option<ReasoningConfig>,
    /// Sampling parameters, rendered into `generationConfig`.
    sampling: SamplingConfig,
    tool_choice: Option<ToolChoice>,
    structured_output: Option<StructuredOutput>,
}

impl GeminiProvider {
    /// Create a new provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.into(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.into(),
            max_tokens: None,
            user_agent: format!("sweet/{}", SWEET_VERSION),
            thought_signatures: Arc::new(Mutex::new(HashMap::new())),
            tool_names: Arc::new(Mutex::new(HashMap::new())),
            reasoning: None,
            sampling: SamplingConfig::default(),
            tool_choice: None,
            structured_output: None,
        }
    }

    /// Set cross-provider sampling parameters, rendered into `generationConfig`.
    pub fn with_sampling(mut self, sampling: SamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }

    /// Constrain how the model chooses among tools. Only sent when the request
    /// carries tools.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Constrain the response to JSON matching a schema (`responseMimeType` +
    /// `responseSchema`).
    pub fn with_structured_output(mut self, output: StructuredOutput) -> Self {
        self.structured_output = Some(output);
        self
    }

    /// The `toolConfig` wire value, or `None` when unset or no tools present.
    fn tool_config_value(&self, has_tools: bool) -> Option<serde_json::Value> {
        if !has_tools {
            return None;
        }
        let (mode, allowed): (&str, Option<&str>) = match self.tool_choice.as_ref()? {
            ToolChoice::Auto => ("AUTO", None),
            ToolChoice::None => ("NONE", None),
            ToolChoice::Required => ("ANY", None),
            ToolChoice::Tool(name) => ("ANY", Some(name.as_str())),
        };
        let mut cfg = serde_json::json!({ "functionCallingConfig": { "mode": mode } });
        if let Some(name) = allowed {
            cfg["functionCallingConfig"]["allowedFunctionNames"] = serde_json::json!([name]);
        }
        Some(cfg)
    }

    /// Read the API key from the environment variable `GEMINI_API_KEY`.
    pub fn from_env() -> std::result::Result<Self, ProviderError> {
        let key = std::env::var(DEFAULT_API_KEY_ENV).map_err(|_| ProviderError::MissingApiKey {
            var: DEFAULT_API_KEY_ENV,
        })?;
        Ok(Self::new(key))
    }

    /// Set the model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the base URL (for proxying or testing).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the maximum number of output tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Replace the underlying HTTP client.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = client;
        self
    }

    /// Overwrite the User-Agent header sent with every request.
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// Prepend a token to the existing User-Agent header.
    pub fn prepend_user_agent(mut self, prefix: impl Into<String>) -> Self {
        self.user_agent = format!("{} {}", prefix.into(), self.user_agent);
        self
    }

    /// Configure reasoning for this provider.
    ///
    /// Maps the cross-provider [`ReasoningConfig`] to Gemini's
    /// `generationConfig.thinkingConfig`:
    /// - [`Toggle(true)`](ReasoningConfig::Toggle) -> dynamic thinking
    ///   (`thinkingBudget: -1`, `includeThoughts: true`).
    /// - [`Toggle(false)`](ReasoningConfig::Toggle) -> thinking off
    ///   (`thinkingBudget: 0`).
    /// - [`Effort(e)`](ReasoningConfig::Effort) -> discrete `thinkingLevel`
    ///   (`minimal`/`low`/`medium`/`high`; Gemini 3+). Models that only accept a
    ///   numeric budget (Gemini 2.5 and earlier) should be sent an explicit
    ///   [`Budget`](ReasoningConfig::Budget) instead - which variant a model
    ///   wants is the caller's catalog knowledge, never sniffed here.
    /// - [`Budget(n)`](ReasoningConfig::Budget) -> `thinkingBudget: n`.
    pub fn with_reasoning(mut self, config: ReasoningConfig) -> Self {
        self.reasoning = Some(config);
        self
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Build the Gemini `thinkingConfig`, or `None` when reasoning is unset.
    fn thinking_config(&self) -> Option<wire::ThinkingConfig> {
        let config = self.reasoning.as_ref()?;
        Some(match config {
            ReasoningConfig::Toggle(true) => wire::ThinkingConfig {
                thinking_budget: Some(-1),
                include_thoughts: Some(true),
                thinking_level: None,
            },
            ReasoningConfig::Toggle(false) => wire::ThinkingConfig {
                thinking_budget: Some(0),
                include_thoughts: None,
                thinking_level: None,
            },
            // Effort maps to the discrete `thinkingLevel` knob (Gemini 3+).
            // Models that only accept a numeric budget (Gemini 2.5) take an
            // explicit `Budget` from the caller instead; the provider does not
            // sniff the model name.
            ReasoningConfig::Effort(e) => wire::ThinkingConfig {
                thinking_budget: None,
                include_thoughts: Some(true),
                thinking_level: Some(normalize_thinking_level(e)),
            },
            ReasoningConfig::Budget(n) => wire::ThinkingConfig {
                thinking_budget: Some(*n as i32),
                include_thoughts: Some(true),
                thinking_level: None,
            },
        })
    }

    fn build_request_body(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> std::result::Result<serde_json::Value, ProviderError> {
        let sigs = self
            .thought_signatures
            .lock()
            .expect("gemini thought_signatures mutex poisoned");
        let names = self
            .tool_names
            .lock()
            .expect("gemini tool_names mutex poisoned");
        let (system_instruction, contents) = wire::convert_messages(messages, &sigs, &names);
        drop(sigs);
        drop(names);

        let has_tools = !tools.is_empty();
        let tools = if tools.is_empty() {
            None
        } else {
            Some(vec![wire::Tool {
                function_declarations: tools
                    .iter()
                    .map(|t| {
                        let mut params = t.parameters_schema.clone();
                        sanitize_schema(&mut params);
                        wire::FunctionDeclaration {
                            name: &t.name,
                            description: &t.description,
                            parameters: params,
                        }
                    })
                    .collect(),
            }])
        };

        let req = wire::GenerateContentRequest {
            system_instruction,
            contents,
            tools,
            generation_config: Some(wire::GenerationConfig {
                max_output_tokens: self.sampling.max_tokens.or(self.max_tokens),
                thinking_config: self.thinking_config(),
                temperature: self.sampling.temperature,
                top_p: self.sampling.top_p,
                top_k: self.sampling.top_k,
                seed: self.sampling.seed,
                frequency_penalty: self.sampling.frequency_penalty,
                presence_penalty: self.sampling.presence_penalty,
                stop_sequences: self.sampling.stop.clone(),
                response_mime_type: self
                    .structured_output
                    .as_ref()
                    .map(|_| "application/json".to_string()),
                response_schema: self.structured_output.as_ref().map(|so| {
                    let mut schema = so.schema.clone();
                    sanitize_schema(&mut schema);
                    schema
                }),
            }),
            tool_config: self.tool_config_value(has_tools),
        };

        let mut body = serde_json::to_value(&req)?;
        crate::sampling::merge_extra(&mut body, &self.sampling.extra);
        Ok(body)
    }

    fn save_meta(&self, thought_sigs: Vec<(String, String)>, tool_names: Vec<(String, String)>) {
        let mut sigs = self
            .thought_signatures
            .lock()
            .expect("gemini thought_signatures mutex poisoned");
        let mut names = self
            .tool_names
            .lock()
            .expect("gemini tool_names mutex poisoned");
        for (id, sig) in thought_sigs {
            sigs.insert(id, sig);
        }
        for (id, name) in tool_names {
            names.insert(id, name);
        }
    }

    async fn complete_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> std::result::Result<Message, ProviderError> {
        let body = self.build_request_body(messages, tools)?;

        let url = format!(
            "{}/models/{}:generateContent",
            self.base_url.trim_end_matches('/'),
            self.model
        );

        let mut req = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .header("User-Agent", &self.user_agent)
            .json(&body);
        if !self.api_key.is_empty() {
            req = req.header("x-goog-api-key", &self.api_key);
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

        let gemini_resp: wire::GenerateContentResponse = resp.json().await?;
        let parsed = wire::parse_response(gemini_resp)?;
        self.save_meta(parsed.thought_signatures, parsed.tool_names);
        Ok(parsed.message)
    }

    async fn complete_stream_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> std::result::Result<Message, ProviderError> {
        let body = self.build_request_body(messages, tools)?;

        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url.trim_end_matches('/'),
            self.model
        );

        let mut req = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .header("User-Agent", &self.user_agent)
            .json(&body);
        if !self.api_key.is_empty() {
            req = req.header("x-goog-api-key", &self.api_key);
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
        let mut text_acc = String::new();
        let mut thinking_acc = String::new();
        let mut function_call_parts: Vec<wire::Part> = Vec::new();
        let mut usage: Option<wire::UsageMetadata> = None;
        let mut finish_reason: Option<String> = None;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buffer.extend_from_slice(&bytes);

            while let Some(result) = crate::sse::drain_event(&mut buffer) {
                let event_text = result?;
                let Some(data) = crate::sse::data_lines(&event_text).last() else {
                    continue;
                };

                let gemini_chunk: wire::GenerateContentResponse = serde_json::from_str(data)
                    .map_err(|e| {
                        ProviderError::Decode(serde::de::Error::custom(format!(
                            "invalid JSON in SSE data: {e}. raw: {data}",
                        )))
                    })?;

                for candidate in gemini_chunk.candidates {
                    if let Some(fr) = candidate.finish_reason.clone() {
                        finish_reason = Some(fr);
                    }
                    for part in candidate.content.parts {
                        if let Some(ref text) = part.text {
                            if part.thought == Some(true) {
                                thinking_acc.push_str(text);
                                sink.on_thinking_delta(text)
                                    .await
                                    .map_err(provider_error_from_core)?;
                            } else {
                                text_acc.push_str(text);
                                sink.on_content_delta(text)
                                    .await
                                    .map_err(provider_error_from_core)?;
                            }
                        }
                        if part.function_call.is_some() {
                            function_call_parts.push(part);
                        }
                    }
                }
                if let Some(u) = gemini_chunk.usage_metadata {
                    usage = Some(u);
                }
            }
        }

        let content = text_acc;
        let mut tool_calls = Vec::new();
        let mut thought_signatures = Vec::new();
        let mut tool_names = Vec::new();

        for part in function_call_parts {
            if let Some(fc) = part.function_call {
                if let Some(sig) = part.thought_signature {
                    thought_signatures.push((fc.id.clone(), sig));
                }
                tool_names.push((fc.id.clone(), fc.name.clone()));
                let call = ToolCall {
                    id: fc.id,
                    name: fc.name,
                    arguments: fc.args,
                };
                sink.on_tool_call(&call)
                    .await
                    .map_err(provider_error_from_core)?;
                tool_calls.push(call);
            }
        }

        let token_count = usage.as_ref().map(|u| u.total_token_count);
        let context_tokens = usage.as_ref().map(|u| u.prompt_token_count);

        self.save_meta(thought_signatures, tool_names);

        let thinking_content = if thinking_acc.is_empty() {
            Vec::new()
        } else {
            vec![ThinkingContent::new(thinking_acc)]
        };

        Ok(Message {
            role: Role::Assistant,
            content: vec![sweet_core::ContentBlock::text(content)],
            thinking_content,
            tool_calls,
            tool_call_id: None,
            token_count,
            context_tokens,
            compacted: false,
            finish_reason: finish_reason.as_deref().map(wire::map_finish_reason),
        })
    }
}

// ---------------------------------------------------------------------------
// Model trait
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl Model for GeminiProvider {
    async fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> CoreResult<Message> {
        Ok(self.complete_inner(messages, tools).await?)
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> CoreResult<Message> {
        Ok(self.complete_stream_inner(messages, tools, sink).await?)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Clamp a free-form effort string to a Gemini-3 `thinkingLevel`. Gemini 3
/// cannot fully disable thinking, so `none` becomes `minimal`.
fn normalize_thinking_level(effort: &str) -> String {
    match effort {
        "minimal" | "low" | "medium" | "high" => effort.to_string(),
        "none" => "minimal".to_string(),
        "xhigh" | "max" => "high".to_string(),
        _ => "medium".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_user_agent_is_sweet_version() {
        let p = GeminiProvider::new("k");
        assert_eq!(p.user_agent, format!("sweet/{}", SWEET_VERSION));
    }

    #[test]
    fn with_user_agent_overwrites() {
        let p = GeminiProvider::new("k").with_user_agent("custom/1.0");
        assert_eq!(p.user_agent, "custom/1.0");
    }

    #[test]
    fn prepend_user_agent_prepends_with_space() {
        let p = GeminiProvider::new("k").prepend_user_agent("app/1.0");
        assert_eq!(p.user_agent, format!("app/1.0 sweet/{}", SWEET_VERSION));
    }

    #[test]
    fn no_thinking_config_by_default() {
        let p = GeminiProvider::new("k");
        let body = p.build_request_body(&[], &[]).unwrap();
        assert!(body["generationConfig"].get("thinkingConfig").is_none());
    }

    #[test]
    fn toggle_on_emits_dynamic_thinking_budget() {
        let p = GeminiProvider::new("k").with_reasoning(ReasoningConfig::Toggle(true));
        let body = p.build_request_body(&[], &[]).unwrap();
        let tc = &body["generationConfig"]["thinkingConfig"];
        assert_eq!(tc["thinkingBudget"], -1);
        assert_eq!(tc["includeThoughts"], true);
    }

    #[test]
    fn toggle_off_zeroes_thinking_budget() {
        let p = GeminiProvider::new("k").with_reasoning(ReasoningConfig::Toggle(false));
        let body = p.build_request_body(&[], &[]).unwrap();
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            0
        );
    }

    #[test]
    fn effort_emits_thinking_level() {
        // `Effort` maps to the discrete `thinkingLevel` (Gemini 3+); no model
        // name is sniffed. Gemini 2.5 callers send an explicit `Budget` instead.
        let p = GeminiProvider::new("k").with_reasoning(ReasoningConfig::Effort("high".into()));
        let body = p.build_request_body(&[], &[]).unwrap();
        let tc = &body["generationConfig"]["thinkingConfig"];
        assert_eq!(tc["thinkingLevel"], "high");
        assert!(tc.get("thinkingBudget").is_none());
    }

    #[test]
    fn max_output_tokens_omitted_when_unset() {
        let p = GeminiProvider::new("k");
        let body = p.build_request_body(&[], &[]).unwrap();
        assert!(body["generationConfig"].get("maxOutputTokens").is_none());
    }

    #[test]
    fn sampling_renders_into_generation_config() {
        let p = GeminiProvider::new("k").with_sampling(SamplingConfig {
            temperature: Some(0.5),
            top_p: Some(0.25),
            top_k: Some(20),
            stop: vec!["END".into()],
            max_tokens: Some(2048),
            ..Default::default()
        });
        let body = p.build_request_body(&[], &[]).unwrap();
        let gc = &body["generationConfig"];
        assert_eq!(gc["temperature"], 0.5);
        assert_eq!(gc["topP"], 0.25);
        assert_eq!(gc["topK"], 20);
        assert_eq!(gc["maxOutputTokens"], 2048);
        assert_eq!(gc["stopSequences"], serde_json::json!(["END"]));
    }

    #[test]
    fn extra_passthrough_merges_into_body() {
        let mut extra = std::collections::BTreeMap::new();
        extra.insert(
            "safetySettings".to_string(),
            serde_json::json!([{ "category": "X" }]),
        );
        let p = GeminiProvider::new("k").with_sampling(SamplingConfig {
            extra,
            ..Default::default()
        });
        let body = p.build_request_body(&[], &[]).unwrap();
        assert_eq!(
            body["safetySettings"],
            serde_json::json!([{ "category": "X" }])
        );
    }

    #[test]
    fn budget_emits_thinking_budget() {
        let p = GeminiProvider::new("k").with_reasoning(ReasoningConfig::Budget(8000));
        let body = p.build_request_body(&[], &[]).unwrap();
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            8000
        );
    }

    #[test]
    fn tool_config_maps_to_function_calling_config() {
        let p = GeminiProvider::new("k").with_tool_choice(ToolChoice::Tool("f".into()));
        let tc = p.tool_config_value(true).unwrap();
        assert_eq!(tc["functionCallingConfig"]["mode"], "ANY");
        assert_eq!(
            tc["functionCallingConfig"]["allowedFunctionNames"],
            serde_json::json!(["f"])
        );
        // Omitted when the request carries no tools.
        assert!(p.tool_config_value(false).is_none());
    }

    #[test]
    fn structured_output_sets_response_schema() {
        let p = GeminiProvider::new("k").with_structured_output(StructuredOutput::new(
            serde_json::json!({ "type": "object" }),
        ));
        let body = p.build_request_body(&[], &[]).unwrap();
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert_eq!(body["generationConfig"]["responseSchema"]["type"], "object");
    }
}
