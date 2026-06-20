// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Cross-provider structured-output control: constrain the model to emit JSON
//! matching a schema. Each provider's `with_structured_output` maps it to its
//! own wire form (OpenAI `response_format`, Gemini `responseSchema`, Anthropic
//! `output_config.format`).

use serde_json::Value;

/// A request to constrain the model's output to JSON matching `schema`.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredOutput {
    /// JSON Schema the response must conform to.
    pub schema: Value,
    /// Schema name sent to providers that require one (defaults to `response`).
    pub name: Option<String>,
    /// Whether to request strict schema adherence where supported.
    pub strict: bool,
}

impl StructuredOutput {
    /// Build a structured-output request from a JSON Schema, with strict
    /// adherence on and no explicit name.
    pub fn new(schema: Value) -> Self {
        Self {
            schema,
            name: None,
            strict: true,
        }
    }

    /// The schema name, or `"response"` when unset.
    pub(crate) fn name_or_default(&self) -> &str {
        self.name.as_deref().unwrap_or("response")
    }
}
