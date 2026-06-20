// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Cross-provider sampling / generation parameters.
//!
//! Every field defaults to `None`/empty: in that case nothing is sent and the
//! model applies its own server-side default - we never inject a `temperature`
//! or other value the caller didn't ask for. Each provider's `with_sampling`
//! maps the subset it supports to its own wire fields and drops the rest with a
//! warning.
//!
//! `extra` is an escape hatch: arbitrary key/value pairs merged
//! verbatim into the top-level request body, covering provider-specific fields
//! (`logit_bias`, `n`, `user`, `metadata`, ...) not modelled here.

use std::collections::BTreeMap;

use serde_json::Value;

/// Provider-agnostic sampling controls. See the module docs for the
/// omit-when-unset contract.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingConfig {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass.
    pub top_p: Option<f32>,
    /// Top-k sampling (Anthropic / Gemini only).
    pub top_k: Option<u32>,
    /// Deterministic sampling seed (OpenAI / Cerebras / Gemini).
    pub seed: Option<u64>,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Frequency penalty (OpenAI / Cerebras / Gemini).
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (OpenAI / Cerebras / Gemini).
    pub presence_penalty: Option<f32>,
    /// Maximum output tokens. Overrides any per-model ceiling.
    pub max_tokens: Option<usize>,
    /// Arbitrary extra fields merged verbatim into the request body.
    pub extra: BTreeMap<String, Value>,
}

/// Merge `extra` keys into the top-level object of a serialized request `body`,
/// overriding any field already present. No-op when `body` is not an object or
/// `extra` is empty.
pub(crate) fn merge_extra(body: &mut Value, extra: &BTreeMap<String, Value>) {
    if extra.is_empty() {
        return;
    }
    if let Value::Object(map) = body {
        for (k, v) in extra {
            map.insert(k.clone(), v.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_extra_overrides_and_adds() {
        let mut body = serde_json::json!({ "model": "m", "temperature": 0.2 });
        let mut extra = BTreeMap::new();
        extra.insert("temperature".to_string(), serde_json::json!(0.9));
        extra.insert("logit_bias".to_string(), serde_json::json!({ "5": -100 }));
        merge_extra(&mut body, &extra);
        assert_eq!(body["temperature"], 0.9);
        assert_eq!(body["logit_bias"], serde_json::json!({ "5": -100 }));
        assert_eq!(body["model"], "m");
    }

    #[test]
    fn merge_extra_noop_on_empty() {
        let mut body = serde_json::json!({ "model": "m" });
        merge_extra(&mut body, &BTreeMap::new());
        assert_eq!(body, serde_json::json!({ "model": "m" }));
    }
}
