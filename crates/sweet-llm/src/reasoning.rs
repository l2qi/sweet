// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Cross-provider reasoning control.
//!
//! Reasoning is exposed by different providers through different wire fields -
//! OpenAI-compatible endpoints use `reasoning_effort` and a `thinking` object,
//! Anthropic uses a `thinking` object plus a top-level `effort`, Gemini uses a
//! `thinkingConfig`. This enum is the single cross-provider abstraction; each
//! provider's `with_reasoning` maps it to its own wire encoding.
//!
//! The three variants mirror the three `reasoning_options` dialects published
//! in the [models.dev](https://models.dev) catalog (`toggle` / `effort` /
//! `budget_tokens`), so a catalog entry translates directly into a
//! [`ReasoningConfig`].

/// How to control a model's reasoning for a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningConfig {
    /// Toggle dialect - turn reasoning on (`true`) or off (`false`).
    Toggle(bool),
    /// Effort dialect - a discrete level such as `"low"`, `"medium"`, `"high"`,
    /// or `"none"`. Accepted values are provider/model-specific.
    Effort(String),
    /// Budget-tokens dialect - a thinking-token budget.
    Budget(u32),
}
