// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Cross-provider tool-choice control.
//!
//! Like [`ReasoningConfig`](crate::ReasoningConfig), this is a provider-agnostic
//! abstraction each provider's `with_tool_choice` maps to its own wire form. It
//! is only ever sent when the request actually carries tools.

/// How the model should choose among the available tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model decides whether and which tool to call (provider default).
    Auto,
    /// The model must not call a tool.
    None,
    /// The model must call some tool.
    Required,
    /// The model must call the named tool.
    Tool(String),
}
