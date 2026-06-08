// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Thinking-mode configuration for OpenAI-compatible providers.

/// Per-request chain-of-thought reasoning configuration.
///
/// Maps to the `thinking` object in the request body. `enabled` toggles the
/// `type: enabled|disabled` field; `preserve_history` toggles `keep: "all"`,
/// which asks the provider to re-feed prior turns' `reasoning_content` so
/// the model can continue a long chain of thought (Kimi `kimi-k2.6`).
///
/// Most callers want one of [`ThinkingMode::ENABLED`],
/// [`ThinkingMode::DISABLED`], or [`ThinkingMode::PRESERVED`]; the bare
/// struct is exposed for finer control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingMode {
    /// Generate `reasoning_content` for this turn (`thinking.type`).
    pub enabled: bool,
    /// Preserve historical `reasoning_content` across turns (`thinking.keep = "all"`).
    /// Kimi-specific; ignored by providers that do not support it.
    pub preserve_history: bool,
}

impl ThinkingMode {
    /// Generate reasoning for this turn, do not preserve history.
    pub const ENABLED: Self = Self {
        enabled: true,
        preserve_history: false,
    };

    /// Suppress reasoning for this turn.
    pub const DISABLED: Self = Self {
        enabled: false,
        preserve_history: false,
    };

    /// Generate reasoning for this turn and preserve all prior turns' reasoning content
    /// (Kimi `kimi-k2.6` with `keep: "all"`).
    pub const PRESERVED: Self = Self {
        enabled: true,
        preserve_history: true,
    };
}
