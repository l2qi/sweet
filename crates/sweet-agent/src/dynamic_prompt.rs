// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-turn dynamic instructions.
//!
//! A [`DynamicPrompt`] is the dynamic counterpart of the static
//! [`crate::extension::PromptSpec`]: instead of text fixed at construction
//! time, its content is recomputed on every turn and appended to the agent's
//! composed system instructions. Because instructions live outside the session
//! transcript, a dynamic prompt survives history compaction - making it the
//! right channel for state that must keep anchoring the model (e.g. a live todo
//! list) no matter how long a task runs.

/// Text contributed to the system instructions, recomputed every turn.
///
/// Implementations typically read from interior-mutable shared state
/// (`Arc<Mutex<...>>`), so the rendered text reflects the latest state on each
/// call. Return `None` to contribute nothing this turn.
///
/// `render` may be called more than once per turn (message assembly is not 1:1
/// with model calls), so it must be cheap and side-effect-free - a pure
/// projection of current state, not a generator that mutates anything.
pub trait DynamicPrompt: Send + Sync {
    fn render(&self) -> Option<String>;
}
