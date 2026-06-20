// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Platform-neutral computer-use substrate for sweet-based agents.
//!
//! This crate owns the *generic* half of computer use: the bounded set of GUI
//! [`ComputerAction`]s a model may request, the structured [`ComputerObservation`]
//! a platform reports back, the [`ComputerUseProvider`] trait a platform backend
//! implements, and the single model-facing [`computer_use_tool`] that bridges the
//! two. It knows nothing about macOS, Quartz events, or accessibility APIs - that
//! lives behind the provider trait (see `sweet-computer-use-macos`).
//!
//! # Observations are text-first, with an optional screenshot image
//!
//! The observation the model consumes is primarily a **textual** rendering of
//! the GUI: the active app and window, the on-screen window list, and the
//! accessibility tree (element roles, labels, values, and screen frames). This
//! is precise (exact frames to click) and works with text-only models, so it is
//! always the dependable signal.
//!
//! When a screenshot is captured it is *also* attached to the tool result as an
//! image block - via [`ToolOutput`](sweet_core::ToolOutput) returned from the
//! tool's `call_rich` - and saved to disk (its path is surfaced in the text).
//! Whether the image actually reaches the model depends on the provider:
//! Anthropic carries images on tool-result messages, so a vision model sees the
//! screenshot; the OpenAI / Chat Completions protocol has no image content on
//! tool messages, so there the image is dropped and only the text survives.
//!
//! # Shape
//!
//! ```text
//! model -> computer tool -> ComputerAction -> ComputerUseProvider (platform) -> ComputerObservation / ActionOutcome -> text -> model
//! ```
//!
//! A coding agent opts in by registering [`computer_use_tool`] with a concrete
//! provider; the agent framework runs it like any other tool.

mod action;
mod observation;
mod provider;
mod render;
mod tool;

pub use action::{ComputerAction, MouseButton, Point};
pub use observation::{
    crosshair_rects, ActionOutcome, ComputerObservation, ObserveOptions, PixelRect, Rect,
    Screenshot, Size, UiNode, WindowInfo,
};
pub use provider::{ComputerUseError, ComputerUseProvider, SharedProvider};
pub use render::{render_observation, render_outcome};
pub use tool::{computer_use_tool, CoordinateSpace, COMPUTER_TOOL_NAME};
