// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! macOS computer-use backend.
//!
//! Implements [`sweet_computer_use_core::ComputerUseProvider`] for macOS by
//! talking directly to the platform C APIs:
//!
//! - **Accessibility** (`AXUIElement`, ApplicationServices) - the focused
//!   window's element tree, plus `AXPress`/`AXValue` actions.
//! - **Quartz Event Services** (`CGEvent`, CoreGraphics) - synthetic mouse,
//!   keyboard, and scroll input.
//! - **CoreGraphics display + window list** - screen size, the on-screen window
//!   list, and (via ImageIO) PNG screen capture.
//!
//! The bindings are hand-written `extern "C"` declarations against stable
//! system frameworks (in the `macos::ffi` module); the crate links them via
//! `build.rs`.
//! Everything platform-specific is gated behind `#[cfg(target_os = "macos")]`,
//! so on other targets the crate still builds and [`MacComputerUse`] simply
//! reports [`ComputerUseError::Unsupported`](sweet_computer_use_core::ComputerUseError::Unsupported).
//!
//! # Permissions
//!
//! Observing the accessibility tree and posting synthetic input both require the
//! host process to be trusted for **Accessibility**; screen capture requires
//! **Screen Recording**. When a permission is missing, calls fail with an
//! actionable [`ComputerUseError::PermissionDenied`](sweet_computer_use_core::ComputerUseError::PermissionDenied)
//! naming the System Settings pane to grant it.

mod provider;

#[cfg(target_os = "macos")]
mod macos;

pub use provider::MacComputerUse;
