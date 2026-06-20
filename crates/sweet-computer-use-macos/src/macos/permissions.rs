// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! TCC permission checks for Accessibility and Screen Recording.
//!
//! We only *check* (and, for screen recording, optionally *request*) - we do
//! not build the prompt-options dictionary for Accessibility, since the CLI can
//! simply tell the user which System Settings pane to open and retry. The hints
//! are returned as error messages so they reach the model and the user.

use super::ffi;

/// Whether the process is trusted for Accessibility (required for the AX tree
/// and for posting synthetic input).
pub fn accessibility_trusted() -> bool {
    unsafe { ffi::AXIsProcessTrusted() != 0 }
}

/// Actionable message for a missing Accessibility grant.
pub fn accessibility_hint() -> String {
    "Accessibility permission is not granted. Grant it to your terminal app \
     (or the application binary) in System Settings › Privacy & Security › \
     Accessibility, then retry."
        .to_string()
}

/// Whether the process may capture the screen (required for screenshots).
pub fn screen_capture_allowed() -> bool {
    unsafe { ffi::CGPreflightScreenCaptureAccess() }
}

/// Trigger the Screen Recording permission prompt. Returns the current grant
/// state (the prompt's effect only applies after the app is relaunched).
pub fn request_screen_capture() -> bool {
    unsafe { ffi::CGRequestScreenCaptureAccess() }
}

/// Actionable message for a missing Screen Recording grant.
pub fn screen_capture_hint() -> String {
    "Screen Recording permission is not granted (required for screenshots). \
     Grant it in System Settings › Privacy & Security › Screen Recording, then \
     restart the application. Accessibility-based observation still works without it."
        .to_string()
}
