// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! The platform-backend contract.

use std::sync::Arc;

use async_trait::async_trait;

use crate::action::ComputerAction;
use crate::observation::{ActionOutcome, ComputerObservation, ObserveOptions};

/// Errors a computer-use backend can return.
///
/// These map to a tool-execution error when surfaced to the model, with a
/// message specific enough to be actionable (e.g. which permission is missing).
#[derive(Debug, thiserror::Error)]
pub enum ComputerUseError {
    /// The current platform has no computer-use backend (e.g. a non-macOS build).
    #[error("computer use is not supported on this platform ({0})")]
    Unsupported(String),
    /// A required OS permission (Accessibility, Screen Recording, ...) is missing.
    /// The string names the permission and how to grant it.
    #[error("{0}")]
    PermissionDenied(String),
    /// An accessibility element path did not resolve to a live element.
    #[error("UI element not found: {0}")]
    ElementNotFound(String),
    /// The action was malformed or cannot be handled here (e.g. an observe
    /// action reaching `act`).
    #[error("invalid action: {0}")]
    InvalidAction(String),
    /// A platform API call failed.
    #[error("platform error: {0}")]
    Platform(String),
}

/// A platform backend that observes and controls the local GUI.
///
/// Implementors translate the neutral [`ComputerAction`] /
/// [`ComputerObservation`] vocabulary into OS-specific calls. The tool routes
/// observe-style actions to [`observe`](Self::observe) and everything else to
/// [`act`](Self::act).
#[async_trait]
pub trait ComputerUseProvider: Send + Sync {
    /// Snapshot the current GUI state.
    async fn observe(&self, opts: &ObserveOptions)
        -> Result<ComputerObservation, ComputerUseError>;

    /// Apply a single non-observing action.
    ///
    /// Implementations may treat [`ComputerAction::Observe`] /
    /// [`ComputerAction::Screenshot`] as [`ComputerUseError::InvalidAction`]:
    /// the tool never forwards those here.
    async fn act(&self, action: &ComputerAction) -> Result<ActionOutcome, ComputerUseError>;

    /// A short platform identifier for diagnostics, e.g. `"macos"`.
    fn platform(&self) -> &'static str;
}

/// Shared handle to a backend. Cloning yields another reference to the same
/// instance, so the tool and any future consumers share one backend.
pub type SharedProvider = Arc<dyn ComputerUseProvider>;
