// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! The public [`MacComputerUse`] provider and its platform routing.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use sweet_computer_use_core::{
    ActionOutcome, ComputerAction, ComputerObservation, ComputerUseError, ComputerUseProvider,
    ObserveOptions, SharedProvider,
};

/// macOS computer-use backend.
///
/// Holds the directory screenshots are written to; everything else is queried
/// live from the OS per call. Cheap to construct and `Clone`-free (share via
/// [`shared`](Self::shared)).
pub struct MacComputerUse {
    screenshot_dir: PathBuf,
}

impl MacComputerUse {
    /// Create a backend that writes captured screenshots under `screenshot_dir`
    /// (created on demand).
    pub fn new(screenshot_dir: impl Into<PathBuf>) -> Self {
        Self {
            screenshot_dir: screenshot_dir.into(),
        }
    }

    /// Convenience: build the backend already wrapped in a [`SharedProvider`].
    pub fn shared(screenshot_dir: impl Into<PathBuf>) -> SharedProvider {
        Arc::new(Self::new(screenshot_dir))
    }
}

// The OS calls are synchronous, so `observe` and the platform actions resolve
// in a single poll with no raw pointers held across an await point. `Wait` is
// the one exception: it `tokio::time::sleep`s so it yields instead of blocking
// the executor thread driving the provider. (`Send + Sync` still holds:
// `tokio::time::Sleep` is `Send`.)
#[async_trait]
impl ComputerUseProvider for MacComputerUse {
    async fn observe(
        &self,
        opts: &ObserveOptions,
    ) -> Result<ComputerObservation, ComputerUseError> {
        observe_impl(&self.screenshot_dir, opts)
    }

    async fn act(&self, action: &ComputerAction) -> Result<ActionOutcome, ComputerUseError> {
        // `Wait` is platform-neutral (just a pause) and must not block the
        // executor, so resolve it here rather than in the sync platform path.
        if let ComputerAction::Wait { millis } = action {
            tokio::time::sleep(std::time::Duration::from_millis(*millis)).await;
            return Ok(ActionOutcome::ok(format!("waited {millis} ms")));
        }
        act_impl(action)
    }

    fn platform(&self) -> &'static str {
        "macos"
    }
}

#[cfg(target_os = "macos")]
fn observe_impl(
    dir: &Path,
    opts: &ObserveOptions,
) -> Result<ComputerObservation, ComputerUseError> {
    crate::macos::observe(dir, opts)
}

#[cfg(not(target_os = "macos"))]
fn observe_impl(
    _dir: &Path,
    _opts: &ObserveOptions,
) -> Result<ComputerObservation, ComputerUseError> {
    Err(ComputerUseError::Unsupported(
        std::env::consts::OS.to_string(),
    ))
}

#[cfg(target_os = "macos")]
fn act_impl(action: &ComputerAction) -> Result<ActionOutcome, ComputerUseError> {
    crate::macos::act(action)
}

#[cfg(not(target_os = "macos"))]
fn act_impl(_action: &ComputerAction) -> Result<ActionOutcome, ComputerUseError> {
    Err(ComputerUseError::Unsupported(
        std::env::consts::OS.to_string(),
    ))
}
