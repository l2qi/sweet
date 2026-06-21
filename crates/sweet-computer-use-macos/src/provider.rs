// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! The public [`MacComputerUse`] provider and its platform routing.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use sweet_computer_use_core::{
    ActionOutcome, ComputerAction, ComputerObservation, ComputerUseError, ComputerUseProvider,
    ObserveOptions, SharedProvider,
};
use sweet_core::sandbox::CommandRunner;
use sweet_core::DirectRunner;

/// macOS computer-use backend.
///
/// Holds the directory screenshots are written to and the command runner used
/// to launch applications. Everything else is queried live from the OS per
/// call. Cheap to construct and `Clone`-free (share via
/// [`shared`](Self::shared)).
pub struct MacComputerUse {
    screenshot_dir: PathBuf,
    runner: Arc<dyn CommandRunner>,
}

impl MacComputerUse {
    /// Create a backend that writes captured screenshots under `screenshot_dir`
    /// (created on demand), using [`DirectRunner`] (unsandboxed) for shell
    /// commands like `open_app`.
    pub fn new(screenshot_dir: impl Into<PathBuf>) -> Self {
        Self {
            screenshot_dir: screenshot_dir.into(),
            runner: Arc::new(DirectRunner),
        }
    }

    /// Use a custom [`CommandRunner`] for shell-invoking actions (`open_app`).
    /// Pass a sandboxed runner (e.g. from `sweet-sandbox`'s `OsSandbox`) to
    /// enforce the project's filesystem and network policy on those commands.
    pub fn with_command_runner(mut self, runner: Arc<dyn CommandRunner>) -> Self {
        self.runner = runner;
        self
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
            // Cap the pause so a model-requested wait can't hang the agent
            // loop indefinitely.
            const MAX_WAIT_MILLIS: u64 = 60_000;
            let capped = (*millis).min(MAX_WAIT_MILLIS);
            tokio::time::sleep(std::time::Duration::from_millis(capped)).await;
            return Ok(ActionOutcome::ok(format!("waited {capped} ms")));
        }
        // `OpenApp` shells out via `open -a`, which is an async command-run -
        // handled here through the injected [`CommandRunner`] rather than the
        // sync platform path.
        if let ComputerAction::OpenApp { name } = action {
            open_app(self.runner.as_ref(), name).await?;
            return Ok(ActionOutcome::ok(format!("opened {name}")));
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

/// Launch or focus an application by name via `open -a`, run through the
/// injected [`CommandRunner`] so a sandboxed runner can enforce the project's
/// filesystem and network policy. The app `name` is model-controlled, so it is
/// shell-quoted to prevent injection (the runner dispatches via `bash -c`).
///
/// macOS only: the `open` binary is macOS-specific. On other targets the stub
/// below reports [`ComputerUseError::Unsupported`], matching every other action.
#[cfg(target_os = "macos")]
async fn open_app(runner: &dyn CommandRunner, name: &str) -> Result<(), ComputerUseError> {
    let quoted = shell_quote(name);
    let command = format!("open -a {quoted}");
    let output = runner
        .run(&command, None, None)
        .await
        .map_err(|e| ComputerUseError::Platform(format!("failed to run `open`: {e}")))?;
    if output.exit_code == 0 {
        Ok(())
    } else {
        let detail = if output.stderr.is_empty() {
            format!("exit code {}", output.exit_code)
        } else {
            output.stderr.trim().to_string()
        };
        Err(ComputerUseError::Platform(format!(
            "`open -a {quoted}` failed: {detail}"
        )))
    }
}

/// `OpenApp` on a non-macOS target: there is no `open -a`, so report
/// [`ComputerUseError::Unsupported`] like every other action's stub.
#[cfg(not(target_os = "macos"))]
async fn open_app(_runner: &dyn CommandRunner, _name: &str) -> Result<(), ComputerUseError> {
    Err(ComputerUseError::Unsupported(
        std::env::consts::OS.to_string(),
    ))
}

/// Shell-quote a string for safe interpolation into a `bash -c` command: wrap
/// in single quotes and escape embedded single quotes as `'\''`. This prevents
/// a model-controlled app name from injecting shell metacharacters. Only the
/// macOS `open_app` shells out, so this is unused on other targets.
#[cfg(target_os = "macos")]
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn shell_quote_wraps_simple_string() {
        assert_eq!(shell_quote("Safari"), "'Safari'");
        assert_eq!(shell_quote("TextEdit"), "'TextEdit'");
    }

    #[test]
    fn shell_quote_escapes_embedded_single_quotes() {
        // A single quote in the name is escaped as '\'' (end-quote, escaped
        // quote, start-quote) so the result stays a single bash token.
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }

    #[test]
    fn shell_quote_neutralizes_shell_metacharacters() {
        // A malicious name with shell metacharacters is fully quoted, so the
        // resulting string is inert when passed to `bash -c`.
        let quoted = shell_quote("Safari; rm -rf /");
        assert_eq!(quoted, "'Safari; rm -rf /'");
        // No unquoted semicolon survives: the whole thing is one single-quoted
        // token.
        assert!(quoted.starts_with('\'') && quoted.ends_with('\''));
    }
}
