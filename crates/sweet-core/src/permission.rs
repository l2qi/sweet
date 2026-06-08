// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Permission types for tool approval gating.
//!
//! Every tool carries a [`ToolRisk`] level. The agent's [`PermissionMode`]
//! combined with the tool's risk determines whether the user must approve the
//! call before execution. Approval decisions are communicated through the
//! `AgentIo::on_tool_approval` callback in `sweet-agent`.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// How risky a tool invocation is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolRisk {
    /// Read-only — no side effects. Never requires approval.
    ReadOnly,
    /// Writes to the filesystem but does not run arbitrary commands.
    FileWrite,
    /// Arbitrary code execution (bash, shell). Most dangerous.
    #[default]
    Dangerous,
}

/// Controls how aggressively the agent auto-approves tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    /// Ask for approval on every write and dangerous tool.
    #[default]
    Normal = 0,
    /// Auto-approve file writes; still ask for dangerous tools (bash).
    AutoEdit = 1,
    /// Auto-approve everything. Use with caution.
    FullAuto = 2,
}

impl PermissionMode {
    /// Cycle to the next mode.
    pub fn cycle(self) -> Self {
        match self {
            Self::Normal => Self::AutoEdit,
            Self::AutoEdit => Self::FullAuto,
            Self::FullAuto => Self::Normal,
        }
    }

    /// Decode from the `u8` representation used by `AtomicU8` handles.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Normal,
            1 => Self::AutoEdit,
            _ => Self::FullAuto,
        }
    }
}

/// User's response to an approval prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApprovalDecision {
    /// Allow this one call.
    Allow,
    /// Allow and remember for the rest of the session.
    AllowSession,
    /// Reject the tool call.
    Deny,
}

/// Pure decision function: does this (mode, risk) combination require
/// explicit user approval?
pub fn needs_approval(mode: PermissionMode, risk: ToolRisk) -> bool {
    match (mode, risk) {
        // Read-only tools never ask.
        (_, ToolRisk::ReadOnly) => false,
        // Full auto approves everything.
        (PermissionMode::FullAuto, _) => false,
        // Auto-edit approves file writes but still asks for dangerous tools.
        (PermissionMode::AutoEdit, ToolRisk::FileWrite) => false,
        (PermissionMode::AutoEdit, ToolRisk::Dangerous) => true,
        // Normal mode asks for all non-readonly tools.
        (PermissionMode::Normal, ToolRisk::FileWrite) => true,
        (PermissionMode::Normal, ToolRisk::Dangerous) => true,
    }
}

/// The user-meaningful "scope" of a tool call — the bash command, the file
/// path being written, etc.
///
/// Used both as the granularity key for session-level ("Always") approvals
/// and as the preview shown in the approval prompt, so the two never diverge:
/// approving "Always" grants exactly what the prompt displayed, not the whole
/// tool. Falls back to the serialized arguments when no well-known field is
/// present.
pub fn approval_scope(args: &serde_json::Value) -> String {
    for key in ["command", "path", "source"] {
        if let Some(value) = args.get(key).and_then(|v| v.as_str()) {
            return value.to_string();
        }
    }
    serde_json::to_string(args).unwrap_or_else(|_| args.to_string())
}

/// Rich preview content shown in scrollback before an approval prompt.
///
/// For file-edit tools the agent computes a diff or content preview and
/// passes it to the UI, which renders it into scrollback *before* the
/// two-row approval prompt appears in the viewport. This gives the user
/// enough context to make an informed y/n decision.
#[derive(Debug, Clone)]
pub enum ApprovalPreview {
    /// No rich preview — just show the scope string (default behavior).
    None,
    /// A unified diff to show in scrollback before the approval prompt.
    Diff {
        tool_name: String,
        path: String,
        diff: String,
    },
    /// Full file content (for new file creation via write_file).
    NewFile { path: String, content: String },
}

/// Run-scoped permission state, shared across agent switches via an `Arc`.
///
/// Holds the current [`PermissionMode`] plus the set of `(tool, scope)` pairs
/// the user approved for the rest of the run. A new `Agent` built when
/// switching agents mid-run (e.g. changing mode or model) is handed the same
/// handle, so the mode *and* the session approvals survive the switch.
///
/// Deliberately not persisted: "session" here means the current process run,
/// not a resumable conversation — a resumed session must not silently
/// auto-approve dangerous tools.
#[derive(Debug, Default)]
pub struct PermissionState {
    mode: AtomicU8,
    allowed: Mutex<HashSet<(String, String)>>,
}

impl PermissionState {
    /// Create state starting in `mode` with no session approvals.
    pub fn new(mode: PermissionMode) -> Self {
        Self {
            mode: AtomicU8::new(mode as u8),
            allowed: Mutex::new(HashSet::new()),
        }
    }

    /// The current permission mode.
    pub fn mode(&self) -> PermissionMode {
        PermissionMode::from_u8(self.mode.load(Ordering::Relaxed))
    }

    /// Overwrite the permission mode.
    pub fn set_mode(&self, mode: PermissionMode) {
        self.mode.store(mode as u8, Ordering::Relaxed);
    }

    /// Advance to the next permission mode and return the new value.
    pub fn cycle_mode(&self) -> PermissionMode {
        let next = self.mode().cycle();
        self.set_mode(next);
        next
    }

    /// Whether the user approved `(tool, scope)` for the rest of the session.
    pub fn is_allowed(&self, tool: &str, scope: &str) -> bool {
        self.lock().contains(&(tool.to_string(), scope.to_string()))
    }

    /// Remember `(tool, scope)` as approved for the rest of the session.
    pub fn allow(&self, tool: String, scope: String) {
        self.lock().insert((tool, scope));
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashSet<(String, String)>> {
        // Poisoning only happens if a thread panicked mid-update; the set is
        // still readable, so recover rather than cascade the panic.
        self.allowed.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readonly_never_needs_approval() {
        assert!(!needs_approval(PermissionMode::Normal, ToolRisk::ReadOnly));
        assert!(!needs_approval(
            PermissionMode::AutoEdit,
            ToolRisk::ReadOnly
        ));
        assert!(!needs_approval(
            PermissionMode::FullAuto,
            ToolRisk::ReadOnly
        ));
    }

    #[test]
    fn full_auto_never_needs_approval() {
        assert!(!needs_approval(
            PermissionMode::FullAuto,
            ToolRisk::FileWrite
        ));
        assert!(!needs_approval(
            PermissionMode::FullAuto,
            ToolRisk::Dangerous
        ));
    }

    #[test]
    fn normal_mode_asks_for_all_non_readonly() {
        assert!(needs_approval(PermissionMode::Normal, ToolRisk::FileWrite));
        assert!(needs_approval(PermissionMode::Normal, ToolRisk::Dangerous));
    }

    #[test]
    fn auto_edit_approves_writes_but_asks_for_dangerous() {
        assert!(!needs_approval(
            PermissionMode::AutoEdit,
            ToolRisk::FileWrite
        ));
        assert!(needs_approval(
            PermissionMode::AutoEdit,
            ToolRisk::Dangerous
        ));
    }

    #[test]
    fn all_nine_combinations_covered() {
        let modes = [
            PermissionMode::Normal,
            PermissionMode::AutoEdit,
            PermissionMode::FullAuto,
        ];
        let risks = [ToolRisk::ReadOnly, ToolRisk::FileWrite, ToolRisk::Dangerous];
        for mode in &modes {
            for risk in &risks {
                // Just ensure no panic — the individual tests assert values.
                let _ = needs_approval(*mode, *risk);
            }
        }
    }

    #[test]
    fn approval_scope_prefers_command() {
        let args = serde_json::json!({"command": "ls -la", "cwd": "/tmp"});
        assert_eq!(approval_scope(&args), "ls -la");
    }

    #[test]
    fn approval_scope_falls_back_to_path() {
        let args = serde_json::json!({"path": "src/main.rs", "content": "x"});
        assert_eq!(approval_scope(&args), "src/main.rs");
    }

    #[test]
    fn approval_scope_uses_source_for_moves() {
        let args = serde_json::json!({"source": "a.txt", "destination": "b.txt"});
        assert_eq!(approval_scope(&args), "a.txt");
    }

    #[test]
    fn approval_scope_falls_back_to_serialized_args() {
        let args = serde_json::json!({"expr": "2 + 2"});
        let scope = approval_scope(&args);
        assert!(scope.contains("expr"));
        assert!(scope.contains("2 + 2"));
    }

    #[test]
    fn permission_state_cycles_mode() {
        let state = PermissionState::new(PermissionMode::Normal);
        assert_eq!(state.mode(), PermissionMode::Normal);
        assert_eq!(state.cycle_mode(), PermissionMode::AutoEdit);
        assert_eq!(state.mode(), PermissionMode::AutoEdit);
    }

    #[test]
    fn permission_state_remembers_allowed_scopes() {
        let state = PermissionState::default();
        assert_eq!(state.mode(), PermissionMode::Normal);
        assert!(!state.is_allowed("bash", "ls"));
        state.allow("bash".to_string(), "ls".to_string());
        assert!(state.is_allowed("bash", "ls"));
        // Scoped: a different command on the same tool is not covered.
        assert!(!state.is_allowed("bash", "pwd"));
        // Scoped: the same scope on a different tool is not covered.
        assert!(!state.is_allowed("write_file", "ls"));
    }
}
