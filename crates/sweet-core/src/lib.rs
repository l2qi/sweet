// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Core data types and traits for the sweet AI agent framework.
//!
//! This crate intentionally stays minimal: it defines the shared vocabulary
//! (`Message`, `Role`, `ToolCall`, `ToolSpec`, `Model`, `StreamSink`, `Session`)
//! that all other sweet crates build on. Agent behavior lives in `sweet-agent`;
//! LLM provider implementations live in `sweet-llm`.

pub mod error;
pub mod message;
pub mod model;
pub mod permission;
pub mod sandbox;
pub mod session;
pub mod stream;
pub mod tool;
pub mod version;

pub use error::{Error, Result};
pub use message::{ContentBlock, Message, Role, ThinkingContent, ToolCall};
pub use model::Model;
pub use permission::{
    approval_scope, ApprovalDecision, ApprovalPreview, PermissionMode, PermissionState, ToolRisk,
};
pub use sandbox::{
    CommandOutput, CommandRunner, DirEntry, DirectFs, DirectRunner, DirectSandbox, FileMetadata,
    Filesystem, Sandbox, SandboxError, SandboxPolicy, SearchMatch,
};
pub use session::{
    last_context_size, InMemorySession, MemoryItem, Session, SessionError, SessionId,
    SharedSession, SharedSessionHandle,
};
pub use stream::{NoopSink, StreamSink};
pub use tool::{execution_error, ToolError, ToolFn, ToolHandler, ToolSpec};
pub use version::SWEET_VERSION;

pub use async_trait::async_trait;

#[cfg(feature = "derive")]
pub use sweet_tool_derive::Tool;

#[doc(hidden)]
#[cfg(feature = "derive")]
pub mod __private {
    pub use ::async_trait::async_trait;
    pub use ::schemars::schema_for;
    pub use ::serde_json;
}
