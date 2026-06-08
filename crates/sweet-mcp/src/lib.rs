// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! MCP tool provider backed by the official [`rmcp`] SDK.
//!
//! Connects to MCP (Model Context Protocol) servers via stdio (subprocess) or
//! Streamable HTTP, discovers their tools, and exposes them as
//! [`sweet_core::ToolSpec`] instances for registration on an agent.
//!
//! # Tool namespacing
//!
//! Tools are namespaced as `{server}__{tool}` (double-underscore separator)
//! to avoid collisions across servers. The separator is `__` rather than `.`
//! because provider tool-name schemas (Anthropic, OpenAI) only permit
//! `[a-zA-Z0-9_-]`.
//!
//! # Configuration
//!
//! Server connections are described in JSON using the standard `mcpServers`
//! format (compatible with Claude Desktop, VS Code, Cursor) and loaded via
//! [`McpConfig::from_file`] or [`McpConfig::from_json`].
//!
//! # Env var interpolation
//!
//! `${VAR}` placeholders in `env` and `headers` values are resolved by
//! [`McpConfig::resolve_env_vars`] from the caller-supplied map first, then
//! from the process environment. Unresolved placeholders are left as-is.

mod config;
mod error;
mod provider;

pub use config::{McpConfig, McpServerConfig};
pub use error::McpError;
pub use provider::{McpProvider, ToolFilter};
