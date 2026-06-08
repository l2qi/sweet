// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

#[derive(Error, Debug)]
pub enum McpError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("tool call failed: {0}")]
    ToolCall(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
