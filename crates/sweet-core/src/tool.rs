// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::error::Error as StdError;
use std::sync::Arc;

use async_trait::async_trait;

use crate::message::ContentBlock;
use crate::permission::ToolRisk;

/// Describes a model-facing tool and carries its executable handler.
#[derive(Clone)]
pub struct ToolSpec {
    /// Unique tool identifier used in the protocol (e.g. `"http_fetch"`).
    pub name: String,
    /// Human-readable description sent to the model.
    pub description: String,
    /// JSON Schema object describing the tool's parameters.
    pub parameters_schema: serde_json::Value,
    /// Executable tool logic.
    pub handler: Arc<dyn ToolHandler>,
    /// Risk classification for the permission system.
    pub risk: ToolRisk,
}

impl ToolSpec {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters_schema: serde_json::Value,
        handler: impl ToolHandler + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters_schema,
            handler: Arc::new(handler),
            risk: ToolRisk::Dangerous,
        }
    }

    pub fn with_risk(mut self, risk: ToolRisk) -> Self {
        self.risk = risk;
        self
    }

    pub async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        self.handler.call(args).await
    }

    /// Rich variant of [`call`](Self::call) that may return image content
    /// alongside text (e.g. a screenshot). See [`ToolOutput`].
    pub async fn call_rich(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError> {
        self.handler.call_rich(args).await
    }
}

/// The content a tool returns to the model: text, optionally with images.
///
/// [`ToolHandler::call`] yields a plain `String`; the framework wraps it in a
/// single text block. Tools that need to show the model an image override
/// [`ToolHandler::call_rich`] and attach image blocks via
/// [`with_image`](Self::with_image).
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// Ordered content blocks (text and/or images).
    pub blocks: Vec<ContentBlock>,
}

impl ToolOutput {
    /// A text-only output.
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            blocks: vec![ContentBlock::text(s)],
        }
    }

    /// Append an image block (raw bytes + MIME type, e.g. `"image/png"`).
    pub fn with_image(mut self, data: Vec<u8>, media_type: impl Into<String>) -> Self {
        self.blocks.push(ContentBlock::Image {
            data,
            media_type: media_type.into(),
        });
        self
    }

    /// The concatenated text of all text blocks (for display and logging).
    pub fn text_content(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| b.as_text())
            .collect::<Vec<_>>()
            .join("")
    }
}

impl From<String> for ToolOutput {
    fn from(s: String) -> Self {
        Self::text(s)
    }
}

impl From<&str> for ToolOutput {
    fn from(s: &str) -> Self {
        Self::text(s)
    }
}

impl std::fmt::Display for ToolOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for block in &self.blocks {
            write!(f, "{block}")?;
        }
        Ok(())
    }
}

/// Executable handler for a [`ToolSpec`].
#[async_trait]
pub trait ToolHandler: Send + Sync {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError>;

    /// Rich variant that may return images alongside text. The default wraps
    /// [`call`](Self::call)'s text. Tools that surface an image to the model
    /// (e.g. a screenshot) override this. Whether the image survives to the
    /// provider depends on the wire protocol: Anthropic carries images on
    /// tool-result messages; the OpenAI / Chat Completions protocol does not,
    /// so images are dropped there and only the text remains.
    async fn call_rich(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(self.call(args).await?))
    }
}

/// Logic trait implemented by the user. Kept separate from [`ToolHandler`] so
/// the derive macro can generate the dispatch plumbing while the user writes
/// only business logic.
#[async_trait]
pub trait ToolFn: Send + Sync {
    async fn run(self) -> Result<String, ToolError>;
}

/// Errors that can occur during tool execution.
#[derive(thiserror::Error, Debug)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    InvalidArgs(#[from] serde_json::Error),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("execution failed: {0}")]
    Execution(#[source] Box<dyn StdError + Send + Sync>),
    /// A handoff was requested. This is not a true error - it signals that
    /// the agent step loop should be interrupted and control transferred.
    #[error("handoff to {target}")]
    Handoff { target: String, payload: String },

    /// The user denied the tool call via the permission system.
    #[error("permission denied: {0}")]
    PermissionDenied(String),
}

/// Convenience: turn any `E: StdError + Send + Sync + 'static` into a tool error.
pub fn execution_error<E>(e: E) -> ToolError
where
    E: StdError + Send + Sync + 'static,
{
    ToolError::Execution(Box::new(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct CountingHandler {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ToolHandler for CountingHandler {
        async fn call(&self, _args: serde_json::Value) -> Result<String, ToolError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok("ok".to_string())
        }
    }

    #[tokio::test]
    async fn tool_spec_clones_share_handler() {
        let calls = Arc::new(AtomicUsize::new(0));
        let tool = ToolSpec::new(
            "count",
            "Count calls",
            serde_json::json!({"type": "object"}),
            CountingHandler {
                calls: calls.clone(),
            },
        );

        let cloned = tool.clone();
        tool.call(serde_json::json!({})).await.unwrap();
        cloned.call(serde_json::json!({})).await.unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(cloned.name, "count");
    }
}
