// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Handoffs: peer-level agent transfers where one agent delegates to another.
//!
//! Unlike subagents (which are nested tools), handoffs transfer the conversation
//! session from one agent to another at the same level. When agent A hands off to
//! agent B, B receives the full conversation history and takes over user interaction.
//!
//! Handoffs are represented as tools to the LLM. When a handoff tool is called,
//! the framework interrupts the current agent's step loop and returns a
//! [`TurnResult::Handoff`] so the caller can swap agents.

use std::sync::Arc;

use async_trait::async_trait;

use sweet_core::model::Model;
use sweet_core::permission::ToolRisk;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

/// Result of a single agent step - either a normal message or a handoff request.
#[derive(Debug, Clone)]
pub enum TurnResult {
    /// The agent produced a normal assistant message.
    Message(sweet_core::message::Message),
    /// The agent requested a handoff to another agent.
    Handoff {
        /// Identifier for the target agent (matches the handoff tool name).
        target: String,
        /// Optional structured payload from the handoff (e.g., an approved plan).
        payload: Option<String>,
    },
}

/// Specification for a handoff tool.
///
/// Build with [`HandoffSpec::new`] and register on an agent via
/// [`crate::Agent::with_handoff`]. The spec is consumed by
/// [`From<HandoffSpec> for ToolSpec`].
#[derive(Clone)]
pub struct HandoffSpec {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) input_schema: serde_json::Value,
    pub(crate) handler: Arc<dyn HandoffHandler>,
}

impl HandoffSpec {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
        handler: impl HandoffHandler + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            handler: Arc::new(handler),
        }
    }

    /// The handoff's tool-facing name (e.g. `transfer_to_plan`).
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Developer-written logic that handles a handoff invocation.
///
/// The handler receives the tool arguments and returns a [`HandoffResult`].
/// When the result is [`HandoffResult::Transfer`], the framework interrupts
/// the agent step loop and surfaces the handoff to the caller.
#[async_trait]
pub trait HandoffHandler: Send + Sync {
    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: HandoffContext,
    ) -> Result<HandoffResult, ToolError>;
}

/// Runtime context handed to a [`HandoffHandler`] on each invocation.
pub struct HandoffContext {
    /// The parent agent's model handle, present iff the parent was built via
    /// `Agent::new_shared`. Clone it to give the target agent the same model,
    /// or ignore it and use a different one.
    pub parent_model: Option<Arc<dyn Model>>,
}

/// Result of a handoff invocation.
#[derive(Debug, Clone)]
pub enum HandoffResult {
    /// Transfer control to the named target agent with an optional payload.
    Transfer {
        target: String,
        payload: Option<String>,
    },
}

impl From<HandoffSpec> for ToolSpec {
    fn from(spec: HandoffSpec) -> Self {
        let parameters_schema = spec.input_schema.clone();
        let description = spec.description.clone();
        let name = spec.name.clone();
        let tool = HandoffTool {
            handler: spec.handler,
        };
        ToolSpec::new(name, description, parameters_schema, tool).with_risk(ToolRisk::ReadOnly)
    }
}

struct HandoffTool {
    handler: Arc<dyn HandoffHandler>,
}

#[async_trait]
impl ToolHandler for HandoffTool {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let parent_model = crate::subagent::PARENT_MODEL
            .try_with(|m| m.clone())
            .ok()
            .flatten();
        let ctx = HandoffContext { parent_model };
        let result = self.handler.invoke(args, ctx).await?;
        match result {
            HandoffResult::Transfer { target, payload } => {
                // Serialize the handoff result so the agent loop can detect it.
                let payload_str = payload.unwrap_or_default();
                Err(ToolError::Handoff {
                    target,
                    payload: payload_str,
                })
            }
        }
    }
}
