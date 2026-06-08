// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Subagents: configured `Agent`s the parent's model can invoke as tools.
//!
//! A subagent is described by a [`SubagentSpec`] and converted into a regular
//! [`ToolSpec`] so the parent's model invokes it through the normal tool-call
//! mechanism. Each invocation runs the developer-supplied [`SubagentHandler`],
//! which builds a fresh child `Agent` and returns its final assistant message
//! content as the tool result string.
//!
//! The framework adds two safety/ergonomics primitives on top of the tool
//! pattern:
//!
//! - **Nesting depth tracking.** A task-local counter tracks the current
//!   subagent depth; handlers see it on [`SubagentContext::depth`] and the
//!   framework refuses invocations that would exceed the spec's max depth
//!   (see [`SubagentSpec::with_max_nested_depth`]).
//! - **Parent-model handoff.** When the parent was built via
//!   [`crate::Agent::new_shared`], a clone of its `Arc<dyn Model>` is passed
//!   through [`SubagentContext::parent_model`]. Handlers can reuse it or
//!   ignore it in favour of a different model.

use std::sync::Arc;

use async_trait::async_trait;

use sweet_core::model::Model;
use sweet_core::permission::ToolRisk;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

/// Default cap on subagent nesting depth when a spec does not set one.
///
/// The value is high enough for the orchestrator(depth 0) → worker(depth 1)
/// → nested-subagent(depth 2) chain used in headless mode, with slack for
/// future nesting.
pub const DEFAULT_MAX_DEPTH: usize = 5;

tokio::task_local! {
    static SUBAGENT_DEPTH: usize;
    pub(crate) static PARENT_MODEL: Option<Arc<dyn Model>>;
}

/// Declarative description of a subagent that can be invoked as a tool.
///
/// Build with [`SubagentSpec::new`] and (optionally) tune with
/// [`SubagentSpec::with_max_nested_depth`]. The spec is consumed by
/// [`From<SubagentSpec> for ToolSpec`] (typically via
/// [`crate::Agent::with_subagent`]).
pub struct SubagentSpec {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) input_schema: serde_json::Value,
    pub(crate) max_nested_depth: Option<usize>,
    pub(crate) risk: ToolRisk,
    pub(crate) handler: Arc<dyn SubagentHandler>,
}

impl SubagentSpec {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
        handler: impl SubagentHandler + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            max_nested_depth: None,
            risk: ToolRisk::Dangerous,
            handler: Arc::new(handler),
        }
    }

    pub fn with_max_nested_depth(mut self, max: usize) -> Self {
        self.max_nested_depth = Some(max);
        self
    }

    /// Classify the subagent's risk for the permission system.
    ///
    /// Defaults to [`ToolRisk::Dangerous`] — the subagent's child agent could
    /// run anything. Read-only investigation subagents should set
    /// [`ToolRisk::ReadOnly`] so they are not gated behind an approval prompt.
    pub fn with_risk(mut self, risk: ToolRisk) -> Self {
        self.risk = risk;
        self
    }
}

/// Developer-written logic that builds and runs a subagent.
///
/// Handlers should construct a fresh `Agent` each call and return its final
/// assistant message content. Errors map to a `ToolError::Execution` that the
/// parent's model receives as a normal tool error.
#[async_trait]
pub trait SubagentHandler: Send + Sync {
    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: SubagentContext,
    ) -> Result<String, ToolError>;
}

/// Runtime context handed to a [`SubagentHandler`] on each invocation.
pub struct SubagentContext {
    /// Depth at which this subagent is running. Top-level subagents invoked
    /// by a non-subagent parent run at depth 1; nested subagents see 2, etc.
    pub depth: usize,
    /// The parent agent's model handle, present iff the parent was built via
    /// `Agent::new_shared`. Clone it to give the child agent the same model,
    /// or ignore it and use a different one.
    pub parent_model: Option<Arc<dyn Model>>,
}

impl From<SubagentSpec> for ToolSpec {
    fn from(spec: SubagentSpec) -> Self {
        let max_depth = spec.max_nested_depth.unwrap_or(DEFAULT_MAX_DEPTH);
        let parameters_schema = spec.input_schema.clone();
        let description = spec.description.clone();
        let name = spec.name.clone();
        let risk = spec.risk;
        let tool = SubagentTool {
            name: spec.name,
            max_depth,
            handler: spec.handler,
        };
        ToolSpec::new(name, description, parameters_schema, tool).with_risk(risk)
    }
}

struct SubagentTool {
    name: String,
    max_depth: usize,
    handler: Arc<dyn SubagentHandler>,
}

#[async_trait]
impl ToolHandler for SubagentTool {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let depth = SUBAGENT_DEPTH.try_with(|d| *d).unwrap_or(0);
        if depth >= self.max_depth {
            return Err(ToolError::Execution(
                format!(
                    "subagent `{}` would exceed max nesting depth of {}",
                    self.name, self.max_depth
                )
                .into(),
            ));
        }
        let parent_model = PARENT_MODEL.try_with(|m| m.clone()).ok().flatten();
        let ctx = SubagentContext {
            depth: depth + 1,
            parent_model,
        };
        SUBAGENT_DEPTH
            .scope(depth + 1, self.handler.invoke(args, ctx))
            .await
    }
}
