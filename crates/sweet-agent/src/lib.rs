// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Agent loop, hooks, subagents, and handoffs for the sweet AI agent framework.
//!
//! This crate owns the *behavior* of an autonomous agent: the conversation loop
//! (`Agent`, `run()`), tool dispatch, hook/event dispatch, subagent nesting,
//! handoff transfers, and command routing. It depends on `sweet-core` for shared
//! data types (`Message`, `ToolSpec`, `Model` trait, `StreamSink`, etc.) but does
//! not depend on any LLM provider crate — concrete models are injected at
//! construction time by the binary.

pub mod agent;
pub mod commands;
pub mod dynamic_prompt;
pub mod extension;
pub mod handoff;
pub mod hooks;
pub mod memory;
pub mod runloop;
pub mod subagent;

#[cfg(any(test, feature = "test-util"))]
pub mod test_util;

pub use agent::Agent;
pub use agent::IntoContentBlocks;
pub use async_trait::async_trait;
pub use commands::{CommandContext, CommandHandler, CommandRouter, CommandSpec};
pub use dynamic_prompt::DynamicPrompt;
pub use extension::{
    Activation, Capability, CapabilityProvider, Extension, ExtensionRegistry, PromptSpec,
    ToolCapabilities,
};
pub use handoff::{HandoffContext, HandoffHandler, HandoffResult, HandoffSpec, TurnResult};
pub use hooks::{
    HookCapability, HookDispatcher, HookEvent, HookInvocation, ProcedureHandler, ProcedureSpec,
};
pub use memory::{
    memory_distill_capabilities, memory_distiller_capabilities, memory_recall_capabilities,
    DistillConfig, DistillError, DistillReport, MemoryDistiller, MemoryRecall,
    DISTILL_PROCEDURE_ID, RECALL_PROCEDURE_ID,
};
pub use runloop::{run, AgentIo, RunOutcome};
pub use subagent::{SubagentContext, SubagentHandler, SubagentSpec, DEFAULT_MAX_DEPTH};
