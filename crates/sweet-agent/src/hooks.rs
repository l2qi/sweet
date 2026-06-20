// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;

use sweet_core::error::{Error, Result};
use sweet_core::tool::ToolSpec;

use crate::commands::{CommandContext, CommandSpec};

/// Runtime event that may trigger one or more hook capabilities.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookEvent {
    /// Fires once at the start of a turn, after the user message has been
    /// appended to the session and after any orphaned-tool-call repair, but
    /// before the first model call.
    BeforeTurn,
    /// Fires once at the end of a turn, after the loop has settled on a
    /// final assistant message or a handoff. Hooks here see the full
    /// transcript for the turn, including all tool calls and tool results.
    AfterTurn,
    BeforeModelCall,
    AfterModelReply,
    BeforeToolCall,
    AfterToolCall,
    BeforeCommand,
    AfterCommand,
    BeforePermissionCheck,
    AfterPermissionCheck,
    ResourceLoaded,
    Custom(String),
}

/// Declarative hook subscription contributed by an extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookCapability {
    pub event: HookEvent,
    pub handler_id: String,
}

/// Runtime hook invocation payload passed to executable procedure handlers.
#[derive(Debug, Clone)]
pub struct HookInvocation {
    pub event: HookEvent,
    pub payload: serde_json::Value,
}

/// Executable procedure for runtime-invoked behavior.
#[async_trait]
pub trait ProcedureHandler: Send + Sync {
    async fn handle(&self, invocation: &HookInvocation, ctx: &mut dyn CommandContext)
        -> Result<()>;
}

/// Describes a runtime procedure and carries its executable handler.
#[derive(Clone)]
pub struct ProcedureSpec {
    pub id: String,
    pub description: String,
    pub handler: Arc<dyn ProcedureHandler>,
}

impl ProcedureSpec {
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        handler: impl ProcedureHandler + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            handler: Arc::new(handler),
        }
    }
}

/// Resolves declarative hook subscriptions to executable capabilities.
///
/// The dispatcher owns its hook subscriptions and the procedure/command
/// targets they bind to, but it does *not* own the tool registry - the
/// agent does. Tools are passed into [`HookDispatcher::fire`] at dispatch
/// time so there is a single source of truth for "what tools does this
/// agent know about?".
#[derive(Default, Clone)]
pub struct HookDispatcher {
    hooks: Vec<HookCapability>,
    procedures: Vec<ProcedureSpec>,
    commands: Vec<CommandSpec>,
}

impl HookDispatcher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_hook(&mut self, hook: HookCapability) {
        self.hooks.push(hook);
    }

    pub fn register_procedure(&mut self, procedure: ProcedureSpec) {
        self.procedures.push(procedure);
    }

    pub fn register_command(&mut self, command: CommandSpec) {
        self.commands.push(command);
    }

    pub async fn fire(
        &self,
        event: HookEvent,
        payload: serde_json::Value,
        ctx: &mut dyn CommandContext,
        tools: &[ToolSpec],
    ) -> Result<()> {
        let invocation = HookInvocation { event, payload };
        for handler_id in self.handler_ids_for(&invocation.event) {
            match self.target_for(&handler_id, tools) {
                Some(HookTarget::Procedure(procedure)) => {
                    procedure.handler.handle(&invocation, ctx).await?;
                }
                Some(HookTarget::Command(command)) => {
                    let args = command_args(&invocation.payload);
                    command.handler.handle(&args, ctx).await?;
                }
                Some(HookTarget::Tool(tool)) => {
                    tool.call(invocation.payload.clone()).await?;
                }
                None => return Err(Error::UnknownHookHandler(handler_id)),
            }
        }
        Ok(())
    }

    fn handler_ids_for(&self, event: &HookEvent) -> Vec<String> {
        self.hooks
            .iter()
            .filter(|hook| hook.event == *event)
            .map(|hook| hook.handler_id.clone())
            .collect()
    }

    fn target_for(&self, id: &str, tools: &[ToolSpec]) -> Option<HookTarget> {
        self.procedures
            .iter()
            .find(|procedure| procedure.id == id)
            .cloned()
            .map(HookTarget::Procedure)
            .or_else(|| {
                self.commands
                    .iter()
                    .find(|command| command.name == id)
                    .cloned()
                    .map(HookTarget::Command)
            })
            .or_else(|| {
                tools
                    .iter()
                    .find(|tool| tool.name == id)
                    .cloned()
                    .map(HookTarget::Tool)
            })
    }
}

enum HookTarget {
    Procedure(ProcedureSpec),
    Command(CommandSpec),
    Tool(ToolSpec),
}

fn command_args(payload: &serde_json::Value) -> String {
    payload
        .get("args")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::test_util::MockModel;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct CountingProcedure {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ProcedureHandler for CountingProcedure {
        async fn handle(
            &self,
            invocation: &HookInvocation,
            _ctx: &mut dyn CommandContext,
        ) -> Result<()> {
            assert_eq!(invocation.event, HookEvent::BeforeModelCall);
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn dispatcher_invokes_matching_procedure() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut dispatcher = HookDispatcher::new();
        dispatcher.register_procedure(ProcedureSpec::new(
            "count",
            "Count hook invocations",
            CountingProcedure {
                calls: calls.clone(),
            },
        ));
        dispatcher.register_hook(HookCapability {
            event: HookEvent::BeforeModelCall,
            handler_id: "count".to_string(),
        });
        let mut agent = Agent::new(MockModel::with_replies(Vec::<&str>::new()));

        dispatcher
            .fire(
                HookEvent::BeforeModelCall,
                serde_json::json!({}),
                &mut agent,
                &[],
            )
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dispatcher_skips_non_matching_event() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut dispatcher = HookDispatcher::new();
        dispatcher.register_procedure(ProcedureSpec::new(
            "count",
            "Count hook invocations",
            CountingProcedure {
                calls: calls.clone(),
            },
        ));
        dispatcher.register_hook(HookCapability {
            event: HookEvent::BeforeModelCall,
            handler_id: "count".to_string(),
        });
        let mut agent = Agent::new(MockModel::with_replies(Vec::<&str>::new()));

        dispatcher
            .fire(
                HookEvent::AfterModelReply,
                serde_json::json!({}),
                &mut agent,
                &[],
            )
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn unknown_hook_handler_is_typed_error() {
        let mut dispatcher = HookDispatcher::new();
        dispatcher.register_hook(HookCapability {
            event: HookEvent::BeforeModelCall,
            handler_id: "missing".to_string(),
        });
        let mut agent = Agent::new(MockModel::with_replies(Vec::<&str>::new()));

        let err = dispatcher
            .fire(
                HookEvent::BeforeModelCall,
                serde_json::json!({}),
                &mut agent,
                &[],
            )
            .await
            .unwrap_err();

        assert!(matches!(err, Error::UnknownHookHandler(id) if id == "missing"));
    }
}
