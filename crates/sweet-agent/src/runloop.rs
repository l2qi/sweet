// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;

use sweet_core::error::Result;
use sweet_core::message::{Message, ToolCall};
use sweet_core::model::Model;
use sweet_core::permission::{ApprovalDecision, ToolRisk};
use sweet_core::stream::StreamSink;

use crate::agent::Agent;
use crate::handoff::TurnResult;

/// I/O surface used by [`run`] to drive a multi-turn conversation.
///
/// Returning `Ok(None)` from `read_input` signals end-of-input and exits the
/// loop cleanly. Empty/whitespace-only lines are filtered by the caller of
/// `read_input` (e.g. the CLI), not here — the runloop forwards every line it
/// receives.
///
/// Streaming UI events (`on_content_delta`, `on_tool_call`, `on_tool_result`,
/// `on_turn_start`, `on_turn_end`) have no-op default implementations so test
/// helpers and non-interactive backends do not need to implement them.
#[async_trait]
pub trait AgentIo: Send {
    async fn read_input(&mut self) -> Result<Option<String>>;
    async fn write_reply(
        &mut self,
        message: &Message,
        session: &dyn sweet_core::Session,
    ) -> Result<()>;

    async fn on_turn_start(&mut self) -> Result<()> {
        Ok(())
    }

    async fn on_content_delta(&mut self, _delta: &str) -> Result<()> {
        Ok(())
    }

    async fn on_thinking_delta(&mut self, _delta: &str) -> Result<()> {
        Ok(())
    }

    async fn on_tool_call(&mut self, _call: &ToolCall) -> Result<()> {
        Ok(())
    }

    async fn on_tool_result(&mut self, _call: &ToolCall, _result: &str) -> Result<()> {
        Ok(())
    }

    async fn on_turn_end(&mut self, _session: &dyn sweet_core::Session) -> Result<()> {
        Ok(())
    }

    /// Ask the user whether a tool call should be executed.
    ///
    /// Called when the agent's permission mode and the tool's risk level
    /// require explicit approval. The implementation should present the
    /// tool call details to the user and return their decision.
    async fn on_tool_approval(
        &mut self,
        _call: &ToolCall,
        _risk: ToolRisk,
    ) -> Result<ApprovalDecision> {
        Ok(ApprovalDecision::Allow)
    }
}

/// Outcome of the [`run`] loop.
pub enum RunOutcome {
    /// Input was exhausted normally.
    Eof,
    /// A handoff was requested; the caller should swap agents.
    Handoff {
        target: String,
        payload: Option<String>,
    },
}

/// Drive `agent` against `io` until input is exhausted or an error occurs.
pub async fn run<M, Io>(agent: &mut Agent<M>, io: &mut Io) -> Result<RunOutcome>
where
    M: Model,
    Io: AgentIo,
{
    while let Some(line) = io.read_input().await? {
        io.on_turn_start().await?;
        let result = agent.step_stream(line, io).await;
        let session_ref: &dyn sweet_core::Session = agent.session();
        io.on_turn_end(session_ref).await?;
        match result? {
            TurnResult::Message(_) => {}
            TurnResult::Handoff { target, payload } => {
                return Ok(RunOutcome::Handoff { target, payload });
            }
        }
    }
    Ok(RunOutcome::Eof)
}

/// Adapter that lets an [`AgentIo`] act as a [`StreamSink`] for the duration
/// of a single agent step. Forwarding-only — no buffering or transformation.
pub(crate) struct IoStreamSink<'a, Io: AgentIo + ?Sized> {
    io: &'a mut Io,
}

impl<'a, Io: AgentIo + ?Sized> IoStreamSink<'a, Io> {
    pub(crate) fn new(io: &'a mut Io) -> Self {
        Self { io }
    }
}

#[async_trait]
impl<Io: AgentIo + ?Sized> StreamSink for IoStreamSink<'_, Io> {
    async fn on_content_delta(&mut self, delta: &str) -> Result<()> {
        self.io.on_content_delta(delta).await
    }

    async fn on_thinking_delta(&mut self, delta: &str) -> Result<()> {
        self.io.on_thinking_delta(delta).await
    }

    async fn on_tool_call(&mut self, call: &ToolCall) -> Result<()> {
        self.io.on_tool_call(call).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{MockModel, VecIo};
    use sweet_core::message::Role;

    #[tokio::test]
    async fn run_drives_each_input_through_the_model() {
        let model = MockModel::with_replies(["r1", "r2"]);
        let mut agent = Agent::new(model);
        let mut io = VecIo::with_inputs(["q1", "q2"]);

        let outcome = run(&mut agent, &mut io).await.unwrap();
        assert!(matches!(outcome, RunOutcome::Eof));

        let deltas = io.deltas();
        assert_eq!(deltas, vec!["r1".to_string(), "r2".to_string()]);

        let messages = agent.session().messages();
        assert_eq!(messages.len(), 4);
        assert!(matches!(messages[1].role, Role::Assistant));
    }

    #[tokio::test]
    async fn run_returns_cleanly_on_empty_input() {
        let model = MockModel::with_replies(Vec::<&str>::new());
        let mut agent = Agent::new(model);
        let mut io = VecIo::with_inputs(Vec::<&str>::new());

        let outcome = run(&mut agent, &mut io).await.unwrap();
        assert!(matches!(outcome, RunOutcome::Eof));
        assert!(io.deltas().is_empty());
    }
}
