// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Test helpers reusable by downstream crates.
//!
//! Available either when compiling tests for `sweet-agent` itself or when a
//! downstream crate enables the `test-util` feature.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;

use sweet_core::error::{Error, Result};
use sweet_core::message::{Message, Role, ToolCall};
use sweet_core::model::Model;
use sweet_core::session::MemoryItem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

use sweet_core::permission::{ApprovalDecision, ToolRisk};

use crate::runloop::AgentIo;

/// A `Model` impl that returns canned replies in order and records each call's
/// message slice.
pub struct MockModel {
    script: Mutex<VecDeque<MockReply>>,
    calls: Mutex<Vec<Vec<Message>>>,
    context_window: Option<usize>,
}

/// One reply from a `MockModel`.
#[derive(Debug, Clone)]
pub enum MockReply {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}

impl MockModel {
    pub fn with_replies<I, S>(replies: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            script: Mutex::new(
                replies
                    .into_iter()
                    .map(|s| MockReply::Text(s.into()))
                    .collect(),
            ),
            calls: Mutex::new(Vec::new()),
            context_window: None,
        }
    }

    pub fn with_scripted<I>(replies: I) -> Self
    where
        I: IntoIterator<Item = MockReply>,
    {
        Self {
            script: Mutex::new(replies.into_iter().collect()),
            calls: Mutex::new(Vec::new()),
            context_window: None,
        }
    }

    pub fn reply_text(s: impl Into<String>) -> MockReply {
        MockReply::Text(s.into())
    }

    pub fn reply_tool_calls(calls: Vec<ToolCall>) -> MockReply {
        MockReply::ToolCalls(calls)
    }

    pub fn with_context_window(mut self, tokens: usize) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// Snapshot of every `complete` invocation's message slice, in order.
    pub fn calls(&self) -> Vec<Vec<Message>> {
        self.calls
            .lock()
            .expect("MockModel::calls poisoned")
            .clone()
    }
}

#[async_trait]
impl Model for MockModel {
    async fn complete(&self, messages: &[Message], _tools: &[ToolSpec]) -> Result<Message> {
        self.calls
            .lock()
            .expect("MockModel::calls poisoned")
            .push(messages.to_vec());
        let next = self
            .script
            .lock()
            .expect("MockModel::script poisoned")
            .pop_front()
            .ok_or(Error::Unsupported("MockModel ran out of canned replies"))?;
        match next {
            MockReply::Text(t) => Ok(Message::assistant(t)),
            MockReply::ToolCalls(calls) => Ok(Message::with_tool_calls(calls)),
        }
    }

    fn context_window(&self) -> Option<usize> {
        self.context_window
    }
}

/// An `AgentIo` impl backed by an in-memory input queue and output buffer.
///
/// `outputs` collects the *final* assistant message at each turn (taken from
/// the session in [`AgentIo::on_turn_end`]) plus any messages written via
/// [`AgentIo::write_reply`] (e.g. command results). `deltas` records every
/// content delta emitted by the model, in order.
pub struct VecIo {
    inputs: VecDeque<String>,
    outputs: Vec<Message>,
    deltas: Vec<String>,
    tool_calls: Vec<ToolCall>,
    tool_results: Vec<(String, String)>,
    /// Configurable approval decision returned by `on_tool_approval`.
    /// Defaults to `Allow`.
    pub approval_decision: ApprovalDecision,
    /// Record of all approval requests received.
    pub approval_requests: Vec<(ToolCall, ToolRisk)>,
}

impl VecIo {
    pub fn with_inputs<I, S>(inputs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            inputs: inputs.into_iter().map(Into::into).collect(),
            outputs: Vec::new(),
            deltas: Vec::new(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            approval_decision: ApprovalDecision::Allow,
            approval_requests: Vec::new(),
        }
    }

    pub fn outputs(&self) -> &[Message] {
        &self.outputs
    }

    pub fn into_outputs(self) -> Vec<Message> {
        self.outputs
    }

    /// Every content delta the IO has seen, in order.
    pub fn deltas(&self) -> &[String] {
        &self.deltas
    }

    /// Every tool-call announcement the IO has seen, in order.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    /// Every (tool_call_id, result) pair the IO has seen, in order.
    pub fn tool_results(&self) -> &[(String, String)] {
        &self.tool_results
    }
}

#[async_trait]
impl AgentIo for VecIo {
    async fn read_input(&mut self) -> Result<Option<String>> {
        Ok(self.inputs.pop_front())
    }

    async fn write_reply(
        &mut self,
        message: &Message,
        _session: &dyn sweet_core::Session,
    ) -> Result<()> {
        self.outputs.push(message.clone());
        Ok(())
    }

    async fn on_content_delta(&mut self, delta: &str) -> Result<()> {
        self.deltas.push(delta.to_string());
        Ok(())
    }

    async fn on_tool_call(&mut self, call: &ToolCall) -> Result<()> {
        self.tool_calls.push(call.clone());
        Ok(())
    }

    async fn on_tool_result(&mut self, call: &ToolCall, result: &str) -> Result<()> {
        self.tool_results
            .push((call.id.clone(), result.to_string()));
        Ok(())
    }

    async fn on_turn_end(&mut self, session: &dyn sweet_core::Session) -> Result<()> {
        for item in session.items().iter().rev() {
            let MemoryItem::Message(m) = item;
            if m.role == Role::Assistant {
                self.outputs.push(m.clone());
                break;
            }
        }
        Ok(())
    }

    async fn on_tool_approval(
        &mut self,
        call: &ToolCall,
        risk: ToolRisk,
    ) -> Result<ApprovalDecision> {
        self.approval_requests.push((call.clone(), risk));
        Ok(self.approval_decision)
    }
}

/// A tool handler that records invocations and returns the JSON arguments as a
/// pretty-printed string. Useful for asserting that the agent passed the right
/// arguments.
#[derive(Debug)]
pub struct MockTool {
    name: &'static str,
    invocations: Mutex<Vec<serde_json::Value>>,
}

impl Clone for MockTool {
    fn clone(&self) -> Self {
        Self {
            name: self.name,
            invocations: Mutex::new(self.invocations.lock().unwrap().clone()),
        }
    }
}

impl MockTool {
    pub fn echoing(name: &'static str) -> Self {
        Self {
            name,
            invocations: Mutex::new(Vec::new()),
        }
    }

    pub fn invocations(&self) -> Vec<serde_json::Value> {
        self.invocations.lock().unwrap().clone()
    }
}

#[async_trait]
impl ToolHandler for MockTool {
    async fn call(&self, args: serde_json::Value) -> std::result::Result<String, ToolError> {
        self.invocations.lock().unwrap().push(args.clone());
        Ok(serde_json::to_string_pretty(&args).unwrap_or_else(|_| args.to_string()))
    }
}

impl From<MockTool> for ToolSpec {
    fn from(tool: MockTool) -> Self {
        let name = tool.name;
        ToolSpec::new(
            name,
            "A mock tool that echoes its arguments back.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "msg": { "type": "string" }
                }
            }),
            tool,
        )
    }
}
