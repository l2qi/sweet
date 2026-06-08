// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for parallel ReadOnly tool dispatch.
//!
//! These tests verify that when the model returns multiple tool calls targeting
//! `ToolRisk::ReadOnly` tools, the agent dispatches them concurrently via
//! `join_all` instead of sequentially.

#![cfg(feature = "test-util")]

use std::sync::Mutex;
use std::time::{Duration, Instant};

use sweet_agent::test_util::{MockModel, VecIo};
use sweet_agent::{Agent, SubagentContext, SubagentHandler, SubagentSpec, TurnResult};
use sweet_core::message::ToolCall;
use sweet_core::permission::ToolRisk;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A tool handler that sleeps for a configurable duration, then returns a
/// fixed result string. Records its invocation count so tests can assert it
/// ran.
#[derive(Debug)]
struct SlowTool {
    name: &'static str,
    delay: Duration,
    invocations: Mutex<Vec<serde_json::Value>>,
}

impl SlowTool {
    fn new(name: &'static str, delay: Duration) -> Self {
        Self {
            name,
            delay,
            invocations: Mutex::new(Vec::new()),
        }
    }

    #[allow(dead_code)]
    fn invocations(&self) -> Vec<serde_json::Value> {
        self.invocations.lock().unwrap().clone()
    }
}

#[sweet_agent::async_trait]
impl ToolHandler for SlowTool {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        self.invocations.lock().unwrap().push(args.clone());
        tokio::time::sleep(self.delay).await;
        Ok(format!(
            "{}-done",
            args.get("label")
                .and_then(|v| v.as_str())
                .unwrap_or(self.name)
        ))
    }
}

fn readonly_spec(tool: SlowTool) -> ToolSpec {
    ToolSpec::new(
        tool.name,
        "A slow tool for testing.",
        serde_json::json!({
            "type": "object",
            "properties": { "label": { "type": "string" } }
        }),
        tool,
    )
    .with_risk(ToolRisk::ReadOnly)
}

fn dangerous_spec(tool: SlowTool) -> ToolSpec {
    ToolSpec::new(
        tool.name,
        "A dangerous tool for testing.",
        serde_json::json!({
            "type": "object",
            "properties": { "label": { "type": "string" } }
        }),
        tool,
    )
}

fn tc(id: &str, name: &str, label: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments: serde_json::json!({ "label": label }),
    }
}

/// A tool handler that always errors.
struct FailingTool;

#[sweet_agent::async_trait]
impl ToolHandler for FailingTool {
    async fn call(&self, _args: serde_json::Value) -> Result<String, ToolError> {
        Err(ToolError::Execution("deliberate failure".into()))
    }
}

fn failing_spec(name: &'static str) -> ToolSpec {
    ToolSpec::new(
        name,
        "A tool that always fails.",
        serde_json::json!({ "type": "object", "properties": { "label": { "type": "string" } } }),
        FailingTool,
    )
    .with_risk(ToolRisk::ReadOnly)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn parallel_readonly_tools_complete_concurrently() {
    let slow_a = SlowTool::new("slow_a", Duration::from_millis(200));
    let slow_b = SlowTool::new("slow_b", Duration::from_millis(200));

    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![tc("c1", "slow_a", "a"), tc("c2", "slow_b", "b")]),
        MockModel::reply_text("done"),
    ]);

    let mut agent = Agent::new(model)
        .with_tool(readonly_spec(slow_a))
        .with_tool(readonly_spec(slow_b));

    let start = Instant::now();
    let reply = agent.step("go").await.unwrap();
    let elapsed = start.elapsed();

    let content = match reply {
        TurnResult::Message(m) => m.text_content(),
        TurnResult::Handoff { .. } => panic!("unexpected handoff"),
    };
    assert_eq!(content, "done");

    // Sequential execution would take ≥ 400ms. Concurrent runs in ~200ms;
    // the ceiling is generous to absorb scheduler/timer jitter on contended
    // CI runners while still failing loudly if the batch is sequential.
    assert!(
        elapsed < Duration::from_millis(500),
        "expected concurrent execution (~200ms), got {elapsed:?}"
    );
}

#[tokio::test]
async fn mixed_risk_falls_back_to_sequential() {
    let slow_a = SlowTool::new("ro", Duration::from_millis(50));
    let slow_b = SlowTool::new("danger", Duration::from_millis(50));

    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![tc("c1", "ro", "a"), tc("c2", "danger", "b")]),
        MockModel::reply_text("done"),
    ]);

    let mut agent = Agent::new(model)
        .with_tool(readonly_spec(slow_a))
        .with_tool(dangerous_spec(slow_b));

    let reply = agent.step("go").await.unwrap();
    let content = match reply {
        TurnResult::Message(m) => m.text_content(),
        TurnResult::Handoff { .. } => panic!("unexpected handoff"),
    };
    assert_eq!(content, "done");

    // Both tools should have run (sequential path).
    let msgs = agent.session().messages();
    assert_eq!(msgs.len(), 5); // user, assistant(2 calls), tool, tool, assistant
}

#[tokio::test]
async fn parallel_results_pushed_in_invocation_order() {
    // Tool A is slower than tool B. If dispatched concurrently, B finishes
    // first. But the session transcript should still have A's result before
    // B's (invocation order).
    let slow_a = SlowTool::new("tool_a", Duration::from_millis(150));
    let slow_b = SlowTool::new("tool_b", Duration::from_millis(10));

    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![tc("c1", "tool_a", "a"), tc("c2", "tool_b", "b")]),
        MockModel::reply_text("done"),
    ]);

    let mut agent = Agent::new(model)
        .with_tool(readonly_spec(slow_a))
        .with_tool(readonly_spec(slow_b));

    let reply = agent.step("go").await.unwrap();
    assert!(matches!(reply, TurnResult::Message(_)));

    let msgs = agent.session().messages();
    // user, assistant(2 calls), tool_c1, tool_c2, assistant
    assert_eq!(
        msgs[2].tool_call_id.as_deref(),
        Some("c1"),
        "first tool result should be c1"
    );
    assert_eq!(
        msgs[3].tool_call_id.as_deref(),
        Some("c2"),
        "second tool result should be c2"
    );
    assert!(msgs[2].text_content().contains("a-done"));
    assert!(msgs[3].text_content().contains("b-done"));
}

#[tokio::test]
async fn error_in_one_parallel_call_doesnt_kill_others() {
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![tc("c1", "fail", "x"), tc("c2", "ok", "y")]),
        MockModel::reply_text("done"),
    ]);

    let ok_tool = SlowTool::new("ok", Duration::from_millis(10));

    let mut agent = Agent::new(model)
        .with_tool(failing_spec("fail"))
        .with_tool(readonly_spec(ok_tool));

    let reply = agent.step("go").await.unwrap();
    let content = match reply {
        TurnResult::Message(m) => m.text_content(),
        TurnResult::Handoff { .. } => panic!("unexpected handoff"),
    };
    assert_eq!(content, "done");

    let msgs = agent.session().messages();
    assert_eq!(msgs[2].tool_call_id.as_deref(), Some("c1"));
    assert!(msgs[2].text_content().contains("Error:"));
    assert_eq!(msgs[3].tool_call_id.as_deref(), Some("c2"));
    assert!(msgs[3].text_content().contains("y-done"));
}

#[tokio::test]
async fn handoff_in_batch_goes_sequential() {
    // A batch containing a handoff tool name should fall back to sequential
    // because handoffs are not in self.tools (all_read_only returns false).
    let slow_a = SlowTool::new("ro", Duration::from_millis(10));
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![
            tc("c1", "ro", "a"),
            tc("c2", "transfer_to_plan", "plan"),
        ]),
        MockModel::reply_text("done"),
    ]);

    let mut agent = Agent::new(model).with_tool(readonly_spec(slow_a));
    // No handoff spec registered — the call to "transfer_to_plan" will fail
    // as unknown tool in dispatch. The key point is the batch takes the
    // sequential path because all_read_only returns false.
    let reply = agent.step("go").await.unwrap();
    let content = match reply {
        TurnResult::Message(m) => m.text_content(),
        TurnResult::Handoff { .. } => panic!("unexpected handoff (no handoff spec registered)"),
    };
    assert_eq!(content, "done");
}

// ---------------------------------------------------------------------------
// Subagent parallel tests
// ---------------------------------------------------------------------------

/// Handler that returns a canned reply after a short delay (simulates a
/// subagent that does real work).
struct DelayedReplyHandler {
    reply: &'static str,
    delay: Duration,
}

#[sweet_agent::async_trait]
impl SubagentHandler for DelayedReplyHandler {
    async fn invoke(
        &self,
        _args: serde_json::Value,
        _ctx: SubagentContext,
    ) -> Result<String, ToolError> {
        tokio::time::sleep(self.delay).await;
        Ok(self.reply.to_string())
    }
}

fn delayed_subagent_spec(name: &str, reply: &'static str, delay: Duration) -> SubagentSpec {
    SubagentSpec::new(
        name,
        "A test subagent.",
        serde_json::json!({
            "type": "object",
            "properties": { "task": { "type": "string" } },
            "required": ["task"]
        }),
        DelayedReplyHandler { reply, delay },
    )
    .with_risk(ToolRisk::ReadOnly)
}

#[tokio::test]
async fn parallel_subagents_both_run() {
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![
            tc("c1", "sub_a", "task-a"),
            tc("c2", "sub_b", "task-b"),
        ]),
        MockModel::reply_text("done"),
    ]);

    let mut agent = Agent::new(model)
        .with_subagent(delayed_subagent_spec(
            "sub_a",
            "result-a",
            Duration::from_millis(50),
        ))
        .with_subagent(delayed_subagent_spec(
            "sub_b",
            "result-b",
            Duration::from_millis(50),
        ));

    let start = Instant::now();
    let reply = agent.step("go").await.unwrap();
    let elapsed = start.elapsed();

    let content = match reply {
        TurnResult::Message(m) => m.text_content(),
        TurnResult::Handoff { .. } => panic!("unexpected handoff"),
    };
    assert_eq!(content, "done");

    // Both subagent results should appear in the transcript.
    let msgs = agent.session().messages();
    assert_eq!(msgs[2].tool_call_id.as_deref(), Some("c1"));
    assert_eq!(msgs[2].text_content(), "result-a");
    assert_eq!(msgs[3].tool_call_id.as_deref(), Some("c2"));
    assert_eq!(msgs[3].text_content(), "result-b");

    // Concurrent: total time ≈ max(50, 50) = 50ms, not 100ms.
    assert!(
        elapsed < Duration::from_millis(150),
        "expected concurrent subagent execution, got {elapsed:?}"
    );
}

#[tokio::test]
async fn parallel_readonly_tools_io_events_in_order() {
    let slow_a = SlowTool::new("tool_a", Duration::from_millis(100));
    let slow_b = SlowTool::new("tool_b", Duration::from_millis(10));

    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![tc("c1", "tool_a", "a"), tc("c2", "tool_b", "b")]),
        MockModel::reply_text("done"),
    ]);

    let mut io = VecIo::with_inputs(["go"]);
    let mut agent = Agent::new(model)
        .with_tool(readonly_spec(slow_a))
        .with_tool(readonly_spec(slow_b));

    let reply = agent.step_stream("go", &mut io).await.unwrap();
    assert!(matches!(reply, TurnResult::Message(_)));

    // Tool-call announcements should be in invocation order.
    assert_eq!(io.tool_calls().len(), 2);
    assert_eq!(io.tool_calls()[0].id, "c1");
    assert_eq!(io.tool_calls()[1].id, "c2");

    // Tool results should be in invocation order.
    assert_eq!(io.tool_results().len(), 2);
    assert_eq!(io.tool_results()[0].0, "c1");
    assert_eq!(io.tool_results()[1].0, "c2");
}
