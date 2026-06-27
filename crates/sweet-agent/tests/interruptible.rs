// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Tests for the interruptible-approval turn: pause on `Defer`, resume without
//! re-invoking the model.
#![cfg(feature = "test-util")]

use sweet_agent::test_util::{MockModel, MockTool, VecIo};
use sweet_agent::{Agent, TurnOutcome, TurnResult};
use sweet_core::message::{Role, ToolCall};
use sweet_core::permission::ApprovalDecision;

fn tool_call(id: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: "echo".into(),
        arguments: serde_json::json!({ "msg": "hi" }),
    }
}

#[tokio::test]
async fn interruptible_turn_pauses_then_resumes_without_re_invoking_model() {
    // First model reply requests a (dangerous) tool call; second is plain text.
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![tool_call("call_1")]),
        MockModel::reply_text("done"),
    ]);
    let mut agent = Agent::new(model).with_tool(MockTool::echoing("echo"));

    let mut io = VecIo::with_inputs(Vec::<&str>::new());
    io.approval_decision = ApprovalDecision::Defer;

    // The turn pauses at the deferred call.
    let outcome = agent
        .step_stream_interruptible("go", &mut io)
        .await
        .unwrap();
    let pending = match outcome {
        TurnOutcome::Paused { pending } => pending,
        other => panic!("expected pause, got {other:?}"),
    };
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].tool_call.id, "call_1");

    // The model ran exactly once; the assistant message (with tool_calls) is
    // persisted, but no tool result yet — the side effect has not run.
    assert_eq!(agent.model().calls().len(), 1);
    let msgs = agent.session().messages();
    assert!(msgs
        .iter()
        .any(|m| m.role == Role::Assistant && !m.tool_calls.is_empty()));
    assert!(!msgs.iter().any(|m| m.role == Role::Tool));

    // Approve and resume: the tool runs, the model is called once more to
    // continue (not to re-do the first turn), and the turn completes.
    io.approval_decision = ApprovalDecision::Allow;
    let outcome = agent.resume_with_approvals(&mut io).await.unwrap();
    let msg = match outcome {
        TurnOutcome::Turn(TurnResult::Message(m)) => m,
        other => panic!("expected completion, got {other:?}"),
    };
    assert_eq!(msg.text_content(), "done");
    assert_eq!(
        agent.model().calls().len(),
        2,
        "first turn's model call was repeated"
    );
    let msgs = agent.session().messages();
    assert!(msgs
        .iter()
        .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("call_1")));
}

#[tokio::test]
async fn plain_step_treats_defer_as_deny() {
    // The non-interruptible `step_stream` cannot pause, so `Defer` is a denial
    // and the loop continues to a final message.
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![tool_call("c")]),
        MockModel::reply_text("after deny"),
    ]);
    let mut agent = Agent::new(model).with_tool(MockTool::echoing("echo"));

    let mut io = VecIo::with_inputs(Vec::<&str>::new());
    io.approval_decision = ApprovalDecision::Defer;

    let result = agent.step_stream("go", &mut io).await.unwrap();
    match result {
        TurnResult::Message(m) => assert_eq!(m.text_content(), "after deny"),
        TurnResult::Handoff { .. } => panic!("unexpected handoff"),
    }
    let msgs = agent.session().messages();
    let tool_msg = msgs.iter().find(|m| m.role == Role::Tool).unwrap();
    assert!(tool_msg.text_content().to_lowercase().contains("denied"));
}
