// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the permission/approval system.

use sweet_agent::test_util::{MockModel, MockTool, VecIo};
use sweet_agent::{Agent, TurnResult};
use sweet_core::permission::{ApprovalDecision, PermissionMode, ToolRisk};
use sweet_core::tool::ToolSpec;
use sweet_core::{Role, ToolCall};

/// A `MockTool` configured with a specific risk level.
fn tool_with_risk(name: &'static str, risk: ToolRisk) -> ToolSpec {
    let spec: ToolSpec = MockTool::echoing(name).into();
    spec.with_risk(risk)
}

#[tokio::test]
async fn readonly_tool_always_executes_in_normal_mode() {
    let tool = tool_with_risk("read", ToolRisk::ReadOnly);
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "read".into(),
            arguments: serde_json::json!({"msg": "hello"}),
        }]),
        MockModel::reply_text("done"),
    ]);
    let mut agent = Agent::new(model)
        .with_tool(tool)
        .with_permission_mode(PermissionMode::Normal);
    let mut io = VecIo::with_inputs(Vec::<&str>::new());
    io.approval_decision = ApprovalDecision::Deny; // should never be asked

    let result = agent.step_stream("go", &mut io).await.unwrap();
    assert!(matches!(result, TurnResult::Message(_)));
    // No approval requests - readonly tools skip the gate.
    assert!(io.approval_requests.is_empty());
}

#[tokio::test]
async fn write_tool_blocked_when_denied_in_normal_mode() {
    let tool = tool_with_risk("write", ToolRisk::FileWrite);
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "write".into(),
            arguments: serde_json::json!({"msg": "data"}),
        }]),
        MockModel::reply_text("ok"),
    ]);
    let mut agent = Agent::new(model)
        .with_tool(tool)
        .with_permission_mode(PermissionMode::Normal);
    let mut io = VecIo::with_inputs(Vec::<&str>::new());
    io.approval_decision = ApprovalDecision::Deny;

    let result = agent.step_stream("go", &mut io).await.unwrap();
    assert!(matches!(result, TurnResult::Message(_)));

    // The approval was requested and denied.
    assert_eq!(io.approval_requests.len(), 1);
    assert_eq!(io.approval_requests[0].0.name, "write");

    // The tool result should contain the permission denied error.
    let tool_results = io.tool_results();
    assert_eq!(tool_results.len(), 1);
    assert!(
        tool_results[0].1.contains("permission denied"),
        "expected permission denied, got: {}",
        tool_results[0].1
    );
}

#[tokio::test]
async fn write_tool_executes_when_approved_in_normal_mode() {
    let tool = tool_with_risk("write", ToolRisk::FileWrite);
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "write".into(),
            arguments: serde_json::json!({"msg": "data"}),
        }]),
        MockModel::reply_text("done"),
    ]);
    let mut agent = Agent::new(model)
        .with_tool(tool)
        .with_permission_mode(PermissionMode::Normal);
    let mut io = VecIo::with_inputs(Vec::<&str>::new());
    io.approval_decision = ApprovalDecision::Allow;

    let result = agent.step_stream("go", &mut io).await.unwrap();
    assert!(matches!(result, TurnResult::Message(_)));

    // Approval was requested.
    assert_eq!(io.approval_requests.len(), 1);

    // The tool actually executed - result contains the echoed args.
    let tool_results = io.tool_results();
    assert_eq!(tool_results.len(), 1);
    assert!(tool_results[0].1.contains("msg"));
}

#[tokio::test]
async fn write_tool_auto_approved_in_auto_edit_mode() {
    let tool = tool_with_risk("write", ToolRisk::FileWrite);
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "write".into(),
            arguments: serde_json::json!({"msg": "data"}),
        }]),
        MockModel::reply_text("done"),
    ]);
    let mut agent = Agent::new(model)
        .with_tool(tool)
        .with_permission_mode(PermissionMode::AutoEdit);
    let mut io = VecIo::with_inputs(Vec::<&str>::new());
    io.approval_decision = ApprovalDecision::Deny; // should never be asked

    let result = agent.step_stream("go", &mut io).await.unwrap();
    assert!(matches!(result, TurnResult::Message(_)));

    // No approval requests - auto-edit mode auto-approves file writes.
    assert!(io.approval_requests.is_empty());
    // Tool executed.
    assert_eq!(io.tool_results().len(), 1);
}

#[tokio::test]
async fn dangerous_tool_still_asks_in_auto_edit_mode() {
    let tool = tool_with_risk("bash", ToolRisk::Dangerous);
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"msg": "rm -rf /"}),
        }]),
        MockModel::reply_text("ok"),
    ]);
    let mut agent = Agent::new(model)
        .with_tool(tool)
        .with_permission_mode(PermissionMode::AutoEdit);
    let mut io = VecIo::with_inputs(Vec::<&str>::new());
    io.approval_decision = ApprovalDecision::Deny;

    let result = agent.step_stream("go", &mut io).await.unwrap();
    assert!(matches!(result, TurnResult::Message(_)));

    // Approval was requested - dangerous tools still ask in auto-edit.
    assert_eq!(io.approval_requests.len(), 1);
}

#[tokio::test]
async fn all_tools_auto_approved_in_full_auto_mode() {
    let write_tool = tool_with_risk("write", ToolRisk::FileWrite);
    let bash_tool = tool_with_risk("bash", ToolRisk::Dangerous);
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![
            ToolCall {
                id: "call_1".into(),
                name: "write".into(),
                arguments: serde_json::json!({"msg": "a"}),
            },
            ToolCall {
                id: "call_2".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"msg": "b"}),
            },
        ]),
        MockModel::reply_text("done"),
    ]);
    let mut agent = Agent::new(model)
        .with_tool(write_tool)
        .with_tool(bash_tool)
        .with_permission_mode(PermissionMode::FullAuto);
    let mut io = VecIo::with_inputs(Vec::<&str>::new());
    io.approval_decision = ApprovalDecision::Deny; // should never be asked

    let result = agent.step_stream("go", &mut io).await.unwrap();
    assert!(matches!(result, TurnResult::Message(_)));

    // No approval requests - full auto approves everything.
    assert!(io.approval_requests.is_empty());
    // Both tools executed.
    assert_eq!(io.tool_results().len(), 2);
}

#[tokio::test]
async fn allow_session_persists_across_calls() {
    // The same bash command twice: "Always" on the first call auto-approves
    // the second, since the (tool, scope) key matches.
    let tool = tool_with_risk("bash", ToolRisk::Dangerous);
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
        }]),
        MockModel::reply_tool_calls(vec![ToolCall {
            id: "call_2".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
        }]),
        MockModel::reply_text("done"),
    ]);
    let mut agent = Agent::new(model)
        .with_tool(tool)
        .with_permission_mode(PermissionMode::Normal);
    let mut io = VecIo::with_inputs(Vec::<&str>::new());

    // First call: approve for session. Second call: auto-approved.
    io.approval_decision = ApprovalDecision::AllowSession;

    let result = agent.step_stream("go", &mut io).await.unwrap();
    assert!(matches!(result, TurnResult::Message(_)));

    // Only the first call triggered an approval request.
    assert_eq!(io.approval_requests.len(), 1);
    assert_eq!(io.approval_requests[0].0.name, "bash");
    // Both tools executed.
    assert_eq!(io.tool_results().len(), 2);
}

#[tokio::test]
async fn allow_session_is_scoped_to_the_call() {
    // Approving "Always" for one bash command must not whitelist a different
    // command - the grant is keyed by (tool, scope), not the bare tool name.
    let tool = tool_with_risk("bash", ToolRisk::Dangerous);
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
        }]),
        MockModel::reply_tool_calls(vec![ToolCall {
            id: "call_2".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "rm -rf /"}),
        }]),
        MockModel::reply_text("done"),
    ]);
    let mut agent = Agent::new(model)
        .with_tool(tool)
        .with_permission_mode(PermissionMode::Normal);
    let mut io = VecIo::with_inputs(Vec::<&str>::new());
    io.approval_decision = ApprovalDecision::AllowSession;

    let result = agent.step_stream("go", &mut io).await.unwrap();
    assert!(matches!(result, TurnResult::Message(_)));

    // Both calls prompted - a different command is a different scope.
    assert_eq!(io.approval_requests.len(), 2);
    assert_eq!(io.tool_results().len(), 2);
}

#[tokio::test]
async fn permission_denied_returns_error_to_model() {
    let tool = tool_with_risk("bash", ToolRisk::Dangerous);
    let model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"msg": "rm"}),
        }]),
        // Model sees the error and recovers with text.
        MockModel::reply_text("I'll use a different approach"),
    ]);
    let mut agent = Agent::new(model)
        .with_tool(tool)
        .with_permission_mode(PermissionMode::Normal);
    let mut io = VecIo::with_inputs(Vec::<&str>::new());
    io.approval_decision = ApprovalDecision::Deny;

    let result = agent.step_stream("go", &mut io).await.unwrap();
    if let TurnResult::Message(msg) = &result {
        assert_eq!(msg.text_content(), "I'll use a different approach");
    } else {
        panic!("expected Message, got handoff");
    }

    // The session should have the denied tool result.
    let messages = agent.session().messages();
    let tool_msgs: Vec<_> = messages.iter().filter(|m| m.role == Role::Tool).collect();
    assert_eq!(tool_msgs.len(), 1);
    assert!(tool_msgs[0].text_content().contains("permission denied"));
}

#[tokio::test]
async fn default_permission_mode_is_normal() {
    let agent: Agent<MockModel> = Agent::new(MockModel::with_replies::<[_; 0], &str>([]));
    assert_eq!(agent.permission_mode(), PermissionMode::Normal);
}

#[tokio::test]
async fn with_permission_mode_builder_works() {
    let agent: Agent<MockModel> = Agent::new(MockModel::with_replies::<[_; 0], &str>([]))
        .with_permission_mode(PermissionMode::FullAuto);
    assert_eq!(agent.permission_mode(), PermissionMode::FullAuto);
}

#[tokio::test]
async fn set_permission_mode_in_place_works() {
    let agent: Agent<MockModel> = Agent::new(MockModel::with_replies::<[_; 0], &str>([]));
    assert_eq!(agent.permission_mode(), PermissionMode::Normal);
    agent.set_permission_mode(PermissionMode::AutoEdit);
    assert_eq!(agent.permission_mode(), PermissionMode::AutoEdit);
}
