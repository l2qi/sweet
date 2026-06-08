// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use sweet_agent::test_util::MockModel;
use sweet_agent::{Agent, HandoffContext, HandoffHandler, HandoffResult, HandoffSpec, TurnResult};
use sweet_core::{Role, ToolCall, ToolError};

struct PlanHandoff;

#[sweet_agent::async_trait]
impl HandoffHandler for PlanHandoff {
    async fn invoke(
        &self,
        _args: serde_json::Value,
        _ctx: HandoffContext,
    ) -> Result<HandoffResult, ToolError> {
        Ok(HandoffResult::Transfer {
            target: "plan".to_string(),
            payload: Some("user wants to plan".to_string()),
        })
    }
}

fn handoff_spec() -> HandoffSpec {
    HandoffSpec::new(
        "transfer_to_plan",
        "Hand off to the plan agent.",
        serde_json::json!({"type": "object"}),
        PlanHandoff,
    )
}

#[tokio::test]
async fn handoff_returns_turn_result_handoff() {
    let model = MockModel::with_scripted([MockModel::reply_tool_calls(vec![ToolCall {
        id: "call_1".into(),
        name: "transfer_to_plan".into(),
        arguments: serde_json::json!({}),
    }])]);
    let mut agent = Agent::new(model).with_handoff(handoff_spec());

    let result = agent.step("plan this task").await.unwrap();
    match result {
        TurnResult::Handoff { target, payload } => {
            assert_eq!(target, "plan");
            assert_eq!(payload, Some("user wants to plan".to_string()));
        }
        TurnResult::Message(_) => panic!("expected handoff, got message"),
    }
}

#[tokio::test]
async fn handoff_appends_synthetic_tool_result_to_session() {
    let model = MockModel::with_scripted([MockModel::reply_tool_calls(vec![ToolCall {
        id: "call_1".into(),
        name: "transfer_to_plan".into(),
        arguments: serde_json::json!({}),
    }])]);
    let mut agent = Agent::new(model).with_handoff(handoff_spec());

    agent.step("plan this task").await.unwrap();

    let messages = agent.session().messages();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, Role::User);
    assert_eq!(messages[1].role, Role::Assistant);
    assert!(messages[1]
        .tool_calls
        .iter()
        .any(|c| c.name == "transfer_to_plan"));
    assert_eq!(messages[2].role, Role::Tool);
    assert!(messages[2].text_content().contains("Handoff to plan"));
}

#[tokio::test]
async fn take_session_moves_history_out() {
    let model = MockModel::with_replies(["hello"]);
    let mut agent = Agent::new(model);
    agent.step("hi").await.unwrap();

    assert_eq!(agent.session().messages().len(), 2);

    let session = agent.take_session();
    assert_eq!(session.messages().len(), 2);
    assert!(agent.session().messages().is_empty());
}
