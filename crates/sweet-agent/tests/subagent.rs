// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the subagent primitive.

use std::sync::Arc;

use sweet_agent::test_util::{MockModel, MockReply};
use sweet_agent::{Agent, SubagentContext, SubagentHandler, SubagentSpec, TurnResult};
use sweet_core::{Model, ToolCall, ToolError};

/// Handler that ignores its args, builds a fresh child `Agent<MockModel>`
/// scripted with the supplied replies, runs one step, and returns the
/// content of the final assistant message.
struct StaticReplyHandler {
    replies: Vec<MockReply>,
    /// If set, also spawn this inner subagent on the child agent (to exercise
    /// nesting). The inner spec is consumed on first invocation.
    inner: std::sync::Mutex<Option<SubagentSpec>>,
}

#[sweet_agent::async_trait]
impl SubagentHandler for StaticReplyHandler {
    async fn invoke(
        &self,
        _args: serde_json::Value,
        _ctx: SubagentContext,
    ) -> Result<String, ToolError> {
        let model = MockModel::with_scripted(self.replies.clone());
        let mut agent = Agent::new(model);
        if let Some(inner) = self.inner.lock().unwrap().take() {
            agent = agent.with_subagent(inner);
        }
        let msg = agent
            .step("go")
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;
        match msg {
            TurnResult::Message(m) => Ok(m.text_content()),
            TurnResult::Handoff { .. } => Err(ToolError::Execution("unexpected handoff".into())),
        }
    }
}

fn echo_subagent_spec() -> SubagentSpec {
    SubagentSpec::new(
        "echo_sub",
        "Returns a canned answer from a mock child agent.",
        serde_json::json!({
            "type": "object",
            "properties": { "task": { "type": "string" } },
            "required": ["task"]
        }),
        StaticReplyHandler {
            replies: vec![MockReply::Text("hello from child".to_string())],
            inner: std::sync::Mutex::new(None),
        },
    )
}

fn tool_call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments: args,
    }
}

#[tokio::test]
async fn parent_invokes_subagent_and_sees_result_as_tool_output() {
    let parent_model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![tool_call(
            "call_1",
            "echo_sub",
            serde_json::json!({"task": "say hi"}),
        )]),
        MockModel::reply_text("got it"),
    ]);
    let mut agent = Agent::new(parent_model).with_subagent(echo_subagent_spec());

    let reply = agent.step("kick off").await.unwrap();
    let content = match reply {
        TurnResult::Message(m) => m.text_content(),
        TurnResult::Handoff { .. } => panic!("unexpected handoff"),
    };
    assert_eq!(content, "got it");

    // The tool result message in the transcript should be the subagent's
    // final assistant content.
    let messages = agent.session().messages();
    let tool_result = messages
        .iter()
        .find(|m| m.tool_call_id.as_deref() == Some("call_1"))
        .expect("tool result message present");
    assert_eq!(tool_result.text_content(), "hello from child");
}

#[tokio::test]
async fn depth_exceeded_errors_without_crashing_loop() {
    // Outer spec, max depth 1: its handler tries to spawn an inner subagent
    // (would be depth 2). That nested invocation must error, the parent's
    // loop must keep running and reach the second model reply.
    let inner_spec = SubagentSpec::new(
        "inner_sub",
        "Inner subagent that should never actually run.",
        serde_json::json!({"type": "object"}),
        StaticReplyHandler {
            replies: vec![MockReply::Text("inner ran".to_string())],
            inner: std::sync::Mutex::new(None),
        },
    );

    // Child agent scripted to immediately call the inner subagent, then
    // observe its (error) result and produce a final message.
    let child_replies = vec![
        MockModel::reply_tool_calls(vec![tool_call(
            "inner_call",
            "inner_sub",
            serde_json::json!({}),
        )]),
        MockModel::reply_text("outer-final".to_string()),
    ];

    let outer_spec = SubagentSpec::new(
        "outer_sub",
        "Outer subagent that tries to nest.",
        serde_json::json!({"type": "object"}),
        StaticReplyHandler {
            replies: child_replies,
            inner: std::sync::Mutex::new(Some(inner_spec)),
        },
    )
    .with_max_nested_depth(1);

    let parent_model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![tool_call(
            "call_1",
            "outer_sub",
            serde_json::json!({}),
        )]),
        MockModel::reply_text("parent-final".to_string()),
    ]);

    let mut agent = Agent::new(parent_model).with_subagent(outer_spec);
    let reply = agent.step("go").await.unwrap();
    let content = match reply {
        TurnResult::Message(m) => m.text_content(),
        TurnResult::Handoff { .. } => panic!("unexpected handoff"),
    };
    assert_eq!(content, "parent-final");

    // The outer subagent's child loop should have seen the inner call return
    // an error string (depth-exceeded message), and recovered to emit the
    // outer-final reply, which the parent saw as the tool result.
    let messages = agent.session().messages();
    let tool_result = messages
        .iter()
        .find(|m| m.tool_call_id.as_deref() == Some("call_1"))
        .expect("tool result message present");
    assert_eq!(tool_result.text_content(), "outer-final");
}

/// Handler that asserts `ctx.parent_model.is_some()` and uses the parent's
/// model to run the child agent. Records the pointer value (as `usize`) it
/// received so the test can assert it matches the parent's Arc.
struct InheritingHandler {
    seen_parent_ptr: Arc<std::sync::Mutex<Option<usize>>>,
}

#[sweet_agent::async_trait]
impl SubagentHandler for InheritingHandler {
    async fn invoke(
        &self,
        _args: serde_json::Value,
        ctx: SubagentContext,
    ) -> Result<String, ToolError> {
        let parent = ctx
            .parent_model
            .expect("parent_model must be Some when parent uses new_shared");
        *self.seen_parent_ptr.lock().unwrap() = Some(Arc::as_ptr(&parent) as *const () as usize);
        // Use the parent's model for the child agent.
        let mut agent = Agent::new(parent);
        let msg = agent
            .step("child go")
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;
        match msg {
            TurnResult::Message(m) => Ok(m.text_content()),
            TurnResult::Handoff { .. } => Err(ToolError::Execution("unexpected handoff".into())),
        }
    }
}

#[tokio::test]
async fn parent_model_is_handed_to_subagent_when_using_new_shared() {
    // Parent model is scripted with three replies: one tool call to the
    // subagent, then the subagent's child uses the same model (one reply),
    // then the parent reads the result and gives a final answer.
    let parent_model: Arc<dyn Model> = Arc::new(MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![tool_call(
            "call_1",
            "inheritor",
            serde_json::json!({}),
        )]),
        MockModel::reply_text("child reply".to_string()),
        MockModel::reply_text("parent done".to_string()),
    ]));
    let expected_ptr = Arc::as_ptr(&parent_model) as *const () as usize;

    let seen = Arc::new(std::sync::Mutex::new(None));
    let spec = SubagentSpec::new(
        "inheritor",
        "Subagent that inherits the parent's model.",
        serde_json::json!({"type": "object"}),
        InheritingHandler {
            seen_parent_ptr: seen.clone(),
        },
    );

    let mut agent = Agent::new_shared(parent_model).with_subagent(spec);
    let reply = agent.step("go").await.unwrap();
    let content = match reply {
        TurnResult::Message(m) => m.text_content(),
        TurnResult::Handoff { .. } => panic!("unexpected handoff"),
    };
    assert_eq!(content, "parent done");

    let seen_ptr = seen.lock().unwrap().expect("handler ran and saw a parent");
    assert_eq!(seen_ptr, expected_ptr, "subagent received the parent's Arc");
}

#[tokio::test]
async fn parent_model_is_none_when_parent_was_built_with_plain_new() {
    let parent_model = MockModel::with_scripted([
        MockModel::reply_tool_calls(vec![tool_call(
            "call_1",
            "inspector",
            serde_json::json!({}),
        )]),
        MockModel::reply_text("ok".to_string()),
    ]);
    let saw_none = Arc::new(std::sync::Mutex::new(false));

    struct InspectorHandler {
        saw_none: Arc<std::sync::Mutex<bool>>,
    }
    #[sweet_agent::async_trait]
    impl SubagentHandler for InspectorHandler {
        async fn invoke(
            &self,
            _args: serde_json::Value,
            ctx: SubagentContext,
        ) -> Result<String, ToolError> {
            *self.saw_none.lock().unwrap() = ctx.parent_model.is_none();
            assert_eq!(ctx.depth, 1);
            Ok("done".to_string())
        }
    }

    let spec = SubagentSpec::new(
        "inspector",
        "Checks ctx fields.",
        serde_json::json!({"type": "object"}),
        InspectorHandler {
            saw_none: saw_none.clone(),
        },
    );

    let mut agent = Agent::new(parent_model).with_subagent(spec);
    agent.step("go").await.unwrap();
    assert!(
        *saw_none.lock().unwrap(),
        "ctx.parent_model was None as expected"
    );
}
