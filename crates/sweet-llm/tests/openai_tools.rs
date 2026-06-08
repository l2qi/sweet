// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "openai")]

use sweet_core::{Message, Model, ToolError, ToolHandler, ToolSpec};
use sweet_llm::OpenAIProvider;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A minimal tool for wiremock testing.
struct EchoTool;

#[async_trait::async_trait]
impl ToolHandler for EchoTool {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        Ok(args.to_string())
    }
}

impl From<EchoTool> for ToolSpec {
    fn from(tool: EchoTool) -> Self {
        ToolSpec::new(
            "echo",
            "Echoes the input back.",
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

#[tokio::test]
async fn complete_serializes_tools_and_parses_tool_calls() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("user-agent", format!("sweet/{}", sweet_core::SWEET_VERSION)))
        .and(body_partial_json(serde_json::json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echoes the input back.",
                    "parameters": { "type": "object", "properties": { "msg": { "type": "string" } } }
                }
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"msg\":\"hello\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("test-key")
        .with_base_url(server.uri())
        .with_model("gpt-test");

    let tool = ToolSpec::from(EchoTool);
    let reply = provider
        .complete(&[Message::user("go")], &[tool])
        .await
        .expect("complete should succeed");

    assert_eq!(reply.role, sweet_core::Role::Assistant);
    assert_eq!(reply.tool_calls.len(), 1);
    assert_eq!(reply.tool_calls[0].id, "call_1");
    assert_eq!(reply.tool_calls[0].name, "echo");
    assert_eq!(
        reply.tool_calls[0].arguments,
        serde_json::json!({"msg": "hello"})
    );
}

#[tokio::test]
async fn complete_serializes_tool_result_messages() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .and(body_partial_json(serde_json::json!({
            "model": "gpt-test",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{\"msg\":\"hi\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "result"}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-test2",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop"
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("test-key")
        .with_base_url(server.uri())
        .with_model("gpt-test");

    let history = [
        Message::user("go"),
        Message::with_tool_calls(vec![sweet_core::ToolCall {
            id: "call_1".into(),
            name: "echo".into(),
            arguments: serde_json::json!({"msg": "hi"}),
        }]),
        Message::tool_result("call_1", "result"),
    ];

    let reply = provider
        .complete(&history, &[])
        .await
        .expect("complete should succeed");

    assert_eq!(reply, Message::assistant("done"));
}
