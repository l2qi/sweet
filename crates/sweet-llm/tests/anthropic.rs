// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "anthropic")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serial_test::serial;
use sweet_core::message::ToolCall;
use sweet_core::stream::StreamSink;
use sweet_core::{Message, Model, ToolError, ToolHandler, ToolSpec};
use sweet_llm::{anthropic, AnthropicProvider, ProviderError, ReasoningConfig};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Minimal tool for wiremock testing.
struct EchoTool;

#[async_trait::async_trait]
impl ToolHandler for EchoTool {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        Ok(args.to_string())
    }
}

fn canned_response_text(content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "msg_01Test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    })
}

fn canned_response_with_tool() -> serde_json::Value {
    serde_json::json!({
        "id": "msg_01Test",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me search"},
            {"type": "tool_use", "id": "tu_01", "name": "web_search", "input": {"query": "foo"}}
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 15, "output_tokens": 20}
    })
}

#[tokio::test]
async fn complete_posts_correct_request_and_returns_assistant_message() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .and(body_partial_json(serde_json::json!({
            "model": "claude-test",
            "max_tokens": 4096,
            "system": "be terse",
            "messages": [
                {"role": "user", "content": "hello"},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response_text("hi there")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new("test-key")
        .with_base_url(server.uri())
        .with_model("claude-test")
        // Assert the base request shape; caching is covered separately.
        .with_prompt_caching(false);

    let reply = provider
        .complete(&[Message::system("be terse"), Message::user("hello")], &[])
        .await
        .expect("complete should succeed");

    assert_eq!(reply.role, sweet_core::Role::Assistant);
    assert_eq!(reply.text_content(), "hi there");
    assert!(reply.tool_calls.is_empty());
    assert_eq!(reply.token_count, Some(15));
}

#[tokio::test]
async fn complete_with_tools_sends_correct_tool_spec_and_parses_tool_use() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "k"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .and(body_partial_json(serde_json::json!({
            "tools": [{
                "name": "echo",
                "description": "echoes",
                "input_schema": {"type": "object"}
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response_with_tool()))
        .expect(1)
        .mount(&server)
        .await;

    let tool = ToolSpec::new(
        "echo",
        "echoes",
        serde_json::json!({"type": "object"}),
        EchoTool,
    );

    let provider = AnthropicProvider::new("k").with_base_url(server.uri());
    let reply = provider
        .complete(&[Message::user("search")], &[tool])
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "Let me search");
    assert_eq!(reply.tool_calls.len(), 1);
    assert_eq!(reply.tool_calls[0].id, "tu_01");
    assert_eq!(reply.tool_calls[0].name, "web_search");
    assert_eq!(
        reply.tool_calls[0].arguments,
        serde_json::json!({"query": "foo"})
    );
    assert_eq!(reply.token_count, Some(35));
}

#[tokio::test]
async fn complete_groups_consecutive_tool_results() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("user-agent", format!("sweet/{}", sweet_core::SWEET_VERSION)))
        .and(body_partial_json(serde_json::json!({
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [{"type":"tool_use","id":"tu_a","name":"a","input":{}}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_a", "content": "result-a"},
                        {"type": "tool_result", "tool_use_id": "tu_b", "content": "result-b"}
                    ]
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response_text("done")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new("k").with_base_url(server.uri());
    let reply = provider
        .complete(
            &[
                Message::user("go"),
                Message::with_tool_calls(vec![ToolCall {
                    id: "tu_a".into(),
                    name: "a".into(),
                    arguments: serde_json::json!({}),
                }]),
                Message::tool_result("tu_a", "result-a"),
                Message::tool_result("tu_b", "result-b"),
            ],
            &[],
        )
        .await
        .unwrap();

    assert_eq!(reply.role, sweet_core::Role::Assistant);
    assert_eq!(reply.text_content(), "done");
    assert!(reply.tool_calls.is_empty());
}

#[tokio::test]
async fn file_user_message_sends_document_block() {
    // User messages carrying a PDF must serialize as a `document` content
    // block with base64 source, matching Anthropic's document API.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "k"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .and(body_partial_json(serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "review this"},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "JVBERi0xLjQ="
                        }
                    }
                ]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response_text("looks good")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new("k").with_base_url(server.uri());

    let msg = Message::user_blocks(vec![
        sweet_core::ContentBlock::text("review this"),
        sweet_core::ContentBlock::File {
            data: b"%PDF-1.4".to_vec(),
            media_type: "application/pdf".to_string(),
            filename: "report.pdf".to_string(),
        },
    ]);

    let reply = provider
        .complete(&[msg], &[])
        .await
        .expect("complete should succeed");

    assert_eq!(reply.text_content(), "looks good");
}

#[tokio::test]
async fn non_2xx_response_yields_provider_error_with_body() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid key"))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new("nope").with_base_url(server.uri());

    let err = provider
        .complete(&[Message::user("x")], &[])
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("provider error"), "got: {msg}");

    let inner = std::error::Error::source(&err).expect("source");
    let provider_err = inner.downcast_ref::<ProviderError>().expect("downcast");
    match provider_err {
        ProviderError::Http { status, body } => {
            assert_eq!(status.as_u16(), 401);
            assert_eq!(body, "invalid key");
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[tokio::test]
async fn complete_stream_emits_content_deltas_and_captures_usage() {
    let server = MockServer::start().await;

    let body = sse_body(&[
        r#"{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-test","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", "}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world!"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}"#,
        r#"{"type":"message_stop"}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "k"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .and(body_partial_json(serde_json::json!({"stream": true})))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new("k").with_base_url(server.uri());
    let mut sink = CapturingSink::new();
    let reply = provider
        .complete_stream(&[Message::user("hi")], &[], &mut sink)
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "Hello, world!");
    assert_eq!(reply.token_count, Some(15));
    assert!(reply.tool_calls.is_empty());
    assert_eq!(sink.deltas(), vec!["Hello", ", ", "world!"]);
}

#[tokio::test]
async fn complete_stream_assembles_tool_use_from_json_deltas() {
    let server = MockServer::start().await;

    let body = sse_body(&[
        r#"{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-test","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_01","name":"echo","input":{}}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"msg\":"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"hi\""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"}"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":15}}"#,
        r#"{"type":"message_stop"}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new("k").with_base_url(server.uri());
    let mut sink = CapturingSink::new();
    let reply = provider
        .complete_stream(&[Message::user("hi")], &[], &mut sink)
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "");
    assert_eq!(reply.tool_calls.len(), 1);
    assert_eq!(reply.tool_calls[0].id, "tu_01");
    assert_eq!(reply.tool_calls[0].name, "echo");
    assert_eq!(
        reply.tool_calls[0].arguments,
        serde_json::json!({"msg": "hi"})
    );
    assert!(sink.deltas().is_empty());
    assert_eq!(sink.tool_calls().len(), 1);
}

#[test]
fn defaults_match_published_constants() {
    let provider = AnthropicProvider::new("k");
    assert_eq!(provider.base_url(), anthropic::DEFAULT_BASE_URL);
    assert_eq!(provider.model_name(), anthropic::DEFAULT_MODEL);
}

#[test]
#[serial]
fn from_env_reads_anthropic_api_key() {
    std::env::set_var(anthropic::DEFAULT_API_KEY_ENV, "from-env-key");
    let provider = AnthropicProvider::from_env().expect("env var present");
    assert_eq!(provider.base_url(), anthropic::DEFAULT_BASE_URL);
    assert_eq!(provider.model_name(), anthropic::DEFAULT_MODEL);
    assert_eq!(provider.max_tokens(), anthropic::DEFAULT_MAX_TOKENS);
    std::env::remove_var(anthropic::DEFAULT_API_KEY_ENV);
}

#[test]
#[serial]
fn from_env_errors_when_unset() {
    std::env::remove_var(anthropic::DEFAULT_API_KEY_ENV);
    let err = AnthropicProvider::from_env().unwrap_err();
    let inner = std::error::Error::source(&err).unwrap();
    let provider_err = inner.downcast_ref::<ProviderError>().unwrap();
    assert!(matches!(
        provider_err,
        ProviderError::MissingApiKey { var } if *var == anthropic::DEFAULT_API_KEY_ENV
    ));
}

// ---------------------------------------------------------------------------
// Thinking tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn complete_with_thinking_enabled_sends_thinking_field_and_captures_thinking_blocks() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("user-agent", format!("sweet/{}", sweet_core::SWEET_VERSION)))
        .and(body_partial_json(serde_json::json!({
            "thinking": {"type": "enabled", "budget_tokens": 10000}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_thinking",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me reason about this...", "signature": "sig_abc123"},
                {"type": "text", "text": "The answer is 42."}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 30}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new("k")
        .with_base_url(server.uri())
        // max_tokens is the output ceiling the budget is carved from; set it
        // above the budget so the `budget_tokens < max_tokens` clamp keeps 10000.
        .with_max_tokens(64_000)
        .with_reasoning(ReasoningConfig::Budget(10000));

    let reply = provider
        .complete(&[Message::user("what is 6*7?")], &[])
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "The answer is 42.");
    assert_eq!(reply.thinking_content.len(), 1);
    assert_eq!(
        reply.thinking_content[0].text,
        "Let me reason about this..."
    );
    assert_eq!(
        reply.thinking_content[0].signature.as_deref(),
        Some("sig_abc123")
    );
    assert_eq!(reply.token_count, Some(50));
}

#[tokio::test]
async fn complete_with_adaptive_thinking_sends_adaptive_field() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(serde_json::json!({
            "thinking": {"type": "adaptive"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_adaptive",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig_xyz"},
                {"type": "text", "text": "done"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 5}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new("k")
        .with_base_url(server.uri())
        // Adaptive thinking requires a 4.6+ model; the toggle dialect maps to
        // `{type: adaptive}` only there (older models get an explicit budget).
        .with_model("claude-opus-4-8")
        .with_reasoning(ReasoningConfig::Toggle(true));

    let reply = provider
        .complete(&[Message::user("hi")], &[])
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "done");
    assert_eq!(reply.thinking_content.len(), 1);
    assert_eq!(reply.thinking_content[0].text, "hmm");
    assert_eq!(
        reply.thinking_content[0].signature.as_deref(),
        Some("sig_xyz")
    );
}

#[tokio::test]
async fn complete_stream_with_thinking_emits_thinking_deltas_and_captures_thinking() {
    let server = MockServer::start().await;

    let body = sse_body(&[
        r#"{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-test","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"I need to "}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think hard."}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_789"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer: 42"}}"#,
        r#"{"type":"content_block_stop","index":1}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":10}}"#,
        r#"{"type":"message_stop"}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(serde_json::json!({"stream": true})))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new("k")
        .with_base_url(server.uri())
        .with_reasoning(ReasoningConfig::Budget(5000));

    let mut sink = CapturingSink::new();
    let reply = provider
        .complete_stream(&[Message::user("think")], &[], &mut sink)
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "Answer: 42");
    assert_eq!(reply.thinking_content.len(), 1);
    assert_eq!(reply.thinking_content[0].text, "I need to think hard.");
    assert_eq!(
        reply.thinking_content[0].signature.as_deref(),
        Some("sig_789")
    );
    assert_eq!(sink.thinking_deltas(), vec!["I need to ", "think hard."]);
    assert_eq!(sink.deltas(), vec!["Answer: 42"]);
}

/// Multi-turn thinking: signatures ride along on `Message.thinking_content`,
/// so a previous turn's thinking block is re-emitted verbatim when the
/// caller passes the prior assistant `Message` back into `complete`.
#[tokio::test]
async fn complete_round_trips_thinking_blocks_in_multi_turn() {
    let server = MockServer::start().await;

    // Each mock matches one call (FIFO) - assertion happens after the fact
    // against `received_requests`, so mock routing stays independent of body
    // shape.
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "pondering", "signature": "sig_old"},
                {"type": "text", "text": "first reply"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 10}
        })))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "more thinking", "signature": "sig_new"},
                {"type": "text", "text": "second reply"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 30, "output_tokens": 15}
        })))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new("k")
        .with_base_url(server.uri())
        .with_reasoning(ReasoningConfig::Budget(10000))
        // Multi-turn shape assertions; caching is covered separately.
        .with_prompt_caching(false);

    let first_reply = provider
        .complete(&[Message::user("first")], &[])
        .await
        .unwrap();
    assert_eq!(first_reply.text_content(), "first reply");
    assert_eq!(first_reply.thinking_content.len(), 1);
    assert_eq!(first_reply.thinking_content[0].text, "pondering");
    assert_eq!(
        first_reply.thinking_content[0].signature.as_deref(),
        Some("sig_old")
    );

    let second_reply = provider
        .complete(
            &[Message::user("first"), first_reply, Message::user("second")],
            &[],
        )
        .await
        .unwrap();

    assert_eq!(second_reply.text_content(), "second reply");
    assert_eq!(second_reply.thinking_content.len(), 1);
    assert_eq!(second_reply.thinking_content[0].text, "more thinking");
    assert_eq!(
        second_reply.thinking_content[0].signature.as_deref(),
        Some("sig_new")
    );

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should record requests");
    assert_eq!(requests.len(), 2);

    let ua = format!("sweet/{}", sweet_core::SWEET_VERSION);
    for req in &requests {
        let ua_header = req
            .headers
            .get("user-agent")
            .expect("user-agent header present");
        assert_eq!(ua_header.to_str().unwrap(), ua);
    }

    let first_body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(
        first_body["messages"],
        serde_json::json!([{"role": "user", "content": "first"}])
    );

    let second_body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        second_body["messages"],
        serde_json::json!([
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "pondering", "signature": "sig_old"},
                    {"type": "text", "text": "first reply"}
                ]
            },
            {"role": "user", "content": "second"}
        ])
    );
}

#[test]
fn with_reasoning_configures_provider() {
    let _ = AnthropicProvider::new("k").with_reasoning(ReasoningConfig::Budget(8000));
    let _ = AnthropicProvider::new("k").with_reasoning(ReasoningConfig::Toggle(true));
    let _ = AnthropicProvider::new("k").with_reasoning(ReasoningConfig::Toggle(false));
    let _ = AnthropicProvider::new("k").with_reasoning(ReasoningConfig::Effort("high".to_string()));
}

#[tokio::test]
async fn prompt_caching_on_by_default_sets_cache_control() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(serde_json::json!({
            "system": [
                {"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ]}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "m",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-test",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 5}
        })))
        .expect(1)
        .mount(&server)
        .await;

    // Caching is on by default - no `with_prompt_caching` call.
    let provider = AnthropicProvider::new("k").with_base_url(server.uri());
    provider
        .complete(&[Message::system("be terse"), Message::user("hi")], &[])
        .await
        .unwrap();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Default)]
struct CapturingSink {
    deltas: Arc<Mutex<Vec<String>>>,
    tool_calls: Arc<Mutex<Vec<ToolCall>>>,
    thinking_deltas: Arc<Mutex<Vec<String>>>,
}

impl CapturingSink {
    fn new() -> Self {
        Self::default()
    }

    fn deltas(&self) -> Vec<String> {
        self.deltas.lock().unwrap().clone()
    }

    fn tool_calls(&self) -> Vec<ToolCall> {
        self.tool_calls.lock().unwrap().clone()
    }

    fn thinking_deltas(&self) -> Vec<String> {
        self.thinking_deltas.lock().unwrap().clone()
    }
}

#[async_trait]
impl StreamSink for CapturingSink {
    async fn on_content_delta(&mut self, delta: &str) -> sweet_core::Result<()> {
        self.deltas.lock().unwrap().push(delta.to_string());
        Ok(())
    }

    async fn on_thinking_delta(&mut self, delta: &str) -> sweet_core::Result<()> {
        self.thinking_deltas.lock().unwrap().push(delta.to_string());
        Ok(())
    }

    async fn on_tool_call(&mut self, call: &ToolCall) -> sweet_core::Result<()> {
        self.tool_calls.lock().unwrap().push(call.clone());
        Ok(())
    }
}

fn sse_body(events: &[&str]) -> String {
    let mut s = String::new();
    for e in events {
        s.push_str("data: ");
        s.push_str(e);
        s.push_str("\n\n");
    }
    s
}
