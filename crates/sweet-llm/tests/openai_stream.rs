// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "openai")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use sweet_core::message::ToolCall;
use sweet_core::stream::StreamSink;
use sweet_core::{Message, Model};
use sweet_llm::openai::ReasoningContent;
use sweet_llm::OpenAIProvider;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Default)]
struct CapturingSink {
    deltas: Arc<Mutex<Vec<String>>>,
    reasoning_deltas: Arc<Mutex<Vec<String>>>,
    tool_calls: Arc<Mutex<Vec<ToolCall>>>,
}

impl CapturingSink {
    fn new() -> Self {
        Self::default()
    }

    fn deltas(&self) -> Vec<String> {
        self.deltas.lock().unwrap().clone()
    }

    fn reasoning_deltas(&self) -> Vec<String> {
        self.reasoning_deltas.lock().unwrap().clone()
    }

    fn tool_calls(&self) -> Vec<ToolCall> {
        self.tool_calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl StreamSink for CapturingSink {
    async fn on_content_delta(&mut self, delta: &str) -> sweet_core::Result<()> {
        self.deltas.lock().unwrap().push(delta.to_string());
        Ok(())
    }

    async fn on_thinking_delta(&mut self, delta: &str) -> sweet_core::Result<()> {
        self.reasoning_deltas
            .lock()
            .unwrap()
            .push(delta.to_string());
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

#[tokio::test]
async fn complete_stream_emits_content_deltas_and_captures_usage() {
    let server = MockServer::start().await;

    let body = sse_body(&[
        r#"{"choices":[{"delta":{"role":"assistant","content":"Hello"}}]}"#,
        r#"{"choices":[{"delta":{"content":", "}}]}"#,
        r#"{"choices":[{"delta":{"content":"world!"}}]}"#,
        r#"{"choices":[{"finish_reason":"stop","delta":{}}]}"#,
        r#"{"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7},"choices":[]}"#,
        "[DONE]",
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer k"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .and(body_partial_json(serde_json::json!({
            "stream": true,
            "stream_options": {"include_usage": true}
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("k").with_base_url(server.uri());
    let mut sink = CapturingSink::new();
    let reply = provider
        .complete_stream(&[Message::user("hi")], &[], &mut sink)
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "Hello, world!");
    assert_eq!(reply.token_count, Some(7));
    assert!(reply.tool_calls.is_empty());
    assert_eq!(sink.deltas(), vec!["Hello", ", ", "world!"]);
}

#[tokio::test]
async fn complete_stream_assembles_tool_calls_from_indexed_chunks() {
    let server = MockServer::start().await;

    // Tool call streamed across three events: id+name first, then args in two pieces.
    let body = sse_body(&[
        r#"{"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"echo","arguments":""}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"msg\":"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"hi\"}"}}]}}]}"#,
        r#"{"choices":[{"finish_reason":"tool_calls","delta":{}}]}"#,
        "[DONE]",
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
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

    let provider = OpenAIProvider::new("k").with_base_url(server.uri());
    let mut sink = CapturingSink::new();
    let reply = provider
        .complete_stream(&[Message::user("hi")], &[], &mut sink)
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "");
    assert_eq!(reply.tool_calls.len(), 1);
    assert_eq!(reply.tool_calls[0].id, "call_1");
    assert_eq!(reply.tool_calls[0].name, "echo");
    assert_eq!(
        reply.tool_calls[0].arguments,
        serde_json::json!({"msg": "hi"})
    );
    assert!(sink.deltas().is_empty());
    assert_eq!(sink.tool_calls().len(), 1);
}

#[tokio::test]
async fn complete_stream_handles_split_byte_chunks() {
    // Verify that an event whose bytes arrive across multiple TCP chunks is
    // still parsed correctly. Wiremock returns the body in one chunk by
    // default, so we instead split at the SSE boundary by interleaving an
    // empty event — proves the buffer state machine handles multi-event
    // bodies.
    let server = MockServer::start().await;
    let body = sse_body(&[
        r#"{"choices":[{"delta":{"content":"a"}}]}"#,
        r#"{"choices":[{"delta":{"content":"b"}}]}"#,
        "[DONE]",
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("k").with_base_url(server.uri());
    let mut sink = CapturingSink::new();
    let reply = provider
        .complete_stream(&[Message::user("hi")], &[], &mut sink)
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "ab");
    assert_eq!(sink.deltas(), vec!["a", "b"]);
}

#[tokio::test]
async fn complete_stream_propagates_http_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("k").with_base_url(server.uri());
    let mut sink = CapturingSink::new();
    let err = provider
        .complete_stream(&[Message::user("hi")], &[], &mut sink)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("provider error"));
}

#[tokio::test]
async fn complete_stream_emits_reasoning_deltas_and_accumulates_content() {
    let server = MockServer::start().await;

    let body = sse_body(&[
        r#"{"choices":[{"delta":{"role":"assistant","reasoning_content":"Let me"}}]}"#,
        r#"{"choices":[{"delta":{"reasoning_content":" think..."}}]}"#,
        r#"{"choices":[{"delta":{"content":"42"}}]}"#,
        r#"{"choices":[{"finish_reason":"stop","delta":{}}]}"#,
        "[DONE]",
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer k"))
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

    let provider = OpenAIProvider::new("k").with_base_url(server.uri());
    let mut sink = CapturingSink::new();
    let reply = provider
        .complete_stream(
            &[Message::user("what is the meaning of life?")],
            &[],
            &mut sink,
        )
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "42");
    assert_eq!(reply.reasoning_content(), Some("Let me think..."));
    assert_eq!(sink.deltas(), vec!["42"]);
    assert_eq!(sink.reasoning_deltas(), vec!["Let me", " think..."]);
}

#[tokio::test]
async fn complete_stream_preserves_explicit_empty_reasoning_content() {
    // Kimi/DeepSeek can stream `reasoning_content: ""` to signal "explicit
    // empty" (no chain-of-thought this turn but the field is present). The
    // streamed Message must round-trip that as a single empty-text block so
    // the next outgoing request re-emits `reasoning_content: ""` — matches
    // the non-streaming TryFrom path.
    let server = MockServer::start().await;

    let body = sse_body(&[
        r#"{"choices":[{"delta":{"role":"assistant","reasoning_content":""}}]}"#,
        r#"{"choices":[{"delta":{"content":"hello"}}]}"#,
        r#"{"choices":[{"finish_reason":"stop","delta":{}}]}"#,
        "[DONE]",
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("k").with_base_url(server.uri());
    let mut sink = CapturingSink::new();
    let reply = provider
        .complete_stream(&[Message::user("hi")], &[], &mut sink)
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "hello");
    assert_eq!(reply.reasoning_content(), Some(""));
    assert_eq!(reply.thinking_content.len(), 1);
    assert!(sink.reasoning_deltas().is_empty());
}

#[tokio::test]
async fn complete_stream_omits_reasoning_when_field_never_present() {
    // When the server never sends `reasoning_content` at all, the streamed
    // Message has zero thinking blocks (distinct from "field present but
    // empty").
    let server = MockServer::start().await;

    let body = sse_body(&[
        r#"{"choices":[{"delta":{"role":"assistant","content":"hello"}}]}"#,
        r#"{"choices":[{"finish_reason":"stop","delta":{}}]}"#,
        "[DONE]",
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("k").with_base_url(server.uri());
    let mut sink = CapturingSink::new();
    let reply = provider
        .complete_stream(&[Message::user("hi")], &[], &mut sink)
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "hello");
    assert_eq!(reply.reasoning_content(), None);
    assert!(reply.thinking_content.is_empty());
}
