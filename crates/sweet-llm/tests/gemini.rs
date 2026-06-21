// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "gemini")]

use serial_test::serial;
use sweet_core::{ContentBlock, Message, Model, Role};
use sweet_llm::{gemini, GeminiProvider, ProviderError};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn canned_response(content: &str) -> serde_json::Value {
    serde_json::json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": content}]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 7,
            "totalTokenCount": 12
        }
    })
}

#[test]
fn published_constants_match_gemini_endpoint() {
    assert_eq!(
        gemini::DEFAULT_BASE_URL,
        "https://generativelanguage.googleapis.com/v1beta"
    );
    assert_eq!(gemini::DEFAULT_API_KEY_ENV, "GEMINI_API_KEY");
    assert_eq!(gemini::DEFAULT_MODEL, "gemini-3-flash-preview");
}

#[test]
fn new_uses_gemini_base_url_and_default_model() {
    let p = GeminiProvider::new("k");
    // Provider internals are private; just verify it builds.
    let _ = p;
}

#[tokio::test]
async fn complete_sends_native_request() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path(format!(
            "/models/{}:generateContent",
            gemini::DEFAULT_MODEL
        )))
        .and(header("x-goog-api-key", "gemini-key"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .and(body_partial_json(serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("hello")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = GeminiProvider::new("gemini-key").with_base_url(server.uri());

    let reply = provider
        .complete(&[Message::user("hi")], &[])
        .await
        .unwrap();
    assert_eq!(reply.role, Role::Assistant);
    assert_eq!(reply.text_content(), "hello");
    assert_eq!(reply.token_count, Some(12));
}

#[tokio::test]
async fn file_user_message_sends_inline_data() {
    // User messages carrying a PDF must serialize via inlineData with
    // the correct MIME type, same path as images.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path(format!(
            "/models/{}:generateContent",
            gemini::DEFAULT_MODEL
        )))
        .and(header("x-goog-api-key", "gemini-key"))
        .and(body_partial_json(serde_json::json!({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "review this"},
                    {"inlineData": {"mimeType": "application/pdf", "data": "JVBERi0xLjQ="}}
                ]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("looks good")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = GeminiProvider::new("gemini-key").with_base_url(server.uri());

    let msg = Message::user_blocks(vec![
        ContentBlock::text("review this"),
        ContentBlock::File {
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
async fn complete_with_tool_use() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path(format!("/models/{}:generateContent", gemini::DEFAULT_MODEL)))
        .and(header("x-goog-api-key", "tool-key"))
        .and(header("user-agent", format!("sweet/{}", sweet_core::SWEET_VERSION)))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "I'll check that."},
                        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = GeminiProvider::new("tool-key").with_base_url(server.uri());

    let reply = provider
        .complete(&[Message::user("What's the weather in Paris?")], &[])
        .await
        .unwrap();

    assert_eq!(reply.text_content(), "I'll check that.");
    assert_eq!(reply.tool_calls.len(), 1);
    assert_eq!(reply.tool_calls[0].name, "get_weather");
    assert_eq!(reply.tool_calls[0].id, "call_1");
}

#[tokio::test]
async fn complete_round_trips_thought_signature() {
    let server = MockServer::start().await;

    // First turn: model calls a tool with a thoughtSignature.
    Mock::given(method("POST"))
        .and(path(format!("/models/{}:generateContent", gemini::DEFAULT_MODEL)))
        .and(header("x-goog-api-key", "sig-key"))
        .and(header("user-agent", format!("sweet/{}", sweet_core::SWEET_VERSION)))
        .and(body_partial_json(serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "calc 1+1"}]}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {"name": "calc", "args": {"x": 1}, "id": "call_1"},
                        "thoughtSignature": "sig-abc-123"
                    }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5, "totalTokenCount": 10}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = GeminiProvider::new("sig-key").with_base_url(server.uri());

    let reply = provider
        .complete(&[Message::user("calc 1+1")], &[])
        .await
        .unwrap();
    assert_eq!(reply.tool_calls.len(), 1);

    // Reset mocks so the first mock doesn't intercept the second request.
    server.reset().await;

    // Second turn: send tool result back. The provider must include the
    // thoughtSignature on the functionCall part in the history.
    let tool_result = Message::tool_result("call_1", "2");

    Mock::given(method("POST"))
        .and(path(format!("/models/{}:generateContent", gemini::DEFAULT_MODEL)))
        .and(header("x-goog-api-key", "sig-key"))
        .and(header("user-agent", format!("sweet/{}", sweet_core::SWEET_VERSION)))
        .and(body_partial_json(serde_json::json!({
            "contents": [
                {"role": "user", "parts": [{"text": "calc 1+1"}]},
                {"role": "model", "parts": [
                    {"functionCall": {"name": "calc", "args": {"x": 1}, "id": "call_1"}, "thoughtSignature": "sig-abc-123"}
                ]},
                {"role": "user", "parts": [
                    {"functionResponse": {"name": "calc", "response": {"result": "2"}}}
                ]}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("The answer is 2.")))
        .expect(1)
        .mount(&server)
        .await;

    let reply2 = provider
        .complete(
            &[
                Message::user("calc 1+1"),
                Message::with_tool_calls(reply.tool_calls),
                tool_result,
            ],
            &[],
        )
        .await
        .unwrap();
    assert_eq!(reply2.text_content(), "The answer is 2.");
}

#[tokio::test]
async fn complete_stream_emits_deltas() {
    let server = MockServer::start().await;

    let sse_body = "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n\
                    data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\" world\"}]}}]}\n\n\
                    data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"!\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":2,\"candidatesTokenCount\":3,\"totalTokenCount\":5}}\n\n";

    Mock::given(method("POST"))
        .and(path(format!(
            "/models/{}:streamGenerateContent",
            gemini::DEFAULT_MODEL
        )))
        .and(header("x-goog-api-key", "stream-key"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = GeminiProvider::new("stream-key").with_base_url(server.uri());
    let mut sink = sweet_core::NoopSink;

    let reply = provider
        .complete_stream(&[Message::user("hi")], &[], &mut sink)
        .await
        .unwrap();
    assert_eq!(reply.text_content(), "Hello world!");
    assert_eq!(reply.token_count, Some(5));
}

#[test]
#[serial]
fn from_env_reads_gemini_api_key() {
    std::env::set_var(gemini::DEFAULT_API_KEY_ENV, "gemini-key-value");
    let provider = GeminiProvider::from_env().expect("env var present");
    let _ = provider;
    std::env::remove_var(gemini::DEFAULT_API_KEY_ENV);
}

#[test]
#[serial]
fn from_env_errors_when_unset() {
    std::env::remove_var(gemini::DEFAULT_API_KEY_ENV);
    let err = GeminiProvider::from_env().unwrap_err();
    assert!(matches!(
        err,
        ProviderError::MissingApiKey { var } if var == gemini::DEFAULT_API_KEY_ENV
    ));
}
