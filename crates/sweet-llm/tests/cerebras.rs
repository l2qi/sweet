// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "cerebras")]

//! Hermetic wiremock integration tests for [`CerebrasProvider`].
//!
//! These exercise the actual HTTP request/response cycle: they assert that the
//! request body carries `reasoning_effort` (and never the `thinking` object
//! Cerebras 400s on), and that prior assistant reasoning is omitted entirely -
//! Cerebras accepts neither `reasoning_content` nor `reasoning` on a request and
//! 400s `wrong_api_format` if either is sent. The provider's internal state is
//! covered by unit tests in `cerebras.rs`.

use sweet_core::{Message, Model};
use sweet_llm::openai::ReasoningContent;
use sweet_llm::{CerebrasProvider, ReasoningConfig};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn canned_response(content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    })
}

#[tokio::test]
async fn effort_sends_reasoning_effort_without_thinking() {
    // The defining Cerebras contract: `reasoning_effort` is sent, the
    // `thinking` object is not (Cerebras rejects it with HTTP 400).
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(serde_json::json!({
            "reasoning_effort": "high"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("ok")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = CerebrasProvider::new("test-key")
        .with_base_url(server.uri())
        .with_model("gpt-oss-120b")
        .with_reasoning(ReasoningConfig::Effort("high".to_string()));

    provider
        .complete(&[Message::user("hi")], &[])
        .await
        .expect("complete should succeed");
}

#[tokio::test]
async fn request_body_never_contains_thinking_object() {
    // Assert the `thinking` key is entirely absent from the body — even with
    // reasoning configured — by inspecting the recorded request.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("ok")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = CerebrasProvider::new("test-key")
        .with_base_url(server.uri())
        .with_model("gpt-oss-120b")
        .with_reasoning(ReasoningConfig::Effort("high".to_string()));

    provider
        .complete(&[Message::user("hi")], &[])
        .await
        .expect("complete should succeed");

    let body = server
        .received_requests()
        .await
        .expect("request was recorded")
        .into_iter()
        .find(|r| r.url.path() == "/chat/completions")
        .expect("completions request")
        .body;
    let json: serde_json::Value = serde_json::from_slice(&body).expect("body is JSON");
    assert!(
        json.get("thinking").is_none(),
        "Cerebras must never send a `thinking` object; got {json}"
    );
}

#[tokio::test]
async fn toggle_off_maps_to_effort_none() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(serde_json::json!({
            "reasoning_effort": "none"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("ok")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = CerebrasProvider::new("test-key")
        .with_base_url(server.uri())
        .with_model("zai-glm-4.7")
        .with_reasoning(ReasoningConfig::Toggle(false));

    provider
        .complete(&[Message::user("hi")], &[])
        .await
        .expect("complete should succeed");
}

#[tokio::test]
async fn no_reasoning_sends_no_reasoning_field() {
    // With nothing configured, no reasoning parameter is sent at all (Cerebras
    // reasons by default). The `thinking` object is also absent.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("ok")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = CerebrasProvider::new("test-key")
        .with_base_url(server.uri())
        .with_model("gpt-oss-120b");

    provider
        .complete(&[Message::user("hi")], &[])
        .await
        .expect("complete should succeed");

    let body = server
        .received_requests()
        .await
        .expect("request was recorded")
        .into_iter()
        .find(|r| r.url.path() == "/chat/completions")
        .expect("completions request")
        .body;
    let json: serde_json::Value = serde_json::from_slice(&body).expect("body is JSON");
    assert!(
        json.get("reasoning_effort").is_none(),
        "no reasoning config should omit `reasoning_effort`; got {json}"
    );
    assert!(
        json.get("thinking").is_none(),
        "no reasoning config should omit `thinking`; got {json}"
    );
}

#[tokio::test]
async fn reasoning_history_is_omitted() {
    // Cerebras accepts no replayed reasoning property on a request (it 400s
    // `wrong_api_format` on either `reasoning_content` or `reasoning`). When a
    // prior assistant turn carries reasoning, the replayed message must carry
    // neither field. This is the wire-level behavior the unit test can't see.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("ok")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = CerebrasProvider::new("test-key")
        .with_base_url(server.uri())
        .with_model("gpt-oss-120b")
        .with_reasoning(ReasoningConfig::Effort("high".to_string()));

    let mut prior = Message::assistant("prev");
    prior.set_reasoning_content("hidden chain of thought");

    provider
        .complete(&[prior, Message::user("next")], &[])
        .await
        .expect("complete should succeed");

    let body = server
        .received_requests()
        .await
        .expect("request was recorded")
        .into_iter()
        .find(|r| r.url.path() == "/chat/completions")
        .expect("completions request")
        .body;
    let json: serde_json::Value = serde_json::from_slice(&body).expect("body is JSON");
    let assistant_msg = json["messages"]
        .as_array()
        .expect("messages array")
        .iter()
        .find(|m| m["role"] == "assistant")
        .expect("assistant message in history");
    assert!(
        assistant_msg.get("reasoning").is_none(),
        "Cerebras must not send `reasoning`; got {assistant_msg}"
    );
    assert!(
        assistant_msg.get("reasoning_content").is_none(),
        "Cerebras must not send `reasoning_content`; got {assistant_msg}"
    );
}
