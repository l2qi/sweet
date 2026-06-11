// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "openai")]

use serial_test::serial;
use sweet_core::Embedder;
use sweet_llm::{openai, OpenAIEmbedder};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn id_reflects_model() {
    assert_eq!(
        OpenAIEmbedder::new("k").id(),
        format!("openai/{}", openai::DEFAULT_EMBEDDING_MODEL)
    );
    assert_eq!(
        OpenAIEmbedder::new("k")
            .with_model("text-embedding-3-large")
            .id(),
        "openai/text-embedding-3-large"
    );
}

#[tokio::test]
async fn embed_sends_request_and_orders_by_index() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_partial_json(serde_json::json!({
            "model": openai::DEFAULT_EMBEDDING_MODEL,
            "input": ["first", "second"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}
            ],
            "model": openai::DEFAULT_EMBEDDING_MODEL,
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let embedder = OpenAIEmbedder::new("test-key").with_base_url(server.uri());
    let vectors = embedder
        .embed(&["first".to_string(), "second".to_string()])
        .await
        .unwrap();

    // Out-of-order `index` values are reordered to match the inputs.
    assert_eq!(vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
}

#[tokio::test]
async fn embed_empty_input_skips_network() {
    // No mock server at all: an empty input must not hit the network.
    let embedder = OpenAIEmbedder::new("k").with_base_url("http://127.0.0.1:1");
    assert!(embedder.embed(&[]).await.unwrap().is_empty());
}

#[tokio::test]
async fn http_error_is_surfaced() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;

    let embedder = OpenAIEmbedder::new("k").with_base_url(server.uri());
    let err = embedder.embed(&["x".to_string()]).await.unwrap_err();
    let text = err.to_string();
    assert!(text.contains("429"), "unexpected error: {text}");
}

#[test]
#[serial]
fn from_env_reads_openai_api_key() {
    // SAFETY: tests in this file using env are gated behind `#[serial]`.
    std::env::set_var("OPENAI_API_KEY", "from-env");
    let embedder = OpenAIEmbedder::from_env().expect("env var present");
    let _ = embedder;
    std::env::remove_var("OPENAI_API_KEY");
}

#[test]
#[serial]
fn from_env_errors_when_unset() {
    std::env::remove_var("OPENAI_API_KEY");
    assert!(OpenAIEmbedder::from_env().is_err());
}
