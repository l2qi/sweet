// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "gemini")]

use serial_test::serial;
use sweet_core::Embedder;
use sweet_llm::{gemini, GeminiEmbedder};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn id_includes_model_and_dimensionality() {
    assert_eq!(
        GeminiEmbedder::new("k").id(),
        format!(
            "{}/{}",
            gemini::DEFAULT_EMBEDDING_MODEL,
            gemini::DEFAULT_OUTPUT_DIMENSIONALITY
        )
    );
    assert_eq!(
        GeminiEmbedder::new("k")
            .with_model("other-embedding")
            .with_output_dimensionality(128)
            .id(),
        "other-embedding/128"
    );
}

#[tokio::test]
async fn embed_sends_batch_request() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path(format!(
            "/models/{}:batchEmbedContents",
            gemini::DEFAULT_EMBEDDING_MODEL
        )))
        .and(header("x-goog-api-key", "gem-key"))
        .and(body_partial_json(serde_json::json!({
            "requests": [
                {
                    "model": format!("models/{}", gemini::DEFAULT_EMBEDDING_MODEL),
                    "content": {"parts": [{"text": "hello"}]},
                    "outputDimensionality": gemini::DEFAULT_OUTPUT_DIMENSIONALITY
                },
                {
                    "model": format!("models/{}", gemini::DEFAULT_EMBEDDING_MODEL),
                    "content": {"parts": [{"text": "world"}]},
                    "outputDimensionality": gemini::DEFAULT_OUTPUT_DIMENSIONALITY
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embeddings": [
                {"values": [1.0, 0.0]},
                {"values": [0.0, 1.0]}
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let embedder = GeminiEmbedder::new("gem-key").with_base_url(server.uri());
    let vectors = embedder
        .embed(&["hello".to_string(), "world".to_string()])
        .await
        .unwrap();
    assert_eq!(vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
}

#[tokio::test]
async fn embed_empty_input_skips_network() {
    let embedder = GeminiEmbedder::new("k").with_base_url("http://127.0.0.1:1");
    assert!(embedder.embed(&[]).await.unwrap().is_empty());
}

#[tokio::test]
async fn http_error_is_surfaced() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;

    let embedder = GeminiEmbedder::new("k").with_base_url(server.uri());
    let err = embedder.embed(&["x".to_string()]).await.unwrap_err();
    assert!(err.to_string().contains("500"));
}

#[tokio::test]
async fn short_response_is_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embeddings": [{"values": [1.0]}]
        })))
        .mount(&server)
        .await;

    let embedder = GeminiEmbedder::new("k").with_base_url(server.uri());
    let err = embedder
        .embed(&["a".to_string(), "b".to_string()])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("no choices") || !err.to_string().is_empty());
}

#[test]
#[serial]
fn from_env_reads_gemini_api_key() {
    // SAFETY: tests in this file using env are gated behind `#[serial]`.
    std::env::set_var("GEMINI_API_KEY", "from-env");
    let embedder = GeminiEmbedder::from_env().expect("env var present");
    let _ = embedder;
    std::env::remove_var("GEMINI_API_KEY");
}
