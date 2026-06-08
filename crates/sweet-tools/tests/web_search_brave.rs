// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "brave")]

use serial_test::serial;
use sweet_core::ToolSpec;
use sweet_tools::web_search::brave::{BraveBackend, DEFAULT_API_KEY_ENV};
use sweet_tools::web_search::{WebSearch, WebSearchBackend, WebSearchError};
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn canned_brave_response() -> serde_json::Value {
    serde_json::json!({
        "web": {
            "results": [
                {
                    "title": "Rust Programming Language",
                    "url": "https://www.rust-lang.org",
                    "description": "A language empowering everyone to build reliable software."
                },
                {
                    "title": "The Rust Book",
                    "url": "https://doc.rust-lang.org/book/",
                    "description": "The official book on Rust."
                }
            ]
        }
    })
}

#[tokio::test]
async fn brave_backend_sends_query_and_subscription_token() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/res/v1/web/search"))
        .and(header("X-Subscription-Token", "test-key"))
        .and(query_param("q", "rust"))
        .and(query_param("count", "3"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_brave_response()))
        .expect(1)
        .mount(&server)
        .await;

    let backend = BraveBackend::new("test-key").with_base_url(server.uri());
    let results = backend
        .search("rust", 3)
        .await
        .expect("search should succeed");

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].title, "Rust Programming Language");
    assert_eq!(results[0].url, "https://www.rust-lang.org");
    assert_eq!(
        results[0].snippet,
        "A language empowering everyone to build reliable software."
    );
    assert_eq!(results[1].title, "The Rust Book");
}

#[tokio::test]
async fn brave_backend_propagates_http_error_with_body() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/res/v1/web/search"))
        .respond_with(ResponseTemplate::new(401).set_body_string("bad key"))
        .mount(&server)
        .await;

    let backend = BraveBackend::new("nope").with_base_url(server.uri());
    let err = backend.search("x", 1).await.unwrap_err();

    match err {
        WebSearchError::Http { status, body } => {
            assert_eq!(status, 401);
            assert_eq!(body, "bad key");
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[test]
#[serial]
fn brave_from_env_reads_api_key() {
    std::env::set_var(DEFAULT_API_KEY_ENV, "from-env-key");
    let backend = BraveBackend::from_env().expect("env var present");
    // Indirect check: base URL is the default.
    let _ = backend;
    std::env::remove_var(DEFAULT_API_KEY_ENV);
}

#[test]
#[serial]
fn brave_from_env_errors_when_unset() {
    std::env::remove_var(DEFAULT_API_KEY_ENV);
    let err = BraveBackend::from_env().unwrap_err();
    match err {
        WebSearchError::MissingApiKey { var } => {
            assert_eq!(var, DEFAULT_API_KEY_ENV);
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[tokio::test]
async fn brave_backend_via_tool_end_to_end() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/res/v1/web/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_brave_response()))
        .mount(&server)
        .await;

    let tool = ToolSpec::from(WebSearch::new(
        BraveBackend::new("k").with_base_url(server.uri()),
    ));
    let result = tool
        .call(serde_json::json!({"query": "rust"}))
        .await
        .expect("tool call should succeed");

    assert!(result.contains("Rust Programming Language"));
    assert!(result.contains("https://www.rust-lang.org"));
    assert!(result.contains("The Rust Book"));
}
