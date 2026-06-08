// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "tavily")]

use serial_test::serial;
use sweet_core::ToolSpec;
use sweet_tools::web_search::tavily::{TavilyBackend, DEFAULT_API_KEY_ENV};
use sweet_tools::web_search::{WebSearch, WebSearchBackend, WebSearchError};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn canned_tavily_response() -> serde_json::Value {
    serde_json::json!({
        "query": "rust",
        "results": [
            {
                "title": "Rust Programming Language",
                "url": "https://www.rust-lang.org",
                "content": "A language empowering everyone to build reliable software.",
                "score": 0.99
            },
            {
                "title": "The Rust Book",
                "url": "https://doc.rust-lang.org/book/",
                "content": "The official book on Rust.",
                "score": 0.95
            }
        ],
        "response_time": "1.23"
    })
}

#[tokio::test]
async fn tavily_backend_sends_query_and_bearer_token() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/search"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_partial_json(serde_json::json!({
            "query": "rust",
            "max_results": 3,
            "search_depth": "basic"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_tavily_response()))
        .expect(1)
        .mount(&server)
        .await;

    let backend = TavilyBackend::new("test-key").with_base_url(server.uri());
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
async fn tavily_backend_propagates_http_error_with_body() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/search"))
        .respond_with(ResponseTemplate::new(401).set_body_string("bad key"))
        .mount(&server)
        .await;

    let backend = TavilyBackend::new("nope").with_base_url(server.uri());
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
fn tavily_from_env_reads_api_key() {
    std::env::set_var(DEFAULT_API_KEY_ENV, "from-env-key");
    let backend = TavilyBackend::from_env().expect("env var present");
    let _ = backend;
    std::env::remove_var(DEFAULT_API_KEY_ENV);
}

#[test]
#[serial]
fn tavily_from_env_errors_when_unset() {
    std::env::remove_var(DEFAULT_API_KEY_ENV);
    let err = TavilyBackend::from_env().unwrap_err();
    match err {
        WebSearchError::MissingApiKey { var } => {
            assert_eq!(var, DEFAULT_API_KEY_ENV);
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[tokio::test]
async fn tavily_backend_via_tool_end_to_end() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_tavily_response()))
        .mount(&server)
        .await;

    let tool = ToolSpec::from(WebSearch::new(
        TavilyBackend::new("k").with_base_url(server.uri()),
    ));
    let result = tool
        .call(serde_json::json!({"query": "rust"}))
        .await
        .expect("tool call should succeed");

    assert!(result.contains("Rust Programming Language"));
    assert!(result.contains("https://www.rust-lang.org"));
    assert!(result.contains("The Rust Book"));
}
