// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Write;
use std::time::Instant;

use sweet_core::{async_trait, ToolError, ToolHandler, ToolSpec};

/// A pluggable web-search backend.
///
/// Implementations live in submodules of `web_search` and are gated behind
/// per-backend Cargo features (e.g. `brave`).
#[async_trait]
pub trait WebSearchBackend: Send + Sync {
    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, WebSearchError>;
}

#[async_trait]
impl<B: WebSearchBackend + ?Sized> WebSearchBackend for Box<B> {
    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, WebSearchError> {
        (**self).search(query, max_results).await
    }
}

#[async_trait]
impl<B: WebSearchBackend + ?Sized> WebSearchBackend for std::sync::Arc<B> {
    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, WebSearchError> {
        (**self).search(query, max_results).await
    }
}

/// One hit returned by a `WebSearchBackend`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Errors that a `WebSearchBackend` can produce.
#[derive(thiserror::Error, Debug)]
pub enum WebSearchError {
    #[error("required environment variable `{var}` is not set")]
    MissingApiKey { var: &'static str },
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("decode error: {0}")]
    Decode(#[from] serde_json::Error),
}

impl From<WebSearchError> for ToolError {
    fn from(e: WebSearchError) -> Self {
        ToolError::Execution(Box::new(e))
    }
}

/// The model-facing search tool. Generic over the backend so operators can
/// swap engines without changing prompts.
pub struct WebSearch<B: WebSearchBackend> {
    backend: B,
    default_max_results: usize,
}

impl<B: WebSearchBackend> WebSearch<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            default_max_results: 5,
        }
    }

    pub fn with_default_max_results(mut self, n: usize) -> Self {
        self.default_max_results = n;
        self
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct WebSearchArgs {
    /// The search query.
    query: String,
    /// Maximum number of results to return. Defaults to 5.
    #[serde(default)]
    max_results: Option<usize>,
}

#[async_trait]
impl<B: WebSearchBackend + 'static> ToolHandler for WebSearch<B> {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let parsed: WebSearchArgs = serde_json::from_value(args).map_err(ToolError::InvalidArgs)?;
        let n = parsed.max_results.unwrap_or(self.default_max_results);
        tracing::debug!(
            target: "sweet_tools::observability",
            event = "web_search.start",
            query = %parsed.query,
            max_results = n,
            "web search start"
        );
        let started = Instant::now();
        let results = match self.backend.search(&parsed.query, n).await {
            Ok(results) => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_tools::observability",
                    event = "web_search",
                    query = %parsed.query,
                    max_results = n,
                    returned_results = results.len(),
                    duration_ms,
                    status = "ok",
                    results = %json_string(&results),
                    "web search"
                );
                results
            }
            Err(err) => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_tools::observability",
                    event = "web_search",
                    query = %parsed.query,
                    max_results = n,
                    returned_results = 0usize,
                    duration_ms,
                    status = "error",
                    error = %err,
                    "web search failed"
                );
                return Err(err.into());
            }
        };
        Ok(format_markdown(&results))
    }
}

impl<B: WebSearchBackend + 'static> From<WebSearch<B>> for ToolSpec {
    fn from(tool: WebSearch<B>) -> Self {
        ToolSpec::new(
            "web_search",
            "Search the web and return the top results as a markdown list.",
            web_search_schema(),
            tool,
        )
        .with_risk(sweet_core::ToolRisk::ReadOnly)
    }
}

fn web_search_schema() -> serde_json::Value {
    let schema = schemars::schema_for!(WebSearchArgs);
    serde_json::to_value(schema).unwrap_or_else(|_| serde_json::json!({"type": "object"}))
}

/// Convenience alias for a runtime-pluggable backend.
pub type BoxedWebSearch = WebSearch<Box<dyn WebSearchBackend>>;

fn format_markdown(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No results.".to_string();
    }
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        let _ = writeln!(out, "{}. [{}]({})", i + 1, r.title, r.url);
        if !r.snippet.is_empty() {
            let _ = writeln!(out, "   {}", r.snippet);
        }
    }
    out
}

fn json_string<T: serde::Serialize + ?Sized>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|e| format!("observability serialization failed: {e}"))
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(feature = "brave")]
pub mod brave;
#[cfg(feature = "tavily")]
pub mod tavily;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::{Arc, Mutex};

    #[test]
    fn format_markdown_with_empty_results() {
        assert_eq!(format_markdown(&[]), "No results.");
    }

    #[test]
    fn format_markdown_with_one_result() {
        let results = vec![SearchResult {
            title: "Rust Programming Language".to_string(),
            url: "https://www.rust-lang.org".to_string(),
            snippet: "A language empowering everyone to build reliable software.".to_string(),
        }];
        let md = format_markdown(&results);
        assert_eq!(
            md,
            "1. [Rust Programming Language](https://www.rust-lang.org)\n   A language empowering everyone to build reliable software.\n"
        );
    }

    #[test]
    fn format_markdown_with_multiple_results() {
        let results = vec![
            SearchResult {
                title: "First".to_string(),
                url: "https://first.example".to_string(),
                snippet: "Snippet one".to_string(),
            },
            SearchResult {
                title: "Second".to_string(),
                url: "https://second.example".to_string(),
                snippet: "Snippet two".to_string(),
            },
        ];
        let md = format_markdown(&results);
        assert!(md.contains("1. [First](https://first.example)"));
        assert!(md.contains("2. [Second](https://second.example)"));
        assert!(md.contains("   Snippet one"));
        assert!(md.contains("   Snippet two"));
    }

    #[test]
    fn format_markdown_omits_blank_snippet_line() {
        let results = vec![SearchResult {
            title: "No Snippet".to_string(),
            url: "https://example.com".to_string(),
            snippet: "".to_string(),
        }];
        let md = format_markdown(&results);
        assert_eq!(md, "1. [No Snippet](https://example.com)\n");
    }

    struct MockBackend(Vec<SearchResult>);

    #[async_trait]
    impl WebSearchBackend for MockBackend {
        async fn search(
            &self,
            _query: &str,
            _max_results: usize,
        ) -> Result<Vec<SearchResult>, WebSearchError> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn web_search_tool_calls_backend_and_formats() {
        let tool = ToolSpec::from(WebSearch::new(MockBackend(vec![SearchResult {
            title: "Hello".to_string(),
            url: "https://hello.com".to_string(),
            snippet: "World".to_string(),
        }])));
        let result = tool
            .call(serde_json::json!({"query": "hello"}))
            .await
            .unwrap();
        assert!(result.contains("[Hello](https://hello.com)"));
        assert!(result.contains("World"));
    }

    #[tokio::test]
    async fn web_search_tool_uses_default_max_results_when_omitted() {
        let (tool, counter) = counting_tool();
        let _ = tool.call(serde_json::json!({"query": "x"})).await.unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn web_search_tool_passes_through_max_results_override() {
        let (tool, counter) = counting_tool();
        let _ = tool
            .call(serde_json::json!({"query": "x", "max_results": 12}))
            .await
            .unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 12);
    }

    fn counting_tool() -> (ToolSpec, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tool = ToolSpec::from(WebSearch::new(CountingBackend(counter.clone())));
        (tool, counter)
    }

    struct CountingBackend(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    #[async_trait]
    impl WebSearchBackend for CountingBackend {
        async fn search(
            &self,
            _query: &str,
            max_results: usize,
        ) -> Result<Vec<SearchResult>, WebSearchError> {
            self.0
                .store(max_results, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn web_search_tool_rejects_invalid_args() {
        let tool = ToolSpec::from(WebSearch::new(MockBackend(vec![])));
        let err = tool
            .call(serde_json::json!("not an object"))
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidArgs(_)),
            "expected InvalidArgs, got: {err:?}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn web_search_tool_logs_query_and_full_results() {
        let tool = ToolSpec::from(WebSearch::new(MockBackend(vec![SearchResult {
            title: "Rust Programming Language".to_string(),
            url: "https://www.rust-lang.org".to_string(),
            snippet: "A language empowering everyone to build reliable software.".to_string(),
        }])));

        let (result, logs) = capture_observability(async {
            tool.call(serde_json::json!({"query": "rust", "max_results": 1}))
                .await
        })
        .await;

        assert!(result.unwrap().contains("Rust Programming Language"));
        assert!(logs.contains("web_search.start"), "{logs}");
        assert!(logs.contains("web_search"), "{logs}");
        assert!(logs.contains("rust"), "{logs}");
        assert!(logs.contains("Rust Programming Language"), "{logs}");
        assert!(logs.contains("https://www.rust-lang.org"), "{logs}");
        assert!(logs.contains("A language empowering everyone"), "{logs}");
    }

    async fn capture_observability<F, T>(future: F) -> (T, String)
    where
        F: std::future::Future<Output = T>,
    {
        let writer = SharedWriter::default();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(writer.clone())
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("set test tracing subscriber");
        tracing::callsite::rebuild_interest_cache();
        let result = future.await;
        (result, writer.contents())
    }

    #[derive(Clone, Default)]
    struct SharedWriter {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl SharedWriter {
        fn contents(&self) -> String {
            String::from_utf8(self.bytes.lock().unwrap().clone()).unwrap()
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedWriter {
        type Writer = SharedWrite;

        fn make_writer(&'a self) -> Self::Writer {
            SharedWrite {
                bytes: self.bytes.clone(),
            }
        }
    }

    struct SharedWrite {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl io::Write for SharedWrite {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.bytes.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}
