// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use sweet_core::async_trait;

use super::{SearchResult, WebSearchBackend, WebSearchError};

pub const DEFAULT_BASE_URL: &str = "https://api.tavily.com";
pub const DEFAULT_API_KEY_ENV: &str = "TAVILY_API_KEY";

/// Tavily search depth. `Basic` is faster and cheaper; `Advanced` costs more
/// credits but returns higher-quality LLM-grounded snippets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchDepth {
    Basic,
    Advanced,
}

impl SearchDepth {
    fn as_str(self) -> &'static str {
        match self {
            SearchDepth::Basic => "basic",
            SearchDepth::Advanced => "advanced",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TavilyBackend {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    search_depth: SearchDepth,
}

impl TavilyBackend {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            search_depth: SearchDepth::Basic,
        }
    }

    pub fn from_env() -> Result<Self, WebSearchError> {
        let key =
            std::env::var(DEFAULT_API_KEY_ENV).map_err(|_| WebSearchError::MissingApiKey {
                var: DEFAULT_API_KEY_ENV,
            })?;
        Ok(Self::new(key))
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    pub fn with_search_depth(mut self, depth: SearchDepth) -> Self {
        self.search_depth = depth;
        self
    }
}

#[async_trait]
impl WebSearchBackend for TavilyBackend {
    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, WebSearchError> {
        let url = format!("{}/search", self.base_url.trim_end_matches('/'));

        let body = serde_json::json!({
            "query": query,
            "max_results": max_results,
            "search_depth": self.search_depth.as_str(),
        });

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(WebSearchError::Http {
                status: status.as_u16(),
                body,
            });
        }

        let parsed: TavilyResponse = resp.json().await?;
        Ok(parsed.results.into_iter().map(SearchResult::from).collect())
    }
}

#[derive(serde::Deserialize)]
struct TavilyResponse {
    #[serde(default)]
    results: Vec<TavilyResult>,
}

#[derive(serde::Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    #[serde(default)]
    content: String,
}

impl From<TavilyResult> for SearchResult {
    fn from(r: TavilyResult) -> Self {
        Self {
            title: r.title,
            url: r.url,
            snippet: r.content,
        }
    }
}
