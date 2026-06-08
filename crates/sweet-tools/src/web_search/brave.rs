// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use sweet_core::async_trait;

use super::{SearchResult, WebSearchBackend, WebSearchError};

pub const DEFAULT_BASE_URL: &str = "https://api.search.brave.com";
pub const DEFAULT_API_KEY_ENV: &str = "BRAVE_SEARCH_API_KEY";

#[derive(Debug)]
pub struct BraveBackend {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl BraveBackend {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
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
}

#[async_trait]
impl WebSearchBackend for BraveBackend {
    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, WebSearchError> {
        let url = format!("{}/res/v1/web/search", self.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .get(&url)
            .query(&[("q", query), ("count", &max_results.to_string())])
            .header("X-Subscription-Token", &self.api_key)
            .header("Accept", "application/json")
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

        let parsed: BraveResponse = resp.json().await?;
        Ok(parsed
            .web
            .results
            .into_iter()
            .map(SearchResult::from)
            .collect())
    }
}

#[derive(serde::Deserialize)]
struct BraveResponse {
    #[serde(default)]
    web: BraveWebResults,
}

#[derive(serde::Deserialize, Default)]
struct BraveWebResults {
    #[serde(default)]
    results: Vec<BraveResult>,
}

#[derive(serde::Deserialize)]
struct BraveResult {
    title: String,
    url: String,
    #[serde(default)]
    description: String,
}

impl From<BraveResult> for SearchResult {
    fn from(r: BraveResult) -> Self {
        Self {
            title: r.title,
            url: r.url,
            snippet: r.description,
        }
    }
}
