// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Embeddings via OpenAI's `/v1/embeddings` API.

use async_trait::async_trait;
use sweet_core::{Embedder, Result, SWEET_VERSION};

use crate::error::ProviderError;

use super::{DEFAULT_API_KEY_ENV, DEFAULT_BASE_URL};

pub const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-3-small";

/// [`Embedder`] backed by OpenAI's embeddings endpoint.
///
/// Compatible with any OpenAI-protocol endpoint — point
/// [`with_base_url`](Self::with_base_url) at the right URL.
#[derive(Debug, Clone)]
pub struct OpenAIEmbedder {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    user_agent: String,
    /// Cached `"openai/{model}"`, kept in sync with `model` so
    /// [`Embedder::id`] can return a borrow.
    id: String,
}

impl OpenAIEmbedder {
    /// Construct an embedder with an explicit API key, using built-in
    /// defaults for everything else.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            model: DEFAULT_EMBEDDING_MODEL.to_string(),
            user_agent: format!("sweet/{SWEET_VERSION}"),
            id: format!("openai/{DEFAULT_EMBEDDING_MODEL}"),
        }
    }

    /// Construct an embedder by reading the API key from the standard
    /// `OPENAI_API_KEY` environment variable.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var(DEFAULT_API_KEY_ENV).map_err(|_| ProviderError::MissingApiKey {
            var: DEFAULT_API_KEY_ENV,
        })?;
        Ok(Self::new(key))
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.id = format!("openai/{}", self.model);
        self
    }

    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    async fn embed_inner(
        &self,
        texts: &[String],
    ) -> std::result::Result<Vec<Vec<f32>>, ProviderError> {
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let body = wire::EmbeddingsRequest {
            model: &self.model,
            input: texts,
        };

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .header("User-Agent", &self.user_agent)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Http { status, body });
        }

        let parsed: wire::EmbeddingsResponse = resp.json().await?;
        if parsed.data.len() != texts.len() {
            return Err(ProviderError::EmptyResponse);
        }
        // The API documents `index` for out-of-order responses.
        let mut data = parsed.data;
        data.sort_by_key(|d| d.index);
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        Ok(self.embed_inner(texts).await?)
    }

    fn id(&self) -> &str {
        &self.id
    }
}

mod wire {
    #[derive(serde::Serialize)]
    pub(super) struct EmbeddingsRequest<'a> {
        pub model: &'a str,
        pub input: &'a [String],
    }

    #[derive(serde::Deserialize)]
    pub(super) struct EmbeddingsResponse {
        pub data: Vec<EmbeddingObject>,
    }

    #[derive(serde::Deserialize)]
    pub(super) struct EmbeddingObject {
        pub embedding: Vec<f32>,
        #[serde(default)]
        pub index: usize,
    }
}
