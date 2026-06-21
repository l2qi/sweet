// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Embeddings via Gemini's `models/{model}:batchEmbedContents` API.

use async_trait::async_trait;
use sweet_core::{Embedder, Result, SWEET_VERSION};

use crate::error::ProviderError;

use super::{DEFAULT_API_KEY_ENV, DEFAULT_BASE_URL};

pub const DEFAULT_EMBEDDING_MODEL: &str = "gemini-embedding-001";

/// Default requested vector size. `gemini-embedding-001` natively produces
/// 3072 dimensions; 768 keeps stored vectors ~4x smaller at a negligible
/// quality cost for memory-recall workloads.
pub const DEFAULT_OUTPUT_DIMENSIONALITY: usize = 768;

/// [`Embedder`] backed by Gemini's batch embeddings endpoint.
#[derive(Debug, Clone)]
pub struct GeminiEmbedder {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    output_dimensionality: usize,
    user_agent: String,
    /// Cached `"{model}/{dims}"`. Dimensionality is part of the identity:
    /// vectors of different sizes are not comparable.
    id: String,
}

impl GeminiEmbedder {
    /// Construct an embedder with an explicit API key, using built-in
    /// defaults for everything else.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            model: DEFAULT_EMBEDDING_MODEL.to_string(),
            output_dimensionality: DEFAULT_OUTPUT_DIMENSIONALITY,
            user_agent: format!("sweet/{SWEET_VERSION}"),
            id: format!("{DEFAULT_EMBEDDING_MODEL}/{DEFAULT_OUTPUT_DIMENSIONALITY}"),
        }
    }

    /// Construct an embedder by reading the API key from the standard
    /// `GEMINI_API_KEY` environment variable.
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
        self.refresh_id();
        self
    }

    /// Request truncated vectors of this size (Matryoshka-style).
    pub fn with_output_dimensionality(mut self, dims: usize) -> Self {
        self.output_dimensionality = dims;
        self.refresh_id();
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

    fn refresh_id(&mut self) {
        self.id = format!("{}/{}", self.model, self.output_dimensionality);
    }

    async fn embed_inner(
        &self,
        texts: &[String],
    ) -> std::result::Result<Vec<Vec<f32>>, ProviderError> {
        let url = format!(
            "{}/models/{}:batchEmbedContents",
            self.base_url.trim_end_matches('/'),
            self.model
        );
        let model_path = format!("models/{}", self.model);
        let body = wire::BatchRequest {
            requests: texts
                .iter()
                .map(|text| wire::EmbedRequest {
                    model: &model_path,
                    content: wire::Content {
                        parts: vec![wire::Part { text }],
                    },
                    output_dimensionality: self.output_dimensionality,
                })
                .collect(),
        };

        let mut req = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .header("User-Agent", &self.user_agent)
            .json(&body);
        if !self.api_key.is_empty() {
            req = req.header("x-goog-api-key", &self.api_key);
        }
        let resp = req.send().await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Http { status, body });
        }

        let parsed: wire::BatchResponse = resp.json().await?;
        if parsed.embeddings.len() != texts.len() {
            return Err(ProviderError::EmptyResponse);
        }
        Ok(parsed.embeddings.into_iter().map(|e| e.values).collect())
    }
}

#[async_trait]
impl Embedder for GeminiEmbedder {
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
    pub(super) struct BatchRequest<'a> {
        pub requests: Vec<EmbedRequest<'a>>,
    }

    #[derive(serde::Serialize)]
    pub(super) struct EmbedRequest<'a> {
        pub model: &'a str,
        pub content: Content<'a>,
        #[serde(rename = "outputDimensionality")]
        pub output_dimensionality: usize,
    }

    #[derive(serde::Serialize)]
    pub(super) struct Content<'a> {
        pub parts: Vec<Part<'a>>,
    }

    #[derive(serde::Serialize)]
    pub(super) struct Part<'a> {
        pub text: &'a str,
    }

    #[derive(serde::Deserialize)]
    pub(super) struct BatchResponse {
        pub embeddings: Vec<EmbeddingValues>,
    }

    #[derive(serde::Deserialize)]
    pub(super) struct EmbeddingValues {
        pub values: Vec<f32>,
    }
}
