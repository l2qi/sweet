// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;

/// A backend that turns text into fixed-dimension vectors for semantic
/// search, the embedding counterpart of [`Model`](crate::Model).
///
/// Provider implementations live in `sweet-llm`; memory stores accept an
/// `Arc<dyn Embedder>` to add semantic recall.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed each input text, returning one vector per input, in order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Stable identifier for the model/dimensions combination, e.g.
    /// `"text-embedding-3-small"` or `"gemini-embedding-001/768"`. Stores
    /// persist it next to each vector: vectors from different embedders are
    /// not comparable, so this is what keeps them from being mixed.
    fn id(&self) -> &str;
}

#[async_trait]
impl<E: Embedder + ?Sized> Embedder for Box<E> {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        (**self).embed(texts).await
    }

    fn id(&self) -> &str {
        (**self).id()
    }
}

#[async_trait]
impl<E: Embedder + ?Sized> Embedder for Arc<E> {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        (**self).embed(texts).await
    }

    fn id(&self) -> &str {
        (**self).id()
    }
}
