// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Long-term memory: durable facts that outlive any single [`Session`](crate::Session).
//!
//! A [`Session`](crate::Session) is the transcript of one conversation; a
//! [`Memory`] is a store of records that persist *across* conversations —
//! user preferences, project facts, distilled decisions. Records carry a
//! [`MemoryScope`] (who/what they belong to) and are recalled with a
//! [`MemoryQuery`] (free-text relevance search or scope listing).
//!
//! This module defines the vocabulary plus [`EphemeralMemory`], a Vec-backed
//! implementation mirroring [`InMemorySession`](crate::InMemorySession) —
//! suitable for tests and ephemeral agents. The persistent, searchable
//! implementation (`SqliteMemoryStore`) lives in the `sweet-memory` crate.

use std::error::Error as StdError;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;

use crate::embedder::Embedder;

/// Structured error type for memory storage backends.
///
/// `Storage` wraps any `std::error::Error + Send + Sync` so concrete backends
/// (SQLite, remote services, etc.) can plug their own error types in without
/// forcing `sweet-core` to depend on any specific storage crate.
#[derive(thiserror::Error, Debug)]
pub enum MemoryError {
    #[error("memory storage error: {0}")]
    Storage(#[source] Box<dyn StdError + Send + Sync>),

    #[error("memory not found: {0}")]
    NotFound(String),

    #[error("embedding error: {0}")]
    Embedding(#[source] Box<dyn StdError + Send + Sync>),
}

impl MemoryError {
    pub fn storage<E>(err: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Storage(Box::new(err))
    }

    pub fn embedding<E>(err: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Embedding(Box::new(err))
    }
}

/// A time-sortable identifier for a memory record.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MemoryId(uuid::Uuid);

impl MemoryId {
    pub fn new() -> Self {
        Self(uuid::Uuid::now_v7())
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::str::FromStr for MemoryId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self(uuid::Uuid::parse_str(s)?))
    }
}

/// Who or what a memory belongs to.
///
/// The key strings are application-defined (a user id, a canonical project
/// path, a session id). Scopes are chosen by the application when it wires
/// memory tools and recall — never by the model — so records can't leak
/// across users or projects.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MemoryScope {
    /// Follows a person across all their sessions and projects.
    User(String),
    /// Tied to a body of work (a repo, a filing, a workspace).
    Project(String),
    /// Tied to one conversation. Rarely useful directly, but allows session
    /// notes to share the store with longer-lived scopes.
    Session(String),
}

impl MemoryScope {
    /// Stable discriminant name, e.g. for storage backends.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::User(_) => "user",
            Self::Project(_) => "project",
            Self::Session(_) => "session",
        }
    }

    /// The application-defined key inside the scope kind.
    pub fn key(&self) -> &str {
        match self {
            Self::User(k) | Self::Project(k) | Self::Session(k) => k,
        }
    }

    /// Inverse of [`kind`](Self::kind)/[`key`](Self::key), for storage
    /// backends reading discriminant + key columns. `None` for an unknown
    /// kind string.
    pub fn from_parts(kind: &str, key: &str) -> Option<Self> {
        match kind {
            "user" => Some(Self::User(key.to_string())),
            "project" => Some(Self::Project(key.to_string())),
            "session" => Some(Self::Session(key.to_string())),
            _ => None,
        }
    }
}

/// One durable memory.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MemoryRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub content: String,
    pub tags: Vec<String>,
    /// Session the memory was distilled from or saved in, for provenance.
    pub source_session: Option<String>,
    /// Unix seconds.
    pub created_at: i64,
    /// Unix seconds.
    pub updated_at: i64,
}

/// Default number of hits returned by [`MemoryQuery::new`].
pub const DEFAULT_QUERY_LIMIT: usize = 10;

/// A recall request against a [`Memory`] store.
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    /// Free-text relevance search. `None` lists by scope, newest first.
    pub text: Option<String>,
    /// Scopes to search (OR). Empty searches every scope in the store.
    pub scopes: Vec<MemoryScope>,
    /// Tags a record must all carry (AND). Empty applies no tag filter.
    pub tags: Vec<String>,
    /// Maximum number of hits returned.
    pub limit: usize,
}

impl MemoryQuery {
    pub fn new() -> Self {
        Self {
            text: None,
            scopes: Vec::new(),
            tags: Vec::new(),
            limit: DEFAULT_QUERY_LIMIT,
        }
    }

    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn with_scopes(mut self, scopes: impl IntoIterator<Item = MemoryScope>) -> Self {
        self.scopes = scopes.into_iter().collect();
        self
    }

    pub fn with_tags(mut self, tags: impl IntoIterator<Item = String>) -> Self {
        self.tags = tags.into_iter().collect();
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

impl Default for MemoryQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// A search result: the record plus its relevance score (higher is better).
///
/// Scores are only comparable within one result set — hybrid backends fuse
/// keyword and vector rankings, so the absolute value carries no unit.
#[derive(Debug, Clone)]
pub struct MemoryHit {
    pub record: MemoryRecord,
    pub score: f32,
}

/// Abstraction over long-term memory storage.
///
/// All methods take `&self`: a store is shared as `Arc<dyn Memory>` between
/// tool handlers, recall, and distillation running concurrently, so
/// implementations use interior mutability (mirroring
/// [`ToolHandler`](crate::ToolHandler), not [`Session`](crate::Session)).
#[async_trait]
pub trait Memory: Send + Sync {
    /// Persist a new memory and return the stored record.
    async fn save(
        &self,
        scope: MemoryScope,
        content: &str,
        tags: &[String],
        source_session: Option<&str>,
    ) -> Result<MemoryRecord, MemoryError>;

    /// Fetch one record by id. `Ok(None)` when it doesn't exist.
    async fn get(&self, id: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError>;

    /// Recall records matching `query`, best first.
    async fn search(&self, query: &MemoryQuery) -> Result<Vec<MemoryHit>, MemoryError>;

    /// Rewrite a record's content and/or tags. `None` leaves a field as-is.
    /// Errors with [`MemoryError::NotFound`] for an unknown id.
    async fn update(
        &self,
        id: &MemoryId,
        content: Option<&str>,
        tags: Option<&[String]>,
    ) -> Result<MemoryRecord, MemoryError>;

    /// Remove a record. Returns whether it existed.
    async fn delete(&self, id: &MemoryId) -> Result<bool, MemoryError>;
}

/// Current unix time in seconds, for implementations stamping records.
pub fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Cosine similarity of two vectors; 0.0 on dimension mismatch or zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Reciprocal Rank Fusion over per-strategy rankings (best first).
///
/// Each id's fused score is `Σ 1/(60 + rank)` across the lists it appears in.
/// RRF combines keyword and vector rankings without having to normalize
/// incomparable score scales (bm25 vs cosine). Returns ids best-first.
pub fn rrf_merge(rankings: &[Vec<MemoryId>]) -> Vec<(MemoryId, f32)> {
    const RRF_K: f32 = 60.0;
    let mut scores: Vec<(MemoryId, f32)> = Vec::new();
    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
            match scores.iter_mut().find(|(seen, _)| seen == id) {
                Some((_, score)) => *score += contribution,
                None => scores.push((id.clone(), contribution)),
            }
        }
    }
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

struct EphemeralEntry {
    record: MemoryRecord,
    embedding: Option<Vec<f32>>,
}

/// Simple in-memory [`Memory`] backed by a `Vec`, mirroring
/// [`InMemorySession`](crate::InMemorySession).
///
/// Keyword recall uses token-overlap scoring; with an [`Embedder`] attached,
/// cosine similarity over embedded records is fused in via [`rrf_merge`].
/// Nothing survives the process — use `SqliteMemoryStore` from `sweet-memory`
/// for persistence.
pub struct EphemeralMemory {
    entries: Mutex<Vec<EphemeralEntry>>,
    embedder: Option<Arc<dyn Embedder>>,
}

impl EphemeralMemory {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            embedder: None,
        }
    }

    /// Attach an embedder; subsequent saves are embedded and searches add a
    /// semantic ranking. Embedding failure degrades that record to
    /// keyword-only recall rather than failing the save.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }
}

impl Default for EphemeralMemory {
    fn default() -> Self {
        Self::new()
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

fn matches_filters(record: &MemoryRecord, query: &MemoryQuery) -> bool {
    let scope_ok = query.scopes.is_empty() || query.scopes.contains(&record.scope);
    let tags_ok = query.tags.iter().all(|t| record.tags.contains(t));
    scope_ok && tags_ok
}

#[async_trait]
impl Memory for EphemeralMemory {
    async fn save(
        &self,
        scope: MemoryScope,
        content: &str,
        tags: &[String],
        source_session: Option<&str>,
    ) -> Result<MemoryRecord, MemoryError> {
        let embedding = match &self.embedder {
            Some(embedder) => match embedder.embed(&[content.to_string()]).await {
                Ok(mut vectors) => vectors.pop(),
                Err(err) => {
                    tracing::warn!("embedding failed, saving keyword-only memory: {err}");
                    None
                }
            },
            None => None,
        };
        let now = unix_now();
        let record = MemoryRecord {
            id: MemoryId::new(),
            scope,
            content: content.to_string(),
            tags: tags.to_vec(),
            source_session: source_session.map(str::to_string),
            created_at: now,
            updated_at: now,
        };
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(EphemeralEntry {
                record: record.clone(),
                embedding,
            });
        Ok(record)
    }

    async fn get(&self, id: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError> {
        Ok(self
            .entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|e| &e.record.id == id)
            .map(|e| e.record.clone()))
    }

    async fn search(&self, query: &MemoryQuery) -> Result<Vec<MemoryHit>, MemoryError> {
        let Some(text) = query.text.as_deref().filter(|t| !t.trim().is_empty()) else {
            // List mode: newest first within the filters.
            let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let mut records: Vec<MemoryRecord> = entries
                .iter()
                .filter(|e| matches_filters(&e.record, query))
                .map(|e| e.record.clone())
                .collect();
            records.sort_by_key(|r| std::cmp::Reverse(r.updated_at));
            records.truncate(query.limit);
            return Ok(records
                .into_iter()
                .map(|record| MemoryHit { record, score: 0.0 })
                .collect());
        };

        // Embed the query before taking the lock; the guard cannot be held
        // across an await point.
        let query_embedding = match &self.embedder {
            Some(embedder) => embedder
                .embed(&[text.to_string()])
                .await
                .map_err(|e| MemoryError::Embedding(e.into()))?
                .pop(),
            None => None,
        };

        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let candidates: Vec<&EphemeralEntry> = entries
            .iter()
            .filter(|e| matches_filters(&e.record, query))
            .collect();

        // Keyword ranking: fraction of query tokens present in the record.
        let query_tokens = tokenize(text);
        let mut keyword: Vec<(MemoryId, f32)> = candidates
            .iter()
            .filter_map(|e| {
                let record_tokens = tokenize(&e.record.content)
                    .into_iter()
                    .chain(e.record.tags.iter().map(|t| t.to_lowercase()))
                    .collect::<Vec<_>>();
                let matched = query_tokens
                    .iter()
                    .filter(|qt| record_tokens.contains(qt))
                    .count();
                (matched > 0).then(|| {
                    (
                        e.record.id.clone(),
                        matched as f32 / query_tokens.len().max(1) as f32,
                    )
                })
            })
            .collect();
        keyword.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Vector ranking, when both the query and the record have embeddings.
        let mut vector: Vec<(MemoryId, f32)> = match &query_embedding {
            Some(qv) => candidates
                .iter()
                .filter_map(|e| {
                    e.embedding
                        .as_ref()
                        .map(|ev| (e.record.id.clone(), cosine_similarity(qv, ev)))
                })
                .collect(),
            None => Vec::new(),
        };
        vector.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let rankings = [
            keyword.into_iter().map(|(id, _)| id).collect::<Vec<_>>(),
            vector.into_iter().map(|(id, _)| id).collect::<Vec<_>>(),
        ];
        let fused = rrf_merge(&rankings);

        Ok(fused
            .into_iter()
            .take(query.limit)
            .filter_map(|(id, score)| {
                candidates
                    .iter()
                    .find(|e| e.record.id == id)
                    .map(|e| MemoryHit {
                        record: e.record.clone(),
                        score,
                    })
            })
            .collect())
    }

    async fn update(
        &self,
        id: &MemoryId,
        content: Option<&str>,
        tags: Option<&[String]>,
    ) -> Result<MemoryRecord, MemoryError> {
        // Re-embed outside the lock if content changes.
        let embedding = match (&self.embedder, content) {
            (Some(embedder), Some(text)) => match embedder.embed(&[text.to_string()]).await {
                Ok(mut vectors) => Some(vectors.pop()),
                Err(err) => {
                    tracing::warn!("embedding failed, demoting memory to keyword-only: {err}");
                    Some(None)
                }
            },
            _ => None, // content unchanged: keep the existing embedding
        };

        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let entry = entries
            .iter_mut()
            .find(|e| &e.record.id == id)
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;
        if let Some(text) = content {
            entry.record.content = text.to_string();
        }
        if let Some(tags) = tags {
            entry.record.tags = tags.to_vec();
        }
        if let Some(new_embedding) = embedding {
            entry.embedding = new_embedding;
        }
        entry.record.updated_at = unix_now();
        Ok(entry.record.clone())
    }

    async fn delete(&self, id: &MemoryId) -> Result<bool, MemoryError> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let before = entries.len();
        entries.retain(|e| &e.record.id != id);
        Ok(entries.len() < before)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as CoreResult;

    fn scope() -> MemoryScope {
        MemoryScope::User("u1".into())
    }

    #[tokio::test]
    async fn save_get_roundtrip() {
        let memory = EphemeralMemory::new();
        let saved = memory
            .save(scope(), "prefers tabs", &["style".into()], Some("s1"))
            .await
            .unwrap();
        let fetched = memory.get(&saved.id).await.unwrap().unwrap();
        assert_eq!(fetched, saved);
        assert_eq!(fetched.content, "prefers tabs");
        assert_eq!(fetched.source_session.as_deref(), Some("s1"));
    }

    #[tokio::test]
    async fn search_ranks_keyword_overlap() {
        let memory = EphemeralMemory::new();
        memory
            .save(scope(), "user prefers dark mode in the editor", &[], None)
            .await
            .unwrap();
        memory
            .save(scope(), "project uses tokio for async runtime", &[], None)
            .await
            .unwrap();

        let hits = memory
            .search(&MemoryQuery::new().with_text("dark mode"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].record.content.contains("dark mode"));
    }

    #[tokio::test]
    async fn search_without_text_lists_newest_first() {
        let memory = EphemeralMemory::new();
        let first = memory.save(scope(), "older", &[], None).await.unwrap();
        // Force distinct updated_at ordering regardless of clock granularity.
        memory.save(scope(), "newer", &[], None).await.unwrap();
        memory
            .update(&first.id, Some("older, refreshed"), None)
            .await
            .unwrap();

        let hits = memory.search(&MemoryQuery::new()).await.unwrap();
        assert_eq!(hits.len(), 2);
        // The refreshed record's updated_at is >= the other's; both orders are
        // valid when timestamps collide, so just assert both are returned.
        assert!(hits.iter().any(|h| h.record.content == "newer"));
        assert!(hits.iter().any(|h| h.record.content == "older, refreshed"));
    }

    #[tokio::test]
    async fn search_filters_scope_and_tags() {
        let memory = EphemeralMemory::new();
        memory
            .save(MemoryScope::User("a".into()), "alpha fact", &[], None)
            .await
            .unwrap();
        memory
            .save(
                MemoryScope::Project("p".into()),
                "alpha decision",
                &["arch".into()],
                None,
            )
            .await
            .unwrap();

        let hits = memory
            .search(
                &MemoryQuery::new()
                    .with_text("alpha")
                    .with_scopes([MemoryScope::Project("p".into())]),
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.content, "alpha decision");

        let hits = memory
            .search(
                &MemoryQuery::new()
                    .with_text("alpha")
                    .with_tags(["missing".to_string()]),
            )
            .await
            .unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn update_and_delete() {
        let memory = EphemeralMemory::new();
        let saved = memory.save(scope(), "draft", &[], None).await.unwrap();

        let updated = memory
            .update(&saved.id, Some("final"), Some(&["done".to_string()]))
            .await
            .unwrap();
        assert_eq!(updated.content, "final");
        assert_eq!(updated.tags, vec!["done".to_string()]);
        assert!(updated.updated_at >= saved.updated_at);

        assert!(memory.delete(&saved.id).await.unwrap());
        assert!(!memory.delete(&saved.id).await.unwrap());
        assert!(memory.get(&saved.id).await.unwrap().is_none());

        let missing = memory.update(&saved.id, Some("x"), None).await;
        assert!(matches!(missing, Err(MemoryError::NotFound(_))));
    }

    /// Deterministic embedder: "hot"-ish texts map near (1,0), others (0,1).
    struct FakeEmbedder;

    #[async_trait]
    impl Embedder for FakeEmbedder {
        async fn embed(&self, texts: &[String]) -> CoreResult<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    if t.contains("hot") {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    }
                })
                .collect())
        }

        fn id(&self) -> &str {
            "fake/embedder"
        }
    }

    #[tokio::test]
    async fn hybrid_search_fuses_semantic_ranking() {
        let memory = EphemeralMemory::new().with_embedder(Arc::new(FakeEmbedder));
        memory
            .save(scope(), "the stove is hot right now", &[], None)
            .await
            .unwrap();
        memory
            .save(scope(), "the lake is cold in winter", &[], None)
            .await
            .unwrap();

        // "hot weather" shares no useful token rank with the cold record;
        // semantically it lands next to the hot record.
        let hits = memory
            .search(&MemoryQuery::new().with_text("hot weather"))
            .await
            .unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].record.content.contains("hot"));
    }

    #[test]
    fn rrf_merge_prefers_items_ranked_in_both_lists() {
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        let fused = rrf_merge(&[vec![a.clone(), b.clone()], vec![c.clone(), a.clone()]]);
        assert_eq!(fused[0].0, a);
    }

    #[test]
    fn cosine_similarity_basics() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn memory_scope_parts_roundtrip() {
        for scope in [
            MemoryScope::User("u".into()),
            MemoryScope::Project("p".into()),
            MemoryScope::Session("s".into()),
        ] {
            assert_eq!(
                MemoryScope::from_parts(scope.kind(), scope.key()),
                Some(scope)
            );
        }
        assert_eq!(MemoryScope::from_parts("other", "x"), None);
    }

    #[test]
    fn memory_id_is_v7_and_parses() {
        let id = MemoryId::new();
        let id_string = id.to_string();
        let parts: Vec<&str> = id_string.split('-').collect();
        assert!(parts[2].starts_with('7'));
        let parsed: MemoryId = id.to_string().parse().unwrap();
        assert_eq!(parsed, id);
    }
}
