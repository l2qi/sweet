// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "sqlite")]

use std::sync::Arc;

use sweet_core::{
    async_trait, Embedder, Memory, MemoryError, MemoryQuery, MemoryScope, Result as CoreResult,
};
use sweet_memory::SqliteMemoryStore;

fn user_scope() -> MemoryScope {
    MemoryScope::User("u1".into())
}

#[tokio::test]
async fn crud_roundtrip() {
    let store = SqliteMemoryStore::open(":memory:").unwrap();

    let saved = store
        .save(user_scope(), "prefers tabs", &["style".into()], Some("s1"))
        .await
        .unwrap();
    assert_eq!(store.get(&saved.id).await.unwrap().unwrap(), saved);

    let updated = store
        .update(&saved.id, Some("prefers spaces"), None)
        .await
        .unwrap();
    assert_eq!(updated.content, "prefers spaces");
    assert_eq!(updated.tags, vec!["style".to_string()]);

    assert!(store.delete(&saved.id).await.unwrap());
    assert!(!store.delete(&saved.id).await.unwrap());
    assert!(store.get(&saved.id).await.unwrap().is_none());

    let missing = store.update(&saved.id, Some("x"), None).await;
    assert!(matches!(missing, Err(MemoryError::NotFound(_))));
}

#[tokio::test]
async fn persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.db");

    let saved = {
        let store = SqliteMemoryStore::open(&path).unwrap();
        store
            .save(user_scope(), "durable fact", &[], None)
            .await
            .unwrap()
    };

    let store = SqliteMemoryStore::open(&path).unwrap();
    let fetched = store.get(&saved.id).await.unwrap().unwrap();
    assert_eq!(fetched, saved);

    // FTS index survives reopen too.
    let hits = store
        .search(&MemoryQuery::new().with_text("durable"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
}

#[tokio::test]
async fn fts_search_ranks_and_filters() {
    let store = SqliteMemoryStore::open(":memory:").unwrap();
    store
        .save(user_scope(), "the user prefers dark mode", &[], None)
        .await
        .unwrap();
    store
        .save(
            MemoryScope::Project("p1".into()),
            "project uses dark color tokens",
            &["design".into()],
            None,
        )
        .await
        .unwrap();
    store
        .save(user_scope(), "unrelated note about lunch", &[], None)
        .await
        .unwrap();

    // Plain text search across all scopes.
    let hits = store
        .search(&MemoryQuery::new().with_text("dark"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);

    // Scope filter.
    let hits = store
        .search(
            &MemoryQuery::new()
                .with_text("dark")
                .with_scopes([user_scope()]),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].record.content.contains("prefers dark mode"));

    // Tag filter.
    let hits = store
        .search(
            &MemoryQuery::new()
                .with_text("dark")
                .with_tags(["design".to_string()]),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].record.tags.contains(&"design".to_string()));
}

#[tokio::test]
async fn list_mode_returns_newest_first() {
    let store = SqliteMemoryStore::open(":memory:").unwrap();
    store.save(user_scope(), "first", &[], None).await.unwrap();
    store.save(user_scope(), "second", &[], None).await.unwrap();

    let hits = store
        .search(&MemoryQuery::new().with_limit(1))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    // Timestamps may collide within a second; the id tiebreak (uuid v7 is
    // time-ordered) keeps the newest insert first.
    assert_eq!(hits[0].record.content, "second");
}

#[tokio::test]
async fn malicious_fts_queries_are_treated_as_text() {
    let store = SqliteMemoryStore::open(":memory:").unwrap();
    store
        .save(user_scope(), "plain content here", &[], None)
        .await
        .unwrap();

    for query in ["NEAR(", "a\" OR \"b", "col:value", "(((", "x*"] {
        // Must not error out as FTS5 syntax.
        let result = store.search(&MemoryQuery::new().with_text(query)).await;
        assert!(result.is_ok(), "query {query:?} errored: {result:?}");
    }
}

/// Deterministic embedder: vectors depend only on whether the text mentions
/// heat, so semantic neighbors are predictable.
struct FakeEmbedder {
    id: &'static str,
}

#[async_trait]
impl Embedder for FakeEmbedder {
    async fn embed(&self, texts: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                if t.contains("hot") || t.contains("warm") {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                }
            })
            .collect())
    }

    fn id(&self) -> &str {
        self.id
    }
}

#[tokio::test]
async fn hybrid_search_fuses_semantic_ranking() {
    let store = SqliteMemoryStore::open(":memory:")
        .unwrap()
        .with_embedder(Arc::new(FakeEmbedder { id: "fake/v1" }));
    store
        .save(user_scope(), "the stove is hot", &[], None)
        .await
        .unwrap();
    store
        .save(user_scope(), "the lake is cold", &[], None)
        .await
        .unwrap();

    // No keyword overlap with the hot record ("warm weather"), but the fake
    // embedder puts them on the same axis.
    let hits = store
        .search(&MemoryQuery::new().with_text("warm weather"))
        .await
        .unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0].record.content.contains("hot"));
}

#[tokio::test]
async fn mismatched_embedder_rows_stay_keyword_searchable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.db");

    // Saved under embedder v1...
    {
        let store = SqliteMemoryStore::open(&path)
            .unwrap()
            .with_embedder(Arc::new(FakeEmbedder { id: "fake/v1" }));
        store
            .save(user_scope(), "the stove is hot", &[], None)
            .await
            .unwrap();
    }

    // ...reopened under embedder v2: the old vector must not participate in
    // the semantic pass, but keyword recall still finds the record.
    let store = SqliteMemoryStore::open(&path)
        .unwrap()
        .with_embedder(Arc::new(FakeEmbedder { id: "fake/v2" }));

    let hits = store
        .search(&MemoryQuery::new().with_text("warm weather")) // semantic-only match
        .await
        .unwrap();
    assert!(hits.is_empty());

    let hits = store
        .search(&MemoryQuery::new().with_text("stove")) // keyword match
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
}
