// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! SQLite-backed [`Memory`] with hybrid keyword + semantic recall.

use std::sync::{Arc, Mutex};

use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use sweet_core::{
    cosine_similarity, rrf_merge, unix_now, Embedder, Memory, MemoryError, MemoryHit, MemoryId,
    MemoryQuery, MemoryRecord, MemoryScope,
};

use async_trait::async_trait;

/// How many candidates each ranking strategy contributes before fusion.
const CANDIDATE_LIMIT: usize = 50;

const RECORD_COLUMNS: &str =
    "id, scope_kind, scope_key, content, tags, source_session, created_at, updated_at";

/// [`RECORD_COLUMNS`] qualified for queries that join `memories_fts` (both
/// tables have `content`/`tags` columns).
const RECORD_COLUMNS_QUALIFIED: &str = "memories.id, memories.scope_kind, memories.scope_key, \
     memories.content, memories.tags, memories.source_session, memories.created_at, \
     memories.updated_at";

/// Persistent [`Memory`] store: one database file holds every scope.
///
/// Keyword recall uses an external-content FTS5 index (kept in sync by
/// triggers, so it survives any write path). With an [`Embedder`] attached,
/// saves are embedded and searches fuse a brute-force cosine ranking with the
/// keyword ranking via Reciprocal Rank Fusion. Vectors are tagged with
/// [`Embedder::id`]; rows embedded by a different embedder simply don't
/// participate in the semantic pass (they remain keyword-searchable).
///
/// Opens in WAL mode with a busy timeout so multiple processes can share the
/// file.
pub struct SqliteMemoryStore {
    conn: Mutex<Connection>,
    embedder: Option<Arc<dyn Embedder>>,
}

impl SqliteMemoryStore {
    /// Open (or create) the store at `path`. Pass `":memory:"` for a
    /// transient store.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, MemoryError> {
        let conn = Connection::open(path).map_err(MemoryError::storage)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(MemoryError::storage)?;
        // WAL lets concurrent readers coexist with a writer (returns the
        // resulting mode, hence query_row; ":memory:" stays "memory").
        let _mode: String = conn
            .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
            .map_err(MemoryError::storage)?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedder: None,
        })
    }

    /// Attach an embedder; subsequent saves are embedded and searches add a
    /// semantic ranking. Embedding failure during save degrades that record
    /// to keyword-only recall rather than failing the save.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    fn init_schema(conn: &Connection) -> Result<(), MemoryError> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id              TEXT PRIMARY KEY,
                scope_kind      TEXT NOT NULL,
                scope_key       TEXT NOT NULL,
                content         TEXT NOT NULL,
                tags            TEXT NOT NULL DEFAULT '[]',
                source_session  TEXT,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL,
                embedding       BLOB,
                embedding_model TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories(scope_kind, scope_key);

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, tags, content='memories', content_rowid='rowid');

            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, tags)
                VALUES (new.rowid, new.content, new.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                VALUES ('delete', old.rowid, old.content, old.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                VALUES ('delete', old.rowid, old.content, old.tags);
                INSERT INTO memories_fts(rowid, content, tags)
                VALUES (new.rowid, new.content, new.tags);
            END;",
        )
        .map_err(MemoryError::storage)
    }

    /// Embed `text` if an embedder is attached; `None` (with a warning) when
    /// embedding fails — memory durability beats vector coverage.
    async fn try_embed(&self, text: &str) -> Option<Vec<f32>> {
        let embedder = self.embedder.as_ref()?;
        match embedder.embed(&[text.to_string()]).await {
            Ok(mut vectors) => vectors.pop(),
            Err(err) => {
                tracing::warn!("embedding failed, saving keyword-only memory: {err}");
                None
            }
        }
    }

    /// Keyword candidates, best (lowest bm25) first.
    fn fts_candidates(
        &self,
        text: &str,
        query: &MemoryQuery,
    ) -> Result<Vec<MemoryRecord>, MemoryError> {
        let match_expr = fts_match_expr(text);
        if match_expr.is_empty() {
            return Ok(Vec::new());
        }
        let (scope_clause, scope_params) = scope_filter(&query.scopes);
        let sql = format!(
            "SELECT {RECORD_COLUMNS_QUALIFIED} FROM memories_fts
             JOIN memories ON memories.rowid = memories_fts.rowid
             WHERE memories_fts MATCH ?1{scope_clause}
             ORDER BY bm25(memories_fts) LIMIT {CANDIDATE_LIMIT}"
        );
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = conn.prepare(&sql).map_err(MemoryError::storage)?;
        let params_iter = std::iter::once(match_expr).chain(scope_params);
        let rows = stmt
            .query_map(params_from_iter(params_iter), row_to_record)
            .map_err(MemoryError::storage)?;
        collect_records(rows, query)
    }

    /// Semantic candidates, most similar first.
    fn vector_candidates(
        &self,
        query_vector: &[f32],
        embedder_id: &str,
        query: &MemoryQuery,
    ) -> Result<Vec<MemoryRecord>, MemoryError> {
        let (scope_clause, scope_params) = scope_filter(&query.scopes);
        let sql = format!(
            "SELECT {RECORD_COLUMNS}, embedding FROM memories
             WHERE embedding IS NOT NULL AND embedding_model = ?1{scope_clause}"
        );
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = conn.prepare(&sql).map_err(MemoryError::storage)?;
        let params_iter = std::iter::once(embedder_id.to_string()).chain(scope_params);
        let rows = stmt
            .query_map(params_from_iter(params_iter), |row| {
                let record = row_to_record(row)?;
                let blob: Vec<u8> = row.get(8)?;
                Ok((record, blob))
            })
            .map_err(MemoryError::storage)?;

        let mut scored: Vec<(MemoryRecord, f32)> = Vec::new();
        for row in rows {
            let (record, blob) = row.map_err(MemoryError::storage)?;
            if let Some(record) = filter_record(record, query) {
                let similarity = cosine_similarity(query_vector, &blob_to_vec(&blob));
                scored.push((record, similarity));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(CANDIDATE_LIMIT);
        Ok(scored.into_iter().map(|(record, _)| record).collect())
    }
}

#[async_trait]
impl Memory for SqliteMemoryStore {
    async fn save(
        &self,
        scope: MemoryScope,
        content: &str,
        tags: &[String],
        source_session: Option<&str>,
    ) -> Result<MemoryRecord, MemoryError> {
        // Embed before taking the lock; the guard can't be held across await.
        let embedding = self.try_embed(content).await;
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
        let tags_json = serde_json::to_string(&record.tags).map_err(MemoryError::storage)?;
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT INTO memories
             (id, scope_kind, scope_key, content, tags, source_session,
              created_at, updated_at, embedding, embedding_model)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                record.id.to_string(),
                record.scope.kind(),
                record.scope.key(),
                record.content,
                tags_json,
                record.source_session,
                record.created_at,
                record.updated_at,
                embedding.as_deref().map(vec_to_blob),
                embedding
                    .is_some()
                    .then(|| self.embedder.as_ref().map(|e| e.id().to_string()))
                    .flatten(),
            ],
        )
        .map_err(MemoryError::storage)?;
        Ok(record)
    }

    async fn get(&self, id: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        conn.query_row(
            &format!("SELECT {RECORD_COLUMNS} FROM memories WHERE id = ?1"),
            params![id.to_string()],
            row_to_record,
        )
        .optional()
        .map_err(MemoryError::storage)
    }

    async fn search(&self, query: &MemoryQuery) -> Result<Vec<MemoryHit>, MemoryError> {
        let text = query.text.as_deref().filter(|t| !t.trim().is_empty());

        let Some(text) = text else {
            // List mode: newest first within the filters.
            let (scope_clause, scope_params) = scope_filter(&query.scopes);
            let sql = format!(
                "SELECT {RECORD_COLUMNS} FROM memories WHERE 1=1{scope_clause}
                 ORDER BY updated_at DESC, id DESC"
            );
            let records = {
                let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
                let mut stmt = conn.prepare(&sql).map_err(MemoryError::storage)?;
                let rows = stmt
                    .query_map(params_from_iter(scope_params), row_to_record)
                    .map_err(MemoryError::storage)?;
                collect_records(rows, query)?
            };
            return Ok(records
                .into_iter()
                .take(query.limit)
                .map(|record| MemoryHit { record, score: 0.0 })
                .collect());
        };

        // Embed the query before any lock is taken.
        let query_embedding = match &self.embedder {
            Some(embedder) => Some((
                embedder
                    .embed(&[text.to_string()])
                    .await
                    .map_err(|e| MemoryError::Embedding(e.into()))?
                    .pop()
                    .unwrap_or_default(),
                embedder.id().to_string(),
            )),
            None => None,
        };

        let keyword = self.fts_candidates(text, query)?;
        let vector = match &query_embedding {
            Some((qv, embedder_id)) => self.vector_candidates(qv, embedder_id, query)?,
            None => Vec::new(),
        };

        let mut by_id: Vec<MemoryRecord> = Vec::new();
        for record in keyword.iter().chain(vector.iter()) {
            if !by_id.iter().any(|r| r.id == record.id) {
                by_id.push(record.clone());
            }
        }
        let rankings = [
            keyword.into_iter().map(|r| r.id).collect::<Vec<_>>(),
            vector.into_iter().map(|r| r.id).collect::<Vec<_>>(),
        ];
        let fused = rrf_merge(&rankings);

        Ok(fused
            .into_iter()
            .take(query.limit)
            .filter_map(|(id, score)| {
                by_id.iter().find(|r| r.id == id).map(|record| MemoryHit {
                    record: record.clone(),
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
        let mut record = self
            .get(id)
            .await?
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;

        // Re-embed only when the content changes.
        let new_embedding = match content {
            Some(text) => Some(self.try_embed(text).await),
            None => None,
        };

        if let Some(text) = content {
            record.content = text.to_string();
        }
        if let Some(tags) = tags {
            record.tags = tags.to_vec();
        }
        record.updated_at = unix_now();
        let tags_json = serde_json::to_string(&record.tags).map_err(MemoryError::storage)?;

        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let updated = match new_embedding {
            Some(embedding) => conn
                .execute(
                    "UPDATE memories SET content = ?2, tags = ?3, updated_at = ?4,
                     embedding = ?5, embedding_model = ?6 WHERE id = ?1",
                    params![
                        id.to_string(),
                        record.content,
                        tags_json,
                        record.updated_at,
                        embedding.as_deref().map(vec_to_blob),
                        embedding
                            .is_some()
                            .then(|| self.embedder.as_ref().map(|e| e.id().to_string()))
                            .flatten(),
                    ],
                )
                .map_err(MemoryError::storage)?,
            None => conn
                .execute(
                    "UPDATE memories SET content = ?2, tags = ?3, updated_at = ?4 WHERE id = ?1",
                    params![id.to_string(), record.content, tags_json, record.updated_at],
                )
                .map_err(MemoryError::storage)?,
        };
        if updated == 0 {
            return Err(MemoryError::NotFound(id.to_string()));
        }
        Ok(record)
    }

    async fn delete(&self, id: &MemoryId) -> Result<bool, MemoryError> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let deleted = conn
            .execute(
                "DELETE FROM memories WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(MemoryError::storage)?;
        Ok(deleted > 0)
    }
}

/// FTS5 MATCH expression for free text: each whitespace token is quoted (with
/// internal quotes doubled) so model-generated punctuation can't be parsed as
/// FTS5 query syntax. Tokens are AND-ed (FTS5's implicit operator).
fn fts_match_expr(text: &str) -> String {
    text.split_whitespace()
        .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `AND (...)` scope filter plus its positional parameters; empty scopes mean
/// no filter.
fn scope_filter(scopes: &[MemoryScope]) -> (String, Vec<String>) {
    if scopes.is_empty() {
        return (String::new(), Vec::new());
    }
    let conditions = vec!["(scope_kind = ? AND scope_key = ?)"; scopes.len()].join(" OR ");
    let params = scopes
        .iter()
        .flat_map(|s| [s.kind().to_string(), s.key().to_string()])
        .collect();
    (format!(" AND ({conditions})"), params)
}

/// Map a row selected with [`RECORD_COLUMNS`] to a record. Unknown scope
/// kinds or malformed ids/tags surface as conversion errors rather than being
/// silently skipped.
fn row_to_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
    let conversion = |e: Box<dyn std::error::Error + Send + Sync>, idx: usize| {
        rusqlite::Error::FromSqlConversionFailure(idx, rusqlite::types::Type::Text, e)
    };
    let id: String = row.get(0)?;
    let scope_kind: String = row.get(1)?;
    let scope_key: String = row.get(2)?;
    let tags: String = row.get(4)?;
    Ok(MemoryRecord {
        id: id.parse().map_err(|e| conversion(Box::new(e), 0))?,
        scope: MemoryScope::from_parts(&scope_kind, &scope_key)
            .ok_or_else(|| conversion(format!("unknown scope kind: {scope_kind}").into(), 1))?,
        content: row.get(3)?,
        tags: serde_json::from_str(&tags).map_err(|e| conversion(Box::new(e), 4))?,
        source_session: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
    })
}

/// Drain mapped rows, applying the Rust-side tag filter (tags live as a JSON
/// column; filtering here keeps the SQL simple).
fn collect_records(
    rows: impl Iterator<Item = rusqlite::Result<MemoryRecord>>,
    query: &MemoryQuery,
) -> Result<Vec<MemoryRecord>, MemoryError> {
    let mut records = Vec::new();
    for row in rows {
        let record = row.map_err(MemoryError::storage)?;
        if let Some(record) = filter_record(record, query) {
            records.push(record);
        }
    }
    Ok(records)
}

fn filter_record(record: MemoryRecord, query: &MemoryQuery) -> Option<MemoryRecord> {
    query
        .tags
        .iter()
        .all(|t| record.tags.contains(t))
        .then_some(record)
}

/// Little-endian f32 bytes; the inverse of [`blob_to_vec`].
fn vec_to_blob(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn blob_to_vec(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts5_is_available_in_bundled_sqlite() {
        // Guards against a rusqlite/libsqlite3-sys bump silently dropping
        // -DSQLITE_ENABLE_FTS5 from the bundled build.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE VIRTUAL TABLE t USING fts5(content)")
            .expect("FTS5 must be compiled into the bundled sqlite");
    }

    #[test]
    fn fts_match_expr_escapes_query_syntax() {
        assert_eq!(fts_match_expr("foo bar"), "\"foo\" \"bar\"");
        assert_eq!(fts_match_expr("a\"b"), "\"a\"\"b\"");
        assert_eq!(fts_match_expr("NEAR(x) OR *"), "\"NEAR(x)\" \"OR\" \"*\"");
        assert_eq!(fts_match_expr("  "), "");
    }

    #[test]
    fn blob_roundtrip() {
        let v = vec![0.5f32, -1.25, 3.0];
        assert_eq!(blob_to_vec(&vec_to_blob(&v)), v);
    }
}
