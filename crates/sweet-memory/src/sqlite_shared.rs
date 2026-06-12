// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for the SQLite-backed memory implementations.

use std::time::Duration;

use rusqlite::{params_from_iter, Connection};
use sweet_core::{
    rrf_merge, Embedder, MemoryError, MemoryHit, MemoryQuery, MemoryRecord, MemoryScope,
};

/// How many candidates each ranking strategy contributes before fusion.
pub(crate) const CANDIDATE_LIMIT: usize = 50;

pub(crate) const RECORD_COLUMNS: &str =
    "id, scope_kind, scope_key, content, tags, source_session, created_at, updated_at";

/// [`RECORD_COLUMNS`] qualified for queries that join `memories_fts` (both
/// tables have `content`/`tags` columns).
pub(crate) const RECORD_COLUMNS_QUALIFIED: &str =
    "memories.id, memories.scope_kind, memories.scope_key, \
     memories.content, memories.tags, memories.source_session, memories.created_at, \
     memories.updated_at";

/// Open a SQLite connection at `path` with WAL mode and a 5-second busy
/// timeout. Both stores use this as the first step in their `open()`
/// constructors.
pub(crate) fn open_conn(path: impl AsRef<std::path::Path>) -> Result<Connection, MemoryError> {
    let conn = Connection::open(path).map_err(MemoryError::storage)?;
    conn.busy_timeout(Duration::from_secs(5))
        .map_err(MemoryError::storage)?;
    // WAL lets concurrent readers coexist with a writer.
    let _mode: String = conn
        .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
        .map_err(MemoryError::storage)?;
    Ok(conn)
}

/// FTS5 MATCH expression for free text: each whitespace token is quoted (with
/// internal quotes doubled) so model-generated punctuation can't be parsed as
/// FTS5 query syntax. Tokens are AND-ed (FTS5's implicit operator).
pub(crate) fn fts_match_expr(text: &str) -> String {
    text.split_whitespace()
        .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `AND (...)` scope filter plus its positional parameters; empty scopes mean
/// no filter.
pub(crate) fn scope_filter(scopes: &[MemoryScope]) -> (String, Vec<String>) {
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
pub(crate) fn row_to_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
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
pub(crate) fn collect_records(
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

/// Rust-side scope + tag filter for one candidate record. Query paths that
/// already filter scope in SQL pass through unchanged; the sqlite-vec KNN
/// path relies on this for scope enforcement (its scan can't carry extra
/// WHERE clauses).
pub(crate) fn filter_record(record: MemoryRecord, query: &MemoryQuery) -> Option<MemoryRecord> {
    let scope_ok = query.scopes.is_empty() || query.scopes.contains(&record.scope);
    let tags_ok = query.tags.iter().all(|t| record.tags.contains(t));
    (scope_ok && tags_ok).then_some(record)
}

/// Embed the query text, tagging the vector with the embedder's id.
/// `Ok(None)` when there is no embedder or it returns no vector (the search
/// degrades to keyword-only); an embedding *error* fails the search.
pub(crate) async fn embed_query(
    embedder: Option<&dyn Embedder>,
    text: &str,
) -> Result<Option<(Vec<f32>, String)>, MemoryError> {
    let Some(embedder) = embedder else {
        return Ok(None);
    };
    Ok(embedder
        .embed(&[text.to_string()])
        .await
        .map_err(|e| MemoryError::Embedding(e.into()))?
        .pop()
        .map(|v| (v, embedder.id().to_string())))
}

/// List mode (a query without text): newest first within the filters, with a
/// zero relevance score.
pub(crate) fn list_newest(
    conn: &Connection,
    query: &MemoryQuery,
) -> Result<Vec<MemoryHit>, MemoryError> {
    let (scope_clause, scope_params) = scope_filter(&query.scopes);
    let sql = format!(
        "SELECT {RECORD_COLUMNS} FROM memories WHERE 1=1{scope_clause}
         ORDER BY updated_at DESC, id DESC"
    );
    let mut stmt = conn.prepare(&sql).map_err(MemoryError::storage)?;
    let rows = stmt
        .query_map(params_from_iter(scope_params), row_to_record)
        .map_err(MemoryError::storage)?;
    Ok(collect_records(rows, query)?
        .into_iter()
        .take(query.limit)
        .map(|record| MemoryHit { record, score: 0.0 })
        .collect())
}

/// Keyword candidates from the FTS5 index, best (lowest bm25) first.
pub(crate) fn fts_candidates(
    conn: &Connection,
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
    let mut stmt = conn.prepare(&sql).map_err(MemoryError::storage)?;
    let params_iter = std::iter::once(match_expr).chain(scope_params);
    let rows = stmt
        .query_map(params_from_iter(params_iter), row_to_record)
        .map_err(MemoryError::storage)?;
    collect_records(rows, query)
}

/// Fuse the keyword and semantic rankings with Reciprocal Rank Fusion and
/// return the top `limit` hits.
pub(crate) fn fuse_hits(
    keyword: Vec<MemoryRecord>,
    vector: Vec<MemoryRecord>,
    limit: usize,
) -> Vec<MemoryHit> {
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
    rrf_merge(&rankings)
        .into_iter()
        .take(limit)
        .filter_map(|(id, score)| {
            by_id.iter().find(|r| r.id == id).map(|record| MemoryHit {
                record: record.clone(),
                score,
            })
        })
        .collect()
}

/// Little-endian f32 bytes; the inverse of [`blob_to_vec`].
pub(crate) fn vec_to_blob(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Deserialize a little-endian f32 blob back into a vector.
///
/// Only needed by the brute-force cosine path in `SqliteMemory` (feature
/// `sqlite`). `SqliteVecMemory` delegates vector I/O to the vec0 virtual
/// table and never reads raw blobs.
#[cfg(feature = "sqlite")]
pub(crate) fn blob_to_vec(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// The memories table DDL shared by both SQLite store implementations, plus
/// the FTS5 virtual table and its sync triggers.
pub(crate) const MEMORIES_TABLE_DDL: &str = "
    CREATE TABLE IF NOT EXISTS memories (
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
    END;";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_match_expr_escapes_query_syntax() {
        assert_eq!(fts_match_expr("foo bar"), "\"foo\" \"bar\"");
        assert_eq!(fts_match_expr("a\"b"), "\"a\"\"b\"");
        assert_eq!(fts_match_expr("NEAR(x) OR *"), "\"NEAR(x)\" \"OR\" \"*\"");
        assert_eq!(fts_match_expr("  "), "");
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn blob_roundtrip() {
        let v = vec![0.5f32, -1.25, 3.0];
        assert_eq!(blob_to_vec(&vec_to_blob(&v)), v);
    }
}
