// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use rusqlite::{params, Connection, OptionalExtension};
use std::sync::Mutex;

use sweet_core::{MemoryItem, Message, Session, SessionError, SessionId};

/// SQLite-backed session.
///
/// Maintains an in-memory cache alongside the database for fast reads.
/// By default opens an in-memory transient store (`:memory:`); pass a file
/// path to `open()` for persistence.
///
/// Each database file stores a single session. Opening an existing file loads
/// the most recent session's items.
pub struct SqliteSession {
    id: SessionId,
    conn: Mutex<Connection>,
    cache: Vec<MemoryItem>,
}

impl SqliteSession {
    /// Create a new transient in-memory session.
    pub fn new() -> Result<Self, rusqlite::Error> {
        Self::open(":memory:")
    }

    /// Open (or create) a session stored at `path`.
    ///
    /// If the database already contains a session, the most recent one is loaded
    /// into the cache. Otherwise a new session row is created.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, rusqlite::Error> {
        Self::open_with_id(path, SessionId::new())
    }

    /// Open (or create) a session stored at `path`, using `id` when a new
    /// session row must be inserted.
    ///
    /// If the database already contains a session row, that row's id is used
    /// regardless of the supplied `id`.
    pub fn open_with_id(
        path: impl AsRef<std::path::Path>,
        id: SessionId,
    ) -> Result<Self, rusqlite::Error> {
        let mut conn = Connection::open(path)?;
        Self::init_schema(&conn)?;

        let tx = conn.transaction()?;
        let existing: Option<String> = tx
            .query_row(
                "SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;

        let id = match existing {
            Some(id_str) => id_str.parse::<SessionId>().map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?,
            None => {
                tx.execute(
                    "INSERT INTO sessions (id, created_at) VALUES (?1, datetime('now'))",
                    params![id.to_string()],
                )?;
                id
            }
        };
        tx.commit()?;

        let mut cache = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT kind, data, tokens FROM items WHERE session_id = ?1 ORDER BY rowid",
            )?;
            let rows = stmt.query_map(params![id.to_string()], |row| {
                let kind: String = row.get(0)?;
                let data: String = row.get(1)?;
                let _tokens: Option<i64> = row.get(2)?;
                Ok((kind, data))
            })?;

            for row in rows {
                let (kind, data) = row?;
                if kind == "message" {
                    let msg: Message = serde_json::from_str(&data).map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            0,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?;
                    cache.push(MemoryItem::Message(msg));
                }
            }
        }

        Ok(Self {
            id,
            conn: Mutex::new(conn),
            cache,
        })
    }

    fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS items (
                rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                kind       TEXT NOT NULL,
                data       TEXT NOT NULL,
                tokens     INTEGER,
                metadata   TEXT
            )",
            [],
        )?;
        Ok(())
    }
}

impl Session for SqliteSession {
    fn id(&self) -> &SessionId {
        &self.id
    }

    fn push(&mut self, item: MemoryItem) -> sweet_core::error::Result<()> {
        {
            let conn = self
                .conn
                .lock()
                .expect("sqlite connection mutex not poisoned");
            match &item {
                MemoryItem::Message(msg) => {
                    let data = serde_json::to_string(msg).map_err(SessionError::storage)?;
                    conn.execute(
                        "INSERT INTO items (session_id, kind, data, tokens) VALUES (?1, 'message', ?2, ?3)",
                        params![self.id.to_string(), data, msg.token_count.map(|t| t as i64)],
                    )
                    .map_err(SessionError::storage)?;
                }
            }
        }
        self.cache.push(item);
        Ok(())
    }

    fn items(&self) -> &[MemoryItem] {
        &self.cache
    }

    fn messages(&self) -> Vec<Message> {
        self.cache
            .iter()
            .map(|item| match item {
                MemoryItem::Message(msg) => msg.clone(),
            })
            .collect()
    }

    fn clear(&mut self) -> sweet_core::error::Result<()> {
        {
            let conn = self
                .conn
                .lock()
                .expect("sqlite connection mutex not poisoned");
            conn.execute(
                "DELETE FROM items WHERE session_id = ?1",
                params![self.id.to_string()],
            )
            .map_err(SessionError::storage)?;
        }
        self.cache.clear();
        Ok(())
    }

    fn token_count(&self) -> usize {
        self.cache
            .iter()
            .map(|item| match item {
                MemoryItem::Message(msg) => msg.text_content().chars().count() / 4,
            })
            .sum()
    }

    fn total_tokens(&self) -> usize {
        {
            let conn = self
                .conn
                .lock()
                .expect("sqlite connection mutex not poisoned");
            let total: i64 = conn
                .query_row(
                    "SELECT COALESCE(SUM(tokens), 0) FROM items WHERE session_id = ?1",
                    params![self.id.to_string()],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            total as usize
        }
    }

    fn context_size(&self) -> usize {
        sweet_core::last_context_size(&self.cache).unwrap_or_else(|| self.token_count())
    }

    fn replace_range(
        &mut self,
        range: std::ops::Range<usize>,
        replacement: Vec<MemoryItem>,
    ) -> sweet_core::error::Result<()> {
        // SQLite assigns rowids monotonically via AUTOINCREMENT, and we use
        // rowid order on read to reconstruct the cache. A naïve "delete the
        // affected rows and append replacements" leaves the new rows at the
        // tail, so on reopen the order would no longer match the in-memory
        // cache. Instead, build the new cache up front and wipe-and-rebuild
        // the session's rows so disk order matches cache order exactly.
        let mut new_cache: Vec<MemoryItem> =
            Vec::with_capacity(self.cache.len() + replacement.len() - (range.end - range.start));
        new_cache.extend_from_slice(&self.cache[..range.start]);
        new_cache.extend(replacement);
        new_cache.extend_from_slice(&self.cache[range.end..]);

        let mut conn = self
            .conn
            .lock()
            .expect("sqlite connection mutex not poisoned");
        let tx = conn.transaction().map_err(SessionError::storage)?;

        tx.execute(
            "DELETE FROM items WHERE session_id = ?1",
            params![self.id.to_string()],
        )
        .map_err(SessionError::storage)?;

        for item in &new_cache {
            insert_item(&tx, &self.id, item)?;
        }

        tx.commit().map_err(SessionError::storage)?;
        drop(conn);

        self.cache = new_cache;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn insert_item(
    tx: &rusqlite::Transaction<'_>,
    session_id: &SessionId,
    item: &MemoryItem,
) -> sweet_core::error::Result<()> {
    match item {
        MemoryItem::Message(msg) => {
            let data = serde_json::to_string(msg).map_err(SessionError::storage)?;
            tx.execute(
                "INSERT INTO items (session_id, kind, data, tokens) VALUES (?1, 'message', ?2, ?3)",
                params![
                    session_id.to_string(),
                    data,
                    msg.token_count.map(|t| t as i64)
                ],
            )
            .map_err(SessionError::storage)?;
        }
    }
    Ok(())
}
