// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Long-term memory implementations for the sweet AI agent framework.
//!
//! Re-exports the [`Memory`] trait and related types from `sweet-core` for
//! convenience, and provides:
//!
//! - [`SqliteMemoryStore`] (feature `sqlite`): a persistent store with hybrid
//!   recall — SQLite FTS5 keyword search fused with embedding cosine
//!   similarity when an [`Embedder`](sweet_core::Embedder) is attached.
//! - Tool factories ([`memory_tools`]) exposing save/search/update/delete to
//!   the model, scope-bound by the application.
//!
//! The in-memory implementation, [`EphemeralMemory`], lives in `sweet-core`
//! next to the trait (mirroring `InMemorySession`).

pub use sweet_core::{
    EphemeralMemory, Memory, MemoryError, MemoryHit, MemoryId, MemoryQuery, MemoryRecord,
    MemoryScope,
};

mod tools;

pub use tools::{
    memory_delete_tool, memory_save_tool, memory_search_tool, memory_tools, memory_update_tool,
    MemoryToolset,
};

#[cfg(feature = "sqlite")]
pub mod sqlite;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteMemoryStore;
