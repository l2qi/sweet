// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Session implementations for the sweet AI agent framework.
//!
//! Re-exports the [`Session`] trait and related types from `sweet-core` for
//! convenience, and provides concrete storage backends.

pub use sweet_core::{InMemorySession, MemoryItem, Session, SessionId};

#[cfg(feature = "sqlite")]
pub mod sqlite;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteSession;
