// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use sweet_core::{MemoryItem, Message, Session};
use sweet_session::InMemorySession;

#[test]
fn in_memory_session_push_and_messages() {
    let mut session = InMemorySession::new();
    session
        .push(MemoryItem::Message(Message::user("hello")))
        .unwrap();
    session
        .push(MemoryItem::Message(Message::assistant("hi")))
        .unwrap();

    assert_eq!(session.messages().len(), 2);
    assert_eq!(session.messages()[0].role, sweet_core::Role::User);
    assert_eq!(session.messages()[1].role, sweet_core::Role::Assistant);
}

#[test]
fn in_memory_session_clear() {
    let mut session = InMemorySession::new();
    session
        .push(MemoryItem::Message(Message::user("hello")))
        .unwrap();
    session.clear().unwrap();
    assert!(session.messages().is_empty());
    assert!(session.items().is_empty());
}

#[test]
fn in_memory_session_token_count() {
    let mut session = InMemorySession::new();
    session
        .push(MemoryItem::Message(Message::user("abcd".repeat(4))))
        .unwrap();
    assert_eq!(session.token_count(), 4);
}

#[cfg(feature = "sqlite")]
mod sqlite_tests {
    use super::*;
    use sweet_session::SqliteSession;

    #[test]
    fn sqlite_session_in_memory_push_and_messages() {
        let mut session = SqliteSession::new().expect("new in-memory session");
        session
            .push(MemoryItem::Message(Message::user("hello")))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::assistant("hi")))
            .unwrap();

        assert_eq!(session.messages().len(), 2);
        assert_eq!(session.messages()[0].role, sweet_core::Role::User);
        assert_eq!(session.messages()[1].role, sweet_core::Role::Assistant);
    }

    #[test]
    fn sqlite_session_clear() {
        let mut session = SqliteSession::new().expect("new in-memory session");
        session
            .push(MemoryItem::Message(Message::user("hello")))
            .unwrap();
        session.clear().unwrap();
        assert!(session.messages().is_empty());
        assert!(session.items().is_empty());
    }

    #[test]
    fn sqlite_session_rebuilds_from_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("session.db");

        {
            let mut session = SqliteSession::open(&path).expect("open session");
            session
                .push(MemoryItem::Message(Message::user("hello")))
                .unwrap();
            session
                .push(MemoryItem::Message(Message::assistant("hi")))
                .unwrap();
            assert_eq!(session.messages().len(), 2);
        }

        // Re-open the same file and verify items are reloaded.
        let session = SqliteSession::open(&path).expect("reopen session");
        assert_eq!(session.messages().len(), 2);
        assert_eq!(session.messages()[0].role, sweet_core::Role::User);
        assert_eq!(session.messages()[1].role, sweet_core::Role::Assistant);
    }

    #[test]
    fn sqlite_session_persists_across_instances() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("session.db");

        let mut session = SqliteSession::open(&path).expect("open session");
        session
            .push(MemoryItem::Message(Message::user("first")))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::assistant("second")))
            .unwrap();

        assert_eq!(session.messages().len(), 2);

        let session2 = SqliteSession::open(&path).expect("reopen session");
        assert_eq!(session2.messages().len(), 2);
        assert_eq!(session2.messages()[0].text_content(), "first");
    }
}
