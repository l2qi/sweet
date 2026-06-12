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

    fn texts(messages: &[Message]) -> Vec<String> {
        messages.iter().map(|m| m.text_content()).collect()
    }

    fn push_users(session: &mut SqliteSession, texts: &[&str]) {
        for t in texts {
            session
                .push(MemoryItem::Message(Message::user(*t)))
                .unwrap();
        }
    }

    #[test]
    fn replace_range_archives_instead_of_deleting() {
        let mut session = SqliteSession::new().unwrap();
        push_users(&mut session, &["a", "b", "c"]);

        let mut summary = Message::user("summary");
        summary.compacted = true;
        session
            .replace_range(0..2, vec![MemoryItem::Message(summary)])
            .unwrap();

        // Live view: summary then c.
        assert_eq!(texts(&session.messages()), ["summary", "c"]);
        // Full transcript: originals in place, summary after the span.
        assert_eq!(
            texts(&session.full_messages().unwrap()),
            ["a", "b", "summary", "c"]
        );
        let archived_flags: Vec<bool> = session
            .full_items()
            .unwrap()
            .iter()
            .map(|(_, archived)| *archived)
            .collect();
        assert_eq!(archived_flags, [true, true, false, false]);
    }

    #[test]
    fn mid_range_replacement_keeps_orders() {
        let mut session = SqliteSession::new().unwrap();
        push_users(&mut session, &["a", "b", "c", "d"]);

        // Replace "b" in the middle (like tool-result clearing).
        session
            .replace_range(1..2, vec![MemoryItem::Message(Message::user("b'"))])
            .unwrap();

        assert_eq!(texts(&session.messages()), ["a", "b'", "c", "d"]);
        assert_eq!(
            texts(&session.full_messages().unwrap()),
            ["a", "b", "b'", "c", "d"]
        );
    }

    #[test]
    fn repeated_compaction_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.db");

        {
            let mut session = SqliteSession::open(&path).unwrap();
            push_users(&mut session, &["a", "b", "c"]);
            session
                .replace_range(0..2, vec![MemoryItem::Message(Message::user("s1"))])
                .unwrap();
            push_users(&mut session, &["d"]);
            session
                .replace_range(0..2, vec![MemoryItem::Message(Message::user("s2"))])
                .unwrap();
            assert_eq!(texts(&session.messages()), ["s2", "d"]);
        }

        let session = SqliteSession::open(&path).unwrap();
        assert_eq!(texts(&session.messages()), ["s2", "d"]);
        assert_eq!(
            texts(&session.full_messages().unwrap()),
            ["a", "b", "s1", "c", "s2", "d"]
        );

        // Appends after reopen land after everything, archived tail included.
        let mut session = session;
        push_users(&mut session, &["e"]);
        assert_eq!(texts(&session.messages()), ["s2", "d", "e"]);
        assert_eq!(
            *texts(&session.full_messages().unwrap()).last().unwrap(),
            "e"
        );
    }

    #[test]
    fn clear_deletes_archived_rows_too() {
        let mut session = SqliteSession::new().unwrap();
        push_users(&mut session, &["a", "b"]);
        session
            .replace_range(0..1, vec![MemoryItem::Message(Message::user("s"))])
            .unwrap();
        session.clear().unwrap();
        assert!(session.messages().is_empty());
        assert!(session.full_messages().unwrap().is_empty());
    }

    #[test]
    fn total_tokens_excludes_archived() {
        let mut session = SqliteSession::new().unwrap();
        let mut a = Message::user("a");
        a.token_count = Some(100);
        let mut b = Message::user("b");
        b.token_count = Some(7);
        session.push(MemoryItem::Message(a)).unwrap();
        session.push(MemoryItem::Message(b)).unwrap();
        assert_eq!(session.total_tokens(), 107);

        let mut summary = Message::user("s");
        summary.token_count = Some(5);
        session
            .replace_range(0..1, vec![MemoryItem::Message(summary)])
            .unwrap();
        assert_eq!(session.total_tokens(), 12);
    }
}
