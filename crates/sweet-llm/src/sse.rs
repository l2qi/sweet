// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by providers that consume SSE (Server-Sent Events) streams.

/// Locate the end (exclusive) of the next SSE event in `buffer`.
///
/// SSE events are terminated by `\n\n` (or `\r\n\r\n`). Returns `None` if no
/// boundary has been received yet, in which case the caller should keep
/// reading bytes into the buffer and try again.
pub(crate) fn find_event_end(buffer: &[u8]) -> Option<usize> {
    let lf = buffer.windows(2).position(|w| w == b"\n\n").map(|i| i + 2);
    let crlf = buffer
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4);
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_none_when_no_boundary() {
        assert_eq!(find_event_end(b""), None);
        assert_eq!(find_event_end(b"data: partial"), None);
    }

    #[test]
    fn finds_lf_lf_boundary() {
        assert_eq!(find_event_end(b"data: x\n\nrest"), Some(9));
    }

    #[test]
    fn finds_crlf_crlf_boundary() {
        assert_eq!(find_event_end(b"data: x\r\n\r\nrest"), Some(11));
    }

    #[test]
    fn picks_earliest_when_both_present() {
        // \n\n appears at index 7 (end-exclusive 9), \r\n\r\n appears later.
        let buf = b"data: x\n\ndata: y\r\n\r\n";
        assert_eq!(find_event_end(buf), Some(9));
    }
}
