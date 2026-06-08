// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Universal built-in tools for the sweet AI agent framework.

#[cfg(feature = "calculator")]
pub mod calculator;
#[cfg(feature = "http-fetch")]
pub mod http_fetch;
#[cfg(feature = "time")]
pub mod time;
#[cfg(feature = "web-search")]
pub mod web_search;

#[cfg(feature = "calculator")]
pub use calculator::Calculator;
#[cfg(feature = "http-fetch")]
pub use http_fetch::HttpFetch;
#[cfg(feature = "time")]
pub use time::CurrentTime;

#[cfg(feature = "web-search")]
pub use web_search::{BoxedWebSearch, SearchResult, WebSearch, WebSearchBackend, WebSearchError};

#[cfg(feature = "brave")]
pub use web_search::brave::BraveBackend;
#[cfg(feature = "tavily")]
pub use web_search::tavily::TavilyBackend;
