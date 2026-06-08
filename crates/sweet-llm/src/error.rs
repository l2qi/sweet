// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use reqwest::StatusCode;
use sweet_core::Error as CoreError;

#[derive(thiserror::Error, Debug)]
pub enum ProviderError {
    #[error("required environment variable `{var}` is not set")]
    MissingApiKey { var: &'static str },

    #[error("environment variable `{0}` is configured but missing or empty")]
    EmptyApiKey(String),

    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("HTTP {status}: {body}")]
    Http { status: StatusCode, body: String },

    #[error("failed to decode response: {0}")]
    Decode(#[from] serde_json::Error),

    #[error("response contained no choices")]
    EmptyResponse,

    #[error("unsupported role from upstream: {0}")]
    UnknownRole(String),
}

impl From<ProviderError> for CoreError {
    fn from(err: ProviderError) -> Self {
        CoreError::provider(err)
    }
}
