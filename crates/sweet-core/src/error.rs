// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::error::Error as StdError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("provider error: {0}")]
    Provider(#[source] Box<dyn StdError + Send + Sync>),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("unsupported: {0}")]
    Unsupported(&'static str),

    #[error("tool error: {0}")]
    Tool(#[from] crate::tool::ToolError),

    #[error("unknown hook handler: {0}")]
    UnknownHookHandler(String),

    #[error("session error: {0}")]
    Session(#[from] crate::session::SessionError),
}

impl Error {
    pub fn provider<E>(err: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Error::Provider(Box::new(err))
    }
}

pub type Result<T> = std::result::Result<T, Error>;
