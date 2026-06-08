// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingConfig {
    Enabled { budget_tokens: usize },
    Adaptive,
}

impl ThinkingConfig {
    pub fn enabled(budget_tokens: usize) -> Self {
        Self::Enabled { budget_tokens }
    }

    pub fn adaptive() -> Self {
        Self::Adaptive
    }
}
