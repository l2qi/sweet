// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use chrono::Local;
use sweet_core::{ToolError, ToolFn};

/// Return the current local time as an ISO 8601 string.
#[derive(Default, serde::Deserialize, schemars::JsonSchema, sweet_core::Tool)]
#[tool(
    description = "Return the current local time as an ISO 8601 string",
    risk = "readonly"
)]
pub struct CurrentTime {}

#[sweet_core::async_trait]
impl ToolFn for CurrentTime {
    async fn run(self) -> Result<String, ToolError> {
        Ok(Local::now().to_rfc3339())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_core::ToolSpec;

    #[test]
    fn name_and_description_are_present() {
        let t = ToolSpec::from(CurrentTime::default());
        assert_eq!(t.name, "current_time");
        assert!(!t.description.is_empty());
    }

    #[tokio::test]
    async fn returns_iso8601() {
        let tool = ToolSpec::from(CurrentTime::default());
        let result = tool.call(serde_json::json!({})).await.unwrap();
        // Basic RFC3339 shape check.
        assert!(result.contains('T'));
        assert!(result.contains('+') || result.contains('-') || result.contains('Z'));
    }
}
