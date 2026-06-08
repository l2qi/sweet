// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use sweet_core::{execution_error, ToolError, ToolFn};

/// Fetch the contents of a URL via HTTP GET.
#[derive(Default, serde::Deserialize, schemars::JsonSchema, sweet_core::Tool)]
#[tool(
    description = "Fetch the contents of a URL via HTTP GET",
    risk = "readonly"
)]
pub struct HttpFetch {
    /// URL to fetch.
    pub url: String,
    /// Maximum number of bytes to read from the response body.
    #[serde(default)]
    pub max_bytes: Option<usize>,
}

#[sweet_core::async_trait]
impl ToolFn for HttpFetch {
    async fn run(self) -> Result<String, ToolError> {
        let resp = reqwest::get(&self.url).await.map_err(execution_error)?;
        let status = resp.status();
        let bytes = resp.bytes().await.map_err(execution_error)?;
        let body = if let Some(max) = self.max_bytes {
            String::from_utf8_lossy(&bytes[..bytes.len().min(max)]).into_owned()
        } else {
            String::from_utf8_lossy(&bytes).into_owned()
        };
        Ok(format!("Status: {status}\n\n{body}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_core::ToolSpec;

    #[tokio::test]
    async fn fetch_returns_status_and_body() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/hello"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("world"))
            .mount(&server)
            .await;

        let tool = ToolSpec::from(HttpFetch::default());
        let result = tool
            .call(serde_json::json!({"url": format!("{}/hello", server.uri())}))
            .await
            .unwrap();
        assert!(result.contains("Status: 200 OK"));
        assert!(result.contains("world"));
    }

    #[tokio::test]
    async fn fetch_respects_max_bytes() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("hello world"))
            .mount(&server)
            .await;

        let tool = ToolSpec::from(HttpFetch::default());
        let result = tool
            .call(serde_json::json!({"url": server.uri(), "max_bytes": 5}))
            .await
            .unwrap();
        assert!(result.contains("hello"));
        assert!(!result.contains("world"));
    }
}
