// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;

use http::{HeaderName, HeaderValue};
use rmcp::model::CallToolRequestParams;
use rmcp::service::{RunningService, ServiceExt};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::ClientHandler;
use sweet_core::tool::{ToolHandler, ToolSpec};

use crate::error::McpError;

/// Filter for which tools to expose from an MCP server.
#[derive(Debug, Clone, Default)]
pub struct ToolFilter {
    pub allow: Vec<String>,
    pub block: Vec<String>,
}

impl ToolFilter {
    pub fn new(allow: Vec<String>, block: Vec<String>) -> Self {
        Self { allow, block }
    }

    pub fn allows(&self, tool_name: &str) -> bool {
        if self.block.iter().any(|b| b == tool_name) {
            return false;
        }
        if !self.allow.is_empty() && !self.allow.iter().any(|a| a == tool_name) {
            return false;
        }
        true
    }
}

#[derive(Clone)]
struct NoopHandler;

#[async_trait::async_trait]
impl ClientHandler for NoopHandler {}

type Client = RunningService<rmcp::service::RoleClient, NoopHandler>;

/// A live connection to one MCP server, exposing its tools as [`ToolSpec`]s.
///
/// On construction, connects to the server, lists tools, and creates a
/// [`ToolSpec`] for each (namespaced as `{server}__{tool}`). Call [`specs`]
/// to register them on an agent with `Agent::with_tool`.
///
/// [`specs`]: McpProvider::specs
pub struct McpProvider {
    tool_specs: Vec<ToolSpec>,
    /// Held for the provider's lifetime: keeps the MCP session — and, for
    /// stdio, the child process — alive. Dropping it closes the connection.
    #[allow(dead_code)]
    connection: Client,
}

impl McpProvider {
    /// Connect to an MCP server over stdio, list tools, and build the provider.
    pub async fn connect_stdio(
        server_name: &str,
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
        filter: &ToolFilter,
    ) -> Result<Self, McpError> {
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args);
        for (k, v) in env {
            cmd.env(k, v);
        }

        // Discard the child's stderr: inheriting it (rmcp's default) would
        // corrupt a TUI consumer's terminal. `spawn` returns no stderr handle
        // when it is `Stdio::null()`.
        let (transport, _) = TokioChildProcess::builder(cmd)
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| McpError::Transport(format!("failed to launch '{command}': {e}")))?;

        let client = NoopHandler
            .serve(transport)
            .await
            .map_err(|e| McpError::Transport(format!("stdio handshake failed: {e}")))?;

        Self::from_client(server_name, client, filter).await
    }

    /// Connect to an MCP server over Streamable HTTP, list tools, and build the provider.
    pub async fn connect_http(
        server_name: &str,
        url: &str,
        headers: &HashMap<String, String>,
        filter: &ToolFilter,
    ) -> Result<Self, McpError> {
        let mut config = StreamableHttpClientTransportConfig::with_uri(url);
        if !headers.is_empty() {
            let mut custom = HashMap::new();
            for (k, v) in headers {
                let name = HeaderName::try_from(k)
                    .map_err(|e| McpError::Config(format!("invalid header name '{k}': {e}")))?;
                let value = HeaderValue::from_str(v).map_err(|e| {
                    McpError::Config(format!("invalid header value for '{k}': {e}"))
                })?;
                custom.insert(name, value);
            }
            config = config.custom_headers(custom);
        }
        let transport = StreamableHttpClientTransport::<reqwest::Client>::from_config(config);

        let client = NoopHandler
            .serve(transport)
            .await
            .map_err(|e| McpError::Transport(format!("HTTP handshake failed for {url}: {e}")))?;

        Self::from_client(server_name, client, filter).await
    }

    async fn from_client(
        server_name: &str,
        client: Client,
        filter: &ToolFilter,
    ) -> Result<Self, McpError> {
        let peer = client.peer().clone();
        let tools = client
            .list_all_tools()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;

        let caller: Arc<dyn McpCaller> = Arc::new(PeerWrapper(peer));

        let tool_specs = tools
            .into_iter()
            .filter(|t| filter.allows(&t.name))
            .map(|t| {
                let namespaced = format!("{}__{}", server_name, t.name);
                let schema =
                    serde_json::to_value(&*t.input_schema).unwrap_or(serde_json::json!({}));
                let desc = t.description.map(|d| d.into_owned()).unwrap_or_default();
                let handler = McpHandler {
                    client: caller.clone(),
                    tool_name: t.name.clone().into_owned(),
                };
                ToolSpec::new(namespaced, desc, schema, handler)
            })
            .collect();

        Ok(Self {
            tool_specs,
            connection: client,
        })
    }

    pub fn specs(&self) -> &[ToolSpec] {
        &self.tool_specs
    }
}

trait McpCaller: Send + Sync + 'static {
    fn call(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, McpError>> + Send + '_>>;
}

struct PeerWrapper(rmcp::service::Peer<rmcp::service::RoleClient>);

impl McpCaller for PeerWrapper {
    fn call(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, McpError>> + Send + '_>>
    {
        let name = name.to_string();
        let params = CallToolRequestParams::new(name)
            .with_arguments(args.as_object().cloned().unwrap_or_default());
        let peer = self.0.clone();
        Box::pin(async move {
            let result = peer
                .call_tool(params)
                .await
                .map_err(|e| McpError::ToolCall(e.to_string()))?;
            // Only text content is surfaced to the model; image and embedded
            // resource blocks are dropped.
            let text = result
                .content
                .iter()
                .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                .collect::<Vec<_>>()
                .join("\n");
            // A tool result flagged `is_error` is a failed call, not output.
            if result.is_error.unwrap_or(false) {
                Err(McpError::ToolCall(text))
            } else {
                Ok(text)
            }
        })
    }
}

struct McpHandler {
    client: Arc<dyn McpCaller>,
    tool_name: String,
}

#[async_trait::async_trait]
impl ToolHandler for McpHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, sweet_core::ToolError> {
        self.client
            .call(&self.tool_name, args)
            .await
            .map_err(|e| sweet_core::ToolError::Execution(e.to_string().into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_filter_allows_all_when_empty() {
        let filter = ToolFilter::default();
        assert!(filter.allows("anything"));
        assert!(filter.allows("other"));
    }

    #[test]
    fn tool_filter_blocks_specific() {
        let filter = ToolFilter::new(vec![], vec!["dangerous".into()]);
        assert!(filter.allows("safe"));
        assert!(!filter.allows("dangerous"));
    }

    #[test]
    fn tool_filter_allowlist_only() {
        let filter = ToolFilter::new(vec!["a".into(), "b".into()], vec![]);
        assert!(filter.allows("a"));
        assert!(filter.allows("b"));
        assert!(!filter.allows("c"));
    }

    #[test]
    fn tool_filter_block_overrides_allow() {
        let filter = ToolFilter::new(vec!["a".into(), "b".into()], vec!["a".into()]);
        assert!(!filter.allows("a"));
        assert!(filter.allows("b"));
    }
}
