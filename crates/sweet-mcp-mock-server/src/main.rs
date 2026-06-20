// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Mock MCP server for `sweet-mcp`'s hermetic integration tests.
//!
//! Exposes two trivial tools - `echo` and `add` - over either transport:
//!
//! * no args (or `stdio`): serve over stdio (stdin/stdout). Used to exercise
//!   `McpProvider::connect_stdio` with a real child process.
//! * `http`: bind an ephemeral loopback port, print `<addr>` on stdout, then
//!   serve Streamable HTTP. Used to exercise `McpProvider::connect_http`.
//!
//! This crate exists purely as a test fixture; it is never published.

use std::io::Write;
use std::sync::Arc;

use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{stdio, StreamableHttpService};
use rmcp::{ErrorData, RoleServer, ServerHandler, ServiceExt};

#[derive(Clone)]
struct MockServer;

/// Coerce a `json!` literal into the `Arc<JsonObject>` shape `Tool::new` wants.
fn object_schema(value: serde_json::Value) -> Arc<serde_json::Map<String, serde_json::Value>> {
    Arc::new(
        value
            .as_object()
            .cloned()
            .expect("tool schema must be a JSON object"),
    )
}

impl ServerHandler for MockServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let echo = Tool::new(
            "echo",
            "Echo back the provided text.",
            object_schema(serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            })),
        );
        let add = Tool::new(
            "add",
            "Add two integers.",
            object_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "a": { "type": "integer" },
                    "b": { "type": "integer" }
                },
                "required": ["a", "b"]
            })),
        );
        Ok(ListToolsResult::with_all_items(vec![echo, add]))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let args = request.arguments.unwrap_or_default();
        match request.name.as_ref() {
            "echo" => {
                let text = args
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            "add" => {
                let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or_default();
                let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or_default();
                Ok(CallToolResult::success(vec![Content::text(
                    (a + b).to_string(),
                )]))
            }
            other => Err(ErrorData::invalid_params(
                format!("unknown tool: {other}"),
                None,
            )),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    match std::env::args().nth(1).as_deref() {
        None | Some("stdio") => {
            let service = MockServer.serve(stdio()).await?;
            service.waiting().await?;
        }
        Some("http") => {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let addr = listener.local_addr()?;
            let service = StreamableHttpService::new(
                || Ok(MockServer),
                Arc::new(LocalSessionManager::default()),
                Default::default(),
            );
            let app = axum::Router::new().fallback_service(service);

            // Report the bound address so the test harness can connect, then
            // flush - stdout is block-buffered when piped to the parent.
            let mut stdout = std::io::stdout();
            writeln!(stdout, "{addr}")?;
            stdout.flush()?;

            axum::serve(listener, app).await?;
        }
        Some(other) => return Err(format!("unknown mode: {other}").into()),
    }
    Ok(())
}
