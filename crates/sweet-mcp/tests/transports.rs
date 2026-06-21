// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Hermetic integration tests for both MCP transports.
//!
//! Both tests drive the `sweet-mcp-mock-server` fixture crate, which exposes two
//! trivial tools (`echo`, `add`). The stdio tests launch it as a child
//! process; the HTTP tests launch it in `http` mode on a loopback port. No
//! network access and no external tooling are required.
//!
//! The fixture binary is built by `cargo test --workspace` (it is a workspace
//! member). Running `cargo test -p sweet-mcp` in isolation will not build it -
//! `mock_server_bin` asserts a clear message in that case.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;

use sweet_mcp::{McpProvider, ToolFilter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

/// Guard around the spawned mock-server child.
///
/// `Command::kill_on_drop(true)` on its own has been observed to leak the
/// child when the per-test tokio runtime tears down at the same instant the
/// `Child` is dropped - the SIGKILL is never delivered and the mock-server
/// orphans to launchd/init, keeping its inherited stdout pipe open and
/// hanging any pipeline (`cargo test ... | tail`) reading from it.
///
/// `start_kill` calls `libc::kill(pid, SIGKILL)` synchronously and does not
/// touch the runtime, so it works reliably from a `Drop` impl. We do not
/// `wait()` the zombie: the test binary's exit will reap it via init.
struct HttpServer(Child);

impl Drop for HttpServer {
    fn drop(&mut self) {
        let _ = self.0.start_kill();
    }
}

/// Locate the `sweet-mcp-mock-server` fixture binary next to this test binary.
fn mock_server_bin() -> PathBuf {
    // current_exe() is `<target>/<profile>/deps/<test>-<hash>`; the fixture
    // binary sits two levels up at `<target>/<profile>/sweet-mcp-mock-server`.
    let mut dir = std::env::current_exe().expect("current_exe");
    dir.pop();
    dir.pop();
    let bin = dir.join("sweet-mcp-mock-server");
    assert!(
        bin.exists(),
        "sweet-mcp-mock-server fixture not found at {} - run the suite with `cargo test --workspace`",
        bin.display()
    );
    bin
}

/// Connect to the fixture over stdio.
async fn connect_stdio(filter: &ToolFilter) -> McpProvider {
    McpProvider::connect_stdio(
        "mock",
        mock_server_bin().to_str().expect("utf-8 path"),
        &[],
        &HashMap::new(),
        filter,
    )
    .await
    .expect("connect_stdio to the mock server")
}

/// Spawn the fixture in HTTP mode; return the live child (wrapped in a kill
/// guard) and its base URL.
async fn spawn_http_server() -> (HttpServer, String) {
    let mut child = Command::new(mock_server_bin())
        .arg("http")
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn sweet-mcp-mock-server in http mode");

    let stdout = child.stdout.take().expect("child stdout pipe");
    let addr = BufReader::new(stdout)
        .lines()
        .next_line()
        .await
        .expect("read address line from fixture")
        .expect("fixture printed its bound address");

    (HttpServer(child), format!("http://{addr}/"))
}

fn tool_names(provider: &McpProvider) -> Vec<&str> {
    provider.specs().iter().map(|s| s.name.as_str()).collect()
}

#[tokio::test]
async fn stdio_lists_namespaced_tools() {
    let provider = connect_stdio(&ToolFilter::default()).await;
    let mut names = tool_names(&provider);
    names.sort_unstable();
    assert_eq!(names, ["mock__add", "mock__echo"]);
}

#[tokio::test]
async fn stdio_calls_tools_and_returns_results() {
    let provider = connect_stdio(&ToolFilter::default()).await;

    let add = provider
        .specs()
        .iter()
        .find(|s| s.name == "mock__add")
        .expect("mock__add tool");
    let sum = add
        .handler
        .call(serde_json::json!({ "a": 2, "b": 40 }))
        .await
        .expect("call mock__add");
    assert_eq!(sum, "42");

    let echo = provider
        .specs()
        .iter()
        .find(|s| s.name == "mock__echo")
        .expect("mock__echo tool");
    let echoed = echo
        .handler
        .call(serde_json::json!({ "text": "hello" }))
        .await
        .expect("call mock__echo");
    assert_eq!(echoed, "hello");
}

#[tokio::test]
async fn stdio_tool_filter_blocks_tool() {
    // The filter matches the raw (un-namespaced) tool name.
    let filter = ToolFilter::new(vec![], vec!["echo".into()]);
    let provider = connect_stdio(&filter).await;
    let names = tool_names(&provider);
    assert!(!names.contains(&"mock__echo"), "{names:?}");
    assert!(names.contains(&"mock__add"), "{names:?}");
}

#[tokio::test]
async fn http_lists_and_calls_tools() {
    let (_server, url) = spawn_http_server().await;
    let provider = McpProvider::connect_http("mock", &url, &HashMap::new(), &ToolFilter::default())
        .await
        .expect("connect_http to the mock server");

    let mut names = tool_names(&provider);
    names.sort_unstable();
    assert_eq!(names, ["mock__add", "mock__echo"]);

    let add = provider
        .specs()
        .iter()
        .find(|s| s.name == "mock__add")
        .expect("mock__add tool");
    let sum = add
        .handler
        .call(serde_json::json!({ "a": 1, "b": 2 }))
        .await
        .expect("call mock__add over http");
    assert_eq!(sum, "3");
}

#[tokio::test]
async fn http_tool_filter_blocks_tool() {
    let (_server, url) = spawn_http_server().await;
    let filter = ToolFilter::new(vec![], vec!["add".into()]);
    let provider = McpProvider::connect_http("mock", &url, &HashMap::new(), &filter)
        .await
        .expect("connect_http to the mock server");

    let names = tool_names(&provider);
    assert!(!names.contains(&"mock__add"), "{names:?}");
    assert!(names.contains(&"mock__echo"), "{names:?}");
}
