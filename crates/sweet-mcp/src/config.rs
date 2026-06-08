// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use serde::Deserialize;

/// Top-level MCP config file (`mcp.json`).
///
/// ```json
/// {
///   "mcpServers": {
///     "git": {
///       "command": "uvx",
///       "args": ["mcp-server-git", "--repository", "."],
///       "env": {}
///     },
///     "github": {
///       "url": "https://api.githubcopilot.com/mcp/",
///       "headers": { "Authorization": "Bearer ${github_token}" }
///     }
///   }
/// }
/// ```
#[derive(Debug, Deserialize, Clone)]
pub struct McpConfig {
    #[serde(rename = "mcpServers")]
    pub servers: HashMap<String, McpServerConfig>,
}

impl McpConfig {
    pub fn from_json(json: &str) -> Result<Self, crate::McpError> {
        serde_json::from_str(json).map_err(|e| crate::McpError::Config(e.to_string()))
    }

    pub fn from_file(path: &std::path::Path) -> Result<Self, crate::McpError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    pub fn resolve_env_vars(&self, extra_env: &HashMap<String, String>) -> Self {
        let servers = self
            .servers
            .iter()
            .map(|(name, server)| {
                let mut server = server.clone();
                if let Some(env) = &mut server.env {
                    resolve_map(env, extra_env);
                }
                if let Some(headers) = &mut server.headers {
                    resolve_map(headers, extra_env);
                }
                (name.clone(), server)
            })
            .collect();
        Self { servers }
    }
}

fn resolve_map(map: &mut HashMap<String, String>, extra_env: &HashMap<String, String>) {
    for value in map.values_mut() {
        *value = resolve_string(value, extra_env);
    }
}

fn resolve_string(value: &str, extra_env: &HashMap<String, String>) -> String {
    let mut result = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '$' && chars.peek() == Some(&'{') {
            chars.next();
            let mut var_name = String::new();
            loop {
                match chars.next() {
                    Some('}') => break,
                    Some(c) => var_name.push(c),
                    None => {
                        result.push_str("${");
                        result.push_str(&var_name);
                        return result;
                    }
                }
            }
            if var_name.is_empty() {
                result.push_str("${}");
            } else if let Some(v) = extra_env.get(&var_name) {
                result.push_str(v);
            } else if let Ok(v) = std::env::var(&var_name) {
                result.push_str(&v);
            } else {
                result.push_str("${");
                result.push_str(&var_name);
                result.push('}');
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Configuration for a single MCP server.
///
/// A server uses either stdio (`command` + `args`) or HTTP (`url`).
#[derive(Debug, Deserialize, Clone)]
pub struct McpServerConfig {
    /// Command to launch the MCP server (stdio transport).
    #[serde(default)]
    pub command: Option<String>,
    /// Arguments for the command.
    #[serde(default)]
    pub args: Option<Vec<String>>,
    /// Environment variables to set when launching.
    #[serde(default)]
    pub env: Option<HashMap<String, String>>,
    /// URL for Streamable HTTP transport.
    #[serde(default)]
    pub url: Option<String>,
    /// Custom headers for HTTP transport.
    #[serde(default)]
    pub headers: Option<HashMap<String, String>>,
    /// Tool filter: allowlist of tool names. If empty, all tools are allowed.
    #[serde(default)]
    pub allow_tools: Option<Vec<String>>,
    /// Tool filter: blocklist of tool names.
    #[serde(default)]
    pub block_tools: Option<Vec<String>>,
}

impl McpServerConfig {
    pub fn is_stdio(&self) -> bool {
        self.command.is_some()
    }

    pub fn is_http(&self) -> bool {
        self.url.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stdio_server() {
        let json = r#"{"mcpServers":{"git":{"command":"uvx","args":["mcp-server-git"]}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        assert_eq!(config.servers.len(), 1);
        let git = &config.servers["git"];
        assert_eq!(git.command.as_deref(), Some("uvx"));
        assert_eq!(git.args.as_ref().unwrap().len(), 1);
        assert!(git.is_stdio());
        assert!(!git.is_http());
    }

    #[test]
    fn parse_http_server() {
        let json = r#"{"mcpServers":{"github":{"url":"https://example.com/mcp","headers":{"Authorization":"Bearer token"}}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        let github = &config.servers["github"];
        assert_eq!(github.url.as_deref(), Some("https://example.com/mcp"));
        assert!(github
            .headers
            .as_ref()
            .unwrap()
            .contains_key("Authorization"));
        assert!(github.is_http());
        assert!(!github.is_stdio());
    }

    #[test]
    fn parse_tool_filters() {
        let json = r#"{"mcpServers":{"svc":{"command":"cmd","allow_tools":["a","b"],"block_tools":["c"]}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        let svc = &config.servers["svc"];
        assert_eq!(svc.allow_tools.as_ref().unwrap().len(), 2);
        assert_eq!(svc.block_tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn parse_empty_config() {
        let json = r#"{"mcpServers":{}}"#;
        let config = McpConfig::from_json(json).unwrap();
        assert!(config.servers.is_empty());
    }

    #[test]
    fn parse_invalid_json_fails() {
        let result = McpConfig::from_json("not json");
        assert!(result.is_err());
    }

    #[test]
    fn parse_env_vars() {
        let json = r#"{"mcpServers":{"svc":{"command":"cmd","env":{"KEY":"value"}}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        let env = config.servers["svc"].env.as_ref().unwrap();
        assert_eq!(env.get("KEY").unwrap(), "value");
    }

    #[test]
    fn resolve_from_extra_env() {
        let json = r#"{"mcpServers":{"svc":{"command":"cmd","env":{"TOKEN":"${my_token}"}}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        let extra = HashMap::from([("my_token".to_string(), "secret123".to_string())]);
        let resolved = config.resolve_env_vars(&extra);
        assert_eq!(
            resolved.servers["svc"]
                .env
                .as_ref()
                .unwrap()
                .get("TOKEN")
                .unwrap(),
            "secret123"
        );
    }

    #[test]
    fn resolve_from_process_env() {
        let json = r#"{"mcpServers":{"svc":{"command":"cmd","env":{"HOME":"${HOME}"}}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        let resolved = config.resolve_env_vars(&HashMap::new());
        let val = resolved.servers["svc"]
            .env
            .as_ref()
            .unwrap()
            .get("HOME")
            .unwrap();
        assert!(!val.starts_with("${"));
        assert!(!val.is_empty());
    }

    #[test]
    fn resolve_not_found_left_as_is() {
        let json =
            r#"{"mcpServers":{"svc":{"command":"cmd","env":{"KEY":"${NONEXISTENT_VAR_XYZ}"}}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        let resolved = config.resolve_env_vars(&HashMap::new());
        assert_eq!(
            resolved.servers["svc"]
                .env
                .as_ref()
                .unwrap()
                .get("KEY")
                .unwrap(),
            "${NONEXISTENT_VAR_XYZ}"
        );
    }

    #[test]
    fn resolve_no_placeholders_unchanged() {
        let json = r#"{"mcpServers":{"svc":{"command":"cmd","env":{"KEY":"plain_value"}}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        let resolved = config.resolve_env_vars(&HashMap::new());
        assert_eq!(
            resolved.servers["svc"]
                .env
                .as_ref()
                .unwrap()
                .get("KEY")
                .unwrap(),
            "plain_value"
        );
    }

    #[test]
    fn resolve_multiple_vars_in_one_value() {
        let json = r#"{"mcpServers":{"svc":{"command":"cmd","env":{"PATH":"${a}:${b}"}}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        let extra = HashMap::from([
            ("a".to_string(), "foo".to_string()),
            ("b".to_string(), "bar".to_string()),
        ]);
        let resolved = config.resolve_env_vars(&extra);
        assert_eq!(
            resolved.servers["svc"]
                .env
                .as_ref()
                .unwrap()
                .get("PATH")
                .unwrap(),
            "foo:bar"
        );
    }

    #[test]
    fn resolve_lone_dollar_unchanged() {
        let json = r#"{"mcpServers":{"svc":{"command":"cmd","env":{"PRICE":"$5"}}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        let resolved = config.resolve_env_vars(&HashMap::new());
        assert_eq!(
            resolved.servers["svc"]
                .env
                .as_ref()
                .unwrap()
                .get("PRICE")
                .unwrap(),
            "$5"
        );
    }

    #[test]
    fn resolve_extra_env_takes_priority_over_process_env() {
        let json = r#"{"mcpServers":{"svc":{"command":"cmd","env":{"VAL":"${HOME}"}}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        let extra = HashMap::from([("HOME".to_string(), "override".to_string())]);
        let resolved = config.resolve_env_vars(&extra);
        assert_eq!(
            resolved.servers["svc"]
                .env
                .as_ref()
                .unwrap()
                .get("VAL")
                .unwrap(),
            "override"
        );
    }

    #[test]
    fn resolve_headers() {
        let json = r#"{"mcpServers":{"svc":{"url":"http://x","headers":{"Authorization":"Bearer ${github_token}"}}}}"#;
        let config = McpConfig::from_json(json).unwrap();
        let extra = HashMap::from([("github_token".to_string(), "ghp_abc".to_string())]);
        let resolved = config.resolve_env_vars(&extra);
        assert_eq!(
            resolved.servers["svc"]
                .headers
                .as_ref()
                .unwrap()
                .get("Authorization")
                .unwrap(),
            "Bearer ghp_abc"
        );
    }
}
