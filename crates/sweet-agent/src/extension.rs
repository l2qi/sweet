// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use sweet_core::tool::ToolSpec;

use crate::commands::CommandSpec;
use crate::hooks::{HookCapability, HookEvent, ProcedureSpec};

/// A framework-level capability contributed by an extension.
#[derive(Clone)]
pub enum Capability {
    Tool(ToolSpec),
    Prompt(PromptSpec),
    Hook(HookCapability),
    Command(CommandSpec),
    Procedure(ProcedureSpec),
}

impl Capability {
    pub fn tool(tool: impl Into<ToolSpec>) -> Self {
        Self::Tool(tool.into())
    }

    pub fn hook(event: HookEvent, handler_id: impl Into<String>) -> Self {
        Self::Hook(HookCapability {
            event,
            handler_id: handler_id.into(),
        })
    }
}

/// Prompt text that should be composed into the agent's system instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSpec {
    pub id: String,
    pub text: String,
    pub activation: Activation,
}

impl PromptSpec {
    /// An always-on prompt, composed into the system instructions every turn.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            activation: Activation::Always,
        }
    }

    /// A prompt activated by a slash command of the given name (its `id`).
    /// The text is a template the runtime renders and submits as a user turn.
    pub fn command(name: impl Into<String>, text: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            id: name.clone(),
            text: text.into(),
            activation: Activation::ByCommand(name),
        }
    }
}

/// Declares when a prompt spec should be active.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Activation {
    /// Composed into the system instructions on every turn.
    Always,
    /// Triggered by a slash command whose name is carried here. Not composed
    /// into the system prompt; the runtime renders it into a user turn.
    ByCommand(String),
}

/// Something that contributes capabilities to an agent runtime.
pub trait CapabilityProvider: Send + Sync {
    fn id(&self) -> &str;
    fn capabilities(&self) -> Vec<Capability>;
}

/// Higher-level extension marker for capability providers.
pub trait Extension: CapabilityProvider + Send + Sync {}

impl<T> Extension for T where T: CapabilityProvider + Send + Sync {}

/// A named bundle of tools, installable on an agent as a [`CapabilityProvider`].
///
/// Tool-producing crates (`sweet-tools`, `sweet-mcp`, ...) stay `sweet-core`-only
/// and just yield [`ToolSpec`]s. The consumer groups them into a named bundle
/// and installs it with [`Agent::with_capability_provider`], so leaf tools
/// enter the agent through the same capability path as commands and hooks
/// rather than via bare `with_tool` calls.
///
/// [`Agent::with_capability_provider`]: crate::Agent::with_capability_provider
pub struct ToolCapabilities {
    id: String,
    tools: Vec<ToolSpec>,
}

impl ToolCapabilities {
    /// Create an empty bundle. `id` names the source for diagnostics.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tools: Vec::new(),
        }
    }

    /// Add a single tool to the bundle.
    pub fn with_tool(mut self, tool: impl Into<ToolSpec>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Add multiple tools to the bundle.
    pub fn with_tools<I>(mut self, tools: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<ToolSpec>,
    {
        self.tools.extend(tools.into_iter().map(Into::into));
        self
    }
}

impl CapabilityProvider for ToolCapabilities {
    fn id(&self) -> &str {
        &self.id
    }

    fn capabilities(&self) -> Vec<Capability> {
        self.tools.iter().cloned().map(Capability::Tool).collect()
    }
}

/// Registry for extensions that should be installed together.
#[derive(Default)]
pub struct ExtensionRegistry {
    extensions: Vec<Box<dyn Extension>>,
}

impl ExtensionRegistry {
    pub fn new() -> Self {
        Self {
            extensions: Vec::new(),
        }
    }

    pub fn register<E>(&mut self, extension: E)
    where
        E: Extension + 'static,
    {
        self.extensions.push(Box::new(extension));
    }

    pub fn capabilities(&self) -> Vec<Capability> {
        self.extensions
            .iter()
            .flat_map(|extension| extension.capabilities())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_spec_new_defaults_to_always_activation() {
        let prompt = PromptSpec::new("test", "instructions");

        assert_eq!(prompt.id, "test");
        assert_eq!(prompt.text, "instructions");
        assert_eq!(prompt.activation, Activation::Always);
    }

    #[test]
    fn tool_capabilities_expose_each_tool_as_a_tool_capability() {
        use crate::test_util::MockTool;

        let bundle = ToolCapabilities::new("mcp")
            .with_tool(MockTool::echoing("a"))
            .with_tools([MockTool::echoing("b"), MockTool::echoing("c")]);

        assert_eq!(bundle.id(), "mcp");
        let names: Vec<String> = bundle
            .capabilities()
            .into_iter()
            .map(|c| match c {
                Capability::Tool(t) => t.name,
                _ => panic!("expected Tool capability"),
            })
            .collect();
        assert_eq!(names, ["a", "b", "c"]);
    }
}
