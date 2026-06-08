// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;

use sweet_core::model::Model;

use crate::extension::{Activation, Capability};

/// Object-safe command execution context.
///
/// Commands are generic runtime actions. User interfaces decide how commands
/// are invoked, such as a `/name args` slash-command syntax.
pub trait CommandContext: Send {
    fn session(&self) -> &dyn sweet_core::Session;
    fn session_mut(&mut self) -> &mut dyn sweet_core::Session;
    fn replace_session(&mut self, session: Box<dyn sweet_core::Session>);
    fn model(&self) -> &dyn Model;
}

/// Executable handler for a command capability.
#[async_trait]
pub trait CommandHandler: Send + Sync {
    async fn handle(
        &self,
        args: &str,
        ctx: &mut dyn CommandContext,
    ) -> sweet_core::Result<Option<String>>;
}

/// Describes a command and carries its executable handler.
#[derive(Clone)]
pub struct CommandSpec {
    pub name: String,
    pub description: String,
    pub usage: String,
    pub handler: Arc<dyn CommandHandler>,
}

impl CommandSpec {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        usage: impl Into<String>,
        handler: impl CommandHandler + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            usage: usage.into(),
            handler: Arc::new(handler),
        }
    }
}

/// A prompt template invoked by a slash command name (an `Activation::ByCommand`
/// prompt). The runtime renders it and submits the result as a user turn.
struct TemplateEntry {
    name: String,
    template: String,
}

/// Routes a slash command name to either an executable action ([`CommandSpec`])
/// or a prompt template (an `Activation::ByCommand` prompt). A single namespace:
/// a template whose name collides with a registered command is dropped so the
/// command always wins.
#[derive(Default)]
pub struct CommandRouter {
    commands: Vec<CommandSpec>,
    templates: Vec<TemplateEntry>,
}

impl CommandRouter {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            templates: Vec::new(),
        }
    }

    pub fn register(&mut self, command: CommandSpec) {
        self.commands.push(command);
    }

    /// The command specs registered on this router.
    pub fn commands(&self) -> &[CommandSpec] {
        &self.commands
    }

    /// The names of the command-template prompts registered on this router.
    pub fn template_names(&self) -> impl Iterator<Item = &str> {
        self.templates.iter().map(|t| t.name.as_str())
    }

    /// The template entries registered on this router, as `(name, template)` pairs.
    pub fn template_entries(&self) -> impl Iterator<Item = (&str, &str)> {
        self.templates
            .iter()
            .map(|t| (t.name.as_str(), t.template.as_str()))
    }

    /// The template text for a command name, if one is registered.
    pub fn template(&self, name: &str) -> Option<&str> {
        self.templates
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.template.as_str())
    }

    pub fn from_capabilities(capabilities: impl IntoIterator<Item = Capability>) -> Self {
        let mut router = Self::new();
        router.install_capabilities(capabilities);
        router
    }

    /// Installs command and command-template capabilities. Commands are
    /// collected first so a template colliding with a command name is dropped.
    pub fn install_capabilities(&mut self, capabilities: impl IntoIterator<Item = Capability>) {
        let mut templates = Vec::new();
        for capability in capabilities {
            match capability {
                Capability::Command(command) => self.register(command),
                Capability::Prompt(prompt) => {
                    if let Activation::ByCommand(name) = prompt.activation {
                        templates.push(TemplateEntry {
                            name,
                            template: prompt.text,
                        });
                    }
                }
                _ => {}
            }
        }
        for entry in templates {
            if self.commands.iter().any(|c| c.name == entry.name) {
                continue;
            }
            self.templates.push(entry);
        }
    }

    pub fn from_extension_registry(registry: &crate::ExtensionRegistry) -> Self {
        Self::from_capabilities(registry.capabilities())
    }

    pub async fn handle(
        &self,
        name: &str,
        args: &str,
        ctx: &mut dyn CommandContext,
    ) -> sweet_core::Result<Option<String>> {
        for command in &self.commands {
            if command.name == name {
                return command.handler.handle(args, ctx).await;
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::extension::CapabilityProvider;
    use crate::test_util::MockModel;

    struct EchoCommand;

    #[async_trait]
    impl CommandHandler for EchoCommand {
        async fn handle(
            &self,
            args: &str,
            _ctx: &mut dyn CommandContext,
        ) -> sweet_core::Result<Option<String>> {
            Ok(Some(args.to_string()))
        }
    }

    struct ClearSessionCommand;

    #[async_trait]
    impl CommandHandler for ClearSessionCommand {
        async fn handle(
            &self,
            _args: &str,
            ctx: &mut dyn CommandContext,
        ) -> sweet_core::Result<Option<String>> {
            ctx.session_mut().clear()?;
            Ok(Some("cleared".to_string()))
        }
    }

    struct EchoProvider;

    impl CapabilityProvider for EchoProvider {
        fn id(&self) -> &str {
            "echo-provider"
        }

        fn capabilities(&self) -> Vec<Capability> {
            vec![Capability::Command(CommandSpec::new(
                "echo",
                "Echo args",
                "echo <text>",
                EchoCommand,
            ))]
        }
    }

    #[tokio::test]
    async fn router_dispatches_registered_command_by_name() {
        let mut router = CommandRouter::new();
        router.register(CommandSpec::new(
            "echo",
            "Echo args",
            "echo <text>",
            EchoCommand,
        ));
        let mut agent = Agent::new(MockModel::with_replies(Vec::<&str>::new()));

        let result = router
            .handle("echo", "hello world", &mut agent)
            .await
            .unwrap();

        assert_eq!(result.as_deref(), Some("hello world"));
    }

    #[tokio::test]
    async fn router_installs_command_capabilities_from_provider() {
        let provider = EchoProvider;
        let router = CommandRouter::from_capabilities(provider.capabilities());
        let mut agent = Agent::new(MockModel::with_replies(Vec::<&str>::new()));

        let result = router
            .handle("echo", "from provider", &mut agent)
            .await
            .unwrap();

        assert_eq!(result.as_deref(), Some("from provider"));
    }

    #[tokio::test]
    async fn command_handler_can_mutate_session_through_context() {
        let mut router = CommandRouter::new();
        router.register(CommandSpec::new(
            "clear",
            "Clear session",
            "clear",
            ClearSessionCommand,
        ));
        let mut agent = Agent::new(MockModel::with_replies(["ok"]));
        agent.step("hi").await.unwrap();
        assert_eq!(agent.session().messages().len(), 2);

        let result = router.handle("clear", "", &mut agent).await.unwrap();

        assert_eq!(result.as_deref(), Some("cleared"));
        assert!(agent.session().messages().is_empty());
    }

    #[tokio::test]
    async fn unknown_command_returns_none() {
        let router = CommandRouter::new();
        let mut agent = Agent::new(MockModel::with_replies(Vec::<&str>::new()));

        let result = router.handle("missing", "", &mut agent).await.unwrap();

        assert_eq!(result, None);
    }

    #[test]
    fn by_command_prompts_are_collected_as_templates() {
        use crate::extension::PromptSpec;

        let router = CommandRouter::from_capabilities([
            Capability::Command(CommandSpec::new("clear", "Clear", "clear", EchoCommand)),
            Capability::Prompt(PromptSpec::command("summarize", "Summarize:\n\n$ARGUMENTS")),
            // An always-on prompt must not be picked up as a command template.
            Capability::Prompt(PromptSpec::new("catalog", "always text")),
        ]);

        assert_eq!(
            router.template("summarize"),
            Some("Summarize:\n\n$ARGUMENTS")
        );
        assert_eq!(router.template("catalog"), None);
        assert_eq!(router.template("missing"), None);
        assert_eq!(
            router.template_names().collect::<Vec<_>>(),
            vec!["summarize"]
        );
    }

    #[test]
    fn template_colliding_with_command_is_dropped() {
        use crate::extension::PromptSpec;

        let router = CommandRouter::from_capabilities([
            Capability::Command(CommandSpec::new("clear", "Clear", "clear", EchoCommand)),
            Capability::Prompt(PromptSpec::command("clear", "shadow attempt")),
        ]);

        // The command wins; the colliding template is not registered.
        assert_eq!(router.template("clear"), None);
        assert_eq!(router.commands().len(), 1);
    }

    #[test]
    fn command_capability_is_cloneable() {
        let command = CommandSpec::new("echo", "Echo args", "echo <text>", EchoCommand);
        let cloned = Capability::Command(command).clone();

        match cloned {
            Capability::Command(command) => assert_eq!(command.name, "echo"),
            _ => panic!("expected command capability"),
        }
    }
}
