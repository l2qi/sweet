// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures_util::stream::{FuturesUnordered, StreamExt};
use tracing::Instrument;

use sweet_core::message::{ContentBlock, Message, Role, ToolCall};
use sweet_core::model::Model;
use sweet_core::permission::{ApprovalDecision, PermissionMode, PermissionState, ToolRisk};
use sweet_core::session::{InMemorySession, MemoryItem, Session};
use sweet_core::tool::{ToolError, ToolOutput, ToolSpec};
use sweet_core::Result;

/// Trait for types that can be converted into user message content blocks.
///
/// Implemented for `Vec<ContentBlock>` (passthrough) and `String` / `&str`
/// (wrapped in a single `ContentBlock::Text`). This lets callers pass plain
/// text strings or multimodal blocks interchangeably to [`Agent::step`] and
/// [`Agent::step_stream`].
pub trait IntoContentBlocks {
    fn into_content_blocks(self) -> Vec<ContentBlock>;
}

impl IntoContentBlocks for Vec<ContentBlock> {
    fn into_content_blocks(self) -> Vec<ContentBlock> {
        self
    }
}

impl IntoContentBlocks for String {
    fn into_content_blocks(self) -> Vec<ContentBlock> {
        vec![ContentBlock::text(self)]
    }
}

impl IntoContentBlocks for &str {
    fn into_content_blocks(self) -> Vec<ContentBlock> {
        vec![ContentBlock::text(self)]
    }
}

impl IntoContentBlocks for &String {
    fn into_content_blocks(self) -> Vec<ContentBlock> {
        vec![ContentBlock::text(self)]
    }
}

use crate::commands::CommandContext;
use crate::dynamic_prompt::DynamicPrompt;
use crate::extension::{Activation, Capability, CapabilityProvider, ExtensionRegistry, PromptSpec};
use crate::handoff::{HandoffSpec, TurnResult};
use crate::hooks::{HookDispatcher, HookEvent};
use crate::runloop::{AgentIo, IoStreamSink};
use crate::subagent::{SubagentSpec, PARENT_MODEL};

/// A minimal conversational agent backed by a [`Model`].
///
/// `Agent` delegates storage to a [`Session`] and keeps `instructions`
/// (system prompt) separate from the transcript. Each call to [`Agent::step`]
/// appends a user message, asks the model for a reply, appends the reply, and
/// returns it. If the model replies with tool calls, the agent dispatches them
/// automatically and feeds the results back to the model in an inner loop.
pub struct Agent<M: Model> {
    model: M,
    session: Box<dyn Session>,
    instructions: Option<String>,
    prompts: Vec<PromptSpec>,
    /// Prompts recomputed every turn and appended to the composed system
    /// instructions. Unlike `prompts`, their text is not fixed at construction
    /// - see [`crate::dynamic_prompt::DynamicPrompt`].
    dynamic_prompts: Vec<Arc<dyn DynamicPrompt>>,
    tools: Vec<ToolSpec>,
    handoffs: Vec<HandoffSpec>,
    hooks: HookDispatcher,
    /// Type-erased clone of `model` set only when the agent was constructed via
    /// [`Agent::new_shared`]. Threaded through to subagent handlers via
    /// [`crate::subagent::SubagentContext::parent_model`] so they can inherit
    /// the parent's model instead of constructing their own.
    shareable_model: Option<Arc<dyn Model>>,
    /// Run-scoped permission state - mode plus session-level approvals.
    /// Shared via `Arc` so it survives agent switches mid-run.
    permission: Arc<PermissionState>,
    /// Cap on how many tool-result images are sent to the model each turn: only
    /// the most recent N images carried on `Role::Tool` messages are kept in the
    /// outgoing request; older ones are dropped (from the request only - the
    /// session keeps the full history). `None` sends every image. Set this for
    /// agents whose tools return a fresh screenshot each turn (computer use),
    /// where re-sending every past screenshot would balloon context and cost.
    max_tool_result_images: Option<usize>,
}

impl<M: Model> Agent<M> {
    pub fn new(model: M) -> Self {
        Self {
            model,
            session: Box::new(InMemorySession::new()),
            instructions: None,
            prompts: Vec::new(),
            dynamic_prompts: Vec::new(),
            tools: Vec::new(),
            handoffs: Vec::new(),
            hooks: HookDispatcher::new(),
            shareable_model: None,
            permission: Arc::new(PermissionState::default()),
            max_tool_result_images: None,
        }
    }

    /// Set or replace the agent's instructions (system prompt).
    ///
    /// Instructions are kept separate from the session and are automatically
    /// prepended to the message slice sent to the model on each turn.
    pub fn with_instructions(mut self, prompt: impl Into<String>) -> Self {
        self.instructions = Some(prompt.into());
        self
    }

    /// Register a [`DynamicPrompt`] whose text is recomputed every turn and
    /// appended to the composed system instructions (after the static parts, so
    /// it carries the strongest recency). Because instructions live outside the
    /// session, this survives history compaction.
    pub fn with_dynamic_prompt(mut self, prompt: Arc<dyn DynamicPrompt>) -> Self {
        self.dynamic_prompts.push(prompt);
        self
    }

    /// Bound how many tool-result images are sent to the model: on each turn
    /// only the most recent `max` images carried on `Role::Tool` messages are
    /// kept in the outgoing request; older ones are dropped (their text - active
    /// app, accessibility tree, on-disk screenshot path - is retained). The
    /// session keeps the full transcript; this caps only what each request
    /// carries.
    ///
    /// Unset by default (every image is sent). Set this for agents using a tool
    /// that returns a fresh screenshot each turn (e.g. computer use), where
    /// re-sending every past screenshot on every turn balloons context and cost.
    pub fn with_max_tool_result_images(mut self, max: usize) -> Self {
        self.max_tool_result_images = Some(max);
        self
    }

    /// Replace the agent's session.
    pub fn with_session(mut self, session: impl Session + 'static) -> Self {
        self.session = Box::new(session);
        self
    }

    /// Replace the agent's session with an already-boxed trait object.
    ///
    /// Use this when you already hold a [`Box<dyn Session>`] (for example
    /// from [`Agent::take_session`]) and would otherwise need to box-then-rebox
    /// through [`Agent::with_session`].
    pub fn with_session_boxed(mut self, session: Box<dyn Session>) -> Self {
        self.session = session;
        self
    }

    /// Add a single tool to the agent's toolset.
    pub fn with_tool(self, tool: impl Into<ToolSpec>) -> Self {
        self.with_capability(Capability::tool(tool))
    }

    /// Add multiple tools to the agent's toolset.
    pub fn with_tools<I>(self, tools: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<ToolSpec>,
    {
        tools
            .into_iter()
            .fold(self, |agent, tool| agent.with_tool(tool))
    }

    /// Register a subagent as a tool. The parent's model invokes it via the
    /// usual tool-call mechanism; the handler builds and runs a fresh child
    /// agent per invocation.
    pub fn with_subagent(self, spec: SubagentSpec) -> Self {
        self.with_tool(ToolSpec::from(spec))
    }

    /// Register a handoff tool. When the model calls it, the agent step loop
    /// is interrupted and a [`TurnResult::Handoff`] is returned.
    pub fn with_handoff(self, spec: HandoffSpec) -> Self {
        debug_assert!(
            !self.tools.iter().any(|t| t.name == spec.name),
            "handoff name `{}` collides with a registered tool",
            spec.name
        );
        debug_assert!(
            !self.handoffs.iter().any(|h| h.name == spec.name),
            "handoff name `{}` is already registered",
            spec.name
        );
        let mut handoffs = self.handoffs;
        handoffs.push(spec);
        Self { handoffs, ..self }
    }

    /// Move the session out of this agent, replacing it with a fresh empty one.
    ///
    /// Used during handoffs to transfer conversation history from one agent to
    /// another.
    pub fn take_session(&mut self) -> Box<dyn Session> {
        let mut taken: Box<dyn Session> = Box::new(InMemorySession::new());
        std::mem::swap(&mut self.session, &mut taken);
        taken
    }

    /// Install a framework capability on the agent.
    pub fn with_capability(mut self, capability: Capability) -> Self {
        self.install_capability(capability);
        self
    }

    /// Install framework capabilities on the agent.
    pub fn with_capabilities<I>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = Capability>,
    {
        self.install_capabilities(capabilities);
        self
    }

    /// Install all capabilities produced by a provider.
    pub fn with_capability_provider<P>(self, provider: &P) -> Self
    where
        P: CapabilityProvider + ?Sized,
    {
        self.with_capabilities(provider.capabilities())
    }

    /// Install all capabilities from an extension registry.
    pub fn with_extension_registry(self, registry: &ExtensionRegistry) -> Self {
        self.with_capabilities(registry.capabilities())
    }

    /// The tools registered on this agent - native tools, MCP tools, and
    /// subagents (which are registered as tools).
    pub fn tools(&self) -> &[ToolSpec] {
        &self.tools
    }

    /// The handoff specs registered on this agent.
    pub fn handoffs(&self) -> &[HandoffSpec] {
        &self.handoffs
    }

    /// Set the permission mode for tool approval gating.
    ///
    /// **Note:** this creates a fresh [`PermissionState`]. If a handle was
    /// already cloned via `permission_handle()`, it will become stale. Use
    /// `with_permission_handle()` when switching agents to preserve the
    /// shared handle.
    pub fn with_permission_mode(mut self, mode: PermissionMode) -> Self {
        self.permission = Arc::new(PermissionState::new(mode));
        self
    }

    /// Adopt an existing shared [`PermissionState`] handle.
    /// Use this when switching agents to keep the same mode *and* the same
    /// session approvals across the old and new agent.
    pub fn with_permission_handle(mut self, handle: Arc<PermissionState>) -> Self {
        self.permission = handle;
        self
    }

    /// Read the current permission mode.
    pub fn permission_mode(&self) -> PermissionMode {
        self.permission.mode()
    }

    /// Set the permission mode in place. Takes `&self` - safe to call
    /// through a shared reference (e.g. while another task holds the
    /// agent mutex).
    pub fn set_permission_mode(&self, mode: PermissionMode) {
        self.permission.set_mode(mode);
    }

    /// Returns a shared handle to the permission state that can be read and
    /// updated without locking the agent, and carried across agent switches.
    pub fn permission_handle(&self) -> Arc<PermissionState> {
        Arc::clone(&self.permission)
    }

    /// Live-swap the agent's session. Tools and instructions are preserved.
    pub fn new_session(&mut self, session: impl Session + 'static) {
        self.session = Box::new(session);
    }

    /// Access the current session.
    pub fn session(&self) -> &dyn Session {
        &*self.session
    }

    /// Mutable access to the current session.
    pub fn session_mut(&mut self) -> &mut dyn Session {
        &mut *self.session
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    /// Clone of the agent's model as `Arc<dyn Model>`, available iff the agent
    /// was constructed via [`Agent::new_shared`]. Used internally to propagate
    /// the parent model into subagent handlers.
    pub fn shareable_model(&self) -> Option<Arc<dyn Model>> {
        self.shareable_model.clone()
    }

    /// Drive one turn of the conversation, dropping streaming events.
    ///
    /// Equivalent to [`Agent::step_stream`] with a no-op IO. Useful for
    /// callers that just want the final reply (e.g., a non-interactive
    /// HTTP backend).
    pub async fn step(&mut self, user_input: impl IntoContentBlocks) -> Result<TurnResult> {
        let mut io = NoopIo;
        self.step_stream(user_input, &mut io).await
    }

    /// Scan the trailing assistant message for `tool_calls` that lack
    /// corresponding tool-role results and push a synthetic error result for
    /// each orphaned call. Returns `true` if any repairs were made.
    ///
    /// This recovers from task aborts where a tool was dispatched but its
    /// result was never written to the session. Without repair, the
    /// `messages` array would violate the API schema (assistant `tool_calls`
    /// must be immediately followed by one `tool` message per call).
    ///
    /// Only the final assistant message can be orphaned: a completed turn
    /// always resolves every call before the next message is appended, and an
    /// abort can only interrupt the turn in progress. Synthetic results are
    /// appended to the end of the session, so the repaired message must be
    /// the last non-tool message - hence the trailing-only scan.
    pub fn repair_orphaned_tool_calls(&mut self) -> Result<bool> {
        let items = self.session.items();

        let Some(assistant_idx) = items.iter().rposition(|item| {
            let MemoryItem::Message(msg) = item;
            msg.role == Role::Assistant
        }) else {
            return Ok(false);
        };

        let MemoryItem::Message(assistant) = &items[assistant_idx];
        if assistant.tool_calls.is_empty() {
            return Ok(false);
        }

        let mut resolved: HashSet<&str> = HashSet::new();
        for item in &items[assistant_idx + 1..] {
            let MemoryItem::Message(m) = item;
            if let Some(ref id) = m.tool_call_id {
                resolved.insert(id.as_str());
            }
        }

        let orphans: Vec<String> = assistant
            .tool_calls
            .iter()
            .filter(|tc| !resolved.contains(tc.id.as_str()))
            .map(|tc| tc.id.clone())
            .collect();

        if orphans.is_empty() {
            return Ok(false);
        }

        let orphan_count = orphans.len();
        for call_id in orphans {
            self.session.push(MemoryItem::Message(Message::tool_result(
                &call_id,
                "Error: tool execution was interrupted",
            )))?;
        }
        tracing::warn!(orphan_count, "repaired orphaned tool calls in session");
        Ok(true)
    }

    /// Drive one turn of the conversation, forwarding incremental events to
    /// the supplied [`AgentIo`].
    ///
    /// Content deltas from the model and tool-call events are reported through
    /// `io` as they happen. The returned [`TurnResult`] is either the final
    /// assistant message or a handoff request.
    pub async fn step_stream(
        &mut self,
        user_input: impl IntoContentBlocks,
        io: &mut (impl AgentIo + ?Sized),
    ) -> Result<TurnResult> {
        let user_blocks = user_input.into_content_blocks();
        let turn_index = self
            .session
            .messages()
            .iter()
            .filter(|m| m.role == Role::User)
            .count()
            + 1;
        let history_len_before = self.session.items().len();
        let span = tracing::debug_span!(
            target: "sweet_agent::observability",
            "agent.turn",
            turn_index,
            history_len_before
        );
        // Shadow PARENT_MODEL for the duration of this turn so subagent tools
        // dispatched from inside `step_loop` see this agent as their parent.
        let parent_model = self.shareable_model.clone();
        let fut = self
            .step_loop(user_blocks, turn_index, history_len_before, io)
            .instrument(span);
        PARENT_MODEL.scope(parent_model, fut).await
    }

    async fn step_loop(
        &mut self,
        user_input: Vec<ContentBlock>,
        turn_index: usize,
        history_len_before: usize,
        io: &mut (impl AgentIo + ?Sized),
    ) -> Result<TurnResult> {
        // Repair before appending the user message: synthetic tool results
        // are pushed to the end of the session, so they must land directly
        // after the orphaned assistant message, not after the new user turn.
        self.repair_orphaned_tool_calls()?;
        self.session
            .push(MemoryItem::Message(Message::user_blocks(user_input)))?;
        self.fire_hook(
            HookEvent::BeforeTurn,
            serde_json::json!({
                "turn_index": turn_index,
                "history_len_before": history_len_before,
            }),
        )
        .await?;
        let tools = self.all_tools();
        let mut model_call_index = 0usize;
        let mut early_handoff: Option<TurnResult> = None;
        'outer: loop {
            model_call_index += 1;
            self.fire_hook(
                HookEvent::BeforeModelCall,
                serde_json::json!({
                    "turn_index": turn_index,
                    "model_call_index": model_call_index,
                    "message_count": self.session.items().len(),
                    "tool_count": tools.len(),
                }),
            )
            .await?;
            let messages = self.build_messages();
            tracing::debug!(
                target: "sweet_agent::observability",
                event = "agent.turn.model_input",
                turn_index,
                model_call_index,
                history_len = self.session.items().len(),
                transcript = %json_string(&messages),
                "agent turn model input"
            );

            let started = Instant::now();
            let reply = {
                let mut sink = IoStreamSink::new(io);
                match self
                    .model
                    .complete_stream(&messages, &tools, &mut sink)
                    .await
                {
                    Ok(reply) => {
                        let duration_ms = elapsed_ms(started);
                        tracing::debug!(
                            target: "sweet_agent::observability",
                            event = "llm.complete",
                            turn_index,
                            model_call_index,
                            message_count = self.session.items().len(),
                            tool_count = tools.len(),
                            tools = %tool_names_json(&tools),
                            duration_ms,
                            status = "ok",
                            assistant_content_len = reply.text_content().len(),
                            tool_call_count = reply.tool_calls.len(),
                            assistant = %json_string(&reply),
                            "llm complete"
                        );
                        self.fire_hook(HookEvent::AfterModelReply, json_value(&reply))
                            .await?;
                        reply
                    }
                    Err(err) => {
                        let duration_ms = elapsed_ms(started);
                        tracing::debug!(
                            target: "sweet_agent::observability",
                            event = "llm.complete",
                            turn_index,
                            model_call_index,
                            message_count = self.session.items().len(),
                            tool_count = tools.len(),
                            tools = %tool_names_json(&tools),
                            duration_ms,
                            status = "error",
                            error = %err,
                            "llm complete failed"
                        );
                        return Err(err);
                    }
                }
            };
            let has_calls = !reply.tool_calls.is_empty();
            self.session.push(MemoryItem::Message(reply))?;
            if !has_calls {
                break;
            }
            let calls = match self.session.items().last() {
                Some(MemoryItem::Message(m)) => m.tool_calls.clone(),
                _ => panic!("session lost the assistant message we just pushed"),
            };
            if self.all_read_only(&calls) {
                if let Some((target, payload)) =
                    self.dispatch_concurrent(&calls, turn_index, io).await?
                {
                    early_handoff = Some(TurnResult::Handoff {
                        target,
                        payload: Some(payload),
                    });
                    break 'outer;
                }
            } else {
                for call in calls {
                    self.fire_hook(HookEvent::BeforeToolCall, json_value(&call))
                        .await?;

                    // --- Permission gate ---
                    let result = match self.check_permission(&call, io).await {
                        Some(denied) => denied,
                        None => self.dispatch(&call, turn_index).await,
                    };

                    let handoff = match &result {
                        Err(ToolError::Handoff { target, payload }) => {
                            Some((target.clone(), payload.clone()))
                        }
                        _ => None,
                    };
                    let (display, message) = tool_result_to_message(&call.id, &result);
                    self.fire_hook(
                        HookEvent::AfterToolCall,
                        serde_json::json!({
                            "call": json_value(&call),
                            "result": display.clone(),
                        }),
                    )
                    .await?;
                    io.on_tool_result(&call, &display).await?;
                    self.session.push(MemoryItem::Message(message))?;
                    if let Some((target, payload)) = handoff {
                        early_handoff = Some(TurnResult::Handoff {
                            target,
                            payload: Some(payload),
                        });
                        break 'outer;
                    }
                }
            }
        }
        let result = match early_handoff {
            Some(handoff) => handoff,
            None => TurnResult::Message(match self.session.items().last() {
                Some(MemoryItem::Message(m)) => m.clone(),
                _ => panic!(
                    "session lost the assistant message - step_stream loop pushed at least one"
                ),
            }),
        };
        tracing::debug!(
            target: "sweet_agent::observability",
            event = "agent.turn.finished",
            turn_index,
            history_len_before,
            history_len_after = self.session.items().len(),
            transcript = %json_string(&self.build_messages()),
            "agent turn finished"
        );
        let after_payload = match &result {
            TurnResult::Message(m) => serde_json::json!({
                "turn_index": turn_index,
                "kind": "message",
                "history_len_after": self.session.items().len(),
                "reply": json_value(m),
            }),
            TurnResult::Handoff { target, .. } => serde_json::json!({
                "turn_index": turn_index,
                "kind": "handoff",
                "history_len_after": self.session.items().len(),
                "target": target,
            }),
        };
        self.fire_hook(HookEvent::AfterTurn, after_payload).await?;
        Ok(result)
    }

    /// All tools visible to the model, including regular tools and handoffs.
    fn all_tools(&self) -> Vec<ToolSpec> {
        let mut all = self.tools.clone();
        for handoff in &self.handoffs {
            all.push(ToolSpec::from(handoff.clone()));
        }
        all
    }

    fn build_messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();
        if let Some(instructions) = self.composed_instructions() {
            messages.push(Message::system(instructions));
        }
        messages.extend(self.session.messages());
        if let Some(limit) = self.max_tool_result_images {
            cap_tool_result_images(&mut messages, limit);
        }
        messages
    }

    fn composed_instructions(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(instructions) = self.instructions.as_deref() {
            parts.push(instructions.to_string());
        }
        parts.extend(
            self.prompts
                .iter()
                .filter(|prompt| prompt.activation == Activation::Always)
                .map(|prompt| prompt.text.clone()),
        );
        // Dynamic prompts are rendered last so the freshest state (e.g. a live
        // todo list) lands at the end of the system prompt.
        parts.extend(self.dynamic_prompts.iter().filter_map(|dp| dp.render()));
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }

    fn install_capabilities<I>(&mut self, capabilities: I)
    where
        I: IntoIterator<Item = Capability>,
    {
        for capability in capabilities {
            self.install_capability(capability);
        }
    }

    fn install_capability(&mut self, capability: Capability) {
        match capability {
            Capability::Prompt(prompt) => self.prompts.push(prompt),
            Capability::Tool(tool) => self.tools.push(tool),
            Capability::Hook(hook) => self.hooks.register_hook(hook),
            Capability::Command(command) => self.hooks.register_command(command),
            Capability::Procedure(procedure) => self.hooks.register_procedure(procedure),
        }
    }

    async fn fire_hook(&mut self, event: HookEvent, payload: serde_json::Value) -> Result<()> {
        // Clone hooks + tools so the dispatch can borrow `self` mutably as
        // the CommandContext while the dispatcher resolves handler IDs.
        let hooks = self.hooks.clone();
        let tools = self.tools.clone();
        hooks.fire(event, payload, self, &tools).await
    }

    /// Returns `true` if every call in the batch targets a known tool
    /// classified as [`ToolRisk::ReadOnly`]. Handoff tool names and unknown
    /// names yield `false`, causing the batch to fall back to sequential
    /// dispatch.
    fn all_read_only(&self, calls: &[ToolCall]) -> bool {
        calls.iter().all(|call| {
            self.tools
                .iter()
                .find(|t| t.name == call.name)
                .map(|t| t.risk == ToolRisk::ReadOnly)
                .unwrap_or(false)
        })
    }

    /// Dispatch a batch of `ToolRisk::ReadOnly` tool calls concurrently.
    ///
    /// Precondition: [`Self::all_read_only`] returned `true` for this batch.
    /// Handoff tools and unknown tool names must not be present.
    ///
    /// Semantics relative to sequential dispatch:
    /// - `BeforeToolCall` hooks fire for every call in invocation order
    ///   *before* any tool begins executing. This batched-hook view differs
    ///   from sequential dispatch, where each hook sees prior calls' results
    ///   already in the session - an intrinsic consequence of running the
    ///   tools in parallel.
    /// - Tools execute concurrently via [`FuturesUnordered`].
    /// - `AfterToolCall` hooks, [`AgentIo::on_tool_result`], and session
    ///   pushes are driven in invocation order with backpressure: a finished
    ///   call's downstream effects fire as soon as it and every
    ///   earlier-indexed call have finished, so the UI and session both see
    ///   results streamed in invocation order without waiting for the
    ///   slowest call in the batch.
    /// - If any tool returned [`ToolError::Handoff`], the target/payload of
    ///   the first such call (by invocation index) is returned; the caller
    ///   surfaces it as [`TurnResult::Handoff`]. All other calls' downstream
    ///   effects still fire first so the session stays schema-valid.
    ///
    /// Uses [`FuturesUnordered`] (not `tokio::spawn`) so that task-locals
    /// such as [`crate::subagent::PARENT_MODEL`] and
    /// [`crate::subagent::SUBAGENT_DEPTH`] remain visible to subagent
    /// handlers.
    async fn dispatch_concurrent(
        &mut self,
        calls: &[ToolCall],
        turn_index: usize,
        io: &mut (impl AgentIo + ?Sized),
    ) -> Result<Option<(String, String)>> {
        // Phase 1: BeforeToolCall hooks (sequential, invocation order).
        for call in calls {
            self.fire_hook(HookEvent::BeforeToolCall, json_value(call))
                .await?;
        }

        // Phase 2: Launch tool calls concurrently. ReadOnly tools never
        // require approval (see `needs_approval`), so the permission gate is
        // intentionally absent. `all_read_only` guarantees every call resolves
        // to a known tool - `expect` documents that invariant.
        let mut unordered: FuturesUnordered<_> = calls
            .iter()
            .enumerate()
            .map(|(i, call)| {
                let tool = self
                    .tools
                    .iter()
                    .find(|t| t.name == call.name)
                    .cloned()
                    .expect("all_read_only guarantees every call resolves to a known tool");
                let call = call.clone();
                async move {
                    let (call, result) = observe_tool_call(&tool, call, turn_index).await;
                    (i, call, result)
                }
            })
            .collect();

        // Phase 3: Drain completions, but fire AfterToolCall, IO, and session
        // pushes in invocation order with backpressure.
        let mut slots: Vec<Option<(ToolCall, std::result::Result<ToolOutput, ToolError>)>> =
            (0..calls.len()).map(|_| None).collect();
        let mut next_to_emit = 0usize;
        let mut handoff: Option<(String, String)> = None;

        while let Some((i, call, result)) = unordered.next().await {
            slots[i] = Some((call, result));
            while next_to_emit < slots.len() && slots[next_to_emit].is_some() {
                let (call, result) = slots[next_to_emit].take().expect("slot just checked Some");
                if handoff.is_none() {
                    if let Err(ToolError::Handoff { target, payload }) = &result {
                        handoff = Some((target.clone(), payload.clone()));
                    }
                }
                let (display, message) = tool_result_to_message(&call.id, &result);
                self.fire_hook(
                    HookEvent::AfterToolCall,
                    serde_json::json!({
                        "call": json_value(&call),
                        "result": display.clone(),
                    }),
                )
                .await?;
                io.on_tool_result(&call, &display).await?;
                self.session.push(MemoryItem::Message(message))?;
                next_to_emit += 1;
            }
        }

        Ok(handoff)
    }

    async fn dispatch(
        &self,
        call: &ToolCall,
        turn_index: usize,
    ) -> std::result::Result<ToolOutput, ToolError> {
        let tool = self.tools.iter().find(|t| t.name == call.name).cloned();
        let handoff = self.handoffs.iter().find(|h| h.name == call.name).cloned();

        let result = match (tool, handoff) {
            (Some(tool), _) => {
                let (_, res) = observe_tool_call(&tool, call.clone(), turn_index).await;
                res
            }
            (None, Some(handoff)) => {
                let spec = ToolSpec::from(handoff);
                let (_, res) = observe_tool_call(&spec, call.clone(), turn_index).await;
                res
            }
            (None, None) => Err(ToolError::UnknownTool(call.name.clone())),
        };
        result
    }

    /// Check whether a tool call needs user approval. Returns `Some(Err)` if
    /// the call was denied, or `None` if dispatch should proceed normally.
    async fn check_permission(
        &self,
        call: &ToolCall,
        io: &mut (impl AgentIo + ?Sized),
    ) -> Option<std::result::Result<ToolOutput, ToolError>> {
        let risk = match self.tools.iter().find(|t| t.name == call.name) {
            Some(tool) => tool.risk,
            None => {
                // Handoff tools are always read-only. An unknown name is left
                // for `dispatch` to reject - no point gating a call that
                // cannot run.
                if self.handoffs.iter().any(|h| h.name == call.name) {
                    ToolRisk::ReadOnly
                } else {
                    return None;
                }
            }
        };

        if !sweet_core::permission::needs_approval(self.permission.mode(), risk) {
            return None;
        }

        // Session approvals are keyed by (tool, scope) - the same scope shown
        // in the prompt - so "Always" grants exactly what the user saw.
        let scope = sweet_core::permission::approval_scope(&call.arguments);
        if self.permission.is_allowed(&call.name, &scope) {
            return None;
        }

        match io.on_tool_approval(call, risk).await {
            Ok(ApprovalDecision::Allow) => None,
            Ok(ApprovalDecision::AllowSession) => {
                self.permission.allow(call.name.clone(), scope);
                None
            }
            Ok(ApprovalDecision::Deny) => Some(Err(ToolError::PermissionDenied(format!(
                "User denied tool call: {}",
                call.name
            )))),
            Err(e) => Some(Err(ToolError::PermissionDenied(format!(
                "Approval check failed: {e}"
            )))),
        }
    }
}

impl Agent<Arc<dyn Model>> {
    /// Construct an agent whose model is shareable with subagents.
    ///
    /// The model is stored as `Arc<dyn Model>` so the framework can hand a
    /// clone to each [`crate::subagent::SubagentHandler`] via
    /// [`crate::subagent::SubagentContext::parent_model`]. Use this instead of
    /// [`Agent::new`] when you want subagents to default to the same model as
    /// the parent.
    pub fn new_shared(model: Arc<dyn Model>) -> Self {
        let shareable = model.clone();
        Self {
            model,
            session: Box::new(InMemorySession::new()),
            instructions: None,
            prompts: Vec::new(),
            dynamic_prompts: Vec::new(),
            tools: Vec::new(),
            handoffs: Vec::new(),
            hooks: HookDispatcher::new(),
            shareable_model: Some(shareable),
            permission: Arc::new(PermissionState::default()),
            max_tool_result_images: None,
        }
    }
}

impl<M: Model> CommandContext for Agent<M> {
    fn session(&self) -> &dyn Session {
        &*self.session
    }

    fn session_mut(&mut self) -> &mut dyn Session {
        &mut *self.session
    }

    fn replace_session(&mut self, session: Box<dyn Session>) {
        self.session = session;
    }

    fn model(&self) -> &dyn Model {
        &self.model
    }
}

/// Emit the `tool.call.start` and `tool.call` observability spans for a
/// single tool invocation. Shared by both `dispatch` (sequential) and
/// `dispatch_concurrent` (parallel) to keep the schema identical.
async fn observe_tool_call(
    tool: &ToolSpec,
    call: ToolCall,
    turn_index: usize,
) -> (ToolCall, std::result::Result<ToolOutput, ToolError>) {
    tracing::debug!(
        target: "sweet_agent::observability",
        event = "tool.call.start",
        turn_index,
        tool_call_id = %call.id,
        tool_name = %call.name,
        arguments = %json_string(&call.arguments),
        "tool call start"
    );
    let started = Instant::now();
    let result = tool.call_rich(call.arguments.clone()).await;
    let duration_ms = elapsed_ms(started);
    match &result {
        Ok(output) => tracing::debug!(
            target: "sweet_agent::observability",
            event = "tool.call",
            turn_index,
            tool_call_id = %call.id,
            tool_name = %call.name,
            duration_ms,
            status = "ok",
            result = %output,
            "tool call"
        ),
        Err(err) => tracing::debug!(
            target: "sweet_agent::observability",
            event = "tool.call",
            turn_index,
            tool_call_id = %call.id,
            tool_name = %call.name,
            duration_ms,
            status = "error",
            error = %err,
            "tool call failed"
        ),
    }
    (call, result)
}

/// Turn a tool-call result into the display text (for IO + hooks) and the
/// `Role::Tool` session message. On success the message carries the tool's full
/// content blocks (text and any images); errors and handoffs become plain text.
fn tool_result_to_message(
    call_id: &str,
    result: &std::result::Result<ToolOutput, ToolError>,
) -> (String, Message) {
    match result {
        Ok(output) => (
            output.text_content(),
            Message::tool_result_blocks(call_id, output.blocks.clone()),
        ),
        Err(ToolError::Handoff { target, .. }) => {
            let s = format!("Handoff to {target} initiated.");
            (s.clone(), Message::tool_result(call_id, s))
        }
        Err(e) => {
            let s = format!("Error: {e}");
            (s.clone(), Message::tool_result(call_id, s))
        }
    }
}

/// Keep image blocks on only the most recent `limit` tool-result images,
/// stripping older ones in place. Scans messages newest-first so the freshest
/// screenshots survive; non-`Tool` messages (including user-supplied images) are
/// never touched. Stripping leaves each tool result's text intact; if a result
/// was image-only and loses all its images, a short placeholder replaces them so
/// the request never carries an empty tool message.
///
/// This bounds context for tools that return a fresh screenshot every turn
/// (computer use): without it every past screenshot is re-encoded and re-sent on
/// every turn. Operates on the per-request message list, not the session.
fn cap_tool_result_images(messages: &mut [Message], limit: usize) {
    let mut kept = 0usize;
    for msg in messages.iter_mut().rev() {
        if msg.role != Role::Tool {
            continue;
        }
        let mut had_image = false;
        msg.content.retain(|block| {
            if matches!(block, ContentBlock::Image { .. }) {
                had_image = true;
                let keep = kept < limit;
                kept += usize::from(keep);
                keep
            } else {
                true
            }
        });
        if had_image && msg.content.is_empty() {
            msg.content.push(ContentBlock::text(
                "[earlier screenshot omitted to bound context]",
            ));
        }
    }
}

fn json_string<T: serde::Serialize + ?Sized>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|e| format!("observability serialization failed: {e}"))
}

fn json_value<T: serde::Serialize + ?Sized>(value: &T) -> serde_json::Value {
    serde_json::to_value(value)
        .unwrap_or_else(|e| serde_json::json!({ "serialization_error": e.to_string() }))
}

fn tool_names_json(tools: &[ToolSpec]) -> String {
    let names = tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>();
    json_string(&names)
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

/// `AgentIo` impl used internally by [`Agent::step`] when the caller does not
/// care about streaming events. `read_input` returns `Ok(None)` so it cannot
/// drive a runloop by itself.
struct NoopIo;

#[async_trait]
impl AgentIo for NoopIo {
    async fn read_input(&mut self) -> Result<Option<String>> {
        Ok(None)
    }

    async fn write_reply(&mut self, _message: &Message, _session: &dyn Session) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::{HookInvocation, ProcedureHandler, ProcedureSpec};
    use crate::test_util::{MockModel, MockTool};
    use std::io;
    use std::sync::{Arc, Mutex};
    use sweet_core::message::ToolCall;
    use sweet_core::tool::ToolHandler;

    #[tokio::test]
    async fn step_appends_user_then_assistant() {
        let model = MockModel::with_replies(["hello back"]);
        let mut agent = Agent::new(model);

        let reply = match agent.step("hello").await.unwrap() {
            TurnResult::Message(m) => m,
            TurnResult::Handoff { .. } => panic!("unexpected handoff"),
        };
        assert_eq!(reply.role, Role::Assistant);
        assert_eq!(reply.text_content(), "hello back");

        let h = agent.session().messages();
        assert_eq!(h.len(), 2);
        assert_eq!(h[0], Message::user("hello"));
        assert_eq!(h[1], Message::assistant("hello back"));
    }

    #[tokio::test]
    async fn with_instructions_prepended_to_model_input() {
        let model = MockModel::with_replies(["ok"]);
        let mut agent = Agent::new(model).with_instructions("be terse");
        let result = agent.step("hi").await.unwrap();
        assert!(matches!(result, TurnResult::Message(_)));

        let calls = agent.model().calls();
        assert_eq!(
            calls[0],
            vec![Message::system("be terse"), Message::user("hi")]
        );
    }

    #[tokio::test]
    async fn with_instructions_replaces_existing() {
        let model = MockModel::with_replies(["ok"]);
        let mut agent = Agent::new(model)
            .with_instructions("first")
            .with_instructions("second");
        let result = agent.step("hi").await.unwrap();
        assert!(matches!(result, TurnResult::Message(_)));

        let calls = agent.model().calls();
        assert_eq!(calls[0][0], Message::system("second"));
    }

    #[tokio::test]
    async fn extension_registry_installs_prompt_capabilities_in_order() {
        let model = MockModel::with_replies(["ok"]);
        let mut registry = ExtensionRegistry::new();
        registry.register(TestPromptProvider::new("one", "first prompt"));
        registry.register(TestPromptProvider::new("two", "second prompt"));

        let mut agent = Agent::new(model)
            .with_instructions("base")
            .with_extension_registry(&registry);
        let result = agent.step("hi").await.unwrap();
        assert!(matches!(result, TurnResult::Message(_)));

        let calls = agent.model().calls();
        assert_eq!(
            calls[0][0],
            Message::system("base\n\nfirst prompt\n\nsecond prompt")
        );
    }

    #[tokio::test]
    async fn prompt_capabilities_with_declarative_activation_are_not_auto_composed() {
        let model = MockModel::with_replies(["ok"]);
        let mut registry = ExtensionRegistry::new();
        registry.register(TestPromptProvider::new("always", "active prompt"));
        registry.register(
            TestPromptProvider::new("summarize", "command prompt")
                .with_activation(Activation::ByCommand("summarize".to_string())),
        );

        let mut agent = Agent::new(model)
            .with_instructions("base")
            .with_extension_registry(&registry);
        let result = agent.step("hi").await.unwrap();
        assert!(matches!(result, TurnResult::Message(_)));

        let calls = agent.model().calls();
        assert_eq!(calls[0][0], Message::system("base\n\nactive prompt"));
    }

    #[tokio::test]
    async fn dynamic_prompt_is_rendered_fresh_after_static_parts_each_turn() {
        // `render` is a pure projection of shared state (it may be called more
        // than once per turn), so we mutate the state between turns and assert
        // the system prompt tracks it - landing after the static instructions.
        struct Live(Mutex<String>);
        impl crate::dynamic_prompt::DynamicPrompt for Live {
            fn render(&self) -> Option<String> {
                Some(self.0.lock().unwrap().clone())
            }
        }

        let live = Arc::new(Live(Mutex::new("state one".to_string())));
        let mut agent = Agent::new(MockModel::with_replies(["a", "b"]))
            .with_instructions("base")
            .with_dynamic_prompt(live.clone());

        agent.step("hi").await.unwrap();
        *live.0.lock().unwrap() = "state two".to_string();
        agent.step("again").await.unwrap();

        let calls = agent.model().calls();
        assert_eq!(
            calls.first().unwrap()[0],
            Message::system("base\n\nstate one")
        );
        assert_eq!(
            calls.last().unwrap()[0],
            Message::system("base\n\nstate two")
        );
    }

    #[tokio::test]
    async fn dynamic_prompt_returning_none_contributes_nothing() {
        struct Silent;
        impl crate::dynamic_prompt::DynamicPrompt for Silent {
            fn render(&self) -> Option<String> {
                None
            }
        }

        let model = MockModel::with_replies(["ok"]);
        let mut agent = Agent::new(model)
            .with_instructions("base")
            .with_dynamic_prompt(Arc::new(Silent));
        agent.step("hi").await.unwrap();

        let calls = agent.model().calls();
        assert_eq!(calls[0][0], Message::system("base"));
    }

    #[tokio::test]
    async fn capability_provider_installs_tool_capabilities() {
        let model = MockModel::with_scripted([
            MockModel::reply_tool_calls(vec![ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                arguments: serde_json::json!({"msg": "hello"}),
            }]),
            MockModel::reply_text("done"),
        ]);
        let provider = TestToolProvider;
        let mut agent = Agent::new(model).with_capability_provider(&provider);

        let reply = match agent.step("go").await.unwrap() {
            TurnResult::Message(m) => m,
            TurnResult::Handoff { .. } => panic!("unexpected handoff"),
        };
        assert_eq!(reply.text_content(), "done");
        assert_eq!(
            agent.session().messages()[2],
            Message::tool_result("call_1", "{\n  \"msg\": \"hello\"\n}")
        );
    }

    #[tokio::test]
    async fn capability_provider_installs_procedure_hook_capabilities() {
        let model = MockModel::with_replies(["ok"]);
        let calls = Arc::new(Mutex::new(0usize));
        let provider = TestHookProvider {
            calls: calls.clone(),
        };
        let mut agent = Agent::new(model).with_capability_provider(&provider);

        let result = agent.step("hi").await.unwrap();
        assert!(matches!(result, TurnResult::Message(_)));

        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn tool_call_hooks_run_through_procedure_capabilities() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let model = MockModel::with_scripted([
            MockModel::reply_tool_calls(vec![ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                arguments: serde_json::json!({"msg": "hello"}),
            }]),
            MockModel::reply_text("done"),
        ]);
        let mut agent = Agent::new(model)
            .with_tool(MockTool::echoing("echo"))
            .with_capabilities([
                Capability::Procedure(ProcedureSpec::new(
                    "record-events",
                    "Record hook events",
                    RecordingProcedure {
                        events: events.clone(),
                    },
                )),
                Capability::hook(HookEvent::BeforeToolCall, "record-events"),
                Capability::hook(HookEvent::AfterToolCall, "record-events"),
            ]);

        let result = agent.step("go").await.unwrap();
        assert!(matches!(result, TurnResult::Message(_)));

        let recorded = events.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec![HookEvent::BeforeToolCall, HookEvent::AfterToolCall]
        );
    }

    #[tokio::test]
    async fn turn_lifecycle_hooks_fire_once_around_tool_loop() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let model = MockModel::with_scripted([
            MockModel::reply_tool_calls(vec![ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                arguments: serde_json::json!({"msg": "hello"}),
            }]),
            MockModel::reply_text("done"),
        ]);
        let mut agent = Agent::new(model)
            .with_tool(MockTool::echoing("echo"))
            .with_capabilities([
                Capability::Procedure(ProcedureSpec::new(
                    "record-turn",
                    "Record turn-level hook events",
                    RecordingProcedure {
                        events: events.clone(),
                    },
                )),
                Capability::hook(HookEvent::BeforeTurn, "record-turn"),
                Capability::hook(HookEvent::AfterTurn, "record-turn"),
            ]);

        let result = agent.step("go").await.unwrap();
        assert!(matches!(result, TurnResult::Message(_)));

        let recorded = events.lock().unwrap().clone();
        // Exactly one BeforeTurn before everything, one AfterTurn after - even
        // though the turn looped through a tool call (two model replies).
        assert_eq!(recorded, vec![HookEvent::BeforeTurn, HookEvent::AfterTurn]);
    }

    #[tokio::test]
    async fn multi_turn_history_preserves_order() {
        let model = MockModel::with_replies(["a1", "a2"]);
        let mut agent = Agent::new(model).with_instructions("sys");

        let result1 = agent.step("u1").await.unwrap();
        assert!(matches!(result1, TurnResult::Message(_)));
        let result2 = agent.step("u2").await.unwrap();
        assert!(matches!(result2, TurnResult::Message(_)));

        let h = agent.session().messages();
        assert_eq!(h.len(), 4);
        assert_eq!(h[0], Message::user("u1"));
        assert_eq!(h[1], Message::assistant("a1"));
        assert_eq!(h[2], Message::user("u2"));
        assert_eq!(h[3], Message::assistant("a2"));
    }

    #[tokio::test]
    async fn model_sees_full_history_on_each_call() {
        let model = MockModel::with_replies(["a1", "a2"]);
        let mut agent = Agent::new(model);
        let result1 = agent.step("u1").await.unwrap();
        assert!(matches!(result1, TurnResult::Message(_)));
        let result2 = agent.step("u2").await.unwrap();
        assert!(matches!(result2, TurnResult::Message(_)));

        let calls = agent.model().calls();
        assert_eq!(calls.len(), 2);
        // First call: just u1.
        assert_eq!(calls[0], vec![Message::user("u1")]);
        // Second call: u1, a1, u2.
        assert_eq!(
            calls[1],
            vec![
                Message::user("u1"),
                Message::assistant("a1"),
                Message::user("u2"),
            ]
        );
    }

    #[tokio::test]
    async fn step_with_no_tools_passes_through() {
        let model = MockModel::with_replies(["plain text"]);
        let mut agent = Agent::new(model);
        let reply = match agent.step("hi").await.unwrap() {
            TurnResult::Message(m) => m,
            TurnResult::Handoff { .. } => panic!("unexpected handoff"),
        };
        assert_eq!(reply.role, Role::Assistant);
        assert_eq!(reply.text_content(), "plain text");
        assert!(agent.session().messages()[1].tool_calls.is_empty());
    }

    #[tokio::test]
    async fn step_dispatches_single_tool_call() {
        let tool = MockTool::echoing("echo");
        let model = MockModel::with_scripted([
            MockModel::reply_tool_calls(vec![ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                arguments: serde_json::json!({"msg": "hello"}),
            }]),
            MockModel::reply_text("done"),
        ]);
        let mut agent = Agent::new(model).with_tool(tool);
        let reply = match agent.step("go").await.unwrap() {
            TurnResult::Message(m) => m,
            TurnResult::Handoff { .. } => panic!("unexpected handoff"),
        };
        assert_eq!(reply.text_content(), "done");

        let h = agent.session().messages();
        assert_eq!(h.len(), 4); // user, assistant(tool_calls), tool, assistant
        assert_eq!(h[1].role, Role::Assistant);
        assert_eq!(h[1].tool_calls.len(), 1);
        assert_eq!(h[2].role, Role::Tool);
        assert_eq!(h[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(h[2].text_content(), "{\n  \"msg\": \"hello\"\n}");
    }

    /// A tool returning an image via `call_rich` (e.g. a screenshot). The image
    /// rides on a tool-result message; the text-only `call` path drops it.
    struct ImageTool;

    #[async_trait]
    impl ToolHandler for ImageTool {
        async fn call(&self, _args: serde_json::Value) -> std::result::Result<String, ToolError> {
            Ok("snapshot".to_string())
        }

        async fn call_rich(
            &self,
            _args: serde_json::Value,
        ) -> std::result::Result<ToolOutput, ToolError> {
            Ok(ToolOutput::text("snapshot").with_image(vec![1, 2, 3], "image/png"))
        }
    }

    fn image_tool() -> ToolSpec {
        ToolSpec::new(
            "shoot",
            "take a screenshot",
            serde_json::json!({"type": "object"}),
            ImageTool,
        )
    }

    #[tokio::test]
    async fn tool_image_output_is_persisted_to_session() {
        // The agent dispatches via `call_rich`, so an image a tool returns must
        // survive onto the `Role::Tool` session message (not just its text).
        let model = MockModel::with_scripted([
            MockModel::reply_tool_calls(vec![ToolCall {
                id: "call_1".into(),
                name: "shoot".into(),
                arguments: serde_json::json!({}),
            }]),
            MockModel::reply_text("done"),
        ]);
        let mut agent = Agent::new(model).with_tool(image_tool());
        agent.step("go").await.unwrap();

        let h = agent.session().messages();
        let tool_msg = h
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("tool result");
        assert!(tool_msg.has_images(), "image block dropped during dispatch");
        assert_eq!(tool_msg.text_content(), "snapshot");
    }

    #[test]
    fn cap_tool_result_images_keeps_only_most_recent() {
        let img = || ContentBlock::Image {
            data: vec![0u8; 4],
            media_type: "image/png".into(),
        };
        let tool_with_image =
            |id: &str| Message::tool_result_blocks(id, vec![ContentBlock::text("shot"), img()]);
        let mut messages = vec![
            Message::user("go"),
            tool_with_image("a"),
            tool_with_image("b"),
            tool_with_image("c"),
        ];
        cap_tool_result_images(&mut messages, 1);
        // Only the newest tool result keeps its image; older ones keep text.
        assert!(!messages[1].has_images());
        assert!(!messages[2].has_images());
        assert!(messages[3].has_images());
        assert_eq!(messages[1].text_content(), "shot");
    }

    #[test]
    fn cap_tool_result_images_placeholder_when_image_only() {
        // An image-only tool result that loses its image must not become empty.
        let mut messages = vec![
            Message::tool_result_blocks(
                "a",
                vec![ContentBlock::Image {
                    data: vec![0u8; 4],
                    media_type: "image/png".into(),
                }],
            ),
            Message::tool_result_blocks(
                "b",
                vec![ContentBlock::Image {
                    data: vec![0u8; 4],
                    media_type: "image/png".into(),
                }],
            ),
        ];
        cap_tool_result_images(&mut messages, 1);
        assert!(!messages[0].has_images());
        assert!(!messages[0].content.is_empty(), "empty tool result");
        assert!(messages[1].has_images());
    }

    #[test]
    fn cap_tool_result_images_ignores_user_images() {
        // User-supplied images are never stripped, even past the limit.
        let mut messages = vec![Message::user_blocks(vec![ContentBlock::Image {
            data: vec![0u8; 4],
            media_type: "image/png".into(),
        }])];
        cap_tool_result_images(&mut messages, 0);
        assert!(messages[0].has_images());
    }

    #[tokio::test]
    async fn build_messages_applies_image_cap() {
        // The cap is wired through `build_messages`: with a limit of 1, only the
        // most recent screenshot tool result reaches the model.
        let mut agent = Agent::new(MockModel::with_replies(["x"]))
            .with_tool(image_tool())
            .with_max_tool_result_images(1);
        for id in ["a", "b"] {
            agent
                .session
                .push(MemoryItem::Message(Message::tool_result_blocks(
                    id,
                    vec![
                        ContentBlock::text("shot"),
                        ContentBlock::Image {
                            data: vec![0u8; 4],
                            media_type: "image/png".into(),
                        },
                    ],
                )))
                .unwrap();
        }
        let built = agent.build_messages();
        let imaged = built.iter().filter(|m| m.has_images()).count();
        assert_eq!(imaged, 1, "only the most recent tool image should be sent");
    }

    #[tokio::test]
    async fn step_dispatches_multiple_tool_calls_in_one_turn() {
        let tool = MockTool::echoing("echo");
        let model = MockModel::with_scripted([
            MockModel::reply_tool_calls(vec![
                ToolCall {
                    id: "call_a".into(),
                    name: "echo".into(),
                    arguments: serde_json::json!({"x": 1}),
                },
                ToolCall {
                    id: "call_b".into(),
                    name: "echo".into(),
                    arguments: serde_json::json!({"x": 2}),
                },
            ]),
            MockModel::reply_text("ok"),
        ]);
        let mut agent = Agent::new(model).with_tool(tool);
        let reply = match agent.step("go").await.unwrap() {
            TurnResult::Message(m) => m,
            TurnResult::Handoff { .. } => panic!("unexpected handoff"),
        };
        assert_eq!(reply.text_content(), "ok");

        let h = agent.session().messages();
        assert_eq!(h.len(), 5); // user, assistant(2 calls), tool_a, tool_b, assistant
        assert_eq!(h[2].role, Role::Tool);
        assert_eq!(h[2].tool_call_id.as_deref(), Some("call_a"));
        assert_eq!(h[3].role, Role::Tool);
        assert_eq!(h[3].tool_call_id.as_deref(), Some("call_b"));
    }

    #[tokio::test]
    async fn step_handles_unknown_tool_name() {
        let model = MockModel::with_scripted([
            MockModel::reply_tool_calls(vec![ToolCall {
                id: "call_1".into(),
                name: "nonexistent".into(),
                arguments: serde_json::json!({}),
            }]),
            MockModel::reply_text("recovered"),
        ]);
        let mut agent = Agent::new(model);
        let reply = match agent.step("go").await.unwrap() {
            TurnResult::Message(m) => m,
            TurnResult::Handoff { .. } => panic!("unexpected handoff"),
        };
        assert_eq!(reply.text_content(), "recovered");

        let h = agent.session().messages();
        assert_eq!(h[2].role, Role::Tool);
        assert!(h[2].text_content().contains("unknown tool: nonexistent"));
    }

    #[tokio::test]
    async fn step_handles_tool_execution_error() {
        // MockTool always succeeds, so we test with an unknown tool (which
        // produces an execution error) and verify the loop continues.
        let model = MockModel::with_scripted([
            MockModel::reply_tool_calls(vec![ToolCall {
                id: "call_1".into(),
                name: "boom".into(),
                arguments: serde_json::json!({}),
            }]),
            MockModel::reply_text("recovered"),
        ]);
        let mut agent = Agent::new(model);
        let reply = match agent.step("go").await.unwrap() {
            TurnResult::Message(m) => m,
            TurnResult::Handoff { .. } => panic!("unexpected handoff"),
        };
        assert_eq!(reply.text_content(), "recovered");
        assert!(agent.session().messages()[2]
            .text_content()
            .contains("Error:"));
    }

    #[tokio::test]
    async fn new_session_swaps_storage_but_preserves_instructions_and_tools() {
        let model = MockModel::with_replies(["ok", "ok"]);
        let mut agent = Agent::new(model)
            .with_instructions("sys")
            .with_tool(MockTool::echoing("echo"));
        let result1 = agent.step("hi").await.unwrap();
        assert!(matches!(result1, TurnResult::Message(_)));
        assert_eq!(agent.session().messages().len(), 2); // user, assistant

        agent.new_session(InMemorySession::new());
        assert!(agent.session().messages().is_empty());
        assert_eq!(agent.tools.len(), 1);

        // Instructions still reach the model after session swap.
        let result2 = agent.step("ho").await.unwrap();
        assert!(matches!(result2, TurnResult::Message(_)));
        let calls = agent.model().calls();
        assert_eq!(calls.last().unwrap()[0], Message::system("sys"));
    }

    #[tokio::test]
    async fn clear_empties_session_but_preserves_instructions() {
        let model = MockModel::with_replies(["ok", "ok"]);
        let mut agent = Agent::new(model).with_instructions("sys");
        let result1 = agent.step("hi").await.unwrap();
        assert!(matches!(result1, TurnResult::Message(_)));
        assert_eq!(agent.session().messages().len(), 2);

        agent.session_mut().clear().unwrap();
        assert!(agent.session().messages().is_empty());

        // Instructions still reach the model after clear.
        let result2 = agent.step("ho").await.unwrap();
        assert!(matches!(result2, TurnResult::Message(_)));
        let calls = agent.model().calls();
        assert_eq!(calls.last().unwrap()[0], Message::system("sys"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn observability_logs_full_transcript_tool_args_and_tool_result() {
        let tool = MockTool::echoing("echo");
        let model = MockModel::with_scripted([
            MockModel::reply_tool_calls(vec![ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                arguments: serde_json::json!({"msg": "hello"}),
            }]),
            MockModel::reply_text("done"),
        ]);
        let mut agent = Agent::new(model).with_tool(tool);

        let (reply, logs) = capture_observability(async {
            agent.step("go").await.map(|result| match result {
                TurnResult::Message(m) => m.text_content(),
                TurnResult::Handoff { .. } => panic!("unexpected handoff"),
            })
        })
        .await;

        assert_eq!(reply.unwrap(), "done");
        assert!(logs.contains("agent.turn.model_input"), "{logs}");
        assert!(logs.contains("llm.complete"), "{logs}");
        assert!(logs.contains("tool.call"), "{logs}");
        assert!(logs.contains("agent.turn.finished"), "{logs}");
        assert!(logs.contains("go"), "{logs}");
        assert!(logs.contains("hello"), "{logs}");
        assert!(logs.contains("result"), "{logs}");
        assert!(logs.contains("msg"), "{logs}");
    }

    async fn capture_observability<F, T>(future: F) -> (T, String)
    where
        F: std::future::Future<Output = T>,
    {
        let writer = SharedWriter::default();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(writer.clone())
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("set test tracing subscriber");
        tracing::callsite::rebuild_interest_cache();
        let result = future.await;
        (result, writer.contents())
    }

    #[derive(Clone, Default)]
    struct SharedWriter {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl SharedWriter {
        fn contents(&self) -> String {
            String::from_utf8(self.bytes.lock().unwrap().clone()).unwrap()
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedWriter {
        type Writer = SharedWrite;

        fn make_writer(&'a self) -> Self::Writer {
            SharedWrite {
                bytes: self.bytes.clone(),
            }
        }
    }

    struct SharedWrite {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl io::Write for SharedWrite {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.bytes.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct TestPromptProvider {
        id: String,
        text: String,
        activation: Activation,
    }

    impl TestPromptProvider {
        fn new(id: &str, text: &str) -> Self {
            Self {
                id: id.to_string(),
                text: text.to_string(),
                activation: Activation::Always,
            }
        }

        fn with_activation(mut self, activation: Activation) -> Self {
            self.activation = activation;
            self
        }
    }

    impl CapabilityProvider for TestPromptProvider {
        fn id(&self) -> &str {
            &self.id
        }

        fn capabilities(&self) -> Vec<Capability> {
            vec![Capability::Prompt(PromptSpec {
                id: self.id.clone(),
                text: self.text.clone(),
                activation: self.activation.clone(),
            })]
        }
    }

    struct TestToolProvider;

    impl CapabilityProvider for TestToolProvider {
        fn id(&self) -> &str {
            "test-tools"
        }

        fn capabilities(&self) -> Vec<Capability> {
            vec![Capability::tool(MockTool::echoing("echo"))]
        }
    }

    struct CountingProcedure {
        calls: Arc<Mutex<usize>>,
    }

    #[async_trait]
    impl ProcedureHandler for CountingProcedure {
        async fn handle(
            &self,
            invocation: &HookInvocation,
            _ctx: &mut dyn CommandContext,
        ) -> sweet_core::Result<()> {
            assert_eq!(invocation.event, HookEvent::BeforeModelCall);
            *self.calls.lock().unwrap() += 1;
            Ok(())
        }
    }

    struct RecordingProcedure {
        events: Arc<Mutex<Vec<HookEvent>>>,
    }

    #[async_trait]
    impl ProcedureHandler for RecordingProcedure {
        async fn handle(
            &self,
            invocation: &HookInvocation,
            _ctx: &mut dyn CommandContext,
        ) -> sweet_core::Result<()> {
            self.events.lock().unwrap().push(invocation.event.clone());
            Ok(())
        }
    }

    struct TestHookProvider {
        calls: Arc<Mutex<usize>>,
    }

    impl CapabilityProvider for TestHookProvider {
        fn id(&self) -> &str {
            "test-hooks"
        }

        fn capabilities(&self) -> Vec<Capability> {
            vec![
                Capability::Procedure(ProcedureSpec::new(
                    "counting",
                    "Count model-call hooks",
                    CountingProcedure {
                        calls: self.calls.clone(),
                    },
                )),
                Capability::hook(HookEvent::BeforeModelCall, "counting"),
            ]
        }
    }

    #[test]
    fn repair_orphaned_tool_calls_does_nothing_on_clean_session() {
        let model = MockModel::with_replies::<[_; 0], &str>([]);
        let mut agent = Agent::new(model);
        agent
            .session
            .push(MemoryItem::Message(Message::user("hi")))
            .unwrap();
        agent
            .session
            .push(MemoryItem::Message(Message::assistant("hello")))
            .unwrap();
        let repaired = agent.repair_orphaned_tool_calls().unwrap();
        assert!(!repaired);
        assert_eq!(agent.session().items().len(), 2);
    }

    #[test]
    fn repair_orphaned_tool_calls_inserts_synthetic_results() {
        let model = MockModel::with_replies::<[_; 0], &str>([]);
        let mut agent = Agent::new(model);
        agent
            .session
            .push(MemoryItem::Message(Message::user("do things")))
            .unwrap();
        let assistant = Message::with_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::Value::Null,
        }]);
        agent.session.push(MemoryItem::Message(assistant)).unwrap();

        assert_eq!(agent.session().items().len(), 2);
        let repaired = agent.repair_orphaned_tool_calls().unwrap();
        assert!(repaired);

        let items = agent.session().items();
        assert_eq!(items.len(), 3);
        let MemoryItem::Message(tool_result) = &items[2];
        assert_eq!(tool_result.role, Role::Tool);
        assert_eq!(tool_result.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn repair_orphaned_tool_calls_skips_resolved_calls() {
        let model = MockModel::with_replies::<[_; 0], &str>([]);
        let mut agent = Agent::new(model);
        agent
            .session
            .push(MemoryItem::Message(Message::user("do things")))
            .unwrap();
        let assistant = Message::with_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::Value::Null,
        }]);
        agent.session.push(MemoryItem::Message(assistant)).unwrap();
        agent
            .session
            .push(MemoryItem::Message(Message::tool_result("call_1", "done")))
            .unwrap();

        assert_eq!(agent.session().items().len(), 3);
        let repaired = agent.repair_orphaned_tool_calls().unwrap();
        assert!(!repaired);
        assert_eq!(agent.session().items().len(), 3);
    }

    #[test]
    fn repair_orphaned_tool_calls_repairs_only_orphans_in_multi_call() {
        let model = MockModel::with_replies::<[_; 0], &str>([]);
        let mut agent = Agent::new(model);
        agent
            .session
            .push(MemoryItem::Message(Message::user("do things")))
            .unwrap();
        let assistant = Message::with_tool_calls(vec![
            ToolCall {
                id: "call_1".into(),
                name: "bash".into(),
                arguments: serde_json::Value::Null,
            },
            ToolCall {
                id: "call_2".into(),
                name: "read".into(),
                arguments: serde_json::Value::Null,
            },
        ]);
        agent.session.push(MemoryItem::Message(assistant)).unwrap();
        agent
            .session
            .push(MemoryItem::Message(Message::tool_result("call_1", "done")))
            .unwrap();

        assert_eq!(agent.session().items().len(), 3);
        let repaired = agent.repair_orphaned_tool_calls().unwrap();
        assert!(repaired);

        let items = agent.session().items();
        assert_eq!(items.len(), 4);
        let MemoryItem::Message(tool_result) = &items[3];
        assert_eq!(tool_result.role, Role::Tool);
        assert_eq!(tool_result.tool_call_id.as_deref(), Some("call_2"));
    }
}
