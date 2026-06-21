// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Model-facing memory tools.
//!
//! The application binds the scopes: saves land in
//! [`MemoryToolset::default_scope`], recall sees
//! [`MemoryToolset::searchable_scopes`], and update/delete refuse records
//! outside those scopes. The model never chooses a scope, so memories can't
//! leak across users or projects no matter what arguments it produces.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use sweet_core::{
    Memory, MemoryHit, MemoryId, MemoryQuery, MemoryRecord, MemoryScope, ToolError, ToolHandler,
    ToolRisk, ToolSpec,
};

/// Memory store plus the scope binding for one agent's tools.
#[derive(Clone)]
pub struct MemoryToolset {
    store: Arc<dyn Memory>,
    default_scope: MemoryScope,
    searchable_scopes: Vec<MemoryScope>,
    source_session: Option<String>,
}

impl MemoryToolset {
    /// Saves land in (and searches see) `default_scope`.
    pub fn new(store: Arc<dyn Memory>, default_scope: MemoryScope) -> Self {
        Self {
            store,
            searchable_scopes: vec![default_scope.clone()],
            default_scope,
            source_session: None,
        }
    }

    /// Widen what search/list/update/delete see (e.g. the user scope in
    /// addition to a project default). The default scope is always included.
    pub fn with_searchable_scopes(mut self, scopes: impl IntoIterator<Item = MemoryScope>) -> Self {
        self.searchable_scopes = scopes.into_iter().collect();
        if !self.searchable_scopes.contains(&self.default_scope) {
            self.searchable_scopes.insert(0, self.default_scope.clone());
        }
        self
    }

    /// Record this session id as provenance on saves.
    pub fn with_source_session(mut self, session_id: impl Into<String>) -> Self {
        self.source_session = Some(session_id.into());
        self
    }

    /// A record is touchable when its scope is visible to this toolset.
    fn in_scope(&self, record: &MemoryRecord) -> bool {
        self.searchable_scopes.contains(&record.scope)
    }

    /// Fetch a record the model referenced by id, refusing out-of-scope ones
    /// (indistinguishable from not-found, by design).
    async fn fetch_in_scope(&self, id: &str) -> Result<(MemoryId, MemoryRecord), ToolError> {
        let id: MemoryId = id
            .parse()
            .map_err(|_| ToolError::Execution(format!("invalid memory id: {id}").into()))?;
        let record = self
            .store
            .get(&id)
            .await
            .map_err(sweet_core::execution_error)?
            .filter(|r| self.in_scope(r))
            .ok_or_else(|| ToolError::Execution(format!("no memory with id {id}").into()))?;
        Ok((id, record))
    }
}

/// All four memory tools bound to one toolset.
pub fn memory_tools(toolset: MemoryToolset) -> Vec<ToolSpec> {
    vec![
        memory_save_tool(toolset.clone()),
        memory_search_tool(toolset.clone()),
        memory_update_tool(toolset.clone()),
        memory_delete_tool(toolset),
    ]
}

fn format_hit(hit: &MemoryHit) -> String {
    format_record(&hit.record)
}

fn format_record(record: &MemoryRecord) -> String {
    let tags = if record.tags.is_empty() {
        String::new()
    } else {
        format!(" [tags: {}]", record.tags.join(", "))
    };
    format!("- ({}){} {}", record.id, tags, record.content)
}

#[derive(Deserialize)]
struct SaveArgs {
    content: String,
    #[serde(default)]
    tags: Vec<String>,
}

struct SaveHandler(MemoryToolset);

#[async_trait]
impl ToolHandler for SaveHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: SaveArgs = serde_json::from_value(args)?;
        let record = self
            .0
            .store
            .save(
                self.0.default_scope.clone(),
                &args.content,
                &args.tags,
                self.0.source_session.as_deref(),
            )
            .await
            .map_err(sweet_core::execution_error)?;
        Ok(format!("Saved memory ({})", record.id))
    }
}

/// `memory_save`: persist one durable fact in the application-chosen scope.
pub fn memory_save_tool(toolset: MemoryToolset) -> ToolSpec {
    ToolSpec::new(
        "memory_save",
        "Save a durable memory for future sessions: user preferences, decisions, \
         and facts worth remembering long-term. Do not save transient task state, \
         anything already remembered, or secrets/credentials. Keep each memory to \
         one self-contained fact.",
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact to remember, phrased to make sense without this conversation's context."
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional short categorization tags."
                }
            },
            "required": ["content"]
        }),
        SaveHandler(toolset),
    )
    .with_risk(ToolRisk::FileWrite)
}

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
    limit: Option<usize>,
    #[serde(default)]
    tags: Vec<String>,
}

struct SearchHandler(MemoryToolset);

#[async_trait]
impl ToolHandler for SearchHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: SearchArgs = serde_json::from_value(args)?;
        let query = MemoryQuery::new()
            .with_text(args.query)
            .with_scopes(self.0.searchable_scopes.clone())
            .with_tags(args.tags)
            .with_limit(args.limit.unwrap_or(5));
        let hits = self
            .0
            .store
            .search(&query)
            .await
            .map_err(sweet_core::execution_error)?;
        if hits.is_empty() {
            return Ok("No matching memories.".to_string());
        }
        Ok(hits.iter().map(format_hit).collect::<Vec<_>>().join("\n"))
    }
}

/// `memory_search`: recall memories relevant to a query.
pub fn memory_search_tool(toolset: MemoryToolset) -> ToolSpec {
    ToolSpec::new(
        "memory_search",
        "Search long-term memories saved in previous sessions. Use when past \
         preferences, decisions, or facts could be relevant. Results include each \
         memory's id for memory_update/memory_delete.",
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search query."},
                "limit": {"type": "integer", "description": "Maximum results (default 5)."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only return memories carrying all of these tags."
                }
            },
            "required": ["query"]
        }),
        SearchHandler(toolset),
    )
    .with_risk(ToolRisk::ReadOnly)
}

#[derive(Deserialize)]
struct UpdateArgs {
    id: String,
    content: Option<String>,
    tags: Option<Vec<String>>,
}

struct UpdateHandler(MemoryToolset);

#[async_trait]
impl ToolHandler for UpdateHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: UpdateArgs = serde_json::from_value(args)?;
        if args.content.is_none() && args.tags.is_none() {
            return Err(ToolError::Execution(
                "provide content and/or tags to update".into(),
            ));
        }
        let (id, _) = self.0.fetch_in_scope(&args.id).await?;
        let record = self
            .0
            .store
            .update(&id, args.content.as_deref(), args.tags.as_deref())
            .await
            .map_err(sweet_core::execution_error)?;
        Ok(format!("Updated memory:\n{}", format_record(&record)))
    }
}

/// `memory_update`: rewrite an outdated memory in place.
pub fn memory_update_tool(toolset: MemoryToolset) -> ToolSpec {
    ToolSpec::new(
        "memory_update",
        "Update an existing memory when it is outdated or imprecise - prefer this \
         over saving a near-duplicate. Takes the id returned by memory_search or \
         memory_save.",
        json!({
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Id of the memory to update."},
                "content": {"type": "string", "description": "Replacement content."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Replacement tags."
                }
            },
            "required": ["id"]
        }),
        UpdateHandler(toolset),
    )
    .with_risk(ToolRisk::FileWrite)
}

#[derive(Deserialize)]
struct DeleteArgs {
    id: String,
}

struct DeleteHandler(MemoryToolset);

#[async_trait]
impl ToolHandler for DeleteHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: DeleteArgs = serde_json::from_value(args)?;
        let (id, _) = self.0.fetch_in_scope(&args.id).await?;
        self.0
            .store
            .delete(&id)
            .await
            .map_err(sweet_core::execution_error)?;
        Ok(format!("Deleted memory ({id})"))
    }
}

/// `memory_delete`: remove a memory that is wrong or no longer wanted.
pub fn memory_delete_tool(toolset: MemoryToolset) -> ToolSpec {
    ToolSpec::new(
        "memory_delete",
        "Delete a memory that is wrong, obsolete, or that the user asked to forget. \
         Takes the id returned by memory_search.",
        json!({
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Id of the memory to delete."}
            },
            "required": ["id"]
        }),
        DeleteHandler(toolset),
    )
    .with_risk(ToolRisk::FileWrite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_core::EphemeralMemory;

    fn toolset(store: Arc<dyn Memory>) -> MemoryToolset {
        MemoryToolset::new(store, MemoryScope::Project("p".into()))
            .with_searchable_scopes([
                MemoryScope::Project("p".into()),
                MemoryScope::User("u".into()),
            ])
            .with_source_session("sess-1")
    }

    #[tokio::test]
    async fn save_then_search_roundtrip() {
        let store: Arc<dyn Memory> = Arc::new(EphemeralMemory::new());
        let tools = memory_tools(toolset(store.clone()));
        let save = tools.iter().find(|t| t.name == "memory_save").unwrap();
        let search = tools.iter().find(|t| t.name == "memory_search").unwrap();

        let reply = save
            .call(json!({"content": "user prefers dark mode", "tags": ["prefs"]}))
            .await
            .unwrap();
        assert!(reply.starts_with("Saved memory ("));

        let reply = search.call(json!({"query": "dark mode"})).await.unwrap();
        assert!(reply.contains("user prefers dark mode"));
        assert!(reply.contains("[tags: prefs]"));

        // Saves land in the default scope with session provenance.
        let hits = store
            .search(&MemoryQuery::new().with_text("dark"))
            .await
            .unwrap();
        assert_eq!(hits[0].record.scope, MemoryScope::Project("p".into()));
        assert_eq!(hits[0].record.source_session.as_deref(), Some("sess-1"));
    }

    #[tokio::test]
    async fn search_reports_no_matches() {
        let store: Arc<dyn Memory> = Arc::new(EphemeralMemory::new());
        let search = memory_search_tool(toolset(store));
        let reply = search.call(json!({"query": "anything"})).await.unwrap();
        assert_eq!(reply, "No matching memories.");
    }

    #[tokio::test]
    async fn update_and_delete_by_id() {
        let store: Arc<dyn Memory> = Arc::new(EphemeralMemory::new());
        let saved = store
            .save(MemoryScope::Project("p".into()), "draft", &[], None)
            .await
            .unwrap();
        let ts = toolset(store.clone());
        let update = memory_update_tool(ts.clone());
        let delete = memory_delete_tool(ts);

        let reply = update
            .call(json!({"id": saved.id.to_string(), "content": "final"}))
            .await
            .unwrap();
        assert!(reply.contains("final"));

        delete
            .call(json!({"id": saved.id.to_string()}))
            .await
            .unwrap();
        assert!(store.get(&saved.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn update_requires_some_change() {
        let store: Arc<dyn Memory> = Arc::new(EphemeralMemory::new());
        let update = memory_update_tool(toolset(store));
        let err = update.call(json!({"id": "irrelevant"})).await.unwrap_err();
        assert!(err.to_string().contains("content and/or tags"));
    }

    #[tokio::test]
    async fn out_of_scope_records_are_invisible() {
        let store: Arc<dyn Memory> = Arc::new(EphemeralMemory::new());
        // A record in someone else's scope.
        let foreign = store
            .save(MemoryScope::User("other".into()), "secret", &[], None)
            .await
            .unwrap();
        let ts = toolset(store.clone());

        let err = memory_delete_tool(ts.clone())
            .call(json!({"id": foreign.id.to_string()}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no memory with id"));
        assert!(store.get(&foreign.id).await.unwrap().is_some());

        let reply = memory_search_tool(ts)
            .call(json!({"query": "secret"}))
            .await
            .unwrap();
        assert_eq!(reply, "No matching memories.");
    }

    #[tokio::test]
    async fn bad_args_are_invalid_args_errors() {
        let store: Arc<dyn Memory> = Arc::new(EphemeralMemory::new());
        let save = memory_save_tool(toolset(store));
        let err = save.call(json!({"tags": ["x"]})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs(_)));
    }
}
