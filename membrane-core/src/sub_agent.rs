use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::AgentOutput;
use crate::error::Error;
use crate::message::Message;
use crate::tool::ToolDefinition;

/// Pinned boxed future for a single agent execution.
pub type AgentFuture<'a> = Pin<Box<dyn Future<Output = Result<AgentOutput, Error>> + Send + 'a>>;

/// Pinned boxed future for parallel (fan-out) agent execution.
pub type ParallelAgentFuture<'a> =
    Pin<Box<dyn Future<Output = Vec<Result<AgentOutput, Error>>> + Send + 'a>>;

/// Type-erased agent execution interface.
///
/// Allows `Agent<P>` instances with different provider types to be used
/// polymorphically as sub-agents. Uses `Pin<Box<dyn Future>>` for
/// dyn-compatibility (same pattern as the `Tool` trait).
pub trait AgentExecutor: Send + Sync {
    /// Run the agent with the given messages and return the output.
    fn run(&self, messages: Vec<Message>) -> AgentFuture<'_>;

    /// Run the agent with multiple message sets in parallel (fan-out).
    ///
    /// Each message set is executed concurrently via `futures_util::future::join_all`.
    /// Results are returned in the same order as the input message sets.
    /// Individual failures do not affect other executions.
    fn run_parallel(&self, message_sets: Vec<Vec<Message>>) -> ParallelAgentFuture<'_> {
        Box::pin(async move {
            let futs: Vec<_> = message_sets
                .into_iter()
                .map(|msgs| self.run(msgs))
                .collect();
            futures_util::future::join_all(futs).await
        })
    }
}

/// How a sub-agent processes its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubAgentMode {
    /// Single query execution. Input schema: `{ "query": string }`.
    Single,
    /// Parallel fan-out execution. Input schema: `{ "tasks": [string] }`.
    /// Each task is executed as an independent agent run concurrently.
    Parallel,
}

/// Entry for a sub-agent registered in a parent agent.
///
/// Sub-agents are presented to the LLM as tool definitions, but are
/// dispatched separately from deterministic tools internally.
///
/// Use [`SubAgentEntry::single`] for a standard sub-agent (one query at a time)
/// or [`SubAgentEntry::parallel`] for a fan-out sub-agent (multiple tasks concurrently).
pub struct SubAgentEntry {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) input_schema: serde_json::Value,
    pub(crate) agent: Arc<dyn AgentExecutor>,
    pub(crate) mode: SubAgentMode,
}

impl SubAgentEntry {
    /// Create a single-query sub-agent with the default schema (`{ "query": string }`).
    pub fn single(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Arc<dyn AgentExecutor>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: default_single_schema(),
            agent,
            mode: SubAgentMode::Single,
        }
    }

    /// Create a parallel fan-out sub-agent with the default schema (`{ "tasks": [string] }`).
    ///
    /// When invoked, each task string is sent as an independent query to the
    /// agent and all queries are executed concurrently.
    pub fn parallel(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Arc<dyn AgentExecutor>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: default_parallel_schema(),
            agent,
            mode: SubAgentMode::Parallel,
        }
    }

    /// Create a sub-agent entry with a custom input schema and mode.
    pub fn with_schema(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Arc<dyn AgentExecutor>,
        input_schema: serde_json::Value,
        mode: SubAgentMode,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            agent,
            mode,
        }
    }

    /// Convert to a `ToolDefinition` for the LLM.
    pub(crate) fn to_tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }
}

// Keep backward-compatible alias
impl SubAgentEntry {
    /// Alias for [`SubAgentEntry::single`].
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Arc<dyn AgentExecutor>,
    ) -> Self {
        Self::single(name, description, agent)
    }
}

fn default_single_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The task or question to delegate to the sub-agent"
            }
        },
        "required": ["query"]
    })
}

fn default_parallel_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": { "type": "string" },
                "description": "List of tasks or questions to execute in parallel"
            }
        },
        "required": ["tasks"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, AgentConfig};
    use crate::provider::{ChatResponse, LlmProvider, StopReason, Usage};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockProvider {
        responses: Vec<ChatResponse>,
        call_count: AtomicUsize,
    }

    impl LlmProvider for MockProvider {
        async fn chat(
            &self,
            _request: crate::provider::ChatRequest,
        ) -> Result<ChatResponse, Error> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            self.responses
                .get(idx)
                .cloned()
                .ok_or_else(|| Error::Provider("No more mock responses".to_string()))
        }
    }

    #[tokio::test]
    async fn agent_executor_trait_object() {
        let provider = MockProvider {
            responses: vec![ChatResponse {
                content: vec![crate::message::Content::Text {
                    text: "Hello from agent".to_string(),
                }],
                usage: Usage {
                    input_tokens: 5,
                    output_tokens: 3,
                },
                stop_reason: StopReason::EndTurn,
            }],
            call_count: AtomicUsize::new(0),
        };

        let agent = Agent::without_system_prompt(
            provider,
            vec![],
            AgentConfig {
                model: "test".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let executor: Arc<dyn AgentExecutor> = Arc::new(agent);
        let output = executor
            .run(vec![Message::user("test")])
            .await
            .expect("test");
        assert_eq!(output.response, "Hello from agent");
    }

    #[test]
    fn sub_agent_entry_single_schema() {
        let provider = MockProvider {
            responses: vec![],
            call_count: AtomicUsize::new(0),
        };
        let agent = Agent::without_system_prompt(
            provider,
            vec![],
            AgentConfig {
                model: "test".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let entry = SubAgentEntry::single("researcher", "Research topics", Arc::new(agent));
        assert_eq!(entry.name, "researcher");
        assert_eq!(entry.mode, SubAgentMode::Single);

        let schema_str = entry.input_schema.to_string();
        assert!(schema_str.contains("query"));
        assert!(schema_str.contains("required"));
    }

    #[test]
    fn sub_agent_entry_parallel_schema() {
        let provider = MockProvider {
            responses: vec![],
            call_count: AtomicUsize::new(0),
        };
        let agent = Agent::without_system_prompt(
            provider,
            vec![],
            AgentConfig {
                model: "test".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let entry = SubAgentEntry::parallel("researcher", "Research topics", Arc::new(agent));
        assert_eq!(entry.name, "researcher");
        assert_eq!(entry.mode, SubAgentMode::Parallel);

        let schema_str = entry.input_schema.to_string();
        assert!(schema_str.contains("tasks"));
        assert!(schema_str.contains("array"));
    }

    #[test]
    fn sub_agent_entry_to_tool_definition() {
        let provider = MockProvider {
            responses: vec![],
            call_count: AtomicUsize::new(0),
        };
        let agent = Agent::without_system_prompt(
            provider,
            vec![],
            AgentConfig {
                model: "test".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let entry = SubAgentEntry::single("test_agent", "Test description", Arc::new(agent));
        let tool_def = entry.to_tool_definition();

        assert_eq!(tool_def.name, "test_agent");
        assert_eq!(tool_def.description, "Test description");
        assert_eq!(tool_def.input_schema, entry.input_schema);
    }

    #[tokio::test]
    async fn run_parallel_multiple_queries() {
        let provider = MockProvider {
            responses: vec![
                ChatResponse {
                    content: vec![crate::message::Content::Text {
                        text: "Response A".to_string(),
                    }],
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 5,
                    },
                    stop_reason: StopReason::EndTurn,
                },
                ChatResponse {
                    content: vec![crate::message::Content::Text {
                        text: "Response B".to_string(),
                    }],
                    usage: Usage {
                        input_tokens: 8,
                        output_tokens: 4,
                    },
                    stop_reason: StopReason::EndTurn,
                },
                ChatResponse {
                    content: vec![crate::message::Content::Text {
                        text: "Response C".to_string(),
                    }],
                    usage: Usage {
                        input_tokens: 12,
                        output_tokens: 6,
                    },
                    stop_reason: StopReason::EndTurn,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let agent = Agent::without_system_prompt(
            provider,
            vec![],
            AgentConfig {
                model: "test".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let results = agent
            .run_parallel(vec![
                vec![Message::user("Query A")],
                vec![Message::user("Query B")],
                vec![Message::user("Query C")],
            ])
            .await;

        assert_eq!(results.len(), 3);
        let responses: Vec<String> = results
            .into_iter()
            .map(|r| r.expect("should succeed").response)
            .collect();
        assert!(responses.contains(&"Response A".to_string()));
        assert!(responses.contains(&"Response B".to_string()));
        assert!(responses.contains(&"Response C".to_string()));
    }

    #[tokio::test]
    async fn run_parallel_via_agent_executor_trait() {
        let provider = MockProvider {
            responses: vec![
                ChatResponse {
                    content: vec![crate::message::Content::Text {
                        text: "First".to_string(),
                    }],
                    usage: Usage::default(),
                    stop_reason: StopReason::EndTurn,
                },
                ChatResponse {
                    content: vec![crate::message::Content::Text {
                        text: "Second".to_string(),
                    }],
                    usage: Usage::default(),
                    stop_reason: StopReason::EndTurn,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let agent = Agent::without_system_prompt(
            provider,
            vec![],
            AgentConfig {
                model: "test".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let executor: Arc<dyn AgentExecutor> = Arc::new(agent);
        let results = executor
            .run_parallel(vec![vec![Message::user("Q1")], vec![Message::user("Q2")]])
            .await;

        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn run_parallel_partial_failure() {
        let provider = MockProvider {
            responses: vec![
                ChatResponse {
                    content: vec![crate::message::Content::Text {
                        text: "OK".to_string(),
                    }],
                    usage: Usage::default(),
                    stop_reason: StopReason::EndTurn,
                },
                ChatResponse {
                    content: vec![crate::message::Content::Text {
                        text: "OK".to_string(),
                    }],
                    usage: Usage::default(),
                    stop_reason: StopReason::EndTurn,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let agent = Agent::without_system_prompt(
            provider,
            vec![],
            AgentConfig {
                model: "test".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let results = agent
            .run_parallel(vec![
                vec![Message::user("Q1")],
                vec![Message::user("Q2")],
                vec![Message::user("Q3")],
            ])
            .await;

        assert_eq!(results.len(), 3);
        let success_count = results.iter().filter(|r| r.is_ok()).count();
        let error_count = results.iter().filter(|r| r.is_err()).count();
        assert_eq!(success_count, 2);
        assert_eq!(error_count, 1);
    }
}
