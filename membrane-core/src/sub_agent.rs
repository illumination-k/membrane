use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::AgentOutput;
use crate::error::Error;
use crate::message::Message;
use crate::tool::ToolDefinition;

/// Type-erased agent execution interface.
///
/// Allows `Agent<P>` instances with different provider types to be used
/// polymorphically as sub-agents. Uses `Pin<Box<dyn Future>>` for
/// dyn-compatibility (same pattern as the `Tool` trait).
pub trait AgentExecutor: Send + Sync {
    /// Run the agent with the given messages and return the output.
    fn run(
        &self,
        messages: Vec<Message>,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput, Error>> + Send + '_>>;
}

/// Entry for a sub-agent registered in a parent agent.
///
/// Sub-agents are presented to the LLM as tool definitions, but are
/// dispatched separately from deterministic tools internally.
pub struct SubAgentEntry {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) input_schema: serde_json::Value,
    pub(crate) agent: Arc<dyn AgentExecutor>,
}

impl SubAgentEntry {
    /// Create a new sub-agent entry with the default input schema.
    ///
    /// The default schema expects a single `"query"` string parameter.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Arc<dyn AgentExecutor>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: default_sub_agent_schema(),
            agent,
        }
    }

    /// Create a sub-agent entry with a custom input schema.
    pub fn with_schema(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Arc<dyn AgentExecutor>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            agent,
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

fn default_sub_agent_schema() -> serde_json::Value {
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
    fn sub_agent_entry_default_schema() {
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

        let entry = SubAgentEntry::new("researcher", "Research topics", Arc::new(agent));
        assert_eq!(entry.name, "researcher");
        assert_eq!(entry.description, "Research topics");

        let schema_str = entry.input_schema.to_string();
        assert!(schema_str.contains("query"));
        assert!(schema_str.contains("required"));
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

        let entry = SubAgentEntry::new("test_agent", "Test description", Arc::new(agent));
        let tool_def = entry.to_tool_definition();

        assert_eq!(tool_def.name, "test_agent");
        assert_eq!(tool_def.description, "Test description");
        assert_eq!(tool_def.input_schema, entry.input_schema);
    }
}
