use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::Instrument;

use crate::context::{ContextBuilder, DefaultContextBuilder};
use crate::error::Error;
use crate::message::{Content, Message, Role};
use crate::provider::{ChatRequest, ChatResponse, LlmProvider, ResponseFormat, StopReason, Usage};
use crate::stop_condition::{AgentStopReason, StopCondition, StopContext};
use crate::sub_agent::{AgentExecutor, AgentFuture, SubAgentEntry, SubAgentMode};
use crate::tool::Tool;

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub model: String,
    pub max_iterations: usize,
    /// Provider-specific parameters merged into each API request.
    /// Use this for `max_tokens`, `temperature`, `max_completion_tokens`, etc.
    pub extra_params: serde_json::Map<String, serde_json::Value>,
}

pub struct Agent<P: LlmProvider> {
    provider: P,
    tools: Vec<Box<dyn Tool>>,
    sub_agents: Vec<SubAgentEntry>,
    config: AgentConfig,
    context_builder: Box<dyn ContextBuilder>,
    stop_conditions: Vec<Box<dyn StopCondition>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    pub response: String,
    pub steps: Vec<AgentStep>,
    pub total_usage: Usage,
    pub stop_reason: AgentStopReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredAgentOutput<O> {
    pub response: O,
    pub steps: Vec<AgentStep>,
    pub total_usage: Usage,
    pub stop_reason: AgentStopReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStep {
    LlmCall {
        request_messages: usize,
        response: ChatResponse,
    },
    ToolExecution {
        name: String,
        input: serde_json::Value,
        output: String,
        is_error: bool,
    },
    SubAgentExecution {
        name: String,
        input: serde_json::Value,
        output: AgentOutput,
    },
    ParallelSubAgentExecution {
        name: String,
        tasks: Vec<String>,
        outputs: Vec<AgentOutput>,
        errors: Vec<String>,
    },
}

impl<P: LlmProvider> Agent<P> {
    pub fn new(
        provider: P,
        tools: Vec<Box<dyn Tool>>,
        config: AgentConfig,
        context_builder: Box<dyn ContextBuilder>,
    ) -> Self {
        Self {
            provider,
            tools,
            sub_agents: Vec::new(),
            config,
            context_builder,
            stop_conditions: Vec::new(),
        }
    }

    /// Add a single stop condition to this agent.
    ///
    /// Stop conditions are checked after each iteration (after tool execution).
    /// If any condition fires, the loop terminates with `AgentStopReason::StopCondition`.
    /// Multiple conditions act as OR — any one firing causes a stop.
    /// Use `.or()` / `.and()` on the trait for composing conditions before adding.
    pub fn with_stop_condition(mut self, condition: impl StopCondition + 'static) -> Self {
        self.stop_conditions.push(Box::new(condition));
        self
    }

    /// Add multiple stop conditions to this agent.
    pub fn with_stop_conditions(mut self, conditions: Vec<Box<dyn StopCondition>>) -> Self {
        self.stop_conditions = conditions;
        self
    }

    /// Add a sub-agent to this agent.
    ///
    /// Sub-agents are autonomous reasoning entities that the parent agent can
    /// invoke. They are presented to the LLM as tool definitions but dispatched
    /// separately from deterministic tools.
    ///
    /// The default input schema expects a single `"query"` string parameter.
    pub fn with_sub_agent(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        agent: impl AgentExecutor + 'static,
    ) -> Self {
        let entry = SubAgentEntry::single(name, description, Arc::new(agent));
        self.sub_agents.push(entry);
        self
    }

    /// Add a parallel (fan-out) sub-agent to this agent.
    ///
    /// The LLM provides an array of tasks via `{ "tasks": ["...", "..."] }`.
    /// Each task is executed as an independent agent run concurrently, and the
    /// combined results are returned to the LLM as a single tool result.
    pub fn with_parallel_sub_agent(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        agent: impl AgentExecutor + 'static,
    ) -> Self {
        let entry = SubAgentEntry::parallel(name, description, Arc::new(agent));
        self.sub_agents.push(entry);
        self
    }

    /// Create a new agent with a simple system prompt.
    pub fn with_system_prompt(
        provider: P,
        tools: Vec<Box<dyn Tool>>,
        config: AgentConfig,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self::new(
            provider,
            tools,
            config,
            Box::new(DefaultContextBuilder::new(system_prompt)),
        )
    }

    /// Create a new agent without a system prompt.
    pub fn without_system_prompt(
        provider: P,
        tools: Vec<Box<dyn Tool>>,
        config: AgentConfig,
    ) -> Self {
        Self::new(
            provider,
            tools,
            config,
            Box::new(DefaultContextBuilder::default()),
        )
    }

    /// Run multiple message sets in parallel (fan-out) and collect results.
    ///
    /// Each message set is executed as an independent ReAct loop concurrently.
    /// Results are returned in the same order as the input. Individual failures
    /// do not affect other executions.
    #[tracing::instrument(skip_all, fields(model = %self.config.model, count = message_sets.len()))]
    pub async fn run_parallel(
        &self,
        message_sets: Vec<Vec<Message>>,
    ) -> Vec<Result<AgentOutput, Error>> {
        let futs: Vec<_> = message_sets
            .into_iter()
            .map(|msgs| self.run(msgs))
            .collect();
        futures_util::future::join_all(futs).await
    }

    /// Run the ReAct loop with the given messages and return a text response.
    #[tracing::instrument(skip_all, fields(model = %self.config.model))]
    pub async fn run(&self, messages: Vec<Message>) -> Result<AgentOutput, Error> {
        let (response_content, steps, total_usage, stop_reason) =
            self.react_loop(messages, None).await?;

        let response = extract_text(&response_content);
        Ok(AgentOutput {
            response,
            steps,
            total_usage,
            stop_reason,
        })
    }

    /// Run the ReAct loop and parse the final response into a structured type.
    #[tracing::instrument(skip_all, fields(model = %self.config.model))]
    pub async fn run_structured<O>(
        &self,
        messages: Vec<Message>,
    ) -> Result<StructuredAgentOutput<O>, Error>
    where
        O: DeserializeOwned + JsonSchema,
    {
        let schema = schemars::schema_for!(O);
        let schema_value = serde_json::to_value(schema)?;
        let response_format = ResponseFormat {
            name: std::any::type_name::<O>().to_string(),
            schema: schema_value,
        };

        let (response_content, steps, total_usage, stop_reason) =
            self.react_loop(messages, Some(response_format)).await?;

        let text = extract_text(&response_content);
        let response: O = serde_json::from_str(&text)?;

        Ok(StructuredAgentOutput {
            response,
            steps,
            total_usage,
            stop_reason,
        })
    }

    /// Core ReAct loop shared between `run` and `run_structured`.
    async fn react_loop(
        &self,
        messages: Vec<Message>,
        response_format: Option<ResponseFormat>,
    ) -> Result<(Vec<Content>, Vec<AgentStep>, Usage, AgentStopReason), Error> {
        let mut tool_definitions: Vec<_> = self.tools.iter().map(|t| t.definition()).collect();
        tool_definitions.extend(self.sub_agents.iter().map(|sa| sa.to_tool_definition()));

        let mut conversation = self.context_builder.build_initial(messages);

        let mut steps = Vec::new();
        let mut total_usage = Usage::default();
        let start_time = std::time::Instant::now();
        let mut last_response_content: Option<Vec<Content>> = None;

        for iteration in 0..self.config.max_iterations {
            let iter_span = tracing::info_span!("iteration", index = iteration);

            let messages_to_send = if iteration == 0 {
                conversation.clone()
            } else {
                self.context_builder
                    .build_iteration(&conversation, iteration)
            };

            let request = ChatRequest {
                model: self.config.model.clone(),
                messages: messages_to_send,
                tools: tool_definitions.clone(),
                response_format: response_format.clone(),
                extra_params: self.config.extra_params.clone(),
            };

            let message_count = request.messages.len();

            let response = self
                .provider
                .chat(request)
                .instrument(tracing::info_span!(parent: &iter_span, "llm.chat"))
                .await?;

            tracing::info!(
                parent: &iter_span,
                input_tokens = response.usage.input_tokens,
                output_tokens = response.usage.output_tokens,
                stop_reason = ?response.stop_reason,
                "LLM response received"
            );

            total_usage = total_usage.add(&response.usage);

            steps.push(AgentStep::LlmCall {
                request_messages: message_count,
                response: response.clone(),
            });

            // If the LLM did not request tool use, return the final response
            if response.stop_reason != StopReason::ToolUse {
                return Ok((
                    response.content,
                    steps,
                    total_usage,
                    AgentStopReason::NaturalStop,
                ));
            }

            // Add the assistant's response to the conversation
            conversation.push(Message {
                role: Role::Assistant,
                content: response.content.clone(),
            });

            last_response_content = Some(response.content.clone());

            // Execute each tool call / sub-agent invocation and append results
            for content in &response.content {
                if let Content::ToolUse { id, name, input } = content {
                    let (output, is_error) = if let Some(tool) = self.find_tool(name) {
                        // Dispatch to Tool
                        let result = tool
                            .execute(input.clone())
                            .instrument(tracing::info_span!(
                                parent: &iter_span,
                                "tool.exec",
                                tool_name = %name
                            ))
                            .await;

                        match result {
                            Ok(result) => {
                                tracing::info!(parent: &iter_span, tool_name = %name, "Tool executed successfully");
                                steps.push(AgentStep::ToolExecution {
                                    name: name.clone(),
                                    input: input.clone(),
                                    output: result.clone(),
                                    is_error: false,
                                });
                                (result, false)
                            }
                            Err(e) => {
                                let msg = e.to_string();
                                tracing::warn!(parent: &iter_span, tool_name = %name, error = %msg, "Tool execution failed");
                                steps.push(AgentStep::ToolExecution {
                                    name: name.clone(),
                                    input: input.clone(),
                                    output: msg.clone(),
                                    is_error: true,
                                });
                                (msg, true)
                            }
                        }
                    } else if let Some(sub_agent) = self.find_sub_agent(name) {
                        match sub_agent.mode {
                            SubAgentMode::Single => {
                                // Single sub-agent dispatch
                                let query =
                                    input.get("query").and_then(|v| v.as_str()).unwrap_or("");
                                let messages = vec![Message::user(query)];

                                let result = sub_agent
                                    .agent
                                    .run(messages)
                                    .instrument(tracing::info_span!(
                                        parent: &iter_span,
                                        "sub_agent.run",
                                        name = %name
                                    ))
                                    .await;

                                match result {
                                    Ok(agent_output) => {
                                        tracing::info!(
                                            parent: &iter_span,
                                            name = %name,
                                            steps = agent_output.steps.len(),
                                            "Sub-agent executed successfully"
                                        );
                                        let response_text = agent_output.response.clone();
                                        total_usage = total_usage.add(&agent_output.total_usage);
                                        steps.push(AgentStep::SubAgentExecution {
                                            name: name.clone(),
                                            input: input.clone(),
                                            output: agent_output,
                                        });
                                        (response_text, false)
                                    }
                                    Err(e) => {
                                        let msg = e.to_string();
                                        tracing::warn!(parent: &iter_span, name = %name, error = %msg, "Sub-agent execution failed");
                                        (msg, true)
                                    }
                                }
                            }
                            SubAgentMode::Parallel => {
                                // Parallel fan-out dispatch
                                let tasks: Vec<String> = input
                                    .get("tasks")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|v| v.as_str().map(String::from))
                                            .collect()
                                    })
                                    .unwrap_or_default();

                                tracing::info!(
                                    parent: &iter_span,
                                    name = %name,
                                    task_count = tasks.len(),
                                    "Parallel sub-agent fan-out"
                                );

                                let message_sets: Vec<Vec<Message>> = tasks
                                    .iter()
                                    .map(|task| vec![Message::user(task.as_str())])
                                    .collect();

                                let results = sub_agent
                                    .agent
                                    .run_parallel(message_sets)
                                    .instrument(tracing::info_span!(
                                        parent: &iter_span,
                                        "sub_agent.run_parallel",
                                        name = %name,
                                        task_count = tasks.len()
                                    ))
                                    .await;

                                let mut outputs = Vec::new();
                                let mut errors = Vec::new();
                                let mut response_parts = Vec::new();

                                for (i, result) in results.into_iter().enumerate() {
                                    let task_label =
                                        tasks.get(i).map(|s| s.as_str()).unwrap_or("unknown");
                                    match result {
                                        Ok(agent_output) => {
                                            total_usage =
                                                total_usage.add(&agent_output.total_usage);
                                            response_parts.push(format!(
                                                "[Task {}] {}: {}",
                                                i + 1,
                                                task_label,
                                                agent_output.response
                                            ));
                                            outputs.push(agent_output);
                                        }
                                        Err(e) => {
                                            let msg = e.to_string();
                                            response_parts.push(format!(
                                                "[Task {}] {}: ERROR: {}",
                                                i + 1,
                                                task_label,
                                                msg
                                            ));
                                            errors.push(msg);
                                        }
                                    }
                                }

                                tracing::info!(
                                    parent: &iter_span,
                                    name = %name,
                                    successes = outputs.len(),
                                    failures = errors.len(),
                                    "Parallel sub-agent completed"
                                );

                                let is_error = outputs.is_empty();
                                let response_text = response_parts.join("\n\n");

                                steps.push(AgentStep::ParallelSubAgentExecution {
                                    name: name.clone(),
                                    tasks,
                                    outputs,
                                    errors,
                                });

                                (response_text, is_error)
                            }
                        }
                    } else {
                        // Neither tool nor sub-agent found
                        let msg = format!("Tool or sub-agent '{}' not found", name);
                        tracing::warn!(parent: &iter_span, msg = %msg);
                        steps.push(AgentStep::ToolExecution {
                            name: name.clone(),
                            input: input.clone(),
                            output: msg.clone(),
                            is_error: true,
                        });
                        (msg, true)
                    };

                    conversation.push(Message {
                        role: Role::User,
                        content: vec![Content::ToolResult {
                            id: id.clone(),
                            output,
                            is_error,
                        }],
                    });
                }
            }

            // Check user-defined stop conditions
            let stop_ctx = StopContext {
                iteration,
                total_usage: &total_usage,
                steps: &steps,
                elapsed: start_time.elapsed(),
                messages: &conversation,
            };

            for condition in &self.stop_conditions {
                if let Some(description) = condition.should_stop(&stop_ctx) {
                    tracing::info!(%description, "Stop condition triggered");
                    let content = last_response_content.take().unwrap_or_default();
                    return Ok((
                        content,
                        steps,
                        total_usage,
                        AgentStopReason::StopCondition { description },
                    ));
                }
            }
        }

        // max_iterations exhausted — not an error, just a stop reason
        let content = last_response_content.unwrap_or_default();
        Ok((
            content,
            steps,
            total_usage,
            AgentStopReason::MaxIterations {
                max: self.config.max_iterations,
            },
        ))
    }

    fn find_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.definition().name == name)
            .map(|t| t.as_ref())
    }

    fn find_sub_agent(&self, name: &str) -> Option<&SubAgentEntry> {
        self.sub_agents.iter().find(|sa| sa.name == name)
    }
}

impl<P: LlmProvider> AgentExecutor for Agent<P> {
    fn run(&self, messages: Vec<Message>) -> AgentFuture<'_> {
        Box::pin(self.run(messages))
    }
}

/// Extract text content from a list of content blocks.
fn extract_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ChatResponse;
    use crate::tool::ToolDefinition;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockProvider {
        responses: Vec<ChatResponse>,
        call_count: AtomicUsize,
    }

    impl LlmProvider for MockProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, Error> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            self.responses
                .get(idx)
                .cloned()
                .ok_or_else(|| Error::Provider("No more mock responses".to_string()))
        }
    }

    struct EchoTool;

    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".to_string(),
                description: "Echoes the input".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string" }
                    },
                    "required": ["text"]
                }),
            }
        }

        fn execute(
            &self,
            input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
            Box::pin(async move { Ok(input["text"].as_str().unwrap_or("").to_string()) })
        }
    }

    #[tokio::test]
    async fn agent_simple_text_response() {
        let provider = MockProvider {
            responses: vec![ChatResponse {
                content: vec![Content::Text {
                    text: "Hello!".to_string(),
                }],
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
                stop_reason: StopReason::EndTurn,
            }],
            call_count: AtomicUsize::new(0),
        };

        let agent = Agent::without_system_prompt(
            provider,
            vec![],
            AgentConfig {
                model: "test-model".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let output = agent.run(vec![Message::user("Hi")]).await.expect("test");
        assert_eq!(output.response, "Hello!");
        assert_eq!(output.steps.len(), 1);
        assert_eq!(output.total_usage.input_tokens, 10);
        assert_eq!(output.total_usage.output_tokens, 5);
        assert!(matches!(output.stop_reason, AgentStopReason::NaturalStop));
    }

    #[tokio::test]
    async fn agent_tool_use_then_response() {
        let provider = MockProvider {
            responses: vec![
                // First: LLM requests tool use
                ChatResponse {
                    content: vec![Content::ToolUse {
                        id: "call_1".to_string(),
                        name: "echo".to_string(),
                        input: serde_json::json!({"text": "world"}),
                    }],
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 8,
                    },
                    stop_reason: StopReason::ToolUse,
                },
                // Second: LLM gives final answer
                ChatResponse {
                    content: vec![Content::Text {
                        text: "The echo said: world".to_string(),
                    }],
                    usage: Usage {
                        input_tokens: 20,
                        output_tokens: 10,
                    },
                    stop_reason: StopReason::EndTurn,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let agent = Agent::without_system_prompt(
            provider,
            vec![Box::new(EchoTool)],
            AgentConfig {
                model: "test-model".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let output = agent
            .run(vec![Message::user("Echo world")])
            .await
            .expect("test");
        assert_eq!(output.response, "The echo said: world");
        // 2 LLM calls + 1 tool execution = 3 steps
        assert_eq!(output.steps.len(), 3);
        assert_eq!(output.total_usage.input_tokens, 30);
        assert_eq!(output.total_usage.output_tokens, 18);
        assert!(matches!(output.stop_reason, AgentStopReason::NaturalStop));
    }

    #[tokio::test]
    async fn agent_max_iterations_exceeded() {
        // Provider always requests tool use
        let responses: Vec<ChatResponse> = (0..5)
            .map(|i| ChatResponse {
                content: vec![Content::ToolUse {
                    id: format!("call_{}", i),
                    name: "echo".to_string(),
                    input: serde_json::json!({"text": "loop"}),
                }],
                usage: Usage::default(),
                stop_reason: StopReason::ToolUse,
            })
            .collect();

        let provider = MockProvider {
            responses,
            call_count: AtomicUsize::new(0),
        };

        let agent = Agent::without_system_prompt(
            provider,
            vec![Box::new(EchoTool)],
            AgentConfig {
                model: "test-model".to_string(),
                max_iterations: 3,
                extra_params: serde_json::Map::new(),
            },
        );

        let output = agent
            .run(vec![Message::user("Loop forever")])
            .await
            .expect("max_iterations is now a stop reason, not an error");
        assert!(matches!(
            output.stop_reason,
            AgentStopReason::MaxIterations { max: 3 }
        ));
    }

    #[tokio::test]
    async fn agent_sub_agent_execution() {
        let sub_provider = MockProvider {
            responses: vec![ChatResponse {
                content: vec![Content::Text {
                    text: "Sub-agent researched: Rust is great!".to_string(),
                }],
                usage: Usage {
                    input_tokens: 15,
                    output_tokens: 10,
                },
                stop_reason: StopReason::EndTurn,
            }],
            call_count: AtomicUsize::new(0),
        };

        let sub_agent = Agent::without_system_prompt(
            sub_provider,
            vec![],
            AgentConfig {
                model: "sub-model".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let parent_provider = MockProvider {
            responses: vec![
                ChatResponse {
                    content: vec![Content::ToolUse {
                        id: "call_1".to_string(),
                        name: "researcher".to_string(),
                        input: serde_json::json!({"query": "What is Rust?"}),
                    }],
                    usage: Usage {
                        input_tokens: 20,
                        output_tokens: 15,
                    },
                    stop_reason: StopReason::ToolUse,
                },
                ChatResponse {
                    content: vec![Content::Text {
                        text: "Based on research: Rust is great!".to_string(),
                    }],
                    usage: Usage {
                        input_tokens: 30,
                        output_tokens: 12,
                    },
                    stop_reason: StopReason::EndTurn,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let parent = Agent::without_system_prompt(
            parent_provider,
            vec![],
            AgentConfig {
                model: "parent-model".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        )
        .with_sub_agent("researcher", "Research a topic in depth", sub_agent);

        let output = parent
            .run(vec![Message::user("Tell me about Rust")])
            .await
            .expect("test");

        assert_eq!(output.response, "Based on research: Rust is great!");

        // 2 LLM calls (parent) + 1 sub-agent execution = 3 steps
        assert_eq!(output.steps.len(), 3);

        // Verify sub-agent step
        match &output.steps[1] {
            AgentStep::SubAgentExecution {
                name,
                output: sub_output,
                ..
            } => {
                assert_eq!(name, "researcher");
                assert_eq!(sub_output.response, "Sub-agent researched: Rust is great!");
                assert_eq!(sub_output.steps.len(), 1);
            }
            _ => panic!("Expected SubAgentExecution step"),
        }

        // Verify token usage accumulation (parent + sub-agent)
        assert_eq!(output.total_usage.input_tokens, 20 + 30 + 15);
        assert_eq!(output.total_usage.output_tokens, 15 + 12 + 10);
    }

    #[tokio::test]
    async fn agent_sub_agent_error_handling() {
        struct ErrorExecutor;

        impl crate::sub_agent::AgentExecutor for ErrorExecutor {
            fn run(
                &self,
                _messages: Vec<Message>,
            ) -> Pin<Box<dyn Future<Output = Result<AgentOutput, Error>> + Send + '_>> {
                Box::pin(async { Err(Error::Provider("Sub-agent failed".to_string())) })
            }
        }

        let parent_provider = MockProvider {
            responses: vec![
                ChatResponse {
                    content: vec![Content::ToolUse {
                        id: "call_1".to_string(),
                        name: "failing_agent".to_string(),
                        input: serde_json::json!({"query": "test"}),
                    }],
                    usage: Usage::default(),
                    stop_reason: StopReason::ToolUse,
                },
                ChatResponse {
                    content: vec![Content::Text {
                        text: "Sub-agent failed, but I continue".to_string(),
                    }],
                    usage: Usage::default(),
                    stop_reason: StopReason::EndTurn,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let parent = Agent::without_system_prompt(
            parent_provider,
            vec![],
            AgentConfig {
                model: "test".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        )
        .with_sub_agent("failing_agent", "An agent that fails", ErrorExecutor);

        let output = parent.run(vec![Message::user("test")]).await.expect("test");

        assert_eq!(output.response, "Sub-agent failed, but I continue");
        assert!(matches!(output.stop_reason, AgentStopReason::NaturalStop));
    }

    #[tokio::test]
    async fn agent_mixed_tools_and_sub_agents() {
        let sub_provider = MockProvider {
            responses: vec![ChatResponse {
                content: vec![Content::Text {
                    text: "Researched!".to_string(),
                }],
                usage: Usage::default(),
                stop_reason: StopReason::EndTurn,
            }],
            call_count: AtomicUsize::new(0),
        };

        let sub_agent = Agent::without_system_prompt(
            sub_provider,
            vec![],
            AgentConfig {
                model: "sub".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let parent_provider = MockProvider {
            responses: vec![
                // Use tool first
                ChatResponse {
                    content: vec![Content::ToolUse {
                        id: "call_1".to_string(),
                        name: "echo".to_string(),
                        input: serde_json::json!({"text": "hello"}),
                    }],
                    usage: Usage::default(),
                    stop_reason: StopReason::ToolUse,
                },
                // Then use sub-agent
                ChatResponse {
                    content: vec![Content::ToolUse {
                        id: "call_2".to_string(),
                        name: "researcher".to_string(),
                        input: serde_json::json!({"query": "research this"}),
                    }],
                    usage: Usage::default(),
                    stop_reason: StopReason::ToolUse,
                },
                // Final response
                ChatResponse {
                    content: vec![Content::Text {
                        text: "Done!".to_string(),
                    }],
                    usage: Usage::default(),
                    stop_reason: StopReason::EndTurn,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let parent = Agent::without_system_prompt(
            parent_provider,
            vec![Box::new(EchoTool)],
            AgentConfig {
                model: "parent".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        )
        .with_sub_agent("researcher", "Research", sub_agent);

        let output = parent.run(vec![Message::user("test")]).await.expect("test");

        // 3 LLM calls + 1 tool + 1 sub-agent = 5 steps
        assert_eq!(output.steps.len(), 5);

        assert!(matches!(output.steps[0], AgentStep::LlmCall { .. }));
        assert!(matches!(output.steps[1], AgentStep::ToolExecution { .. }));
        assert!(matches!(output.steps[2], AgentStep::LlmCall { .. }));
        assert!(matches!(
            output.steps[3],
            AgentStep::SubAgentExecution { .. }
        ));
        assert!(matches!(output.steps[4], AgentStep::LlmCall { .. }));
    }

    #[tokio::test]
    async fn agent_parallel_sub_agent_execution() {
        // Sub-agent provider with 3 responses (one per parallel task)
        let sub_provider = MockProvider {
            responses: vec![
                ChatResponse {
                    content: vec![Content::Text {
                        text: "Result for task A".to_string(),
                    }],
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 5,
                    },
                    stop_reason: StopReason::EndTurn,
                },
                ChatResponse {
                    content: vec![Content::Text {
                        text: "Result for task B".to_string(),
                    }],
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 5,
                    },
                    stop_reason: StopReason::EndTurn,
                },
                ChatResponse {
                    content: vec![Content::Text {
                        text: "Result for task C".to_string(),
                    }],
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 5,
                    },
                    stop_reason: StopReason::EndTurn,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let sub_agent = Agent::without_system_prompt(
            sub_provider,
            vec![],
            AgentConfig {
                model: "sub".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        );

        let parent_provider = MockProvider {
            responses: vec![
                // Parent calls parallel sub-agent with 3 tasks
                ChatResponse {
                    content: vec![Content::ToolUse {
                        id: "call_1".to_string(),
                        name: "researcher".to_string(),
                        input: serde_json::json!({
                            "tasks": ["task A", "task B", "task C"]
                        }),
                    }],
                    usage: Usage {
                        input_tokens: 20,
                        output_tokens: 15,
                    },
                    stop_reason: StopReason::ToolUse,
                },
                // Parent synthesizes final answer
                ChatResponse {
                    content: vec![Content::Text {
                        text: "All 3 tasks completed.".to_string(),
                    }],
                    usage: Usage {
                        input_tokens: 50,
                        output_tokens: 10,
                    },
                    stop_reason: StopReason::EndTurn,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let parent = Agent::without_system_prompt(
            parent_provider,
            vec![],
            AgentConfig {
                model: "parent".to_string(),
                max_iterations: 10,
                extra_params: serde_json::Map::new(),
            },
        )
        .with_parallel_sub_agent("researcher", "Research topics in parallel", sub_agent);

        let output = parent
            .run(vec![Message::user("Research A, B, C")])
            .await
            .expect("test");

        assert_eq!(output.response, "All 3 tasks completed.");

        // 2 LLM calls + 1 parallel sub-agent = 3 steps
        assert_eq!(output.steps.len(), 3);

        // Verify parallel sub-agent step
        match &output.steps[1] {
            AgentStep::ParallelSubAgentExecution {
                name,
                tasks,
                outputs,
                errors,
            } => {
                assert_eq!(name, "researcher");
                assert_eq!(tasks.len(), 3);
                assert_eq!(outputs.len(), 3);
                assert!(errors.is_empty());
                // Verify responses contain expected content
                let responses: Vec<&str> = outputs.iter().map(|o| o.response.as_str()).collect();
                assert!(responses.contains(&"Result for task A"));
                assert!(responses.contains(&"Result for task B"));
                assert!(responses.contains(&"Result for task C"));
            }
            _ => panic!("Expected ParallelSubAgentExecution step"),
        }

        // Verify token usage: parent (20+50 input, 15+10 output) + 3 sub-agents (10*3 input, 5*3 output)
        assert_eq!(output.total_usage.input_tokens, 20 + 50 + 30);
        assert_eq!(output.total_usage.output_tokens, 15 + 10 + 15);
    }

    // Test the proc macro
    mod macro_tool {
        use membrane_macros::membrane_tool;
        use schemars::JsonSchema;
        use serde::Deserialize;

        #[derive(Debug, Deserialize, JsonSchema)]
        struct GreetInput {
            name: String,
        }

        #[membrane_tool(name = "greet", description = "Greet someone by name")]
        async fn greet(input: GreetInput) -> Result<String, membrane_core::error::Error> {
            Ok(format!("Hello, {}!", input.name))
        }

        #[test]
        fn macro_generates_tool_definition() {
            use crate::tool::Tool;

            let tool = GreetTool;
            let def = tool.definition();
            assert_eq!(def.name, "greet");
            assert_eq!(def.description, "Greet someone by name");
            // Schema should contain "name" property
            assert!(def.input_schema.to_string().contains("name"));
        }

        #[tokio::test]
        async fn macro_generated_tool_executes() {
            use crate::tool::Tool;

            let tool = GreetTool;
            let result = tool
                .execute(serde_json::json!({"name": "World"}))
                .await
                .expect("test");
            assert_eq!(result, "Hello, World!");
        }
    }
}
