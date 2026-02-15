use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::message::{Content, Message, Role};
use crate::provider::{ChatRequest, ChatResponse, LlmProvider, ResponseFormat, StopReason, Usage};
use crate::tool::Tool;

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub model: String,
    pub max_iterations: usize,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

pub struct Agent<P: LlmProvider> {
    provider: P,
    tools: Vec<Box<dyn Tool>>,
    config: AgentConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    pub response: String,
    pub steps: Vec<AgentStep>,
    pub total_usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredAgentOutput<O> {
    pub response: O,
    pub steps: Vec<AgentStep>,
    pub total_usage: Usage,
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
}

impl<P: LlmProvider> Agent<P> {
    pub fn new(provider: P, tools: Vec<Box<dyn Tool>>, config: AgentConfig) -> Self {
        Self {
            provider,
            tools,
            config,
        }
    }

    /// Run the ReAct loop with the given messages and return a text response.
    #[tracing::instrument(skip_all, fields(model = %self.config.model))]
    pub async fn run(&self, messages: Vec<Message>) -> Result<AgentOutput, Error> {
        let (response_content, steps, total_usage) = self.react_loop(messages, None).await?;

        let response = extract_text(&response_content);
        Ok(AgentOutput {
            response,
            steps,
            total_usage,
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

        let (response_content, steps, total_usage) =
            self.react_loop(messages, Some(response_format)).await?;

        let text = extract_text(&response_content);
        let response: O = serde_json::from_str(&text)?;

        Ok(StructuredAgentOutput {
            response,
            steps,
            total_usage,
        })
    }

    /// Core ReAct loop shared between `run` and `run_structured`.
    async fn react_loop(
        &self,
        messages: Vec<Message>,
        response_format: Option<ResponseFormat>,
    ) -> Result<(Vec<Content>, Vec<AgentStep>, Usage), Error> {
        let tool_definitions: Vec<_> = self.tools.iter().map(|t| t.definition()).collect();

        let mut conversation = Vec::new();

        // Prepend system prompt if configured
        if let Some(ref system_prompt) = self.config.system_prompt {
            conversation.push(Message::system(system_prompt));
        }
        conversation.extend(messages);

        let mut steps = Vec::new();
        let mut total_usage = Usage::default();

        for iteration in 0..self.config.max_iterations {
            let span = tracing::info_span!("iteration", index = iteration);
            let _enter = span.enter();

            let request = ChatRequest {
                model: self.config.model.clone(),
                messages: conversation.clone(),
                tools: tool_definitions.clone(),
                response_format: response_format.clone(),
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
            };

            let message_count = request.messages.len();

            let response = {
                let _chat_span = tracing::info_span!("llm.chat").entered();
                let resp = self.provider.chat(request).await?;
                tracing::info!(
                    input_tokens = resp.usage.input_tokens,
                    output_tokens = resp.usage.output_tokens,
                    stop_reason = ?resp.stop_reason,
                    "LLM response received"
                );
                resp
            };

            total_usage = total_usage.add(&response.usage);

            steps.push(AgentStep::LlmCall {
                request_messages: message_count,
                response: response.clone(),
            });

            // If the LLM did not request tool use, return the final response
            if response.stop_reason != StopReason::ToolUse {
                return Ok((response.content, steps, total_usage));
            }

            // Add the assistant's response to the conversation
            conversation.push(Message {
                role: Role::Assistant,
                content: response.content.clone(),
            });

            // Execute each tool call and append results
            for content in &response.content {
                if let Content::ToolUse { id, name, input } = content {
                    let _tool_span = tracing::info_span!("tool.exec", tool_name = %name).entered();

                    let (output, is_error) = match self.find_tool(name) {
                        Some(tool) => match tool.execute(input.clone()).await {
                            Ok(result) => {
                                tracing::info!(tool_name = %name, "Tool executed successfully");
                                (result, false)
                            }
                            Err(e) => {
                                tracing::warn!(tool_name = %name, error = %e, "Tool execution failed");
                                (e.to_string(), true)
                            }
                        },
                        None => {
                            let msg = format!("Tool '{}' not found", name);
                            tracing::warn!(%msg);
                            (msg, true)
                        }
                    };

                    steps.push(AgentStep::ToolExecution {
                        name: name.clone(),
                        input: input.clone(),
                        output: output.clone(),
                        is_error,
                    });

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
        }

        Err(Error::MaxIterations {
            max: self.config.max_iterations,
        })
    }

    fn find_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.definition().name == name)
            .map(|t| t.as_ref())
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

        let agent = Agent::new(
            provider,
            vec![],
            AgentConfig {
                model: "test-model".to_string(),
                max_iterations: 10,
                system_prompt: None,
                max_tokens: None,
                temperature: None,
            },
        );

        let output = agent.run(vec![Message::user("Hi")]).await.expect("test");
        assert_eq!(output.response, "Hello!");
        assert_eq!(output.steps.len(), 1);
        assert_eq!(output.total_usage.input_tokens, 10);
        assert_eq!(output.total_usage.output_tokens, 5);
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

        let agent = Agent::new(
            provider,
            vec![Box::new(EchoTool)],
            AgentConfig {
                model: "test-model".to_string(),
                max_iterations: 10,
                system_prompt: None,
                max_tokens: None,
                temperature: None,
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

        let agent = Agent::new(
            provider,
            vec![Box::new(EchoTool)],
            AgentConfig {
                model: "test-model".to_string(),
                max_iterations: 3,
                system_prompt: None,
                max_tokens: None,
                temperature: None,
            },
        );

        let result = agent.run(vec![Message::user("Loop forever")]).await;
        assert!(matches!(result, Err(Error::MaxIterations { max: 3 })));
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
