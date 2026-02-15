use membrane_core::error::Error;
use membrane_core::message::{Content, Message, Role};
use membrane_core::provider::{ChatRequest, ChatResponse, ResponseFormat, StopReason, Usage};
use membrane_core::tool::ToolDefinition;

use crate::types::*;

/// Convert a membrane `ChatRequest` into an OpenAI API request.
pub(crate) fn to_openai_request(request: ChatRequest) -> Result<OpenAiRequest, Error> {
    let messages = convert_messages(request.messages)?;

    let tools = if request.tools.is_empty() {
        None
    } else {
        Some(
            request
                .tools
                .into_iter()
                .map(convert_tool_definition)
                .collect(),
        )
    };

    let response_format = request.response_format.map(convert_response_format);

    Ok(OpenAiRequest {
        model: request.model,
        messages,
        tools,
        response_format,
        extra: request.extra_params,
    })
}

/// Convert an OpenAI API response into a membrane `ChatResponse`.
pub(crate) fn from_openai_response(response: OpenAiResponse) -> Result<ChatResponse, Error> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::Provider("OpenAI response contained no choices".to_string()))?;

    let mut content = Vec::new();

    if let Some(text) = choice.message.content
        && !text.is_empty()
    {
        content.push(Content::Text { text });
    }

    if let Some(tool_calls) = choice.message.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)?;
            content.push(Content::ToolUse {
                id: tc.id,
                name: tc.function.name,
                input,
            });
        }
    }

    let stop_reason = match choice.finish_reason.as_str() {
        "stop" => StopReason::EndTurn,
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        other => {
            tracing::warn!(finish_reason = %other, "Unknown OpenAI finish_reason, defaulting to EndTurn");
            StopReason::EndTurn
        }
    };

    let usage = Usage {
        input_tokens: response.usage.prompt_tokens,
        output_tokens: response.usage.completion_tokens,
    };

    Ok(ChatResponse {
        content,
        usage,
        stop_reason,
    })
}

/// Convert a list of membrane messages into OpenAI message format.
///
/// Key differences:
/// - membrane stores all content in `Vec<Content>` per message
/// - OpenAI uses `content` (text), `tool_calls` (array) on assistant messages,
///   and separate `role: "tool"` messages for tool results
fn convert_messages(messages: Vec<Message>) -> Result<Vec<OpenAiMessage>, Error> {
    let mut result = Vec::new();

    for message in messages {
        match message.role {
            Role::System => {
                result.push(OpenAiMessage {
                    role: "system".to_string(),
                    content: Some(collect_text(&message.content)),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            Role::User => {
                let mut text_parts = Vec::new();
                let mut tool_results = Vec::new();

                for content in message.content {
                    match content {
                        Content::Text { text } => text_parts.push(text),
                        Content::ToolResult { id, output, .. } => {
                            tool_results.push((id, output));
                        }
                        Content::ToolUse { .. } => {}
                    }
                }

                // Emit text as a user message
                if !text_parts.is_empty() {
                    result.push(OpenAiMessage {
                        role: "user".to_string(),
                        content: Some(text_parts.join("")),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }

                // Emit each tool result as a separate "tool" message
                for (id, output) in tool_results {
                    result.push(OpenAiMessage {
                        role: "tool".to_string(),
                        content: Some(output),
                        tool_calls: None,
                        tool_call_id: Some(id),
                    });
                }
            }
            Role::Assistant => {
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();

                for content in message.content {
                    match content {
                        Content::Text { text } => text_parts.push(text),
                        Content::ToolUse { id, name, input } => {
                            let arguments = serde_json::to_string(&input)?;
                            tool_calls.push(OpenAiToolCall {
                                id,
                                call_type: "function".to_string(),
                                function: OpenAiFunctionCall { name, arguments },
                            });
                        }
                        Content::ToolResult { .. } => {}
                    }
                }

                let content = if text_parts.is_empty() {
                    None
                } else {
                    Some(text_parts.join(""))
                };

                let tool_calls_field = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                };

                result.push(OpenAiMessage {
                    role: "assistant".to_string(),
                    content,
                    tool_calls: tool_calls_field,
                    tool_call_id: None,
                });
            }
        }
    }

    Ok(result)
}

fn convert_tool_definition(def: ToolDefinition) -> OpenAiTool {
    OpenAiTool {
        tool_type: "function".to_string(),
        function: OpenAiFunctionDef {
            name: def.name,
            description: def.description,
            parameters: def.input_schema,
        },
    }
}

fn convert_response_format(fmt: ResponseFormat) -> OpenAiResponseFormat {
    OpenAiResponseFormat {
        format_type: "json_schema".to_string(),
        json_schema: OpenAiJsonSchema {
            name: fmt.name,
            schema: fmt.schema,
            strict: true,
        },
    }
}

fn collect_text(content: &[Content]) -> String {
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
    use membrane_core::message::Message;

    #[test]
    fn convert_system_message() {
        let messages = vec![Message::system("You are a helpful assistant")];
        let result = convert_messages(messages).expect("conversion should succeed");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "system");
        assert_eq!(
            result[0].content.as_deref(),
            Some("You are a helpful assistant")
        );
        assert!(result[0].tool_calls.is_none());
        assert!(result[0].tool_call_id.is_none());
    }

    #[test]
    fn convert_user_text_message() {
        let messages = vec![Message::user("Hello")];
        let result = convert_messages(messages).expect("conversion should succeed");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].content.as_deref(), Some("Hello"));
    }

    #[test]
    fn convert_tool_result_message() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![Content::ToolResult {
                id: "call_1".to_string(),
                output: "result data".to_string(),
                is_error: false,
            }],
        }];
        let result = convert_messages(messages).expect("conversion should succeed");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "tool");
        assert_eq!(result[0].content.as_deref(), Some("result data"));
        assert_eq!(result[0].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn convert_assistant_with_tool_calls() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                Content::Text {
                    text: "Let me search.".to_string(),
                },
                Content::ToolUse {
                    id: "call_1".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({"query": "rust"}),
                },
            ],
        }];
        let result = convert_messages(messages).expect("conversion should succeed");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "assistant");
        assert_eq!(result[0].content.as_deref(), Some("Let me search."));

        let tool_calls = result[0]
            .tool_calls
            .as_ref()
            .expect("should have tool_calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].call_type, "function");
        assert_eq!(tool_calls[0].function.name, "search");
        assert_eq!(tool_calls[0].function.arguments, r#"{"query":"rust"}"#);
    }

    #[test]
    fn convert_assistant_tool_calls_only() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![Content::ToolUse {
                id: "call_1".to_string(),
                name: "search".to_string(),
                input: serde_json::json!({"q": "test"}),
            }],
        }];
        let result = convert_messages(messages).expect("conversion should succeed");

        assert_eq!(result.len(), 1);
        assert!(result[0].content.is_none());
        assert!(result[0].tool_calls.is_some());
    }

    #[test]
    fn convert_tool_definition_wrapping() {
        let def = ToolDefinition {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } }
            }),
        };

        let tool = convert_tool_definition(def);
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "search");
        assert_eq!(tool.function.description, "Search the web");

        // Verify serialization includes "type" field
        let json = serde_json::to_value(&tool).expect("should serialize");
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "search");
    }

    #[test]
    fn convert_response_format_wrapping() {
        let fmt = ResponseFormat {
            name: "MyOutput".to_string(),
            schema: serde_json::json!({"type": "object"}),
        };

        let result = convert_response_format(fmt);
        assert_eq!(result.format_type, "json_schema");
        assert_eq!(result.json_schema.name, "MyOutput");
        assert!(result.json_schema.strict);

        let json = serde_json::to_value(&result).expect("should serialize");
        assert_eq!(json["type"], "json_schema");
        assert_eq!(json["json_schema"]["strict"], true);
    }

    #[test]
    fn from_openai_text_response() {
        let response = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiResponseMessage {
                    content: Some("Hello!".to_string()),
                    tool_calls: None,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: OpenAiUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            },
        };

        let result = from_openai_response(response).expect("should convert");
        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
        assert_eq!(result.content.len(), 1);
        assert!(matches!(&result.content[0], Content::Text { text } if text == "Hello!"));
    }

    #[test]
    fn from_openai_tool_calls_response() {
        let response = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiResponseMessage {
                    content: None,
                    tool_calls: Some(vec![OpenAiToolCall {
                        id: "call_abc".to_string(),
                        call_type: "function".to_string(),
                        function: OpenAiFunctionCall {
                            name: "search".to_string(),
                            arguments: r#"{"query":"rust"}"#.to_string(),
                        },
                    }]),
                },
                finish_reason: "tool_calls".to_string(),
            }],
            usage: OpenAiUsage {
                prompt_tokens: 15,
                completion_tokens: 8,
            },
        };

        let result = from_openai_response(response).expect("should convert");
        assert_eq!(result.stop_reason, StopReason::ToolUse);
        assert_eq!(result.content.len(), 1);
        assert!(
            matches!(&result.content[0], Content::ToolUse { id, name, input } if id == "call_abc" && name == "search" && input["query"] == "rust")
        );
    }

    #[test]
    fn from_openai_max_tokens_reason() {
        let response = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiResponseMessage {
                    content: Some("partial...".to_string()),
                    tool_calls: None,
                },
                finish_reason: "length".to_string(),
            }],
            usage: OpenAiUsage {
                prompt_tokens: 10,
                completion_tokens: 100,
            },
        };

        let result = from_openai_response(response).expect("should convert");
        assert_eq!(result.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn from_openai_unknown_finish_reason_defaults_to_end_turn() {
        let response = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiResponseMessage {
                    content: Some("filtered".to_string()),
                    tool_calls: None,
                },
                finish_reason: "content_filter".to_string(),
            }],
            usage: OpenAiUsage {
                prompt_tokens: 5,
                completion_tokens: 1,
            },
        };

        let result = from_openai_response(response).expect("should convert");
        assert_eq!(result.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn from_openai_empty_choices_returns_error() {
        let response = OpenAiResponse {
            choices: vec![],
            usage: OpenAiUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
            },
        };

        let result = from_openai_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn full_request_conversion() {
        let mut extra_params = serde_json::Map::new();
        extra_params.insert(
            "max_completion_tokens".to_string(),
            serde_json::Value::Number(1000.into()),
        );
        extra_params.insert("temperature".to_string(), serde_json::json!(0.7));

        let request = ChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![Message::system("Be helpful"), Message::user("Hi")],
            tools: vec![ToolDefinition {
                name: "search".to_string(),
                description: "Search".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            response_format: None,
            extra_params,
        };

        let result = to_openai_request(request).expect("should convert");
        assert_eq!(result.model, "gpt-4");
        assert_eq!(result.messages.len(), 2);
        assert!(result.tools.is_some());
        assert_eq!(result.tools.as_ref().map(|t| t.len()), Some(1));
        assert!(result.response_format.is_none());
        assert_eq!(result.extra["max_completion_tokens"], 1000);
        assert_eq!(result.extra["temperature"], 0.7);
    }
}
