use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's input parameters.
    pub input_schema: serde_json::Value,
}

pub trait Tool: Send + Sync {
    /// Return the tool's definition (name, description, input schema).
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the given input and return the result as a string.
    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>>;
}

/// A wrapper that turns any [`Tool`] into a parallel (fan-out) version.
///
/// The wrapped tool's input schema is changed to accept an array of inputs:
/// `{ "inputs": [<original_input>, ...] }`. Each input is executed concurrently
/// via the inner tool's `execute` method, and results are returned as a
/// combined string with `[N]` prefixes.
pub struct ParallelTool {
    inner: Box<dyn Tool>,
}

impl ParallelTool {
    /// Create a parallel wrapper around an existing tool.
    ///
    /// The tool name and description are preserved from the inner tool.
    /// The description is appended with a note about parallel execution.
    pub fn new(tool: Box<dyn Tool>) -> Self {
        Self { inner: tool }
    }
}

impl Tool for ParallelTool {
    fn definition(&self) -> ToolDefinition {
        let inner_def = self.inner.definition();
        ToolDefinition {
            name: inner_def.name,
            description: format!(
                "{} Accepts multiple inputs to execute concurrently.",
                inner_def.description
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "inputs": {
                        "type": "array",
                        "items": inner_def.input_schema,
                        "description": "List of inputs to execute in parallel"
                    }
                },
                "required": ["inputs"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        Box::pin(async move {
            let inputs = input
                .get("inputs")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            if inputs.is_empty() {
                return Ok("No inputs provided.".to_string());
            }

            let futs: Vec<_> = inputs
                .into_iter()
                .map(|inp| self.inner.execute(inp))
                .collect();

            let results = futures_util::future::join_all(futs).await;

            let mut parts = Vec::new();
            for (i, result) in results.into_iter().enumerate() {
                match result {
                    Ok(output) => parts.push(format!("[{}] {}", i + 1, output)),
                    Err(e) => parts.push(format!("[{}] ERROR: {}", i + 1, e)),
                }
            }

            Ok(parts.join("\n\n"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AddTool;

    impl Tool for AddTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "add".to_string(),
                description: "Add two numbers".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "a": { "type": "number" },
                        "b": { "type": "number" }
                    },
                    "required": ["a", "b"]
                }),
            }
        }

        fn execute(
            &self,
            input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
            Box::pin(async move {
                let a = input["a"].as_f64().unwrap_or(0.0);
                let b = input["b"].as_f64().unwrap_or(0.0);
                Ok(format!("{}", a + b))
            })
        }
    }

    #[test]
    fn parallel_tool_definition_wraps_schema() {
        let tool = ParallelTool::new(Box::new(AddTool));
        let def = tool.definition();

        assert_eq!(def.name, "add");
        assert!(def.description.contains("Accepts multiple inputs"));

        let schema = &def.input_schema;
        assert_eq!(schema["properties"]["inputs"]["type"], "array");
        assert_eq!(
            schema["properties"]["inputs"]["items"]["properties"]["a"]["type"],
            "number"
        );
        assert_eq!(schema["required"][0], "inputs");
    }

    #[tokio::test]
    async fn parallel_tool_executes_concurrently() {
        let tool = ParallelTool::new(Box::new(AddTool));
        let result = tool
            .execute(serde_json::json!({
                "inputs": [
                    {"a": 1, "b": 2},
                    {"a": 10, "b": 20},
                    {"a": 100, "b": 200}
                ]
            }))
            .await
            .expect("should succeed");

        assert!(result.contains("[1] 3"));
        assert!(result.contains("[2] 30"));
        assert!(result.contains("[3] 300"));
    }

    #[tokio::test]
    async fn parallel_tool_empty_inputs() {
        let tool = ParallelTool::new(Box::new(AddTool));
        let result = tool
            .execute(serde_json::json!({"inputs": []}))
            .await
            .expect("should succeed");

        assert_eq!(result, "No inputs provided.");
    }

    #[tokio::test]
    async fn parallel_tool_partial_error() {
        struct MaybeFailTool;

        impl Tool for MaybeFailTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "maybe_fail".to_string(),
                    description: "Fails if input has fail=true".to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "fail": { "type": "boolean" }
                        }
                    }),
                }
            }

            fn execute(
                &self,
                input: serde_json::Value,
            ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
                Box::pin(async move {
                    if input["fail"].as_bool().unwrap_or(false) {
                        Err(Error::ToolExecution {
                            tool_name: "maybe_fail".to_string(),
                            message: "Intentional failure".to_string(),
                        })
                    } else {
                        Ok("OK".to_string())
                    }
                })
            }
        }

        let tool = ParallelTool::new(Box::new(MaybeFailTool));
        let result = tool
            .execute(serde_json::json!({
                "inputs": [
                    {"fail": false},
                    {"fail": true},
                    {"fail": false}
                ]
            }))
            .await
            .expect("should succeed even with partial failures");

        assert!(result.contains("[1] OK"));
        assert!(result.contains("[2] ERROR:"));
        assert!(result.contains("[3] OK"));
    }
}
