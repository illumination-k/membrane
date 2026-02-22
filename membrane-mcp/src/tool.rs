use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use membrane_core::error::Error;
use membrane_core::tool::{Tool, ToolDefinition};
use rmcp::model::CallToolRequestParams;
use rmcp::service::RunningService;

/// A membrane [`Tool`] backed by a remote MCP server tool.
///
/// Each `McpTool` holds a reference to the running MCP service and the
/// tool's name, description, and input schema from the server's tool list.
/// When executed, it forwards the call to the MCP server via `call_tool`.
pub struct McpTool {
    service: Arc<RunningService<rmcp::RoleClient, ()>>,
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

impl McpTool {
    pub(crate) fn new(
        service: Arc<RunningService<rmcp::RoleClient, ()>>,
        name: String,
        description: String,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            service,
            name,
            description,
            input_schema,
        }
    }
}

impl Tool for McpTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        Box::pin(async move {
            let arguments = input.as_object().cloned();

            let result = self
                .service
                .call_tool(CallToolRequestParams {
                    meta: None,
                    name: self.name.clone().into(),
                    arguments,
                    task: None,
                })
                .await
                .map_err(|e| Error::ToolExecution {
                    tool_name: self.name.clone(),
                    message: format!("MCP call_tool failed: {e}"),
                })?;

            if result.is_error == Some(true) {
                return Err(Error::ToolExecution {
                    tool_name: self.name.clone(),
                    message: extract_text_content(&result.content),
                });
            }

            Ok(extract_text_content(&result.content))
        })
    }
}

/// Extract text from MCP content blocks, joining multiple text blocks with newlines.
fn extract_text_content(content: &[rmcp::model::Content]) -> String {
    let texts: Vec<&str> = content
        .iter()
        .filter_map(|c| match &c.raw {
            rmcp::model::RawContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();

    if texts.is_empty() {
        // Fall back to debug representation if no text content
        format!("{content:?}")
    } else {
        texts.join("\n")
    }
}
