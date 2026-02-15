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
