use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("LLM provider error: {0}")]
    Provider(String),

    #[error("Tool execution error: {tool_name}: {message}")]
    ToolExecution { tool_name: String, message: String },

    #[error("Max iterations ({max}) exceeded")]
    MaxIterations { max: usize },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Serializable representation of an error for structured output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub kind: String,
    pub message: String,
}

impl From<&Error> for ErrorInfo {
    fn from(err: &Error) -> Self {
        let kind = match err {
            Error::Provider(_) => "provider",
            Error::ToolExecution { .. } => "tool_execution",
            Error::MaxIterations { .. } => "max_iterations",
            Error::Serialization(_) => "serialization",
        };
        ErrorInfo {
            kind: kind.to_string(),
            message: err.to_string(),
        }
    }
}
