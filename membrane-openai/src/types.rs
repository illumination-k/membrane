use serde::{Deserialize, Serialize};

// === Request Types ===

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAiResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAiFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OpenAiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAiFunctionDef,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiFunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub json_schema: OpenAiJsonSchema,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiJsonSchema {
    pub name: String,
    pub schema: serde_json::Value,
    pub strict: bool,
}

// === Response Types ===

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiResponse {
    pub choices: Vec<OpenAiChoice>,
    pub usage: OpenAiUsage,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiChoice {
    pub message: OpenAiResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiResponseMessage {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

// === Error Types ===

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiErrorResponse {
    pub error: OpenAiErrorDetail,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiErrorDetail {
    pub message: String,
}
