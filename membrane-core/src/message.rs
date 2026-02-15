use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        id: String,
        output: String,
        is_error: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

impl Message {
    /// Create a system message with text content.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![Content::Text { text: text.into() }],
        }
    }

    /// Create a user message with text content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text { text: text.into() }],
        }
    }

    /// Create an assistant message with text content.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::Text { text: text.into() }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_roundtrip_serialization() {
        let msg = Message::user("Hello");
        let json = serde_json::to_string(&msg).expect("test");
        let deserialized: Message = serde_json::from_str(&json).expect("test");
        assert_eq!(deserialized.role, Role::User);
    }

    #[test]
    fn content_serialization_tagged() {
        let content = Content::ToolUse {
            id: "call_1".to_string(),
            name: "search".to_string(),
            input: serde_json::json!({"query": "rust"}),
        };
        let json = serde_json::to_string(&content).expect("test");
        assert!(json.contains("\"type\":\"tool_use\""));
    }
}
