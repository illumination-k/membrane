use membrane_core::message::{Content, Message};
use membrane_core::tool::ToolDefinition;

use crate::types::{FunctionCall, FunctionDef, QuestionMessage};

/// Convert a BFCL function definition to a membrane ToolDefinition.
///
/// BFCL uses `"type": "dict"` for object types, which must be mapped to
/// JSON Schema's `"type": "object"`. Property names with dots are replaced
/// with underscores for compatibility.
pub fn function_def_to_tool_definition(func: &FunctionDef) -> ToolDefinition {
    let mut schema = func.parameters.clone();
    normalize_schema_types(&mut schema);

    // Build description, appending response schema if present
    let description = if let Some(resp) = &func.response {
        format!(
            "{}\n\nResponse schema: {}",
            func.description,
            serde_json::to_string(resp).unwrap_or_default()
        )
    } else {
        func.description.clone()
    };

    // Sanitize function name (dots → underscores for OpenAI compatibility)
    let name = func.name.replace('.', "_");

    ToolDefinition {
        name,
        description,
        input_schema: schema,
    }
}

/// Convert BFCL question messages to membrane Messages.
pub fn question_messages_to_membrane(messages: &[QuestionMessage]) -> Vec<Message> {
    messages
        .iter()
        .map(|msg| match msg.role.as_str() {
            "system" => Message::system(&msg.content),
            "assistant" => Message::assistant(&msg.content),
            _ => Message::user(&msg.content),
        })
        .collect()
}

/// Extract function calls from a membrane ChatResponse's content blocks.
pub fn extract_function_calls(content: &[Content]) -> Vec<FunctionCall> {
    content
        .iter()
        .filter_map(|c| {
            if let Content::ToolUse { name, input, .. } = c {
                let arguments = match input.as_object() {
                    Some(obj) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                    None => std::collections::HashMap::new(),
                };
                Some(FunctionCall {
                    name: name.clone(),
                    arguments,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Recursively normalize BFCL schema types to JSON Schema types.
/// - `"dict"` → `"object"`
/// - `"integer"` stays, `"float"` → `"number"`, etc.
fn normalize_schema_types(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            // Fix the type field
            if let Some(type_val) = map.get_mut("type")
                && let Some(s) = type_val.as_str()
            {
                let normalized = match s {
                    "dict" => "object",
                    "float" => "number",
                    "list" => "array",
                    "tuple" => "array",
                    other => other,
                };
                *type_val = serde_json::Value::String(normalized.to_string());
            }
            // Recurse into all values
            for v in map.values_mut() {
                normalize_schema_types(v);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                normalize_schema_types(v);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_dict_to_object() {
        let mut schema = serde_json::json!({
            "type": "dict",
            "properties": {
                "base": {"type": "integer"},
                "height": {"type": "float"}
            }
        });
        normalize_schema_types(&mut schema);
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["height"]["type"], "number");
    }

    #[test]
    fn test_function_def_conversion() {
        let func = FunctionDef {
            name: "math.factorial".to_string(),
            description: "Calculate factorial".to_string(),
            parameters: serde_json::json!({
                "type": "dict",
                "properties": {
                    "number": {"type": "integer", "description": "The number"}
                },
                "required": ["number"]
            }),
            response: None,
        };

        let tool_def = function_def_to_tool_definition(&func);
        assert_eq!(tool_def.name, "math_factorial");
        assert_eq!(tool_def.input_schema["type"], "object");
    }
}
