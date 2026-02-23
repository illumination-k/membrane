use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single BFCL test entry (simple/parallel/multiple/irrelevance categories).
#[derive(Debug, Clone, Deserialize)]
pub struct TestEntry {
    pub id: String,
    /// Nested array: outer = turns, inner = messages per turn.
    /// For single-turn: `[[{"role": "user", "content": "..."}]]`
    pub question: Vec<Vec<QuestionMessage>>,
    /// Available function definitions for this test case.
    pub function: Vec<FunctionDef>,
}

/// A multi-turn BFCL test entry (used by multi_turn_* categories).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct MultiTurnTestEntry {
    pub id: String,
    pub question: Vec<Vec<QuestionMessage>>,
    /// Available function paths (e.g. "GorillaFileSystem.find").
    #[serde(default)]
    pub path: Vec<String>,
    /// Initial environment state.
    #[serde(default)]
    pub initial_config: serde_json::Value,
    /// Functions excluded from use.
    #[serde(default)]
    pub excluded_function: Vec<String>,
}

/// A message within a BFCL question (maps to chat message).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QuestionMessage {
    pub role: String,
    pub content: String,
}

/// A BFCL function definition.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
    /// Optional response schema (appended to description).
    #[serde(default)]
    pub response: Option<serde_json::Value>,
}

/// A BFCL ground truth answer entry.
#[derive(Debug, Clone, Deserialize)]
pub struct GroundTruth {
    pub id: String,
    /// Each element is a function call: `{"func_name": {"param": [accepted_values]}}`.
    pub ground_truth: Vec<HashMap<String, HashMap<String, Vec<serde_json::Value>>>>,
}

/// Result of evaluating a single test case.
#[derive(Debug, Clone, Serialize)]
pub struct EvalResult {
    pub id: String,
    pub passed: bool,
    pub expected_calls: usize,
    pub actual_calls: usize,
    pub details: String,
}

/// Aggregated evaluation scores for a category.
#[derive(Debug, Clone, Serialize)]
pub struct CategoryScore {
    pub category: String,
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
}

/// A model-produced function call (extracted from tool_use content).
#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}
