use std::collections::HashMap;
use std::path::Path;

use membrane_core::message::Content;
use membrane_core::provider::{ChatRequest, ChatResponse, LlmProvider};

use crate::convert::{
    extract_function_calls, function_def_to_tool_definition, question_messages_to_membrane,
};
use crate::eval::{evaluate_ast, evaluate_irrelevance};
use crate::types::{CategoryScore, EvalResult, GroundTruth, TestEntry};

/// Configuration for a BFCL evaluation run.
pub struct RunConfig {
    pub model: String,
    pub data_dir: String,
    pub category: String,
    pub limit: Option<usize>,
    pub extra_params: serde_json::Map<String, serde_json::Value>,
}

/// Load test entries from a BFCL JSON file.
///
/// Each line is a separate JSON object (JSONL format).
pub fn load_test_entries(path: &Path) -> Result<Vec<TestEntry>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;

    // Try JSONL first (one JSON object per line)
    let entries: Vec<TestEntry> = if content.trim_start().starts_with('[') {
        serde_json::from_str(&content)?
    } else {
        content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(entries)
}

/// Load ground truth answers from a BFCL possible_answer JSON file.
pub fn load_ground_truth(
    path: &Path,
) -> Result<HashMap<String, GroundTruth>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;

    let entries: Vec<GroundTruth> = if content.trim_start().starts_with('[') {
        serde_json::from_str(&content)?
    } else {
        content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<Vec<_>, _>>()?
    };

    let map = entries.into_iter().map(|gt| (gt.id.clone(), gt)).collect();
    Ok(map)
}

/// Run a single test entry through the LLM and evaluate the result.
pub async fn run_single_test<P: LlmProvider>(
    provider: &P,
    entry: &TestEntry,
    ground_truth: Option<&GroundTruth>,
    model: &str,
    extra_params: &serde_json::Map<String, serde_json::Value>,
) -> Result<(EvalResult, ChatResponse), Box<dyn std::error::Error>> {
    let is_irrelevance = entry.id.starts_with("irrelevance") || entry.id.starts_with("live_irrel");

    // Convert BFCL function definitions to membrane ToolDefinitions
    let tools = entry
        .function
        .iter()
        .map(function_def_to_tool_definition)
        .collect::<Vec<_>>();

    // Convert first turn's messages to membrane format
    let first_turn = entry.question.first().ok_or("empty question array")?;
    let messages = question_messages_to_membrane(first_turn);

    // Make the LLM call
    let request = ChatRequest {
        model: model.to_string(),
        messages,
        tools,
        response_format: None,
        extra_params: extra_params.clone(),
    };

    let response = provider.chat(request).await?;

    // Extract function calls from the response
    let calls = extract_function_calls(&response.content);

    // Evaluate
    let eval_result = if is_irrelevance {
        evaluate_irrelevance(&entry.id, &calls)
    } else if let Some(gt) = ground_truth {
        evaluate_ast(&entry.id, &calls, gt)
    } else {
        // No ground truth — just record what happened
        EvalResult {
            id: entry.id.clone(),
            passed: false,
            expected_calls: 0,
            actual_calls: calls.len(),
            details: "No ground truth available".to_string(),
        }
    };

    Ok((eval_result, response))
}

/// Run evaluation for a complete category.
pub async fn run_category<P: LlmProvider>(
    provider: &P,
    config: &RunConfig,
) -> Result<CategoryScore, Box<dyn std::error::Error>> {
    let data_dir = Path::new(&config.data_dir);

    // Load test data
    let test_file = data_dir.join(format!("BFCL_v4_{}.json", config.category));
    if !test_file.exists() {
        return Err(format!("Test file not found: {}", test_file.display()).into());
    }
    let entries = load_test_entries(&test_file)?;

    // Load ground truth (if available)
    let gt_file = data_dir.join(format!("possible_answer/BFCL_v4_{}.json", config.category));
    let ground_truth = if gt_file.exists() {
        load_ground_truth(&gt_file)?
    } else {
        tracing::warn!(
            category = %config.category,
            "No ground truth file found at {}",
            gt_file.display()
        );
        HashMap::new()
    };

    let entries = if let Some(limit) = config.limit {
        &entries[..entries.len().min(limit)]
    } else {
        &entries
    };

    let total = entries.len();
    let mut correct = 0;
    let mut results: Vec<EvalResult> = Vec::with_capacity(total);

    for (i, entry) in entries.iter().enumerate() {
        let gt = ground_truth.get(&entry.id);

        match run_single_test(provider, entry, gt, &config.model, &config.extra_params).await {
            Ok((eval_result, response)) => {
                let status = if eval_result.passed {
                    correct += 1;
                    "PASS"
                } else {
                    "FAIL"
                };

                tracing::info!(
                    "[{}/{total}] {status} {} — {} (tokens: {}/{})",
                    i + 1,
                    eval_result.id,
                    eval_result.details,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                );

                results.push(eval_result);
            }
            Err(e) => {
                tracing::error!("[{}/{total}] ERROR {}: {e}", i + 1, entry.id);
                results.push(EvalResult {
                    id: entry.id.clone(),
                    passed: false,
                    expected_calls: 0,
                    actual_calls: 0,
                    details: format!("Error: {e}"),
                });
            }
        }
    }

    let accuracy = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    };

    let score = CategoryScore {
        category: config.category.clone(),
        total,
        correct,
        accuracy,
    };

    // Write detailed results to file
    let results_dir = data_dir.join("results");
    std::fs::create_dir_all(&results_dir)?;
    let results_file = results_dir.join(format!("{}_{}.json", config.model, config.category));
    let output = serde_json::json!({
        "model": config.model,
        "category": config.category,
        "score": score,
        "results": results,
    });
    std::fs::write(&results_file, serde_json::to_string_pretty(&output)?)?;
    tracing::info!("Results written to {}", results_file.display());

    Ok(score)
}

/// Extract a text summary from response content.
#[allow(dead_code)]
pub fn response_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| {
            if let Content::Text { text } = c {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
