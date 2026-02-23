use std::collections::HashMap;

use crate::types::{EvalResult, FunctionCall, GroundTruth};

/// Evaluate a model's function calls against ground truth using AST-style matching.
///
/// BFCL ground truth format:
/// ```json
/// [{"func_name": {"param1": [accepted_val1, accepted_val2], "param2": [val]}}]
/// ```
///
/// The model passes if:
/// 1. It produced the same number of function calls as the ground truth.
/// 2. Each call matches a ground truth entry (name + parameter values).
///
/// Parameter matching: a produced value matches if it equals any value in the
/// ground truth's accepted values list (with type coercion for numbers/strings).
pub fn evaluate_ast(
    test_id: &str,
    calls: &[FunctionCall],
    ground_truth: &GroundTruth,
) -> EvalResult {
    let expected = &ground_truth.ground_truth;
    let expected_count = expected.len();
    let actual_count = calls.len();

    if actual_count != expected_count {
        return EvalResult {
            id: test_id.to_string(),
            passed: false,
            expected_calls: expected_count,
            actual_calls: actual_count,
            details: format!("Call count mismatch: expected {expected_count}, got {actual_count}"),
        };
    }

    // Match each expected call to an actual call (order-sensitive)
    for (i, (expected_call, actual_call)) in expected.iter().zip(calls.iter()).enumerate() {
        let Some((expected_name, expected_params)) = expected_call.iter().next() else {
            continue;
        };

        // Normalize function name (dots → underscores, matching the conversion layer)
        let normalized_expected = expected_name.replace('.', "_");
        let normalized_actual = actual_call.name.replace('.', "_");

        if normalized_expected != normalized_actual {
            return EvalResult {
                id: test_id.to_string(),
                passed: false,
                expected_calls: expected_count,
                actual_calls: actual_count,
                details: format!(
                    "Call {i}: name mismatch: expected '{expected_name}', got '{}'",
                    actual_call.name
                ),
            };
        }

        // Check parameters
        if let Some(mismatch) = check_params(expected_params, &actual_call.arguments) {
            return EvalResult {
                id: test_id.to_string(),
                passed: false,
                expected_calls: expected_count,
                actual_calls: actual_count,
                details: format!("Call {i} ({expected_name}): {mismatch}"),
            };
        }
    }

    EvalResult {
        id: test_id.to_string(),
        passed: true,
        expected_calls: expected_count,
        actual_calls: actual_count,
        details: "All calls match".to_string(),
    }
}

/// Evaluate an irrelevance test case: model should NOT produce any function calls.
pub fn evaluate_irrelevance(test_id: &str, calls: &[FunctionCall]) -> EvalResult {
    let passed = calls.is_empty();
    EvalResult {
        id: test_id.to_string(),
        passed,
        expected_calls: 0,
        actual_calls: calls.len(),
        details: if passed {
            "Correctly refused to call any function".to_string()
        } else {
            format!(
                "Should not have called any function, but called: {}",
                calls
                    .iter()
                    .map(|c| c.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        },
    }
}

/// Check if actual parameters match expected parameters.
/// Returns `None` if all match, or `Some(description)` for the first mismatch.
fn check_params(
    expected: &HashMap<String, Vec<serde_json::Value>>,
    actual: &HashMap<String, serde_json::Value>,
) -> Option<String> {
    // Check each required expected parameter
    for (param_name, accepted_values) in expected {
        // Skip empty accepted values lists (optional params with no constraint)
        if accepted_values.is_empty() {
            continue;
        }

        // If all accepted values are empty string/null, param is optional
        let all_empty = accepted_values
            .iter()
            .all(|v| v.is_null() || v.as_str() == Some(""));
        if all_empty {
            continue;
        }

        let Some(actual_value) = actual.get(param_name) else {
            // Check if all accepted values include empty string (meaning param is optional)
            let has_empty = accepted_values
                .iter()
                .any(|v| v.is_null() || v.as_str() == Some(""));
            if has_empty {
                continue;
            }
            return Some(format!("missing required parameter '{param_name}'"));
        };

        if !value_matches_any(actual_value, accepted_values) {
            return Some(format!(
                "parameter '{param_name}': got {actual_value}, expected one of {accepted_values:?}"
            ));
        }
    }

    None
}

/// Check if an actual value matches any of the accepted values (with type coercion).
fn value_matches_any(actual: &serde_json::Value, accepted: &[serde_json::Value]) -> bool {
    accepted
        .iter()
        .any(|expected| values_match(actual, expected))
}

/// Compare two JSON values with type-tolerant matching.
///
/// - Numbers: compared by numeric value (int/float interchangeable)
/// - Strings: compared case-insensitively after trimming
/// - Arrays: element-wise comparison (order matters)
/// - Objects: recursive comparison
/// - Null/empty string in expected: matches anything (optional param)
fn values_match(actual: &serde_json::Value, expected: &serde_json::Value) -> bool {
    // Empty string or null in expected means "any value is fine"
    if expected.is_null() || expected.as_str() == Some("") {
        return true;
    }

    match (actual, expected) {
        // Both numbers — compare as f64
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            match (a.as_f64(), b.as_f64()) {
                (Some(a_f), Some(b_f)) => (a_f - b_f).abs() < 1e-9,
                _ => false,
            }
        }
        // String vs Number: try parsing the string as a number
        (serde_json::Value::String(s), serde_json::Value::Number(n)) => {
            if let (Ok(s_f), Some(n_f)) = (s.parse::<f64>(), n.as_f64()) {
                (s_f - n_f).abs() < 1e-9
            } else {
                false
            }
        }
        (serde_json::Value::Number(n), serde_json::Value::String(s)) => {
            if let (Ok(s_f), Some(n_f)) = (s.parse::<f64>(), n.as_f64()) {
                (s_f - n_f).abs() < 1e-9
            } else {
                false
            }
        }
        // Both strings
        (serde_json::Value::String(a), serde_json::Value::String(b)) => {
            a.trim().eq_ignore_ascii_case(b.trim())
        }
        // Both arrays
        (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| values_match(x, y))
        }
        // Both objects
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            // All keys in expected must match
            b.iter()
                .all(|(k, v)| a.get(k).is_some_and(|av| values_match(av, v)))
        }
        // Both booleans
        (serde_json::Value::Bool(a), serde_json::Value::Bool(b)) => a == b,
        // Both null
        (serde_json::Value::Null, serde_json::Value::Null) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_values_match_numbers() {
        assert!(values_match(&serde_json::json!(5), &serde_json::json!(5)));
        assert!(values_match(&serde_json::json!(5), &serde_json::json!(5.0)));
        assert!(!values_match(&serde_json::json!(5), &serde_json::json!(6)));
    }

    #[test]
    fn test_values_match_strings() {
        assert!(values_match(
            &serde_json::json!("hello"),
            &serde_json::json!("Hello")
        ));
        assert!(values_match(
            &serde_json::json!("units"),
            &serde_json::json!("units")
        ));
    }

    #[test]
    fn test_empty_string_matches_anything() {
        assert!(values_match(&serde_json::json!(42), &serde_json::json!("")));
        assert!(values_match(
            &serde_json::json!("anything"),
            &serde_json::json!("")
        ));
    }

    #[test]
    fn test_evaluate_simple() {
        let calls = vec![FunctionCall {
            name: "calculate_triangle_area".to_string(),
            arguments: [
                ("base".to_string(), serde_json::json!(10)),
                ("height".to_string(), serde_json::json!(5)),
            ]
            .into_iter()
            .collect(),
        }];

        let gt = GroundTruth {
            id: "simple_0".to_string(),
            ground_truth: vec![{
                let mut m = HashMap::new();
                m.insert(
                    "calculate_triangle_area".to_string(),
                    [
                        ("base".to_string(), vec![serde_json::json!(10)]),
                        ("height".to_string(), vec![serde_json::json!(5)]),
                        (
                            "unit".to_string(),
                            vec![serde_json::json!("units"), serde_json::json!("")],
                        ),
                    ]
                    .into_iter()
                    .collect(),
                );
                m
            }],
        };

        let result = evaluate_ast("simple_0", &calls, &gt);
        assert!(result.passed, "Expected pass, got: {}", result.details);
    }

    #[test]
    fn test_evaluate_irrelevance() {
        let result = evaluate_irrelevance("irrel_0", &[]);
        assert!(result.passed);

        let calls = vec![FunctionCall {
            name: "some_func".to_string(),
            arguments: HashMap::new(),
        }];
        let result = evaluate_irrelevance("irrel_1", &calls);
        assert!(!result.passed);
    }
}
