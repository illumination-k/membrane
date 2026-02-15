use membrane_core::membrane_tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchFilesInput {
    /// Glob pattern to match files (e.g., "**/*.rs", "src/**/*.ts")
    pattern: String,
}

#[membrane_tool(
    name = "search_files",
    description = "Search for files matching a glob pattern. Returns a list of matching file paths."
)]
async fn search_files(input: SearchFilesInput) -> Result<String, membrane_core::error::Error> {
    let paths =
        glob::glob(&input.pattern).map_err(|e| membrane_core::error::Error::ToolExecution {
            tool_name: "search_files".to_string(),
            message: format!("Invalid glob pattern: {}", e),
        })?;

    let mut results = Vec::new();
    for entry in paths {
        match entry {
            Ok(path) => results.push(path.display().to_string()),
            Err(e) => results.push(format!("(error: {})", e)),
        }
    }

    if results.is_empty() {
        Ok("No files found matching the pattern.".to_string())
    } else {
        Ok(format!(
            "{} files found:\n{}",
            results.len(),
            results.join("\n")
        ))
    }
}
