use std::path::Path;

use membrane_core::error::Error;
use membrane_core::membrane_tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct GrepInput {
    /// Regular expression pattern to search for
    pattern: String,
    /// Directory or file path to search in (defaults to current directory)
    path: Option<String>,
    /// Glob pattern to filter files (e.g., "*.rs", "*.ts")
    include: Option<String>,
}

#[membrane_tool(
    name = "grep",
    description = "Search file contents for a regular expression pattern. Returns matching lines with file paths and line numbers. Use this to find code, function definitions, usages, etc."
)]
async fn grep(input: GrepInput) -> Result<String, Error> {
    let search_path = input.path.unwrap_or_else(|| ".".to_string());
    let path = Path::new(&search_path);

    if !path.exists() {
        return Err(Error::ToolExecution {
            tool_name: "grep".to_string(),
            message: format!("Path does not exist: {search_path}"),
        });
    }

    let mut cmd = std::process::Command::new("grep");
    cmd.args(["-rn", "--color=never"]);

    if let Some(ref include) = input.include {
        cmd.arg(format!("--include={include}"));
    }

    // Exclude common non-code directories
    cmd.args([
        "--exclude-dir=.git",
        "--exclude-dir=node_modules",
        "--exclude-dir=target",
        "--exclude-dir=.next",
        "--exclude-dir=dist",
    ]);

    cmd.arg(&input.pattern);
    cmd.arg(&search_path);

    let output = cmd.output().map_err(|e| Error::ToolExecution {
        tool_name: "grep".to_string(),
        message: format!("Failed to execute grep: {e}"),
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    if stdout.is_empty() {
        return Ok(format!(
            "No matches found for pattern '{}' in {}",
            input.pattern, search_path
        ));
    }

    // Truncate output if too large
    let lines: Vec<&str> = stdout.lines().collect();
    let total = lines.len();
    if total > 200 {
        let truncated: String = lines[..200].join("\n");
        Ok(format!(
            "{truncated}\n\n... ({total} total matches, showing first 200)"
        ))
    } else {
        Ok(format!("{total} matches:\n{stdout}"))
    }
}
