use membrane_core::membrane_tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadFileInput {
    /// File path to read
    path: String,
    /// Line offset to start reading from (1-indexed). Defaults to 1.
    offset: Option<usize>,
    /// Maximum number of lines to read. Defaults to all lines.
    limit: Option<usize>,
}

#[membrane_tool(
    name = "read_file",
    description = "Read the contents of a file. Returns lines with line numbers. Use offset and limit for large files."
)]
async fn read_file(input: ReadFileInput) -> Result<String, membrane_core::error::Error> {
    let content = std::fs::read_to_string(&input.path).map_err(|e| {
        membrane_core::error::Error::ToolExecution {
            tool_name: "read_file".to_string(),
            message: format!("{}: {}", input.path, e),
        }
    })?;

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let offset = input.offset.unwrap_or(1).saturating_sub(1);
    let limit = input.limit.unwrap_or(total.saturating_sub(offset));

    let numbered: Vec<String> = lines
        .iter()
        .enumerate()
        .skip(offset)
        .take(limit)
        .map(|(i, line)| format!("{:>4} | {}", i + 1, line))
        .collect();

    let header = format!("({} total lines)", total);
    Ok(format!("{}\n{}", header, numbered.join("\n")))
}
