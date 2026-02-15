use std::path::Path;

use membrane_core::membrane_tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct WriteFileInput {
    /// File path to write to
    path: String,
    /// Content to write
    content: String,
}

#[membrane_tool(
    name = "write_file",
    description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Parent directories are created automatically."
)]
async fn write_file(input: WriteFileInput) -> Result<String, membrane_core::error::Error> {
    let path = Path::new(&input.path);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            membrane_core::error::Error::ToolExecution {
                tool_name: "write_file".to_string(),
                message: format!("Failed to create directories for {}: {}", input.path, e),
            }
        })?;
    }

    std::fs::write(path, &input.content).map_err(|e| {
        membrane_core::error::Error::ToolExecution {
            tool_name: "write_file".to_string(),
            message: format!("{}: {}", input.path, e),
        }
    })?;

    let lines = input.content.lines().count();
    Ok(format!("Wrote {} lines to {}", lines, input.path))
}
