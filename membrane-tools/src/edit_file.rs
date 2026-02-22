use membrane_core::error::Error;
use membrane_core::membrane_tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct EditFileInput {
    /// Path to the file to edit
    path: String,
    /// The exact text to find and replace. Must match exactly.
    old_string: String,
    /// The new text to replace with
    new_string: String,
}

#[membrane_tool(
    name = "edit_file",
    description = "Edit a file by replacing an exact string match. The old_string must appear exactly once in the file. Use read_file first to see the current contents, then provide the exact text to replace."
)]
async fn edit_file(input: EditFileInput) -> Result<String, Error> {
    let content = std::fs::read_to_string(&input.path).map_err(|e| Error::ToolExecution {
        tool_name: "edit_file".to_string(),
        message: format!("{}: {}", input.path, e),
    })?;

    let count = content.matches(&input.old_string).count();
    if count == 0 {
        return Err(Error::ToolExecution {
            tool_name: "edit_file".to_string(),
            message: format!(
                "old_string not found in {}. Make sure it matches exactly (including whitespace).",
                input.path
            ),
        });
    }
    if count > 1 {
        return Err(Error::ToolExecution {
            tool_name: "edit_file".to_string(),
            message: format!(
                "old_string found {count} times in {}. Provide more surrounding context to make the match unique.",
                input.path
            ),
        });
    }

    let new_content = content.replacen(&input.old_string, &input.new_string, 1);
    std::fs::write(&input.path, &new_content).map_err(|e| Error::ToolExecution {
        tool_name: "edit_file".to_string(),
        message: format!("Failed to write {}: {}", input.path, e),
    })?;

    let old_lines = input.old_string.lines().count();
    let new_lines = input.new_string.lines().count();
    Ok(format!(
        "Edited {}. Replaced {old_lines} lines with {new_lines} lines.",
        input.path
    ))
}
