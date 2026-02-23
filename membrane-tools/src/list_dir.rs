use std::path::Path;

use membrane_core::error::Error;
use membrane_core::membrane_tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct ListDirInput {
    /// Directory path to list (defaults to current directory)
    path: Option<String>,
}

#[membrane_tool(
    name = "list_dir",
    description = "List files and directories in a path. Shows file types (file/dir) and sizes. Use this to explore project structure."
)]
async fn list_dir(input: ListDirInput) -> Result<String, Error> {
    let dir_path = input.path.unwrap_or_else(|| ".".to_string());
    let path = Path::new(&dir_path);

    if !path.is_dir() {
        return Err(Error::ToolExecution {
            tool_name: "list_dir".to_string(),
            message: format!("Not a directory: {dir_path}"),
        });
    }

    let mut entries: Vec<(String, bool, u64)> = Vec::new();

    let read_dir = std::fs::read_dir(path).map_err(|e| Error::ToolExecution {
        tool_name: "list_dir".to_string(),
        message: format!("Failed to read directory {dir_path}: {e}"),
    })?;

    for entry in read_dir {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let name = entry.file_name().to_string_lossy().to_string();
        let metadata = entry.metadata();
        let (is_dir, size) = match metadata {
            Ok(m) => (m.is_dir(), m.len()),
            Err(_) => (false, 0),
        };
        entries.push((name, is_dir, size));
    }

    entries.sort_by(|a, b| {
        // Directories first, then alphabetical
        b.1.cmp(&a.1).then(a.0.cmp(&b.0))
    });

    if entries.is_empty() {
        return Ok(format!("{dir_path}: empty directory"));
    }

    let lines: Vec<String> = entries
        .iter()
        .map(|(name, is_dir, size)| {
            if *is_dir {
                format!("  {name}/")
            } else {
                format!("  {name} ({size} bytes)")
            }
        })
        .collect();

    Ok(format!(
        "{dir_path} ({} entries):\n{}",
        entries.len(),
        lines.join("\n")
    ))
}
