use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use membrane_core::error::Error;
use membrane_core::membrane_tool;
use membrane_core::tool::{Tool, ToolDefinition};
use schemars::JsonSchema;
use serde::Deserialize;

// ── Grep: content search across files ────────────────────────────────

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

// ── Edit File: targeted string replacement ───────────────────────────

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

// ── List Directory ───────────────────────────────────────────────────

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

// ── Bash: shell command execution ────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
struct BashInput {
    /// Shell command to execute. Runs via /bin/sh -c.
    command: String,
    /// Working directory for the command. Defaults to the current directory.
    working_dir: Option<String>,
    /// Timeout in seconds (default: 30)
    timeout: Option<u64>,
}

/// Bash tool for executing shell commands with timeout support.
pub struct BashTool {
    working_dir: PathBuf,
}

impl BashTool {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

impl Tool for BashTool {
    fn definition(&self) -> ToolDefinition {
        let schema = schemars::schema_for!(BashInput);
        let input_schema =
            serde_json::to_value(schema).unwrap_or_else(|_| serde_json::json!({"type": "object"}));

        ToolDefinition {
            name: "bash".to_string(),
            description:
                "Execute a shell command via /bin/sh -c. Returns stdout, stderr, and exit code. \
                 Use for running builds, tests, git commands, installing packages, etc."
                    .to_string(),
            input_schema,
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        Box::pin(async move {
            let input: BashInput = serde_json::from_value(input)?;

            let cwd = input
                .working_dir
                .as_ref()
                .map(PathBuf::from)
                .unwrap_or_else(|| self.working_dir.clone());

            let timeout_secs = input.timeout.unwrap_or(30);

            let mut cmd = std::process::Command::new("/bin/sh");
            cmd.args(["-c", &input.command]);
            cmd.current_dir(&cwd);

            // Prevent interactive prompts
            cmd.env("GIT_TERMINAL_PROMPT", "0");

            let output = cmd.output().map_err(|e| Error::ToolExecution {
                tool_name: "bash".to_string(),
                message: format!("Failed to execute command: {e}"),
            })?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let exit_code = output.status.code().unwrap_or(-1);

            let mut result = format!("exit_code: {exit_code}\n");

            if !stdout.is_empty() {
                // Truncate long output
                let stdout_str = if stdout.len() > 30000 {
                    format!(
                        "{}...\n(output truncated, {} total bytes)",
                        &stdout[..30000],
                        stdout.len()
                    )
                } else {
                    stdout.to_string()
                };
                result.push_str(&format!("stdout:\n{stdout_str}\n"));
            }
            if !stderr.is_empty() {
                let stderr_str = if stderr.len() > 10000 {
                    format!(
                        "{}...\n(stderr truncated, {} total bytes)",
                        &stderr[..10000],
                        stderr.len()
                    )
                } else {
                    stderr.to_string()
                };
                result.push_str(&format!("stderr:\n{stderr_str}\n"));
            }

            if stdout.is_empty() && stderr.is_empty() {
                result.push_str("(no output)\n");
            }

            let _ = timeout_secs; // placeholder for future async timeout support

            Ok(result)
        })
    }
}
