use membrane_core::membrane_tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct ExecInput {
    /// Command to execute (e.g., "ls", "git")
    command: String,
    /// Arguments to pass to the command
    args: Option<Vec<String>>,
    /// Working directory for the command. Defaults to the current directory.
    working_dir: Option<String>,
}

#[membrane_tool(
    name = "exec",
    description = "Execute a shell command and return its stdout and stderr. Use this to run CLI tools, scripts, or system commands."
)]
async fn exec(input: ExecInput) -> Result<String, membrane_core::error::Error> {
    let mut cmd = std::process::Command::new(&input.command);

    if let Some(args) = &input.args {
        cmd.args(args);
    }
    if let Some(dir) = &input.working_dir {
        cmd.current_dir(dir);
    }

    let output = cmd
        .output()
        .map_err(|e| membrane_core::error::Error::ToolExecution {
            tool_name: "exec".to_string(),
            message: format!("Failed to execute '{}': {}", input.command, e),
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut result = String::new();
    result.push_str(&format!(
        "exit_code: {}\n",
        output.status.code().unwrap_or(-1)
    ));

    if !stdout.is_empty() {
        result.push_str(&format!("stdout:\n{}\n", stdout));
    }
    if !stderr.is_empty() {
        result.push_str(&format!("stderr:\n{}\n", stderr));
    }

    Ok(result)
}
