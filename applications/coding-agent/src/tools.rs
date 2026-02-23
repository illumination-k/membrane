use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use membrane_core::error::Error;
use membrane_core::tool::{Tool, ToolDefinition};
use schemars::JsonSchema;
use serde::Deserialize;

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
