use std::sync::Arc;

use membrane_core::error::Error;
use membrane_core::plugin::Plugin;
use membrane_core::tool::Tool;
use rmcp::service::{RunningService, ServiceExt};
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use serde_json::json;

use crate::tool::McpTool;

type McpService = RunningService<rmcp::RoleClient, ()>;

/// Transport configuration for connecting to an MCP server.
#[derive(Debug, Clone)]
pub enum McpTransport {
    /// Connect via stdio by spawning a child process.
    ///
    /// The `command` is the executable name (e.g. `"npx"`, `"uvx"`, `"node"`),
    /// and `args` are the arguments to pass (e.g. `["-y", "@modelcontextprotocol/server-everything"]`).
    Stdio { command: String, args: Vec<String> },
    /// Connect via Streamable HTTP to a running MCP server.
    StreamableHttp { url: String },
}

/// Configuration for connecting to an MCP server.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// A human-readable name for this MCP server (used as plugin name).
    pub name: String,
    /// Transport configuration.
    pub transport: McpTransport,
}

/// Plugin that connects to an MCP server and exposes its tools.
///
/// `McpPlugin` implements the membrane [`Plugin`] trait. On construction it
/// connects to the MCP server, discovers all available tools, and wraps each
/// one as an [`McpTool`] so the agent can invoke them transparently.
///
/// # Examples
///
/// ```no_run
/// use membrane_mcp::{McpPlugin, McpServerConfig, McpTransport};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let plugin = McpPlugin::connect(McpServerConfig {
///     name: "git".to_string(),
///     transport: McpTransport::Stdio {
///         command: "uvx".to_string(),
///         args: vec!["mcp-server-git".to_string()],
///     },
/// }).await?;
/// # Ok(())
/// # }
/// ```
pub struct McpPlugin {
    name: String,
    tools: Vec<Box<dyn Tool>>,
    tool_descriptions: Vec<String>,
    _service: Arc<McpService>,
}

impl McpPlugin {
    /// Connect to an MCP server and discover its tools.
    pub async fn connect(config: McpServerConfig) -> Result<Self, Error> {
        let service: McpService = match &config.transport {
            McpTransport::Stdio { command, args } => {
                let args = args.clone();
                let transport = TokioChildProcess::new(
                    tokio::process::Command::new(command).configure(move |cmd| {
                        cmd.args(&args);
                    }),
                )
                .map_err(|e| Error::Provider(format!("Failed to spawn MCP server: {e}")))?;

                <() as ServiceExt<rmcp::RoleClient>>::serve((), transport)
                    .await
                    .map_err(|e| Error::Provider(format!("MCP handshake failed: {e}")))?
            }
            McpTransport::StreamableHttp { url } => {
                let transport =
                    rmcp::transport::StreamableHttpClientTransport::from_uri(url.as_str());

                <() as ServiceExt<rmcp::RoleClient>>::serve((), transport)
                    .await
                    .map_err(|e| Error::Provider(format!("MCP handshake failed: {e}")))?
            }
        };

        let service = Arc::new(service);

        let tools_result = service
            .list_all_tools()
            .await
            .map_err(|e| Error::Provider(format!("Failed to list MCP tools: {e}")))?;

        tracing::info!(
            server = %config.name,
            tool_count = tools_result.len(),
            "Discovered MCP tools"
        );

        let mut tools: Vec<Box<dyn Tool>> = Vec::new();
        let mut tool_descriptions = Vec::new();

        for t in tools_result {
            let name = t.name.to_string();
            let description = t.description.as_deref().unwrap_or("").to_string();
            let input_schema: serde_json::Value = serde_json::to_value(&*t.input_schema)
                .unwrap_or_else(|_| json!({"type": "object"}));

            tool_descriptions.push(format!("- {name}: {description}"));

            tools.push(Box::new(McpTool::new(
                Arc::clone(&service),
                name,
                description,
                input_schema,
            )));
        }

        Ok(Self {
            name: config.name,
            tools,
            tool_descriptions,
            _service: service,
        })
    }
}

impl Plugin for McpPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "MCP server plugin"
    }

    fn tools(&mut self) -> Vec<Box<dyn Tool>> {
        std::mem::take(&mut self.tools)
    }

    fn context(&self) -> Vec<String> {
        if self.tool_descriptions.is_empty() {
            return vec![];
        }
        vec![format!(
            "MCP server \"{}\" provides the following tools:\n{}",
            self.name,
            self.tool_descriptions.join("\n")
        )]
    }
}
