mod prompt;
mod tools;

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use membrane_core::agent::{Agent, AgentConfig, AgentStep};
use membrane_core::message::Message;
use membrane_core::stop_condition::{MaxConsecutiveErrors, Timeout, TokenBudget};
use membrane_openai::OpenAiProvider;
use membrane_tools::{ReadFileTool, SearchFilesTool, WriteFileTool, task_list_tools};

use tools::{BashTool, EditFileTool, GrepTool, ListDirTool};

#[derive(Parser)]
#[command(
    name = "coding-agent",
    about = "An AI-powered coding assistant built with membrane"
)]
struct Cli {
    /// OpenAI-compatible API base URL
    #[arg(
        long,
        env = "OPENAI_BASE_URL",
        default_value = "https://api.openai.com/v1"
    )]
    base_url: String,

    /// API key (reads from OPENAI_API_KEY env var)
    #[arg(long, env = "OPENAI_API_KEY")]
    api_key: String,

    /// Model to use
    #[arg(long, short, default_value = "o4-mini")]
    model: String,

    /// Maximum agent iterations per request
    #[arg(long, default_value = "50")]
    max_iterations: usize,

    /// Token budget (total input + output tokens per request)
    #[arg(long, default_value = "1000000")]
    token_budget: u32,

    /// Timeout per request in seconds
    #[arg(long, default_value = "300")]
    timeout: u64,

    /// Working directory (defaults to current directory)
    #[arg(long, short)]
    dir: Option<String>,

    /// Run a single prompt non-interactively then exit
    #[arg(long, short)]
    prompt: Option<String>,
}

fn build_agent(cli: &Cli, working_dir: &Path) -> Agent<OpenAiProvider> {
    let provider = OpenAiProvider::builder()
        .api_key(&cli.api_key)
        .base_url(&cli.base_url)
        .build();

    let (task_write, task_read) = task_list_tools();

    let tools: Vec<Box<dyn membrane_core::tool::Tool>> = vec![
        Box::new(ReadFileTool),
        Box::new(WriteFileTool),
        Box::new(EditFileTool),
        Box::new(GrepTool),
        Box::new(SearchFilesTool),
        Box::new(ListDirTool),
        Box::new(BashTool::new(working_dir.to_path_buf())),
        Box::new(task_write),
        Box::new(task_read),
    ];

    // Try to load project context from CLAUDE.md or similar
    let project_context = load_project_context(working_dir);
    let system_prompt = prompt::build_system_prompt(
        &working_dir.display().to_string(),
        project_context.as_deref(),
    );

    let config = AgentConfig {
        model: cli.model.clone(),
        max_iterations: cli.max_iterations,
        extra_params: serde_json::Map::new(),
    };

    Agent::with_system_prompt(provider, tools, config, system_prompt)
        .with_stop_condition(Timeout::new(Duration::from_secs(cli.timeout)))
        .with_stop_condition(TokenBudget::new(cli.token_budget))
        .with_stop_condition(MaxConsecutiveErrors::new(5))
}

fn load_project_context(working_dir: &Path) -> Option<String> {
    let candidates = ["CLAUDE.md", "AGENTS.md", ".claude/instructions.md"];
    for name in &candidates {
        let path = working_dir.join(name);
        if let Ok(content) = std::fs::read_to_string(&path)
            && !content.trim().is_empty()
        {
            return Some(content);
        }
    }
    None
}

fn print_step_summary(steps: &[AgentStep]) {
    let mut tool_calls = 0;
    let mut tool_errors = 0;
    for step in steps {
        if let AgentStep::ToolExecution { is_error, .. } = step {
            tool_calls += 1;
            if *is_error {
                tool_errors += 1;
            }
        }
    }
    let llm_calls = steps
        .iter()
        .filter(|s| matches!(s, AgentStep::LlmCall { .. }))
        .count();

    if tool_calls > 0 || llm_calls > 1 {
        eprint!("\n  ({llm_calls} LLM calls, {tool_calls} tool calls");
        if tool_errors > 0 {
            eprint!(", {tool_errors} errors");
        }
        eprintln!(")");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Set up tracing (only show warnings by default, RUST_LOG for more)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(false)
        .init();

    let working_dir = match &cli.dir {
        Some(d) => PathBuf::from(d)
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(d)),
        None => std::env::current_dir()?,
    };

    // Non-interactive mode: single prompt
    if let Some(ref user_prompt) = cli.prompt {
        let agent = build_agent(&cli, &working_dir);
        let output = agent.run(vec![Message::user(user_prompt)]).await?;
        println!("{}", output.response);
        print_step_summary(&output.steps);
        eprintln!(
            "  (tokens: {} in / {} out, stop: {:?})",
            output.total_usage.input_tokens, output.total_usage.output_tokens, output.stop_reason
        );
        return Ok(());
    }

    // Interactive REPL mode
    eprintln!(
        "coding-agent v0.1.0 (model: {}, dir: {})",
        cli.model,
        working_dir.display()
    );
    eprintln!("Type your request. Press Ctrl+D or type /exit to quit.\n");

    let stdin = io::stdin();
    let mut conversation: Vec<Message> = Vec::new();

    loop {
        eprint!("> ");
        io::stderr().flush()?;

        let mut input = String::new();
        let bytes = stdin.lock().read_line(&mut input)?;

        // EOF (Ctrl+D)
        if bytes == 0 {
            eprintln!("\nGoodbye!");
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Commands
        match input {
            "/exit" | "/quit" => {
                eprintln!("Goodbye!");
                break;
            }
            "/clear" => {
                conversation.clear();
                eprintln!("Conversation cleared.");
                continue;
            }
            "/help" => {
                eprintln!("Commands:");
                eprintln!("  /clear  - Clear conversation history");
                eprintln!("  /exit   - Exit the agent");
                eprintln!("  /help   - Show this help");
                continue;
            }
            _ => {}
        }

        conversation.push(Message::user(input));

        let agent = build_agent(&cli, &working_dir);

        match agent.run(conversation.clone()).await {
            Ok(output) => {
                println!("\n{}", output.response);
                print_step_summary(&output.steps);
                eprintln!(
                    "  (tokens: {} in / {} out)\n",
                    output.total_usage.input_tokens, output.total_usage.output_tokens
                );

                // Add assistant response to conversation for multi-turn
                conversation.push(Message::assistant(&output.response));
            }
            Err(e) => {
                eprintln!("\nError: {e}\n");
            }
        }
    }

    Ok(())
}
