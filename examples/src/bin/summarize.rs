use membrane_core::agent::{Agent, AgentConfig};
use membrane_core::message::Message;
use membrane_openai::OpenAiProvider;
use membrane_tools::{ExecTool, ReadFileTool, SearchFilesTool, WriteFileTool};

const SYSTEM_PROMPT: &str = r#"You are a project analyst.
Your task is to explore a Rust project and generate a summary document.

Rules:
- Write in English
- Output the summary to summary.md in the project root
- Include: project overview, crate structure, public API highlights, and dependency graph
- Be concise but informative

Workflow:
1. Search for source files to understand the project structure
2. Read key files (Cargo.toml, lib.rs files) to understand what each crate does
3. Read important source files to understand the public API
4. Write summary.md with your findings
"#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable must be set");

    let provider = OpenAiProvider::builder().api_key(api_key).build();

    let agent = Agent::new(
        provider,
        vec![
            Box::new(ReadFileTool),
            Box::new(WriteFileTool),
            Box::new(SearchFilesTool),
            Box::new(ExecTool),
        ],
        AgentConfig {
            model: "gpt-4o".to_string(),
            max_iterations: 20,
            system_prompt: Some(SYSTEM_PROMPT.to_string()),
            max_tokens: Some(4096),
            temperature: Some(0.0),
        },
    );

    let output = agent
        .run(vec![Message::user(
            "Explore this Rust project and generate summary.md in the project root. \
             The project root is the current working directory.",
        )])
        .await?;

    println!("=== Agent Response ===");
    println!("{}", output.response);
    println!();
    println!("=== Execution Steps ===");
    for (i, step) in output.steps.iter().enumerate() {
        match step {
            membrane_core::agent::AgentStep::LlmCall {
                request_messages,
                response,
            } => {
                println!(
                    "Step {}: LLM call ({} messages, stop_reason={:?})",
                    i + 1,
                    request_messages,
                    response.stop_reason,
                );
            }
            membrane_core::agent::AgentStep::ToolExecution {
                name,
                input,
                output,
                is_error,
            } => {
                println!(
                    "Step {}: Tool '{}' input={} output={}{}",
                    i + 1,
                    name,
                    input,
                    output,
                    if *is_error { " [ERROR]" } else { "" },
                );
            }
        }
    }
    println!();
    println!(
        "Total usage: {} input tokens, {} output tokens",
        output.total_usage.input_tokens, output.total_usage.output_tokens,
    );

    Ok(())
}
