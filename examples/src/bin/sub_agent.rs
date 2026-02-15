use membrane_core::agent::{Agent, AgentConfig, AgentStep};
use membrane_core::membrane_tool;
use membrane_core::message::Message;
use membrane_openai::OpenAiProvider;
use membrane_tools::{ReadFileTool, SearchFilesTool};
use schemars::JsonSchema;
use serde::Deserialize;

// --- Mock weather tool for the weather sub-agent ---

#[derive(Debug, Deserialize, JsonSchema)]
struct GetWeatherInput {
    /// City name to get weather for
    city: String,
}

#[membrane_tool(
    name = "get_weather",
    description = "Get the current weather for a city"
)]
async fn get_weather(input: GetWeatherInput) -> Result<String, membrane_core::error::Error> {
    let weather = match input.city.to_lowercase().as_str() {
        "tokyo" => "Sunny, 22°C, humidity 45%",
        "london" => "Rainy, 14°C, humidity 85%",
        "new york" => "Cloudy, 18°C, humidity 60%",
        "paris" => "Partly cloudy, 16°C, humidity 55%",
        "sydney" => "Clear, 28°C, humidity 40%",
        _ => "No data available",
    };
    Ok(format!("{}: {}", input.city, weather))
}

// --- Helper to build a provider ---

fn build_provider() -> OpenAiProvider {
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable must be set");
    OpenAiProvider::builder().api_key(api_key).build()
}

fn default_extra_params() -> serde_json::Map<String, serde_json::Value> {
    serde_json::json!({
        "max_completion_tokens": 2048,
        "temperature": 0.0,
    })
    .as_object()
    .expect("extra_params must be a JSON object")
    .clone()
}

// --- Main ---

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // 1. Build a "researcher" sub-agent that can explore the codebase
    let researcher = Agent::with_system_prompt(
        build_provider(),
        vec![Box::new(SearchFilesTool), Box::new(ReadFileTool)],
        AgentConfig {
            model: "gpt-4o-mini".to_string(),
            max_iterations: 10,
            extra_params: default_extra_params(),
        },
        "You are a code researcher. When asked a question about a codebase, \
         use your tools to search for relevant files and read their contents. \
         Provide a concise, factual answer based on what you find.",
    );

    // 2. Build a "weather_expert" sub-agent that can check weather
    let weather_expert = Agent::with_system_prompt(
        build_provider(),
        vec![Box::new(GetWeatherTool)],
        AgentConfig {
            model: "gpt-4o-mini".to_string(),
            max_iterations: 5,
            extra_params: default_extra_params(),
        },
        "You are a weather expert. When asked about weather, use the get_weather tool \
         to look up current conditions. Provide a helpful summary with recommendations.",
    );

    // 3. Build the parent "coordinator" agent with sub-agents
    let coordinator = Agent::with_system_prompt(
        build_provider(),
        vec![], // No direct tools — delegates everything to sub-agents
        AgentConfig {
            model: "gpt-4o-mini".to_string(),
            max_iterations: 10,
            extra_params: default_extra_params(),
        },
        "You are a coordinator agent. You have access to specialist sub-agents:\n\
         - 'researcher': Can search and read files in a Rust codebase\n\
         - 'weather_expert': Can look up current weather for cities\n\n\
         Delegate tasks to the appropriate sub-agent and synthesize their responses \
         into a coherent answer for the user.",
    )
    .with_sub_agent(
        "researcher",
        "Research a Rust codebase by searching and reading source files",
        researcher,
    )
    .with_sub_agent(
        "weather_expert",
        "Look up current weather conditions for cities",
        weather_expert,
    );

    // 4. Run the coordinator
    let output = coordinator
        .run(vec![Message::user(
            "I need two things:\n\
             1. What public modules does membrane-core export? (check its lib.rs)\n\
             2. What's the weather like in Tokyo and London?",
        )])
        .await?;

    // 5. Print results
    println!("=== Agent Response ===");
    println!("{}", output.response);
    println!();
    print_steps(&output.steps, 0);
    println!();
    println!("Stop reason: {:?}", output.stop_reason);
    println!(
        "Total usage: {} input tokens, {} output tokens",
        output.total_usage.input_tokens, output.total_usage.output_tokens,
    );

    Ok(())
}

/// Recursively print agent steps with indentation for sub-agent nesting.
fn print_steps(steps: &[AgentStep], depth: usize) {
    let indent = "  ".repeat(depth);
    for (i, step) in steps.iter().enumerate() {
        match step {
            AgentStep::LlmCall {
                request_messages,
                response,
            } => {
                println!(
                    "{indent}Step {}: LLM call ({} messages, stop_reason={:?})",
                    i + 1,
                    request_messages,
                    response.stop_reason,
                );
            }
            AgentStep::ToolExecution {
                name,
                output,
                is_error,
                ..
            } => {
                let truncated = if output.len() > 100 {
                    format!("{}...", &output[..100])
                } else {
                    output.clone()
                };
                println!(
                    "{indent}Step {}: Tool '{name}' -> {truncated}{}",
                    i + 1,
                    if *is_error { " [ERROR]" } else { "" },
                );
            }
            AgentStep::SubAgentExecution { name, output, .. } => {
                println!(
                    "{indent}Step {}: Sub-agent '{name}' ({} steps)",
                    i + 1,
                    output.steps.len(),
                );
                print_steps(&output.steps, depth + 1);
                let response_preview = if output.response.len() > 200 {
                    format!("{}...", &output.response[..200])
                } else {
                    output.response.clone()
                };
                println!("{indent}  => {response_preview}");
            }
        }
    }
}
