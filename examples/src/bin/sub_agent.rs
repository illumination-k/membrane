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

    // 1. Build a "researcher" as a parallel sub-agent (fan-out).
    //    The coordinator can send {"tasks": ["...", "..."]} and each task
    //    runs as an independent agent concurrently.
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

    // 2. Build a "weather_expert" as a single sub-agent (one query at a time).
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

    // 3. Build the parent "coordinator" agent.
    //    - researcher: parallel sub-agent (tasks: [string])
    //    - weather_expert: single sub-agent (query: string)
    let coordinator = Agent::with_system_prompt(
        build_provider(),
        vec![],
        AgentConfig {
            model: "gpt-4o-mini".to_string(),
            max_iterations: 10,
            extra_params: default_extra_params(),
        },
        "You are a coordinator agent. You have access to specialist sub-agents:\n\
         - 'researcher': Accepts {\"tasks\": [\"...\", \"...\"]} to research multiple \
           topics in the Rust codebase concurrently.\n\
         - 'weather_expert': Accepts {\"query\": \"...\"} to check weather for a city.\n\n\
         When you need to research multiple things, send them all as tasks to the \
         researcher in a single call — they will be processed in parallel.\n\
         Synthesize the sub-agent responses into a coherent answer.",
    )
    .with_parallel_sub_agent(
        "researcher",
        "Research multiple topics in the codebase concurrently. \
         Input: {\"tasks\": [\"topic1\", \"topic2\", ...]}",
        researcher,
    )
    .with_sub_agent(
        "weather_expert",
        "Look up current weather conditions for a city",
        weather_expert,
    );

    // 4. Run — the coordinator should fan-out research tasks in parallel
    let output = coordinator
        .run(vec![Message::user(
            "I need to understand this project. Please research all of these concurrently:\n\
             1. What public modules does membrane-core export? (check its lib.rs)\n\
             2. What built-in tools are available in membrane-tools?\n\
             3. What fields does AgentConfig have?\n\
             Also, what's the weather like in Tokyo?",
        )])
        .await?;

    // 5. Print results
    println!("=== Agent Response ===");
    println!("{}", output.response);
    println!();
    println!("=== Execution Steps ===");
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
                let truncated = truncate_str(output, 100);
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
                let preview = truncate_str(&output.response, 200);
                println!("{indent}  => {preview}");
            }
            AgentStep::ParallelSubAgentExecution {
                name,
                tasks,
                outputs,
                errors,
            } => {
                println!(
                    "{indent}Step {}: Parallel sub-agent '{name}' \
                     ({} tasks, {} ok, {} err)",
                    i + 1,
                    tasks.len(),
                    outputs.len(),
                    errors.len(),
                );
                for (j, output) in outputs.iter().enumerate() {
                    let task = tasks.get(j).map(|s| s.as_str()).unwrap_or("?");
                    let preview = truncate_str(&output.response, 150);
                    println!("{indent}  [Task {}] {task}: {preview}", j + 1);
                }
                for error in errors {
                    println!("{indent}  [ERROR] {error}");
                }
            }
        }
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s.to_string()
    }
}
