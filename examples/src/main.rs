use membrane_core::agent::{Agent, AgentConfig};
use membrane_core::membrane_tool;
use membrane_core::message::Message;
use membrane_openai::OpenAiProvider;
use membrane_tools::{ReadFileTool, SearchFilesTool};
use schemars::JsonSchema;
use serde::Deserialize;

// --- Tool definitions ---

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
    // Simulated weather data for demonstration
    let weather = match input.city.to_lowercase().as_str() {
        "tokyo" => "Sunny, 22°C",
        "london" => "Rainy, 14°C",
        "new york" => "Cloudy, 18°C",
        "paris" => "Partly cloudy, 16°C",
        _ => "No data available",
    };
    Ok(format!("{}: {}", input.city, weather))
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CalculateInput {
    /// First operand
    a: f64,
    /// Second operand
    b: f64,
    /// Operator: add, sub, mul, div
    op: String,
}

#[membrane_tool(
    name = "calculate",
    description = "Perform basic arithmetic (add, sub, mul, div)"
)]
async fn calculate(input: CalculateInput) -> Result<String, membrane_core::error::Error> {
    let result = match input.op.as_str() {
        "add" => input.a + input.b,
        "sub" => input.a - input.b,
        "mul" => input.a * input.b,
        "div" => {
            if input.b == 0.0 {
                return Ok("Error: division by zero".to_string());
            }
            input.a / input.b
        }
        _ => return Ok(format!("Unknown operator: {}", input.op)),
    };
    Ok(format!("{result}"))
}

// --- Main ---

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable must be set");

    let provider = OpenAiProvider::builder().api_key(api_key).build();

    let agent = Agent::new(
        provider,
        vec![
            Box::new(GetWeatherTool),
            Box::new(CalculateTool),
            Box::new(ReadFileTool),
            Box::new(SearchFilesTool),
        ],
        AgentConfig {
            model: "gpt-5-nano".to_string(),
            max_iterations: 10,
            system_prompt: Some(
                "You are a helpful assistant. Use the available tools to answer questions."
                    .to_string(),
            ),
            max_tokens: Some(1024),
            temperature: Some(0.7),
        },
    );

    let output = agent
        .run(vec![Message::user(
            "Search for all .rs files under membrane-tools/src/ and then read the contents of lib.rs from the results.",
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
                    "Step {}: Tool '{}' input={} output={} error={}",
                    i + 1,
                    name,
                    input,
                    output,
                    is_error,
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
