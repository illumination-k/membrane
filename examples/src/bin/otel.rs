//! OpenTelemetry tracing example.
//!
//! Demonstrates how to export membrane agent traces to an OTLP-compatible
//! backend (Grafana Tempo, Jaeger, etc.) using the `tracing` → `tracing-opentelemetry`
//! → `opentelemetry-otlp` pipeline.
//!
//! # Prerequisites
//!
//! Start the local observability stack:
//!
//! ```sh
//! docker compose -f examples/docker-compose.otel.yml up -d
//! ```
//!
//! Then run this example:
//!
//! ```sh
//! OPENAI_API_KEY=... cargo run -p membrane-examples --features otel --bin otel
//! ```
//!
//! Open Grafana at <http://localhost:3000> and explore traces in the "Tempo" datasource.

#[cfg(not(feature = "otel"))]
fn main() {
    eprintln!("This example requires the `otel` feature:");
    eprintln!("  cargo run -p membrane-examples --features otel --bin otel");
    std::process::exit(1);
}

#[cfg(feature = "otel")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use membrane_core::agent::{Agent, AgentConfig};
    use membrane_core::membrane_tool;
    use membrane_core::message::Message;
    use membrane_openai::OpenAiProvider;
    use membrane_tools::ReadFileTool;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::{SpanExporter, WithExportConfig};
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use tracing_opentelemetry::OpenTelemetryLayer;
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    // --- OpenTelemetry setup ---

    // OTLP exporter sends traces to the collector (default: http://localhost:4318)
    let otlp_endpoint =
        std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").unwrap_or("http://localhost:4318".into());

    let exporter = SpanExporter::builder()
        .with_http()
        .with_endpoint(&otlp_endpoint)
        .build()?;

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .build();

    let tracer = provider.tracer("membrane-agent");

    // Compose: fmt layer (console) + OpenTelemetry layer (OTLP export)
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,membrane_core=debug")),
        )
        .with(tracing_subscriber::fmt::layer().compact())
        .with(OpenTelemetryLayer::new(tracer))
        .init();

    tracing::info!(endpoint = %otlp_endpoint, "OpenTelemetry exporter configured");

    // --- Tool definition ---

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
            "tokyo" => "Sunny, 22°C",
            "london" => "Rainy, 14°C",
            "new york" => "Cloudy, 18°C",
            "paris" => "Partly cloudy, 16°C",
            _ => "No data available",
        };
        Ok(format!("{}: {}", input.city, weather))
    }

    // --- Agent execution ---

    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable must be set");

    let provider_llm = OpenAiProvider::builder().api_key(api_key).build();

    let agent = Agent::with_system_prompt(
        provider_llm,
        vec![Box::new(GetWeatherTool), Box::new(ReadFileTool)],
        AgentConfig {
            model: "gpt-4o-mini".to_string(),
            max_iterations: 10,
            extra_params: serde_json::Map::new(),
        },
        "You are a helpful assistant. Use the available tools to answer questions.",
    );

    let output = agent
        .run(vec![Message::user(
            "What's the weather in Tokyo and London? Compare them.",
        )])
        .await?;

    println!("\n=== Agent Response ===\n{}", output.response);
    println!(
        "\nTokens: {} in / {} out",
        output.total_usage.input_tokens, output.total_usage.output_tokens
    );
    println!("Stop reason: {:?}", output.stop_reason);

    // Flush traces before exit
    provider.shutdown()?;
    tracing::info!("Traces flushed to OTLP endpoint");

    Ok(())
}
