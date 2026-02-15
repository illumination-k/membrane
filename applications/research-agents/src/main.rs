use clap::Parser;

#[derive(Parser)]
#[command(name = "research-agents", about = "Research agent powered by membrane")]
struct Cli {
    /// The query to research
    query: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    println!("Query: {}", cli.query);
}
