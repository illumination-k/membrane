mod convert;
mod eval;
mod runner;
mod types;

use clap::Parser;
use membrane_openai::OpenAiProvider;

/// BFCL (Berkeley Function Calling Leaderboard) evaluation harness for membrane.
///
/// Downloads test data from the BFCL repository, sends function-calling queries
/// through an OpenAI-compatible LLM provider, and evaluates responses using
/// AST-based matching against ground truth.
///
/// # Data Setup
///
/// Clone the BFCL data into a local directory:
///
/// ```sh
/// git clone --depth 1 --filter=blob:none --sparse \
///   https://github.com/ShishirPatil/gorilla.git /tmp/bfcl
/// cd /tmp/bfcl && git sparse-checkout set berkeley-function-call-leaderboard/bfcl_eval/data
/// ```
///
/// Then point `--data-dir` at the data directory:
///
/// ```sh
/// membrane-eval-bfcl --data-dir /tmp/bfcl/berkeley-function-call-leaderboard/bfcl_eval/data \
///   --category simple_python --model gpt-4o-mini
/// ```
#[derive(Parser)]
#[command(
    name = "membrane-eval-bfcl",
    about = "BFCL evaluation harness for membrane"
)]
struct Cli {
    /// Path to BFCL data directory (containing BFCL_v4_*.json files).
    #[arg(long, env = "BFCL_DATA_DIR")]
    data_dir: String,

    /// Test category to evaluate.
    #[arg(long, default_value = "simple_python")]
    category: String,

    /// Model name for the LLM provider.
    #[arg(long, default_value = "gpt-4o-mini")]
    model: String,

    /// OpenAI-compatible API base URL.
    #[arg(long, env = "OPENAI_BASE_URL")]
    base_url: Option<String>,

    /// Maximum number of test cases to run (for quick testing).
    #[arg(long)]
    limit: Option<usize>,

    /// Temperature for LLM generation.
    #[arg(long, default_value = "0.0")]
    temperature: f64,
}

/// Available single-turn test categories.
const SINGLE_TURN_CATEGORIES: &[&str] = &[
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel",
    "multiple",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    // Build LLM provider
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable required");

    let mut builder = OpenAiProvider::builder().api_key(api_key);
    if let Some(base_url) = &cli.base_url {
        builder = builder.base_url(base_url);
    }
    let provider = builder.build();

    let extra_params = serde_json::json!({
        "temperature": cli.temperature,
    })
    .as_object()
    .expect("extra_params must be an object")
    .clone();

    // Run evaluation
    if cli.category == "all" {
        run_all_categories(&provider, &cli, &extra_params).await?;
    } else {
        let config = runner::RunConfig {
            model: cli.model.clone(),
            data_dir: cli.data_dir.clone(),
            category: cli.category.clone(),
            limit: cli.limit,
            extra_params: extra_params.clone(),
        };

        let score = runner::run_category(&provider, &config).await?;
        print_score(&score);
    }

    Ok(())
}

async fn run_all_categories(
    provider: &OpenAiProvider,
    cli: &Cli,
    extra_params: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut scores = Vec::new();

    for &category in SINGLE_TURN_CATEGORIES {
        let test_file =
            std::path::Path::new(&cli.data_dir).join(format!("BFCL_v4_{category}.json"));
        if !test_file.exists() {
            tracing::warn!("Skipping {category}: test file not found");
            continue;
        }

        tracing::info!("=== Evaluating: {category} ===");
        let config = runner::RunConfig {
            model: cli.model.clone(),
            data_dir: cli.data_dir.clone(),
            category: category.to_string(),
            limit: cli.limit,
            extra_params: extra_params.clone(),
        };

        match runner::run_category(provider, &config).await {
            Ok(score) => {
                print_score(&score);
                scores.push(score);
            }
            Err(e) => {
                tracing::error!("Failed to evaluate {category}: {e}");
            }
        }
    }

    // Print summary
    if !scores.is_empty() {
        println!("\n{}", "=".repeat(60));
        println!("SUMMARY — {}", cli.model);
        println!("{}", "=".repeat(60));

        let mut total_correct = 0;
        let mut total_count = 0;

        for score in &scores {
            total_correct += score.correct;
            total_count += score.total;
            println!(
                "  {:<30} {}/{} ({:.1}%)",
                score.category,
                score.correct,
                score.total,
                score.accuracy * 100.0,
            );
        }

        let overall = if total_count > 0 {
            total_correct as f64 / total_count as f64
        } else {
            0.0
        };
        println!(
            "  {:<30} {total_correct}/{total_count} ({:.1}%)",
            "OVERALL",
            overall * 100.0
        );
    }

    Ok(())
}

fn print_score(score: &types::CategoryScore) {
    println!(
        "\n[{}] {}/{} correct ({:.1}% accuracy)",
        score.category,
        score.correct,
        score.total,
        score.accuracy * 100.0,
    );
}
