# Agent Benchmark Research for membrane

Research on benchmarks suitable for evaluating LLM agent ability, with practical integration guidance for the membrane framework.

## Executive Summary

This document surveys 25+ benchmarks for evaluating LLM agent capabilities. The key trend in the field is a shift from "can the model generate correct output?" to "can the agent accomplish a real task end-to-end using tools, reasoning, and interaction?"

For membrane — a Rust agent framework communicating via OpenAI-compatible chat completion APIs — the most actionable benchmarks are:

1. **BFCL** — function-calling accuracy (direct tool interface match)
2. **tau-bench** — multi-turn tool-use with policy compliance
3. **GAIA** — general assistant tasks with tool use
4. **SWE-bench** — code agent evaluation (file I/O + exec tools)
5. **AgentHarm / Agent-SafetyBench** — safety evaluation

---

## 1. Tool-Use / Function-Calling Benchmarks

### 1.1 Berkeley Function Calling Leaderboard (BFCL)

- **References**: [Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) | [GitHub](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | [ICML 2025](https://proceedings.mlr.press/v267/patil25a.html)
- **What it measures**: LLM accuracy in generating function/tool calls across single-call, parallel-call, multi-turn, and agentic scenarios. Also tests relevance detection (knowing when NOT to call a function).
- **Task format**: 2,000 question-function-answer pairs across Python, Java, JavaScript, and REST APIs. Models receive function schemas and natural language queries and must produce correct invocations. Evaluation uses AST matching.
- **How it runs**: Python evaluation harness (`pip install bfcl-eval`). Models are called via their standard chat completion APIs with tool definitions.
- **Open-source**: Yes (Apache 2.0).
- **Versions**: V1 (AST eval), V2 (enterprise functions), V3 (multi-turn), V4 (holistic agentic).
- **Typical scores**: Top models (Claude Opus 4.1, Claude Sonnet 4) ~70%. Single-turn calls are mostly solved; multi-turn and stateful reasoning remain challenging.
- **Membrane relevance**: **HIGH**. Tests exactly the function-calling interface that membrane's `Tool` trait exposes. The evaluation harness sends chat completions with tool definitions and checks generated tool calls — perfectly aligned with OpenAI-compatible APIs.

### 1.2 Gorilla / APIBench

- **References**: [GitHub](https://github.com/ShishirPatil/gorilla) | [arXiv:2305.15334](https://arxiv.org/abs/2305.15334)
- **What it measures**: LLM ability to invoke real ML APIs (HuggingFace, TorchHub, TensorHub) with correct parameters and minimal hallucination.
- **Task format**: ~1,600 API entries. Evaluation uses AST sub-tree matching. BFCL supersedes this for ongoing evaluation.
- **Open-source**: Yes (Apache 2.0).
- **Membrane relevance**: **MEDIUM**. AST matching methodology applicable, but ML-focused API corpus limits generality.

### 1.3 Nexus Function Calling Benchmark (NFCL)

- **References**: [GitHub](https://github.com/nexusflowai/NexusRaven-V2) | [OpenReview](https://openreview.net/pdf?id=5lcPe6DqfI)
- **What it measures**: Zero-shot function calling across single calls, parallel calls, and nested calls (one function's output feeds another's input).
- **Task format**: 9 real-world API domains. Exact-match accuracy evaluation.
- **Open-source**: Yes.
- **Typical scores**: Nested calling is extremely hard — no model above 10%. Single calls: 50-80%.
- **Membrane relevance**: **HIGH**. Nested calling evaluation tests the multi-step tool chaining that membrane's ReAct loop handles.

### 1.4 ToolBench (OpenBMB / ToolLLM)

- **References**: [GitHub](https://github.com/OpenBMB/ToolBench) | [arXiv:2307.16789](https://arxiv.org/abs/2307.16789) | ICLR 2024 Spotlight
- **What it measures**: Planning and executing sequences of real-world RESTful API calls across 16,464 APIs in 49 categories from RapidAPI Hub.
- **Task format**: Multi-tool and single-tool instructions. ToolEval evaluator (LLM-based, 87% human agreement).
- **Open-source**: Yes. StableToolBench variant adds a virtual API server for reproducibility.
- **Typical scores**: Multi-step planning accuracy below 50% for most models.
- **Membrane relevance**: **MEDIUM-HIGH**. Massive API corpus and multi-step chaining stress-test agent loop capabilities. Python-heavy infrastructure requires adaptation.

### 1.5 API-Bank

- **References**: [arXiv:2304.08244](https://arxiv.org/abs/2304.08244) | EMNLP 2023 | [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
- **What it measures**: Three levels of tool-use: (1) Call, (2) Retrieve+Call, (3) Plan+Retrieve+Call.
- **Task format**: 314 dialogues, 753 API calls, 73 runnable API tools.
- **Open-source**: Yes (HuggingFace datasets).
- **Membrane relevance**: **MEDIUM**. Three-level decomposition provides useful diagnostics. Somewhat dated (2023) but task design remains sound.

### 1.6 ToolTalk

- **References**: [GitHub](https://github.com/microsoft/ToolTalk) | [arXiv:2311.10775](https://arxiv.org/abs/2311.10775)
- **What it measures**: Tool use in multi-turn conversational settings with side-effecting tools (sending emails, updating calendars).
- **Task format**: 78 conversations, 178 turns, 28 tools in 7 categories. Includes simulated tool implementations.
- **Open-source**: Yes (Microsoft).
- **Typical scores**: GPT-4: 50% on hard tasks, 92.8% on easy.
- **Membrane relevance**: **HIGH**. Conversational multi-turn format maps directly to membrane's `Vec<Message>` input. Simulated tools can be wrapped as membrane `Tool` impls.

---

## 2. Code Generation Agent Benchmarks

### 2.1 SWE-bench (and Variants)

- **References**: [GitHub](https://github.com/SWE-bench/SWE-bench) | [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) | [SWE-bench Pro](https://scale.com/leaderboard/swe_bench_pro_public)
- **What it measures**: Resolving real GitHub issues by generating code patches. Tests codebase understanding, bug fixing, and feature implementation.
- **Task format**: 2,294 issue-PR pairs from 12 Python repos. Agent receives issue description + codebase, must produce a patch. Success = FAIL_TO_PASS tests pass AND PASS_TO_PASS tests don't break.
- **How it runs**: Docker-based evaluation.
- **Variants**:
  - **Verified** (500 curated): ~75% for top agents (2025)
  - **Pro** (enterprise-level): Claude Sonnet 4.5 at 43.6%, GPT-5 at 41.8%
  - **SWE-bench+**: Enhanced tests reduce resolution rates by ~50%
  - **SWE-bench-C**: C language variant (179 PRs)
- **Known issues**: 60.83% of resolved issues involve solution leakage.
- **Membrane relevance**: **MEDIUM-HIGH**. Gold standard for code agents. A membrane agent with `ReadFileTool`, `WriteFileTool`, `ExecTool`, `SearchFilesTool` can be evaluated. Python harness requires bridging with Rust agent binary.

### 2.2 HumanEval / BigCodeBench / LiveCodeBench

- **HumanEval**: 164 Python challenges, ~91% pass@1 for top models. Tests raw code generation, not agentic behavior.
- **BigCodeBench** ([ICLR 2025](https://openreview.net/forum?id=YrycTjllL0)): 1,140 tasks with complex multi-library function calls. Up to 60% for top models vs 97% human.
- **LiveCodeBench** ([Website](https://livecodebench.github.io/)): Contamination-free competitive programming evaluation. 1,055+ problems. Top: 91.7%.
- **Membrane relevance**: **LOW-MEDIUM**. These test code generation quality rather than agentic tool-use patterns.

### 2.3 MLE-bench

- **References**: [GitHub](https://github.com/openai/mle-bench) | [arXiv:2410.07095](https://arxiv.org/abs/2410.07095)
- **What it measures**: AI agent performance on ML engineering tasks from 75 Kaggle competitions.
- **How it runs**: Docker-based. Extremely resource-intensive (1,800 GPU-hours full run). Lite version: 22 competitions.
- **Typical scores**: Best achieves bronze in 16.9% of competitions.
- **Membrane relevance**: **MEDIUM**. A membrane agent with `ExecTool` and file tools could be evaluated, but resource requirements are extreme.

---

## 3. Multi-Step Reasoning / Planning Benchmarks

### 3.1 tau-bench / tau2-bench

- **References**: [GitHub (tau-bench)](https://github.com/sierra-research/tau-bench) | [GitHub (tau2-bench)](https://github.com/sierra-research/tau2-bench) | [arXiv:2406.12045](https://arxiv.org/abs/2406.12045)
- **What it measures**: Agent performance in realistic customer service scenarios requiring tool use, policy compliance, and dynamic conversation with simulated users.
- **Task format**: Domains: retail, airline (tau-bench), + telecom (tau2). Agent receives domain-specific API tools and policy guidelines, converses with an LLM-simulated user, and must resolve requests. Evaluation compares final database state with annotated goal state.
- **How it runs**: Python evaluation framework. Introduces pass^k metric measuring reliability over k independent trials.
- **Open-source**: Yes. Leaderboard at taubench.com.
- **Typical scores**: GPT-4o < 50% of tasks. pass^8 < 25% in retail — even high single-trial pass rates mask inconsistency.
- **Membrane relevance**: **VERY HIGH**. Tests exactly the pattern membrane supports: agent with domain-specific tools, multi-step API calls, policy compliance. Tool definitions map directly to membrane `Tool` impls. Anthropic uses tau-bench as a key benchmark for Claude model announcements.

### 3.2 MINT

- **References**: [GitHub](https://github.com/xingyaoww/mint-bench) | [arXiv:2309.10691](https://arxiv.org/abs/2309.10691) | ICLR 2024
- **What it measures**: Multi-turn interaction with tools (Python execution) and natural language feedback.
- **Task format**: Reasoning (HotpotQA, MMLU, GSM8K, MATH), code gen (HumanEval, MBPP), decision-making (ALFWorld).
- **Key finding**: Better single-turn performance does not guarantee better multi-turn performance. RLHF can hurt multi-turn capabilities.
- **Membrane relevance**: **HIGH**. Tests iterative refinement via membrane's ReAct loop. Code execution via `ExecTool`. Multi-turn feedback aligns with `Vec<Message>` model.

### 3.3 AppWorld

- **What it measures**: Interactive coding across 9 simulated apps with 457 APIs.
- **Typical scores**: GPT-4o ReAct: 48.8% normal, 30.2% challenge.
- **Membrane relevance**: **MEDIUM-HIGH**. Multi-API composition and state-based evaluation map well to membrane's tool architecture.

---

## 4. General Agent Benchmarks

### 4.1 GAIA

- **References**: [arXiv:2311.12983](https://arxiv.org/abs/2311.12983) | ICLR 2024 | [HuggingFace](https://huggingface.co/papers/2311.12983)
- **What it measures**: General AI assistant capabilities — reasoning, multi-modality, web browsing, and tool-use on questions simple for humans but hard for AI.
- **Task format**: 466 questions across 3 difficulty levels. Unambiguous answers enable rule-based evaluation (no LLM judge needed).
- **How it runs**: Zero-shot inference. Inspect AI implementation available.
- **Typical scores**: Humans: 92%. Original GPT-4 with plugins: 15%.
- **Membrane relevance**: **HIGH**. Tests the general-purpose assistant membrane is designed to build. Rule-based evaluation makes automated scoring straightforward.

### 4.2 AgentBench

- **References**: [GitHub](https://github.com/THUDM/AgentBench) | [arXiv:2308.03688](https://arxiv.org/abs/2308.03688) | ICLR 2024
- **What it measures**: Agent capabilities across 8 environments: OS, Database, KG, Card Game, Puzzles, House-Holding, Web Shopping, Web Browsing.
- **Task format**: Multi-turn interactive challenges (5-50 turns per problem). Dev: 269, Test: 1,014 problems.
- **Membrane relevance**: **MEDIUM-HIGH**. Multi-environment diversity is valuable. Function-calling version is available.

### 4.3 WebArena

- **References**: [Website](https://webarena.dev/) | [arXiv:2307.13854](https://arxiv.org/abs/2307.13854) | ICLR 2024
- **What it measures**: Autonomous web task completion across e-commerce, forums, GitLab, CMS.
- **Task format**: 812 long-horizon tasks. Self-hosted Docker environment. Functional correctness evaluation.
- **Typical scores**: Humans: 78%. Current top (Gemini 2.5 Pro): >54.8%.
- **Membrane relevance**: **MEDIUM**. Browser-based interaction modality differs from membrane's tool-calling approach, but adaptable via web browsing tools or MCP.

### 4.4 TheAgentCompany

- **References**: [Website](https://the-agent-company.com/) | [GitHub](https://github.com/TheAgentCompany/TheAgentCompany) | [arXiv:2412.14161](https://arxiv.org/abs/2412.14161) | NeurIPS 2025
- **What it measures**: Workplace tasks in a simulated software company (SWE, PM, DS, Admin, HR, Finance).
- **Task format**: 175 diverse tasks. Self-contained Docker environment with simulated colleagues via RocketChat.
- **Typical scores**: Best agent: ~30% task completion. DS/Admin/Finance tasks often 0%.
- **Membrane relevance**: **MEDIUM-HIGH**. Realistic deployment scenario. Multi-tool interaction suits membrane's plugin architecture.

---

## 5. Safety / Alignment Benchmarks

### 5.1 AgentHarm

- **References**: [Emergent Mind](https://www.emergentmind.com/topics/agentharm)
- **What it measures**: Whether LLM agents can be manipulated into harmful actions through tool use.
- **Task format**: Harmful prompts in agentic settings. Measures harm score and refusal rate.
- **Membrane relevance**: **HIGH**. Testing harmful tool-use refusal maps directly to membrane's tool execution pipeline.

### 5.2 Agent-SafetyBench

- **References**: [ResearchGate](https://www.researchgate.net/publication/387263688)
- **What it measures**: Safety across 8 risk categories and 10 failure modes.
- **Task format**: 349 environments, 2,000 test cases.
- **Key finding**: No tested agent achieves safety score above 60%.
- **Membrane relevance**: **HIGH**. Comprehensive risk categorization for pre-deployment safety evaluation.

### 5.3 ST-WebAgentBench

- **References**: [GitHub](https://github.com/segev-shlomov/ST-WebAgentBench) | ICML 2025
- **What it measures**: Safety/trustworthiness across 6 dimensions (consent, boundary, strict execution, hierarchy, robustness, error handling).
- **Task format**: 222 tasks with YAML-based safety policy templates.
- **Membrane relevance**: **MEDIUM-HIGH**. Policy compliance framework is relevant to enterprise deployment.

---

## 6. How to Run Benchmarks with membrane

### 6.1 Architecture Integration Points

membrane provides the following integration points for benchmark evaluation:

#### AgentOutput — Full Execution Trace

```rust
pub struct AgentOutput {
    pub response: String,           // Final text response
    pub steps: Vec<AgentStep>,      // All execution steps
    pub total_usage: Usage,         // Cumulative token usage
    pub stop_reason: AgentStopReason,
}

pub enum AgentStep {
    LlmCall { request_messages: usize, response: ChatResponse },
    ToolExecution { name: String, input: Value, output: String, is_error: bool },
    SubAgentExecution { name: String, input: Value, output: AgentOutput },
    ParallelSubAgentExecution { name: String, tasks: Vec<String>, outputs: Vec<AgentOutput>, errors: Vec<String> },
}
```

Every tool call, LLM response, and sub-agent execution is captured in `AgentStep`, enabling extraction of:
- Tool call sequences and parameters
- Success/failure per tool execution
- Token consumption per step and total
- Iteration count and stop reason

#### Metrics Collection

```rust
// Token usage
output.total_usage.input_tokens + output.total_usage.output_tokens

// Iteration count
output.steps.iter().filter(|s| matches!(s, AgentStep::LlmCall { .. })).count()

// Tool failure count
output.steps.iter().filter(|s| matches!(s, AgentStep::ToolExecution { is_error: true, .. })).count()

// Stop reason
matches!(output.stop_reason, AgentStopReason::NaturalStop)     // success
matches!(output.stop_reason, AgentStopReason::MaxIterations { .. })  // budget exceeded
matches!(output.stop_reason, AgentStopReason::StopCondition { .. }) // custom condition
```

#### Structured Output for Evaluation

```rust
#[derive(Deserialize, JsonSchema)]
struct BenchmarkResult {
    answer: String,
    confidence: f32,
    reasoning: String,
}

let output = agent.run_structured::<BenchmarkResult>(messages).await?;
// output.response is a parsed BenchmarkResult
```

#### Stop Conditions for Resource Budgets

```rust
use membrane_core::stop_condition::{Timeout, TokenBudget, MaxConsecutiveErrors};

let agent = agent
    .with_stop_condition(
        Timeout::new(Duration::from_secs(300))
            .or(TokenBudget::new(100_000))
            .or(MaxConsecutiveErrors::new(5))
    );
```

### 6.2 Benchmark Harness Pattern

The general pattern for running a benchmark with membrane:

```rust
use membrane_core::agent::{Agent, AgentConfig, AgentStep, AgentStopReason};
use membrane_core::message::Message;
use membrane_core::stop_condition::{Timeout, TokenBudget};
use membrane_openai::OpenAiProvider;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Serialize)]
struct BenchmarkMetrics {
    task_id: String,
    success: bool,
    response: String,
    total_tokens: u32,
    input_tokens: u32,
    output_tokens: u32,
    num_llm_calls: usize,
    num_tool_calls: usize,
    num_tool_errors: usize,
    stop_reason: String,
    wall_clock_ms: u128,
    tool_call_trace: Vec<ToolCallRecord>,
}

#[derive(Serialize)]
struct ToolCallRecord {
    name: String,
    input: serde_json::Value,
    output: String,
    is_error: bool,
}

async fn run_benchmark_task(
    agent: &Agent<impl membrane_core::provider::LlmProvider>,
    task_id: &str,
    task_prompt: &str,
    expected_answer: Option<&str>,
) -> BenchmarkMetrics {
    let start = Instant::now();
    let output = agent.run(vec![Message::user(task_prompt)]).await;
    let elapsed = start.elapsed();

    match output {
        Ok(output) => {
            let tool_calls: Vec<ToolCallRecord> = output.steps.iter()
                .filter_map(|step| match step {
                    AgentStep::ToolExecution { name, input, output, is_error } => {
                        Some(ToolCallRecord {
                            name: name.clone(),
                            input: input.clone(),
                            output: output.clone(),
                            is_error: *is_error,
                        })
                    }
                    _ => None,
                })
                .collect();

            let success = expected_answer
                .map(|expected| evaluate_answer(&output.response, expected))
                .unwrap_or(false);

            BenchmarkMetrics {
                task_id: task_id.to_string(),
                success,
                response: output.response,
                total_tokens: output.total_usage.input_tokens + output.total_usage.output_tokens,
                input_tokens: output.total_usage.input_tokens,
                output_tokens: output.total_usage.output_tokens,
                num_llm_calls: output.steps.iter()
                    .filter(|s| matches!(s, AgentStep::LlmCall { .. }))
                    .count(),
                num_tool_calls: tool_calls.len(),
                num_tool_errors: tool_calls.iter().filter(|t| t.is_error).count(),
                stop_reason: format!("{:?}", output.stop_reason),
                wall_clock_ms: elapsed.as_millis(),
                tool_call_trace: tool_calls,
            }
        }
        Err(e) => BenchmarkMetrics {
            task_id: task_id.to_string(),
            success: false,
            response: format!("Error: {}", e),
            total_tokens: 0,
            input_tokens: 0,
            output_tokens: 0,
            num_llm_calls: 0,
            num_tool_calls: 0,
            num_tool_errors: 0,
            stop_reason: "Error".to_string(),
            wall_clock_ms: elapsed.as_millis(),
            tool_call_trace: vec![],
        },
    }
}

fn evaluate_answer(actual: &str, expected: &str) -> bool {
    // Implement benchmark-specific evaluation logic
    // e.g., exact match, fuzzy match, AST comparison
    actual.trim().to_lowercase() == expected.trim().to_lowercase()
}
```

### 6.3 Per-Benchmark Integration Guide

#### BFCL Integration

BFCL is the most straightforward benchmark to integrate:

1. **Load test cases**: Parse BFCL's JSON test files containing function schemas and queries
2. **Map schemas to membrane tools**: Convert BFCL function definitions to `ToolDefinition` structs with mock implementations
3. **Run agent**: Feed each query as `Message::user(query)` with mapped tools
4. **Extract tool calls**: Iterate `AgentStep::ToolExecution` entries from `output.steps`
5. **Evaluate**: Compare extracted tool calls against BFCL ground truth using AST matching

```rust
// Mock tool that records calls but doesn't execute
struct MockBenchmarkTool {
    definition: ToolDefinition,
}

impl Tool for MockBenchmarkTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn execute(&self, input: serde_json::Value)
        -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>>
    {
        Box::pin(async move {
            // Return a synthetic success response
            Ok(serde_json::to_string(&input)?)
        })
    }
}
```

#### tau-bench Integration

tau-bench requires implementing domain-specific tools as membrane `Tool` impls:

1. **Implement domain tools**: Create retail/airline tools (get_order, update_order, search_flights, etc.) backed by the tau-bench database
2. **Implement simulated user**: Use an LLM-powered user simulator that generates conversation turns
3. **Run conversation loop**: Alternate between user messages and agent responses via `agent.run()`
4. **Evaluate**: Compare final database state with tau-bench's expected goal state
5. **Measure pass^k**: Run each task k times and compute reliability metric

#### SWE-bench Integration

1. **Set up environment**: Clone target repo, apply task setup
2. **Create agent**: Equip with `ReadFileTool`, `WriteFileTool`, `SearchFilesTool`, `ExecTool`, `GrepTool`, `EditFileTool`, `ListDirTool`
3. **Run agent**: Feed issue description as user message
4. **Extract patch**: Diff the repo after agent execution
5. **Evaluate**: Run the SWE-bench test suite against the patched repo

#### GAIA Integration

1. **Load questions**: Parse GAIA dataset from HuggingFace
2. **Create agent**: Equip with web browsing tools (via MCP or custom `Tool` impls), file tools, and code execution
3. **Run agent**: Feed each question as user message
4. **Evaluate**: Rule-based exact match of agent's final answer against GAIA ground truth

### 6.4 Observability During Benchmarks

membrane's `tracing` integration provides span-level observability:

```
agent.run              [model, task_id]
├── iteration.0
│   ├── llm.chat       [tokens, stop_reason]
│   ├── tool.exec      [tool_name, success/failure]
│   └── tool.exec      [tool_name, success/failure]
├── iteration.1
│   ├── llm.chat
│   └── sub_agent.run  [name]
│       ├── iteration.0
│       │   ├── llm.chat
│       │   └── tool.exec
│       └── iteration.1
│           └── llm.chat
└── event: stop_reason
```

Use `tracing-subscriber` with JSON formatting for structured benchmark logs. All types derive `Serialize`/`Deserialize`, so `AgentOutput` can be serialized to JSON for post-hoc analysis.

---

## 7. Recommendations

### Tier 1 — Start Here (Direct interface match, highest value)

| Benchmark | Why | Integration Effort |
|-----------|-----|-------------------|
| **BFCL** | Tests exactly the function-calling interface membrane exposes. AST-based evaluation, no LLM judge needed. | **Low** — mock tools + call extraction |
| **tau-bench** | Multi-turn tool-use with policies; the exact pattern membrane supports. pass^k measures reliability. Used by Anthropic for Claude evals. | **Medium** — implement domain tools + user simulator |

### Tier 2 — High Value (Agent loop + tools)

| Benchmark | Why | Integration Effort |
|-----------|-----|-------------------|
| **GAIA** | General assistant tasks with unambiguous answers. Rule-based evaluation. | **Medium** — needs web browsing tools |
| **ToolTalk** | Conversational multi-turn with simulated tool implementations. | **Medium** — wrap simulated tools |
| **SWE-bench** | Gold standard for code agents. Uses membrane's built-in file/exec tools. | **High** — Docker setup + harness bridge |

### Tier 3 — For Specific Needs

| Benchmark | Use Case |
|-----------|----------|
| **NFCL** | Testing nested/chained tool calling specifically |
| **MINT** | Multi-turn interaction with code execution feedback |
| **AgentHarm / Agent-SafetyBench** | Pre-deployment safety evaluation |
| **TheAgentCompany** | Realistic workplace task simulation |
| **AppWorld** | Multi-API composition with state-based evaluation |

### Suggested Implementation Order

1. **BFCL** — lowest effort, highest signal for core tool-calling quality
2. **tau-bench** — tests the full agent loop with policy compliance
3. **GAIA** — tests general reasoning + tool use
4. **Safety benchmarks** (AgentHarm or Agent-SafetyBench) — for production readiness
5. **SWE-bench** — if targeting code agent use cases
