# membrane

Rust library for building LLM agents. See `specs/DESIGN_DOC.md` for full design.

## Language Rules

- Comment language: English
- Commit messages: English

## Project Structure

```
membrane/
├── membrane-core/       # Core traits, types, agent loop, re-exports macro
├── membrane-macros/     # Proc macro crate (#[membrane_tool])
├── membrane-openai/     # OpenAI Chat Completions API provider
├── membrane-tools/      # Built-in utility tools (file I/O, exec, search)
├── examples/            # Example applications
├── applications/        # Standalone agent applications
│   └── research-agents/ # Research agent CLI (WIP)
└── specs/DESIGN_DOC.md  # Design document
```

## Architecture

- **membrane-core**: Runtime-agnostic core. No HTTP client dependency.
  - `message.rs` — Message, Content, Role
  - `provider.rs` — LlmProvider trait, ChatRequest/ChatResponse
  - `tool.rs` — Tool trait (dyn-compatible via `Pin<Box<dyn Future>>`)
  - `agent.rs` — Agent with ReAct loop (`run`, `run_structured`)
  - `context.rs` — ContextBuilder trait, DefaultContextBuilder
  - `stop_condition.rs` — StopCondition trait, built-in conditions (Timeout, TokenBudget, MaxConsecutiveErrors, CustomStop), Or/And combinators
  - `error.rs` — Error enum, ErrorInfo
- **membrane-macros**: `#[membrane_tool]` attribute macro for Tool generation
- **membrane-openai**: OpenAI provider implementing `LlmProvider`
  - `types.rs` — Internal serde types matching OpenAI wire format
  - `convert.rs` — Bidirectional conversion (membrane-core ↔ OpenAI)
  - `lib.rs` — `OpenAiProvider` (builder pattern), `TokenProvider` trait
- **membrane-tools**: Built-in tools using `#[membrane_tool]` macro
  - `read_file.rs` — ReadFileTool (offset/limit support)
  - `write_file.rs` — WriteFileTool (auto directory creation)
  - `search_files.rs` — SearchFilesTool (glob pattern matching)
  - `exec.rs` — ExecTool (shell command execution)
- `extern crate self as membrane_core;` in lib.rs enables macro-generated `membrane_core::` paths to resolve inside the crate itself

## Commands

Task runner is `just` (installed via mise). Prefer `mise exec -- just` over raw cargo commands.

```sh
mise exec -- just fmt             # Format (cargo fmt + dprint)
mise exec -- just fmt-check       # Check formatting without modifying
mise exec -- just lint            # Run clippy (--all-targets -D warnings)
mise exec -- just check           # fmt-check + lint
mise exec -- just test            # Run all tests
mise exec -- just build           # Build all crates
```

For crate-scoped operations, use cargo directly:

```sh
cargo test -p membrane-core
```

## Key Patterns

- All core types derive `Serialize`/`Deserialize` for user-side persistence
- `LlmProvider` uses RPITIT (`impl Future`), `Tool` uses `Pin<Box<dyn Future + Send + '_>>` for dyn-compatibility (`Box<dyn Tool>`)
- `Message.content` is `Vec<Content>` (multiple content blocks per message)
- `Content` uses internally-tagged serde (`#[serde(tag = "type")]`)
- Agent takes `Vec<Message>` as input — no built-in memory management
- `AgentConfig.extra_params` and `ChatRequest.extra_params` (`serde_json::Map`) are `#[serde(flatten)]`-ed into API requests for provider-specific params (temperature, max_tokens, etc.)
- `AgentStopReason` enum distinguishes NaturalStop / MaxIterations / StopCondition (max_iterations is not an error)
- `StopCondition` trait with `or()`/`and()` combinators for composable early termination
- Observability via `tracing` crate spans (`agent.run` → `iteration` → `llm.chat` / `tool.exec`)
- Structured output uses `schemars::JsonSchema` for automatic JSON Schema generation
- Provider crates use builder pattern for construction (e.g. `OpenAiProvider::builder().api_key("...").build()`)
- `TokenProvider` trait enables dynamic auth (e.g. Azure AD token refresh)
- `#[membrane_tool]` macro generates `{FnNamePascalCase}Tool` struct + `Tool` impl from async functions
