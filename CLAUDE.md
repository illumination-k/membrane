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
└── specs/DESIGN_DOC.md  # Design document
```

## Architecture

- **membrane-core**: Runtime-agnostic core. No HTTP client dependency.
  - `message.rs` — Message, Content, Role
  - `provider.rs` — LlmProvider trait, ChatRequest/ChatResponse
  - `tool.rs` — Tool trait (dyn-compatible via `Pin<Box<dyn Future>>`)
  - `agent.rs` — Agent with ReAct loop (`run`, `run_structured`)
  - `error.rs` — Error enum
- **membrane-macros**: `#[membrane_tool]` attribute macro for Tool generation
- `extern crate self as membrane_core;` in lib.rs enables macro-generated `membrane_core::` paths to resolve inside the crate itself

## Commands

```sh
cargo build          # Build all crates
cargo test           # Run all tests
cargo test -p membrane-core  # Test core only
```

## Key Patterns

- All core types derive `Serialize`/`Deserialize` for user-side persistence
- `LlmProvider` and `Tool` traits use RPITIT / `Pin<Box<dyn Future>>` (no async-trait crate)
- `Tool` trait returns `Pin<Box<dyn Future + Send + '_>>` for dyn-compatibility (`Box<dyn Tool>`)
- Agent takes `Vec<Message>` as input — no built-in memory management
- Observability via `tracing` crate spans (`agent.run` → `iteration` → `llm.chat` / `tool.exec`)
- Structured output uses `schemars::JsonSchema` for automatic JSON Schema generation
