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

Task runner is `just` (installed via mise). Prefer `just` over raw cargo commands.

```sh
just fmt             # Format (cargo fmt + dprint)
just fmt-check       # Check formatting without modifying
just lint            # Run clippy (--all-targets -D warnings)
just check           # fmt-check + lint
just test            # Run all tests
just build           # Build all crates
```

For crate-scoped operations, use cargo directly:

```sh
cargo test -p membrane-core
```

## Key Patterns

- All core types derive `Serialize`/`Deserialize` for user-side persistence
- `LlmProvider` and `Tool` traits use RPITIT / `Pin<Box<dyn Future>>` (no async-trait crate)
- `Tool` trait returns `Pin<Box<dyn Future + Send + '_>>` for dyn-compatibility (`Box<dyn Tool>`)
- Agent takes `Vec<Message>` as input — no built-in memory management
- Observability via `tracing` crate spans (`agent.run` → `iteration` → `llm.chat` / `tool.exec`)
- Structured output uses `schemars::JsonSchema` for automatic JSON Schema generation
