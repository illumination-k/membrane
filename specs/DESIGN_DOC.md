# membrane Design Document

## Overview

**membrane** は Rust で LLM Agent を構築するためのライブラリ。LLM 呼び出しの抽象化、ReAct パターン、Tool-use を統一的なインターフェースで提供し、tracing ベースの Observability を全レイヤーに組み込む。

名前の由来: 細胞膜（membrane）のように、アプリケーションと LLM の間に位置し、情報の出入りを制御する境界層。

## Goals

- LLM プロバイダー間の差異を吸収する統一的な抽象化
- ReAct ループ・Tool 実行の仕組みを透明に提供する
- 各ステップを tracing span で追跡可能にする
- async runtime に依存しない core 設計

## Non-Goals

- Web フレームワークとの統合（自前で繋ぐ）
- GUI / CLI ツールの提供
- RAG パイプライン全体の構築（検索部分はユーザー側）

---

## Architecture

### Crate 構成

```
membrane/
├── membrane-core/       # Core traits, types, agent loop
├── membrane-openai/     # OpenAI provider implementation
├── membrane-anthropic/  # Anthropic provider implementation
└── membrane/            # Re-export + convenience (将来)
```

**membrane-core** は HTTP クライアントや特定の async runtime に依存しない。プロバイダー実装クレートが具体的な HTTP 通信を担う。

### 依存関係の方針

```
membrane-core:  serde, serde_json, schemars, tracing, thiserror
membrane-openai:    membrane-core, reqwest, tokio (HTTP通信用)
membrane-anthropic: membrane-core, reqwest, tokio (HTTP通信用)
```

core は runtime 非依存。プロバイダー実装は reqwest (tokio) を使うが、ユーザーが独自に `LlmProvider` trait を実装すれば任意の HTTP クライアント/runtime を利用可能。

---

## Core Abstractions

### 1. Message Types

LLM とのやりとりの基本型。プロバイダー間で共通化する。全 core 型は `Serialize` / `Deserialize` を derive し、ユーザーが任意の方法で永続化できるようにする。永続化の仕組み自体はライブラリのスコープ外。

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

pub struct Message {
    pub role: Role,
    pub content: Content,
}

pub enum Content {
    Text(String),
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { id: String, output: String, is_error: bool },
}
```

### 2. LLM Provider Trait

```rust
pub struct ResponseFormat {
    pub name: String,
    pub schema: serde_json::Value, // JSON Schema
}

pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub response_format: Option<ResponseFormat>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    // ...
}

pub struct ChatResponse {
    pub content: Vec<Content>,
    pub usage: Usage,
    pub stop_reason: StopReason,
}

pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

pub trait LlmProvider: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, Error>;
    // Streaming は将来対応。まずは non-streaming で完成させる
}
```

**設計意図**: async fn in trait (Rust 2024 edition で安定化) を直接使う。`async-trait` マクロは不要。object safety が必要になった場合は `trait-variant` や手動の Boxing で対応する。

### 3. Tool System

```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value, // JSON Schema
}

pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn execute(&self, input: serde_json::Value) -> Result<String, Error>;
}
```

ToolDefinition の `input_schema` は JSON Schema 形式。derive マクロや schemars 連携は将来の拡張として、まずは手書き or serde_json::json! マクロで定義する。

### 4. Agent (ReAct Loop)

会話履歴（Memory）はライブラリでは管理せず、呼び出し側が `Vec<Message>` として渡す設計とする。必要に応じてユーザーがスライディングウィンドウや要約を自前で実装できる。

```rust
pub struct AgentConfig {
    pub max_iterations: usize,
    pub system_prompt: Option<String>,
}

pub struct Agent<P: LlmProvider> {
    provider: P,
    tools: Vec<Box<dyn Tool>>,
    config: AgentConfig,
}

impl<P: LlmProvider> Agent<P> {
    /// メッセージ列を受け取り、ReAct ループを実行する（テキスト応答）
    pub async fn run(&self, messages: Vec<Message>) -> Result<AgentOutput, Error> {
        // 1. System prompt + messages でリクエスト構築
        // 2. Loop:
        //    a. LLM に chat リクエスト送信
        //    b. StopReason が ToolUse なら Tool を実行し、結果を追加して continue
        //    c. StopReason が EndTurn なら最終応答として返す
        //    d. max_iterations 超えたら打ち切り
    }

    /// Structured Output: 最終応答を型 O にパースして返す
    pub async fn run_structured<O>(&self, messages: Vec<Message>) -> Result<StructuredAgentOutput<O>, Error>
    where
        O: DeserializeOwned + JsonSchema,
    {
        // 1. O::json_schema() から ResponseFormat を生成
        // 2. run と同じ ReAct ループ（最終リクエストに response_format を付与）
        // 3. 最終応答の JSON テキストを serde_json::from_str::<O> でパース
    }
}

pub struct AgentOutput {
    pub response: String,
    pub steps: Vec<AgentStep>,
    pub total_usage: Usage,
}

pub struct StructuredAgentOutput<O> {
    pub response: O,
    pub steps: Vec<AgentStep>,
    pub total_usage: Usage,
}

pub enum AgentStep {
    LlmCall { request_messages: usize, response: ChatResponse },
    ToolExecution { name: String, input: serde_json::Value, output: String },
}
```

**ReAct ループの透明性**: `AgentOutput` に全ステップ (`AgentStep`) を記録し、呼び出し側が各ステップの詳細を参照できるようにする。ブラックボックスにしない。

---

## Observability

全レイヤーで `tracing` を活用する。

```
agent.run          [span]          ← Agent 全体の実行
├── iteration.0    [span]          ← ReAct ループの各イテレーション
│   ├── llm.chat   [span]          ← LLM 呼び出し
│   │   └── event: request/response details, token usage
│   └── tool.exec  [span]          ← Tool 実行
│       └── event: tool name, input, output
├── iteration.1    [span]
│   └── llm.chat   [span]
│       └── event: final response
```

- 各 span にはタイミング・token 使用量・エラー情報を記録
- tracing subscriber の選択はユーザーに任せる（`tracing-subscriber`, OpenTelemetry 等）
- ライブラリ側では `tracing::instrument` と `tracing::event!` のみ使用

---

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("LLM provider error: {0}")]
    Provider(String),

    #[error("Tool execution error: {tool_name}: {message}")]
    ToolExecution { tool_name: String, message: String },

    #[error("Max iterations ({max}) exceeded")]
    MaxIterations { max: usize },


    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
```

---

## Implementation Phases

### Phase 1: Core Types & Traits
- Message, Content, Role 型の定義
- LlmProvider trait
- Tool, ToolDefinition
- Error 型
- 基本的な tracing instrumentation

### Phase 2: Agent Loop
- AgentConfig, Agent 構造体
- ReAct ループの実装
- AgentOutput, AgentStep

### Phase 3: OpenAI Provider 実装
- membrane-openai クレート（Chat Completions API）

### Phase 4: 拡張
- membrane-anthropic クレート（Messages API）
- Streaming 対応
- Tool の derive マクロ / schemars 連携
- Memory 抽象化（必要になった場合）
- Multi-agent 対応

---

## Open Questions

1. **Streaming**: Phase 1 では non-streaming のみ。Streaming は `Stream` trait で返す想定だが、runtime 非依存との兼ね合いをどうするか（`futures::Stream` を使うか）
2. **Tool の型安全性**: 現状は `serde_json::Value` ベース。proc macro で型安全な Tool 定義を自動生成する価値はあるか
3. **Agent の構成パターン**: Builder パターン vs 構造体直接構築 vs config ファイル
4. **Multi-agent**: 複数 Agent の連携（chain, parallel, supervisor パターン）は Phase 4 以降の検討事項
