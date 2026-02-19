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
├── membrane-core/       # Core traits, types, agent loop, re-exports macro
├── membrane-macros/     # Proc macro crate (#[membrane_tool])
├── membrane-openai/     # OpenAI Chat Completions API provider
├── membrane-tools/      # Built-in utility tools (file I/O, exec, search)
├── examples/            # Example applications
├── applications/        # Standalone agent applications
│   └── research-agents/ # Research agent CLI (WIP)
└── specs/DESIGN_DOC.md  # This document
```

**membrane-core** は HTTP クライアントや特定の async runtime に依存しない。プロバイダー実装クレートが具体的な HTTP 通信を担う。

### 依存関係の方針

```
membrane-core:      serde, serde_json, schemars, tracing, thiserror, membrane-macros
membrane-macros:    proc-macro2, quote, syn
membrane-openai:    membrane-core, reqwest, tokio (HTTP通信用)
membrane-tools:     membrane-core, glob, schemars, serde, serde_json
```

core は runtime 非依存。プロバイダー実装は reqwest (tokio) を使うが、ユーザーが独自に `LlmProvider` trait を実装すれば任意の HTTP クライアント/runtime を利用可能。

ワークスペース共通依存は `Cargo.toml` の `[workspace.dependencies]` で一元管理。

---

## Core Abstractions

### 1. Message Types

LLM とのやりとりの基本型。プロバイダー間で共通化する。全 core 型は `Serialize` / `Deserialize` を derive し、ユーザーが任意の方法で永続化できるようにする。永続化の仕組み自体はライブラリのスコープ外。

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,  // 複数の Content ブロックを保持
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { id: String, output: String, is_error: bool },
}
```

`Message` は `content: Vec<Content>` を持つ。LLM のレスポンスがテキストとツール呼び出しを同時に含むケースに対応するため。便利コンストラクタ `Message::system()`, `Message::user()`, `Message::assistant()` を提供。

`Content` は internally-tagged enum (`#[serde(tag = "type")]`) で、シリアライズ時に `{"type": "text", "text": "..."}` のような形式になる。

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
    /// Provider-specific parameters merged into the API request JSON.
    /// Use this for `max_tokens`, `temperature`, `max_completion_tokens`, etc.
    #[serde(flatten)]
    pub extra_params: serde_json::Map<String, serde_json::Value>,
}

pub struct ChatResponse {
    pub content: Vec<Content>,
    pub usage: Usage,
    pub stop_reason: StopReason,
}

#[derive(Default)]
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
    fn chat(&self, request: ChatRequest)
        -> impl Future<Output = Result<ChatResponse, Error>> + Send;
}
```

**設計意図**:
- RPITIT (`impl Future`) を使い、`async-trait` マクロは不要。
- `ChatRequest` にはプロバイダー固有パラメータ用の `extra_params` を持たせ、`#[serde(flatten)]` で API リクエスト JSON にマージする。`max_tokens` や `temperature` 等はプロバイダーによってフィールド名・挙動が異なるため、共通フィールドではなく `extra_params` で対応する。
- `AgentConfig` にも同じ `extra_params` があり、各リクエストに自動で渡される。

### 3. Tool System

```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value, // JSON Schema
}

pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, input: serde_json::Value)
        -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>>;
}
```

**設計意図**: `execute` は `Pin<Box<dyn Future + Send + '_>>` を返す。これにより `Box<dyn Tool>` として dyn-compatible に利用できる。RPITIT ではなく手動の Boxing を選択した理由は、Agent が `Vec<Box<dyn Tool>>` でツールを保持するため。

#### `#[membrane_tool]` Proc Macro

`membrane-macros` クレートが提供する属性マクロ。async 関数からToolの定義と実装を自動生成する。

```rust
#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherInput {
    city: String,
}

#[membrane_tool(name = "get_weather", description = "Get current weather")]
async fn get_weather(input: WeatherInput) -> Result<String, Error> {
    Ok(format!("Weather in {}: sunny", input.city))
}
// → GetWeatherTool 構造体が生成され、Tool trait が実装される
// → input_schema は schemars::schema_for!(WeatherInput) から自動生成
```

マクロは以下を生成する:
- 関数名を PascalCase + `Tool` に変換した構造体（例: `get_weather` → `GetWeatherTool`）
- `Tool` trait の `definition()` 実装（name, description, schemars による JSON Schema）
- `Tool` trait の `execute()` 実装（JSON → 入力型のデシリアライズ → 元の async 関数の呼び出し）

#### ParallelTool (Fan-out Wrapper)

任意の `Box<dyn Tool>` をラップし、並列実行版に変換するラッパー。LLM が `{ "inputs": [<original_input>, ...] }` 形式で複数の入力を送ると、各入力を `futures_util::future::join_all` で並列実行し、結果を `[N]` プレフィックス付きで結合して返す。

```rust
pub struct ParallelTool {
    inner: Box<dyn Tool>,
}

impl ParallelTool {
    pub fn new(tool: Box<dyn Tool>) -> Self;
}
```

`ParallelTool` 自体が `Tool` trait を実装するため、ReAct ループの変更は不要。Agent の builder で登録できる:

```rust
let agent = Agent::with_system_prompt(provider, vec![], config, "...")
    .with_parallel_tool(Box::new(SearchFilesTool));
// LLM は {"inputs": [{"pattern": "*.rs"}, {"pattern": "*.toml"}]} で並列検索できる
```

一部の入力が失敗しても他の結果は返る（`[N] ERROR: ...` として含まれる）。

### 4. Context Builder

LLM に渡すコンテキスト（メッセージ列）の構築を制御する抽象。system prompt の挿入、会話履歴の圧縮、外部メモリの注入などをユーザーがカスタマイズできる。

```rust
pub trait ContextBuilder: Send + Sync {
    /// ReActループ開始前。ユーザーのメッセージ列 → 初期コンテキスト。
    fn build_initial(&self, messages: Vec<Message>) -> Vec<Message>;

    /// 2回目以降のイテレーション前。会話履歴 → LLMに送るメッセージ列。
    /// デフォルト: そのまま返す。
    fn build_iteration(&self, conversation: &[Message], _iteration: usize) -> Vec<Message> {
        conversation.to_vec()
    }
}

pub struct DefaultContextBuilder { system_prompt: Option<String> }
```

**設計意図**: 同期 trait にしている。非同期検索（ベクトル DB 等）は `agent.run()` 前に行い、ContextBuilder の state に入れる設計。必要になったら `AsyncContextBuilder` を別 trait として追加可能。

### 5. Stop Conditions

Agent の ReAct ループを早期終了させるためのプラグイン可能な条件システム。

```rust
/// Agent ループの状態スナップショット。各 StopCondition に渡される。
pub struct StopContext<'a> {
    pub iteration: usize,
    pub total_usage: &'a Usage,
    pub steps: &'a [AgentStep],
    pub elapsed: Duration,
    pub messages: &'a [Message],
}

/// Agent ループが終了した理由。
pub enum AgentStopReason {
    NaturalStop,                          // LLM がツール呼び出しなしで応答
    MaxIterations { max: usize },         // max_iterations に到達
    StopCondition { description: String }, // ユーザー定義の条件が発火
}

pub trait StopCondition: Send + Sync {
    fn should_stop(&self, ctx: &StopContext<'_>) -> Option<String>;
    fn or<B: StopCondition>(self, other: B) -> Or<Self, B>;
    fn and<B: StopCondition>(self, other: B) -> And<Self, B>;
}
```

**ビルトイン条件**:
- `Timeout` — 経過時間がタイムアウトを超えたら停止
- `TokenBudget` — 累計トークン使用量が予算を超えたら停止
- `MaxConsecutiveErrors` — 連続するツール実行エラーが閾値に達したら停止
- `CustomStop<F>` — クロージャベースのカスタム条件

**コンビネータ**: `or()` / `and()` メソッドで条件を合成できる。Agent に追加された複数の条件は OR で評価される（いずれかが発火したら停止）。

```rust
let condition = Timeout::new(Duration::from_secs(300))
    .or(TokenBudget::new(100_000));
let agent = agent.with_stop_condition(condition);
```

### 6. Sub-Agent System

Tool と Sub-Agent は明確に異なる概念として分離する。Tool は決定的な関数（ファイル読み書き、コマンド実行など）、Sub-Agent は自律的に推論・行動するエンティティ。

#### AgentExecutor Trait

`Agent<P>` の型消去を行い、異なるプロバイダーの Agent を統一的に扱えるようにする。

```rust
pub trait AgentExecutor: Send + Sync {
    fn run(&self, messages: Vec<Message>)
        -> Pin<Box<dyn Future<Output = Result<AgentOutput, Error>> + Send + '_>>;

    /// Run multiple queries in parallel (fan-out). Default implementation
    /// uses futures_util::future::join_all over self.run().
    fn run_parallel(&self, message_sets: Vec<Vec<Message>>)
        -> Pin<Box<dyn Future<Output = Vec<Result<AgentOutput, Error>>> + Send + '_>> {
        Box::pin(async move {
            let futs: Vec<_> = message_sets.into_iter().map(|msgs| self.run(msgs)).collect();
            futures_util::future::join_all(futs).await
        })
    }
}

impl<P: LlmProvider> AgentExecutor for Agent<P> {
    fn run(&self, messages: Vec<Message>)
        -> Pin<Box<dyn Future<Output = Result<AgentOutput, Error>> + Send + '_>> {
        Box::pin(self.run(messages))
    }
}
```

**設計意図**: `Agent<P>` はジェネリクスのため `Box<dyn Agent>` にできない。`AgentExecutor` で実行能力だけを trait object 化することで、異なるプロバイダーの Agent（OpenAI, Anthropic 等）を `Arc<dyn AgentExecutor>` として混在させられる。テスト用モックや実行ログのラッパーにも使い回せる。

`run_parallel` はデフォルト実装付きで、同一 Agent に複数クエリを並列に投げる fan-out パターンをサポートする。`&self` は `Send + Sync` なので複数の共有借用は安全。`Agent<P>` には二重 boxing を避けるため inherent method としても `run_parallel` を提供する。

#### SubAgentEntry と SubAgentMode

Sub-Agent は LLM には `ToolDefinition` として見せるが、ライブラリ内部では Tool とは別に管理する。2つのモードを持つ:

- **Single**: 1つのクエリを受け取り実行。スキーマ: `{ "query": string }`
- **Parallel**: 複数タスクを受け取り `join_all` で並列実行。スキーマ: `{ "tasks": [string] }`

```rust
pub enum SubAgentMode {
    Single,
    Parallel,
}

pub struct SubAgentEntry {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    agent: Arc<dyn AgentExecutor>,
    mode: SubAgentMode,
}
```

コンストラクタ:

```rust
SubAgentEntry::single(name, description, agent)   // query: string
SubAgentEntry::parallel(name, description, agent)  // tasks: [string]
```

#### Agent 構造体の拡張

```rust
pub struct Agent<P: LlmProvider> {
    provider: P,
    tools: Vec<Box<dyn Tool>>,
    sub_agents: Vec<SubAgentEntry>,
    config: AgentConfig,
    context_builder: Box<dyn ContextBuilder>,
    stop_conditions: Vec<Box<dyn StopCondition>>,
}
```

Builder メソッド:

```rust
impl<P: LlmProvider> Agent<P> {
    pub fn with_sub_agent(self, name, description, agent) -> Self;
    pub fn with_parallel_sub_agent(self, name, description, agent) -> Self;
}
```

#### AgentStep の拡張

Sub-Agent の実行結果を保持する2つの variant:

```rust
pub enum AgentStep {
    LlmCall { request_messages: usize, response: ChatResponse },
    ToolExecution { name: String, input: Value, output: String, is_error: bool },
    SubAgentExecution { name: String, input: Value, output: AgentOutput },
    ParallelSubAgentExecution {
        name: String,
        tasks: Vec<String>,
        outputs: Vec<AgentOutput>,  // 成功した実行結果
        errors: Vec<String>,        // 失敗したタスクのエラーメッセージ
    },
}
```

#### ReAct ループ内の dispatch

LLM が `tool_use` を返したとき、name を tools → sub_agents の順で検索し、該当する方に dispatch する:

1. **Tool にマッチ** → `Tool::execute()` → `String` → `Content::ToolResult` を会話に追加
2. **Single Sub-Agent にマッチ** → `AgentExecutor::run()` → `AgentOutput` を取得
   - `output.response` を `Content::ToolResult` として会話に追加
   - `AgentOutput` 全体を `AgentStep::SubAgentExecution` に記録
   - `output.total_usage` を親の `total_usage` に加算
3. **Parallel Sub-Agent にマッチ** → `input.tasks` 配列を取得
   - 各タスクを `Message::user(task)` にして `AgentExecutor::run_parallel()` で並列実行
   - 結果を `[Task 1] ...\n[Task 2] ...` 形式にまとめて `Content::ToolResult` として返す
   - 各成功した `AgentOutput` の `total_usage` を親に加算
   - `AgentStep::ParallelSubAgentExecution` に全結果を記録

#### Observability

tracing span で Tool 実行・Single Sub-Agent・Parallel Sub-Agent を分離する:

```
agent.run              [parent]
├── iteration.0
│   ├── llm.chat
│   ├── tool.exec              [name=read_file]
│   ├── sub_agent.run          [name=assistant]     ← Single
│   └── sub_agent.run_parallel [name=researcher, task_count=3] ← Parallel
│       ├── agent.run [task 0]
│       ├── agent.run [task 1]
│       └── agent.run [task 2]
```

#### Application 層での利用イメージ

```rust
// Single Sub-Agent
let coder = Agent::with_system_prompt(provider, coding_tools, config, "You are a coder.");

// Parallel Sub-Agent (fan-out)
let researcher = Agent::with_system_prompt(provider, research_tools, config, "You are a researcher.");

let planner = Agent::with_system_prompt(provider, vec![], config, "You are a planner.")
    .with_sub_agent("code", "Write code", coder)
    .with_parallel_sub_agent("research", "Research topics concurrently", researcher);

let output = planner.run(vec![Message::user("Research A, B, C then code")]).await?;

// Parallel Sub-Agent の結果は ParallelSubAgentExecution から参照
for step in &output.steps {
    if let AgentStep::ParallelSubAgentExecution { name, tasks, outputs, .. } = step {
        println!("{}: {} tasks, {} succeeded", name, tasks.len(), outputs.len());
    }
}
```

### 7. Agent (ReAct Loop)

会話履歴（Memory）はライブラリでは管理せず、呼び出し側が `Vec<Message>` として渡す設計とする。ContextBuilder を通じて、ユーザーがスライディングウィンドウや要約を実装できる。

```rust
pub struct AgentConfig {
    pub model: String,
    pub max_iterations: usize,
    pub extra_params: serde_json::Map<String, serde_json::Value>,
}

pub struct Agent<P: LlmProvider> {
    provider: P,
    tools: Vec<Box<dyn Tool>>,
    sub_agents: Vec<SubAgentEntry>,
    config: AgentConfig,
    context_builder: Box<dyn ContextBuilder>,
    stop_conditions: Vec<Box<dyn StopCondition>>,
}

impl<P: LlmProvider> Agent<P> {
    // コンストラクタ
    pub fn new(provider, tools, config, context_builder) -> Self;
    pub fn with_system_prompt(provider, tools, config, prompt) -> Self;
    pub fn without_system_prompt(provider, tools, config) -> Self;

    // Stop condition 追加 (builder pattern)
    pub fn with_stop_condition(self, condition) -> Self;
    pub fn with_stop_conditions(self, conditions) -> Self;

    // Sub-Agent 追加 (builder pattern)
    pub fn with_sub_agent(self, name, description, agent) -> Self;
    pub fn with_parallel_sub_agent(self, name, description, agent) -> Self;

    // Parallel tool 追加 (fan-out wrapper)
    pub fn with_parallel_tool(self, tool: Box<dyn Tool>) -> Self;

    // 実行
    pub async fn run(&self, messages: Vec<Message>) -> Result<AgentOutput, Error>;
    pub async fn run_structured<O: DeserializeOwned + JsonSchema>(
        &self, messages: Vec<Message>,
    ) -> Result<StructuredAgentOutput<O>, Error>;
}
```

**ReAct ループの流れ**:
1. `context_builder.build_initial(messages)` でコンテキスト構築
2. `tools` と `sub_agents` の両方から `ToolDefinition` を収集し、LLM に渡す
3. Loop:
   a. `context_builder.build_iteration()` でメッセージ列を構築（2回目以降）
   b. `extra_params` を含む ChatRequest を構築し LLM に送信
   c. StopReason が EndTurn/MaxTokens → 最終応答として返す (`AgentStopReason::NaturalStop`)
   d. StopReason が ToolUse → name で dispatch:
      - tools にマッチ → `Tool::execute()` → `String` → `ToolResult`
      - sub_agents にマッチ → `AgentExecutor::run()` → `AgentOutput` → response を `ToolResult` に、usage を親に加算
   e. StopCondition をチェック → 発火したら `AgentStopReason::StopCondition` で返す
   f. max_iterations 超過 → `AgentStopReason::MaxIterations` で返す

```rust
pub struct AgentOutput {
    pub response: String,
    pub steps: Vec<AgentStep>,
    pub total_usage: Usage,
    pub stop_reason: AgentStopReason,
}

pub struct StructuredAgentOutput<O> {
    pub response: O,
    pub steps: Vec<AgentStep>,
    pub total_usage: Usage,
    pub stop_reason: AgentStopReason,
}

pub enum AgentStep {
    LlmCall { request_messages: usize, response: ChatResponse },
    ToolExecution { name: String, input: Value, output: String, is_error: bool },
    SubAgentExecution { name: String, input: Value, output: AgentOutput },
}
```

**ReAct ループの透明性**: `AgentOutput` に全ステップ (`AgentStep`) を記録し、呼び出し側が各ステップの詳細を参照できるようにする。`max_iterations` 超過はエラーではなく `AgentStopReason` として返す。Sub-Agent の実行は `AgentStep::SubAgentExecution` としてネストした `AgentOutput` を保持し、再帰的にステップを辿れる。

---

## Built-in Tools (membrane-tools)

ファイル操作やコマンド実行など、Agent が一般的に必要とするツール群を提供するクレート。すべて `#[membrane_tool]` マクロで定義。

- **ReadFileTool** — ファイル読み込み（offset/limit パラメータ対応）
- **WriteFileTool** — ファイル書き込み（ディレクトリ自動作成）
- **SearchFilesTool** — glob パターンによるファイル検索
- **ExecTool** — シェルコマンド実行

---

## OpenAI Provider (membrane-openai)

OpenAI Chat Completions API の実装。

- Builder パターンによる構築 (`OpenAiProvider::builder().api_key("...").build()`)
- `TokenProvider` trait で動的トークン取得に対応（Azure AD 等）
- 内部型 (`types.rs`) + 変換層 (`convert.rs`) で membrane-core ↔ OpenAI wire format を分離
- `extra_params` は OpenAI リクエスト JSON にフラット展開される
- wiremock ベースのテスト（実 API キー不要）

```rust
pub trait TokenProvider: Send + Sync {
    fn token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>>;
}

let provider = OpenAiProvider::builder()
    .api_key("sk-...")
    .base_url("https://api.openai.com/v1")  // optional
    .client(reqwest_client)                   // optional
    .build();
```

---

## Observability

全レイヤーで `tracing` を活用する。

```
agent.run          [span]          ← Agent 全体の実行 (model名を記録)
├── iteration.0    [span]          ← ReAct ループの各イテレーション
│   ├── llm.chat   [span]          ← LLM 呼び出し
│   │   └── event: token usage, stop_reason
│   ├── tool.exec  [span]          ← Tool 実行 (tool_name を記録)
│   │   └── event: success / failure
│   └── sub_agent.run [span]       ← Sub-Agent 実行 (name を記録)
│       ├── iteration.0 [span]     ← ネストした ReAct ループ
│       │   ├── llm.chat [span]
│       │   └── tool.exec [span]
│       └── iteration.1 [span]
│           └── llm.chat [span]
├── iteration.1    [span]
│   └── llm.chat   [span]
│       └── event: final response
└── event: stop condition triggered (if applicable)
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

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Serializable representation of an error for structured output.
pub struct ErrorInfo {
    pub kind: String,
    pub message: String,
}
```

`max_iterations` 超過は `Error` ではなく `AgentStopReason::MaxIterations` として表現する。Agent の実行は常に `Result<AgentOutput, Error>` を返し、`stop_reason` フィールドで終了理由を確認できる。

---

## Implementation Phases

### Phase 1: Core Types & Traits ✅
- Message, Content, Role 型の定義
- LlmProvider trait (RPITIT)
- Tool trait (`Pin<Box<dyn Future>>` for dyn-compatibility), ToolDefinition
- Error 型, ErrorInfo
- 基本的な tracing instrumentation

### Phase 2: Agent Loop ✅
- AgentConfig, Agent 構造体
- ReAct ループの実装 (`run`, `run_structured`)
- AgentOutput, StructuredAgentOutput, AgentStep

### Phase 3: OpenAI Provider ✅
- membrane-openai クレート（Chat Completions API）
- Builder パターンによる構築
- `TokenProvider` trait で動的トークン取得に対応
- 内部型 + 変換層で wire format を分離
- `extra_params` の flatten マージ
- wiremock ベースのテスト

### Phase 4: Context Builder ✅
- `ContextBuilder` trait（コンテキスト構築の抽象化）
- `DefaultContextBuilder`（system prompt の単純挿入）
- Agent に `with_system_prompt()` / `without_system_prompt()` 便利メソッド

### Phase 5: `#[membrane_tool]` Proc Macro ✅
- `membrane-macros` クレート
- async 関数 → Tool 構造体 + trait 実装の自動生成
- schemars 連携で JSON Schema を自動生成
- `membrane-core` から `pub use membrane_macros::membrane_tool` で re-export

### Phase 6: Stop Conditions ✅
- `StopCondition` trait + `StopContext` スナップショット
- `AgentStopReason` enum（NaturalStop / MaxIterations / StopCondition）
- ビルトイン条件: Timeout, TokenBudget, MaxConsecutiveErrors, CustomStop
- Or / And コンビネータ
- `max_iterations` 超過をエラーから AgentStopReason に変更

### Phase 7: Built-in Tools ✅
- `membrane-tools` クレート
- ReadFileTool, WriteFileTool, SearchFilesTool, ExecTool

### Phase 8: Sub-Agent System
- `AgentExecutor` trait（`Agent<P>` の型消去）
- `SubAgentEntry` + `Agent::with_sub_agent()` builder メソッド
- `AgentStep::SubAgentExecution` variant
- ReAct ループ内の Tool / Sub-Agent dispatch 分岐
- Sub-Agent の token usage 親への加算
- `sub_agent.run` tracing span
- テスト（MockProvider ベースの親子 Agent テスト）

### Phase 9: 拡張（未実装）
- membrane-anthropic クレート（Messages API）
- Streaming 対応

---

## Memory Management Research

現在の membrane は `Agent takes Vec<Message> as input — no built-in memory management` という方針で、会話履歴の管理は呼び出し側に委ねている。`ContextBuilder` trait を通じて、ユーザーがスライディングウィンドウや要約を実装できる設計となっている。

本セクションでは、将来的なメモリ管理機能の設計判断の材料として、LLM エージェントのメモリマネジメントに関する主要な論文・アプローチを調査した結果をまとめる。

### メモリの分類体系

人間の認知科学に基づき、LLM エージェントのメモリは以下のように分類される（Zhang et al., 2024; LangMem, 2025）。

#### Short-term Memory（短期記憶）

LLM のコンテキストウィンドウそのものに対応する。現在の会話ターンや直近の tool 実行結果など、即座にアクセス可能な情報。membrane の現在の `Vec<Message>` はこれに相当する。

- **制約**: コンテキストウィンドウのトークン上限
- **対策**: スライディングウィンドウ、要約、truncation

#### Long-term Memory（長期記憶）

コンテキストウィンドウ外に永続化される情報。以下の3種類に細分される:

1. **Semantic Memory（意味記憶）**: 世界知識、ユーザープロファイル、ドメイン固有の事実。ベクトル DB や構造化ストレージに格納。「何を知っているか」。
2. **Episodic Memory（エピソード記憶）**: 過去の具体的な対話や経験の記録。成功した対話パターンや特定のタスク遂行の文脈を保持。「何が起きたか」。Few-shot example として利用されることが多い。
3. **Procedural Memory（手続き記憶）**: タスクの実行方法に関する知識。エージェントのシステムプロンプトやルール、学習された手順。「どう振る舞うか」。

### 主要論文・アプローチ

#### 1. MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023)

- **論文**: [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- **核心**: OS の仮想メモリ管理をメタファーとした階層的メモリアーキテクチャ
- **設計**:
  - **Main Context**（RAM 相当）: LLM のコンテキストウィンドウ内の情報
  - **External Context**（Disk 相当）: コンテキストウィンドウ外の永続ストレージ
  - LLM 自身が function call を通じてメモリの読み書き（ページング）を自律的に実行
  - **Core Memory**: 常にコンテキスト内に保持される重要情報（ユーザープロファイル等）
  - **Archival Memory**: 大容量の外部ストレージ（ベクトル検索可能）
  - **Recall Memory**: 過去の会話履歴の検索可能なアーカイブ
- **評価**: 長文書分析と複数セッションチャットで、コンテキスト上限を超えたテキスト処理と長期対話の一貫性を実証
- **membrane への示唆**: Tool として memory read/write を提供し、エージェント自身にメモリ管理を委ねるアプローチ。membrane-tools に MemoryReadTool / MemoryWriteTool を追加する形で実現可能

#### 2. Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)

- **論文**: [arXiv:2304.03442](https://arxiv.org/abs/2304.03442), UIST '23
- **核心**: Memory Stream + Retrieval + Reflection + Planning の4層アーキテクチャ
- **設計**:
  - **Memory Stream**: 全経験を自然言語で時系列に記録するデータベース。各メモリに作成日時・最終アクセス日時を付与
  - **Retrieval**: recency（時間減衰）、relevance（埋め込みの類似度）、importance（LLM が判定した重要度スコア）の3軸で検索スコアを算出
  - **Reflection**: 蓄積した観察から高次の抽象的な洞察（reflection）を生成。重要度スコアの累計が閾値を超えるとトリガー
  - **Planning**: トップダウンで日次計画 → 時間単位 → 5-15分単位に再帰的に詳細化
- **membrane への示唆**: ContextBuilder の `build_iteration()` で retrieval ロジックを実装できる。importance scoring は LLM 呼び出しが必要なため、`AsyncContextBuilder` trait の導入が有用

#### 3. Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)

- **論文**: [arXiv:2303.11366](https://arxiv.org/abs/2303.11366), NeurIPS 2023
- **核心**: 重み更新なしに、言語的フィードバック（自己反省）でエージェントを強化
- **設計**:
  - **Actor**: 環境と対話してアクション・軌跡を生成
  - **Evaluator**: 軌跡にスコアを付与
  - **Self-Reflection**: 失敗の分析結果を自然言語で生成し、episodic memory buffer に蓄積
  - 次回の試行で、蓄積した反省を参照して意思決定を改善
  - memory buffer はコンテキスト上限に収まるよう、直近 1-3 件に制限
- **成果**: HumanEval で 91% pass@1（GPT-4 単体の 80% を上回る）
- **membrane への示唆**: `AgentOutput` の steps を分析して reflection を生成し、次回の run に渡す episodic memory パターン。アプリケーション層で実装可能

#### 4. A-MEM: Agentic Memory for LLM Agents (Xu et al., 2025)

- **論文**: [arXiv:2502.12110](https://arxiv.org/abs/2502.12110), NeurIPS 2025
- **核心**: Zettelkasten メソッドに着想を得た、エージェント的メモリ組織化
- **設計**:
  - メモリを相互にリンクされたナレッジネットワークとして動的にインデックス化
  - エージェント自身がメモリの整理・関連付けを実行
- **成果**: ベースライン（MemGPT 含む）比で 85-93% のトークン使用量削減
- **membrane への示唆**: 構造化されたメモリストア（グラフベース）と、メモリ管理自体をエージェントのタスクとして扱うアプローチ

#### 5. SCM: Self-Controlled Memory Framework (Liang et al., 2023)

- **論文**: [arXiv:2304.13343](https://arxiv.org/abs/2304.13343)
- **核心**: LLM ベースの Memory Controller がメモリの更新・利用タイミングを制御
- **設計**:
  - **Agent**（バックボーン LLM）+ **Memory Stream** + **Memory Controller** の3コンポーネント
  - Memory Controller がいつ・どのメモリを利用するかを判断
  - ファインチューニング不要で任意の instruction-following LLM に plug-and-play で統合可能
- **membrane への示唆**: ContextBuilder を拡張して Memory Controller の役割を持たせる設計が可能

#### 6. Recursively Summarizing Enables Long-Term Dialogue Memory (Wang et al., 2023)

- **論文**: [arXiv:2308.15022](https://arxiv.org/abs/2308.15022)
- **核心**: 再帰的要約による長期対話メモリ
- **設計**:
  - 小さな対話コンテキストからまず要約を生成
  - 前回の要約 + 新しいコンテキストから再帰的に新しい要約を生成
  - 要約の連鎖により、任意の長さの対話履歴を圧縮
- **membrane への示唆**: `ContextBuilder::build_iteration()` で実装可能な最もシンプルなアプローチ。要約の生成に LLM 呼び出しが必要なため、非同期 ContextBuilder が有用

### コンテキストウィンドウ管理の実践的手法

論文以外の実践的なアプローチも整理する（JetBrains Research, 2025; LangChain, 2025）。

| 手法 | 概要 | トレードオフ |
|------|------|------------|
| **Sliding Window** | 直近 N メッセージのみ保持 | 実装が最も単純。古い情報を完全に喪失 |
| **Conversation Summarization** | 古い対話を LLM で要約して圧縮 | 情報保持率が高い。要約のための追加 LLM 呼び出しコスト |
| **Hybrid Summary + Buffer** | 直近はそのまま保持、古い部分は要約 | バランスが良い。LangChain の ConversationSummaryBufferMemory が代表例 |
| **Hierarchical Summarization** | 情報の古さに応じて段階的に圧縮度を上げる | 91% の重要情報を保持しつつ 60-70% のトークン削減（研究報告） |
| **Observation Masking** | 古い観察結果をプレースホルダーに置換 | LLM 要約と同等の性能（JetBrains 調べ）。ハイパーパラメータ調整が必要 |
| **Vector Store / RAG** | 過去の対話を埋め込みとして保存し、意味的に類似したものを検索注入 | FAQ 的なインタラクションに有効。セットアップコストが高い |
| **Knowledge Graph** | エンティティ・関係を抽出しグラフ構築 | 構造化された知識に強い。抽出精度に依存 |

### membrane への抽象化設計

調査結果を踏まえ、membrane でメモリ管理をどこまで抽象化できるかを分析する。

#### 設計原則

1. **membrane-core はメモリ管理を行わない方針を維持**する。`Vec<Message>` をそのまま受け取る設計の利点（永続化手段の自由、タスク依存の戦略、core の責務の明確さ）は保つ。
2. **抽象化は拡張レイヤーで提供**する。core に trait のみ追加し、実装は別クレートに配置。
3. **既存の `ContextBuilder` との後方互換性**を維持する。

#### レイヤー構成

```
Layer 0: membrane-core（trait のみ追加）
├── ContextBuilder (sync, 既存・変更なし)
└── AsyncContextBuilder (新規、Pin<Box<dyn Future>> で dyn-compatible)

Layer 1: membrane-memory（新規クレート）
├── MemoryStore trait (ストレージ抽象)
├── MemoryEntry, MemoryQuery 型
├── InMemoryStore (メモリ内実装)
├── SlidingWindowContextBuilder
├── SummaryContextBuilder (AsyncContextBuilder, LLM 呼び出し)
└── HybridContextBuilder (Summary + Buffer)

Layer 2: membrane-tools（既存クレートに追加）
├── MemoryStoreTool (MemGPT 方式: エージェントがメモリに書き込み)
├── MemorySearchTool (セマンティック検索 / キーワード検索)
└── MemoryDeleteTool

Layer 3: Application 層（ユーザー実装）
├── ストレージバックエンド (Redis, SQLite, ベクトル DB 等)
├── Episodic memory 構築ロジック (Reflexion パターン)
└── タスク固有のメモリ戦略
```

#### 1. AsyncContextBuilder（membrane-core に追加）

多くのメモリ管理手法は LLM 呼び出しやベクトル検索など非同期処理を必要とする。`Tool` trait と同様に `Pin<Box<dyn Future>>` を使い、dyn-compatible にする。

```rust
/// Async version of ContextBuilder for memory-intensive context building.
///
/// Uses Pin<Box<dyn Future>> (like Tool trait) for dyn-compatibility.
/// Enables retrieval-based memory, LLM summarization, importance scoring, etc.
pub trait AsyncContextBuilder: Send + Sync {
    fn build_initial(&self, messages: Vec<Message>)
        -> Pin<Box<dyn Future<Output = Vec<Message>> + Send + '_>>;

    fn build_iteration(&self, conversation: &[Message], iteration: usize)
        -> Pin<Box<dyn Future<Output = Vec<Message>> + Send + '_>>;
}
```

**Agent 側の統合**: Agent は内部で `Box<dyn AsyncContextBuilder>` を保持する。既存の同期 `ContextBuilder` はアダプタ構造体でラップすることで後方互換性を維持する。

```rust
/// Adapter: wraps a sync ContextBuilder as an AsyncContextBuilder.
struct SyncAdapter(Box<dyn ContextBuilder>);

impl AsyncContextBuilder for SyncAdapter {
    fn build_initial(&self, messages: Vec<Message>)
        -> Pin<Box<dyn Future<Output = Vec<Message>> + Send + '_>> {
        Box::pin(std::future::ready(self.0.build_initial(messages)))
    }

    fn build_iteration(&self, conversation: &[Message], iteration: usize)
        -> Pin<Box<dyn Future<Output = Vec<Message>> + Send + '_>> {
        Box::pin(std::future::ready(self.0.build_iteration(conversation, iteration)))
    }
}
```

`Agent::with_system_prompt()` 等の既存コンストラクタは内部で `SyncAdapter` を使うため、ユーザーコードへの影響は最小限。

**blanket impl を使わない理由**: `impl<T: ContextBuilder> AsyncContextBuilder for T` は、ユーザーが `ContextBuilder` を実装した型に対して `AsyncContextBuilder` も実装したい場合に orphan rule で衝突する。明示的なアダプタ構造体の方が柔軟。

#### 2. MemoryStore trait（membrane-memory に配置）

ストレージバックエンドの抽象。core ではなく別クレートに配置する理由:
- 新しい型（`MemoryEntry`, `MemoryQuery`）を導入する
- 時刻依存（`std::time::SystemTime` or chrono）が入る
- 基本的な Agent 動作には不要
- core の依存を最小限に保てる

```rust
/// A single memory entry stored in the memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub created_at: u64,      // Unix timestamp (seconds)
    pub last_accessed_at: u64, // Unix timestamp (seconds)
}

/// Query parameters for memory retrieval.
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub query: String,
    pub limit: usize,
    pub metadata_filter: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Storage backend abstraction for agent memory.
///
/// Implementations can range from in-memory HashMap to vector databases.
/// Uses Pin<Box<dyn Future>> for dyn-compatibility (same pattern as Tool trait).
pub trait MemoryStore: Send + Sync {
    fn store(&self, entry: MemoryEntry)
        -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;

    fn search(&self, query: MemoryQuery)
        -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, Error>> + Send + '_>>;

    fn delete(&self, id: &str)
        -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>>;
}
```

**InMemoryStore**: テスト・プロトタイピング用のメモリ内実装。キーワード部分一致検索をサポート。

#### 3. 組み込み ContextBuilder 実装（membrane-memory に配置）

よく使われるパターンを実装として提供する。

**SlidingWindowContextBuilder**（同期、`ContextBuilder` を実装）:

```rust
/// Keeps the system prompt + last N non-system messages.
pub struct SlidingWindowContextBuilder {
    system_prompt: Option<String>,
    max_messages: usize,
}

impl ContextBuilder for SlidingWindowContextBuilder {
    fn build_initial(&self, messages: Vec<Message>) -> Vec<Message> { /* ... */ }

    fn build_iteration(&self, conversation: &[Message], _iteration: usize) -> Vec<Message> {
        // Keep system messages + last max_messages non-system messages
    }
}
```

**SummaryContextBuilder**（非同期、`AsyncContextBuilder` を実装）:

```rust
/// Summarizes old conversation turns using an LLM, keeping recent turns verbatim.
///
/// Requires an LlmProvider for summarization.
/// Hybrid strategy: [system] + [summary of old turns] + [last N verbatim turns]
pub struct SummaryContextBuilder<P: LlmProvider> {
    provider: P,
    system_prompt: Option<String>,
    summary_model: String,
    buffer_size: usize,       // Number of recent messages to keep verbatim
    summary: RwLock<Option<String>>,  // Running summary of old conversation
}
```

**MemoryContextBuilder**（非同期、`AsyncContextBuilder` を実装）:

```rust
/// Retrieves relevant memories from a MemoryStore and injects them as context.
///
/// On each iteration:
/// 1. Extracts the latest user query from conversation
/// 2. Searches MemoryStore for relevant memories
/// 3. Injects matched memories as system messages
/// 4. Applies optional sliding window to conversation
pub struct MemoryContextBuilder {
    store: Arc<dyn MemoryStore>,
    system_prompt: Option<String>,
    max_memories: usize,
    max_conversation_messages: usize,
}
```

#### 4. Memory Tool パターン（membrane-tools に追加）

MemGPT の設計に倣い、メモリの読み書きを Tool として提供する。エージェント自身がメモリ管理を行う「自律的メモリ管理」パターン。

```rust
/// Tool for the agent to store memories. Wraps a MemoryStore.
pub struct MemoryStoreTool {
    store: Arc<dyn MemoryStore>,
}

/// Tool for the agent to search its memories.
pub struct MemorySearchTool {
    store: Arc<dyn MemoryStore>,
}

/// Tool for the agent to delete memories.
pub struct MemoryDeleteTool {
    store: Arc<dyn MemoryStore>,
}
```

これにより、同じ `Arc<dyn MemoryStore>` を ContextBuilder（受動的メモリ注入）と Memory Tool（能動的メモリ管理）の両方で共有できる:

```rust
let store = Arc::new(InMemoryStore::new());

let context = MemoryContextBuilder::new(store.clone(), "You are helpful.", 5, 20);
let tools: Vec<Box<dyn Tool>> = vec![
    Box::new(MemoryStoreTool::new(store.clone())),
    Box::new(MemorySearchTool::new(store.clone())),
];

let agent = Agent::new(provider, tools, config, Box::new(context));
```

#### 5. Episodic Memory ユーティリティ（Application 層）

Reflexion パターンに基づく episodic memory 構築は、アプリケーション層のユーティリティとして提供し core には組み込まない。理由:

- Reflection 生成には LLM 呼び出しが必要で、Agent のスコープ外
- 成功/失敗の評価基準がタスク依存
- `AgentOutput` の `steps` は Serialize/Deserialize 可能なので、ユーザーが自由に分析・蓄積できる

```rust
// Application-layer example: building episodic memory from AgentOutput
fn build_episodic_memory(output: &AgentOutput, store: &dyn MemoryStore) {
    for step in &output.steps {
        match step {
            AgentStep::ToolExecution { name, input, output, is_error } => {
                // Record successful/failed tool patterns
            }
            AgentStep::SubAgentExecution { name, output: sub_output, .. } => {
                // Record sub-agent interaction patterns
            }
            _ => {}
        }
    }
}
```

#### 抽象化の判断基準まとめ

| 機能 | 抽象化レベル | 配置先 | 理由 |
|------|------------|--------|------|
| AsyncContextBuilder trait | Core trait | membrane-core | Agent ループが直接呼び出す。dyn-compatible 必須 |
| SyncAdapter | Core internal | membrane-core | 後方互換性のための内部アダプタ |
| MemoryStore trait | Library trait | membrane-memory | 新型導入。core に不要な依存を持ち込まない |
| InMemoryStore | Library impl | membrane-memory | テスト・プロトタイプ用 |
| SlidingWindowContextBuilder | Library impl | membrane-memory | 同期で単純。core に入れても良いが、crate を分ける方が一貫性がある |
| SummaryContextBuilder | Library impl | membrane-memory | LlmProvider 依存。非同期 |
| MemoryContextBuilder | Library impl | membrane-memory | MemoryStore 依存。非同期 |
| MemoryStore/Search/DeleteTool | Library tools | membrane-tools | MemGPT 方式の能動的メモリ管理 |
| Episodic memory 構築 | App utility | examples / applications | タスク依存が強い。ライブラリ化は時期尚早 |

#### 実装順序

1. **AsyncContextBuilder trait** を membrane-core に追加。Agent の内部を `Box<dyn AsyncContextBuilder>` に変更
2. **membrane-memory** クレートを作成。MemoryStore trait + InMemoryStore + SlidingWindowContextBuilder
3. **SummaryContextBuilder** を membrane-memory に追加
4. **Memory Tools**（Store/Search/Delete）を membrane-tools に追加
5. **examples/** に MemGPT 方式のメモリ管理サンプルを追加

### 参考文献

1. Packer, C. et al. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560, 2023.
2. Park, J. S. et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST '23. arXiv:2304.03442, 2023.
3. Shinn, N. et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023. arXiv:2303.11366, 2023.
4. Xu, W. et al. "A-MEM: Agentic Memory for LLM Agents." NeurIPS 2025. arXiv:2502.12110, 2025.
5. Liang, X. et al. "SCM: Enhancing Large Language Model with Self-Controlled Memory Framework." arXiv:2304.13343, 2023.
6. Wang, Q. et al. "Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models." arXiv:2308.15022, 2023.
7. Zhang, Z. et al. "A Survey on the Memory Mechanism of Large Language Model based Agents." ACM TOIS. arXiv:2404.13501, 2024.
8. JetBrains Research. "Cutting Through the Noise: Smarter Context Management for LLM-Powered Agents." 2025.
9. LangChain. "LangMem: Long-term Memory in LLM Applications." 2025.

---

## Open Questions

1. **Streaming**: 現在 non-streaming のみ。Streaming は `Stream` trait で返す想定だが、runtime 非依存との兼ね合いをどうするか（`futures::Stream` を使うか）
2. **Sub-Agent の input_schema**: Sub-Agent に渡す入力のスキーマをどう定義するか。デフォルトは `{ "query": string }` で十分か、カスタムスキーマを許容すべきか
3. **Sub-Agent のエラーハンドリング**: Sub-Agent が `Err` を返した場合、親の会話に `ToolResult { is_error: true }` として渡すか、親の `run` 自体を `Err` にするか。Tool と同じく `is_error: true` で会話に戻す方が自然か
