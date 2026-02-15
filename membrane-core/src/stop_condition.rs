use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::agent::AgentStep;
use crate::message::Message;
use crate::provider::Usage;

/// Snapshot of the agent loop state, passed to each stop condition check.
#[derive(Debug)]
pub struct StopContext<'a> {
    /// Zero-based iteration index (after the iteration completes).
    pub iteration: usize,
    /// Accumulated token usage so far.
    pub total_usage: &'a Usage,
    /// All steps recorded so far in this run.
    pub steps: &'a [AgentStep],
    /// Wall-clock time elapsed since the agent run started.
    pub elapsed: Duration,
    /// Current conversation history (system prompt, user messages, assistant responses, tool results).
    pub messages: &'a [Message],
}

/// Describes why the agent loop terminated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentStopReason {
    /// The LLM produced a final response without requesting more tools.
    NaturalStop,
    /// The `max_iterations` limit in `AgentConfig` was reached.
    MaxIterations { max: usize },
    /// A user-defined `StopCondition` fired.
    StopCondition { description: String },
}

/// A condition that can terminate the agent's ReAct loop early.
///
/// Conditions are checked at the end of each iteration, after the LLM call
/// and any tool executions. If `should_stop` returns `Some(description)`,
/// the loop terminates and the description is included in the output's
/// `stop_reason`.
///
/// All built-in conditions are stateless — they derive their decision solely
/// from the [`StopContext`]. For stateful conditions (e.g., shared state with
/// a tool via `Arc<Mutex<...>>`), use [`CustomStop`].
pub trait StopCondition: Send + Sync {
    /// Check whether the agent should stop. Returns `Some(description)` if so.
    fn should_stop(&self, ctx: &StopContext<'_>) -> Option<String>;

    /// Combine with another condition using OR logic (stop if either fires).
    fn or<B: StopCondition>(self, other: B) -> Or<Self, B>
    where
        Self: Sized,
    {
        Or { a: self, b: other }
    }

    /// Combine with another condition using AND logic (stop only if both fire).
    fn and<B: StopCondition>(self, other: B) -> And<Self, B>
    where
        Self: Sized,
    {
        And { a: self, b: other }
    }
}

/// Delegate to the inner condition for `Box<dyn StopCondition>`.
impl StopCondition for Box<dyn StopCondition> {
    fn should_stop(&self, ctx: &StopContext<'_>) -> Option<String> {
        (**self).should_stop(ctx)
    }
}

// ── Built-in conditions ────────────────────────────────────────────

/// Stop after a wall-clock timeout.
pub struct Timeout {
    duration: Duration,
}

impl Timeout {
    pub fn new(duration: Duration) -> Self {
        Self { duration }
    }
}

impl StopCondition for Timeout {
    fn should_stop(&self, ctx: &StopContext<'_>) -> Option<String> {
        if ctx.elapsed >= self.duration {
            Some(format!("timeout ({:?}) exceeded", self.duration))
        } else {
            None
        }
    }
}

/// Stop when cumulative token usage exceeds a budget.
pub struct TokenBudget {
    max_total_tokens: u32,
}

impl TokenBudget {
    pub fn new(max_total_tokens: u32) -> Self {
        Self { max_total_tokens }
    }
}

impl StopCondition for TokenBudget {
    fn should_stop(&self, ctx: &StopContext<'_>) -> Option<String> {
        let total = ctx.total_usage.input_tokens + ctx.total_usage.output_tokens;
        if total >= self.max_total_tokens {
            Some(format!(
                "token budget ({}) exceeded (used {})",
                self.max_total_tokens, total
            ))
        } else {
            None
        }
    }
}

/// Stop after N consecutive tool execution errors.
pub struct MaxConsecutiveErrors {
    max: usize,
}

impl MaxConsecutiveErrors {
    pub fn new(max: usize) -> Self {
        Self { max }
    }
}

impl StopCondition for MaxConsecutiveErrors {
    fn should_stop(&self, ctx: &StopContext<'_>) -> Option<String> {
        let consecutive_errors = ctx
            .steps
            .iter()
            .rev()
            .take_while(|step| matches!(step, AgentStep::ToolExecution { is_error: true, .. }))
            .count();

        if consecutive_errors >= self.max {
            Some(format!("max consecutive errors ({}) reached", self.max))
        } else {
            None
        }
    }
}

/// A custom predicate stop condition using a closure.
pub struct CustomStop<F> {
    predicate: F,
}

impl<F> CustomStop<F>
where
    F: Fn(&StopContext<'_>) -> Option<String> + Send + Sync,
{
    pub fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

impl<F> StopCondition for CustomStop<F>
where
    F: Fn(&StopContext<'_>) -> Option<String> + Send + Sync,
{
    fn should_stop(&self, ctx: &StopContext<'_>) -> Option<String> {
        (self.predicate)(ctx)
    }
}

// ── Combinators ────────────────────────────────────────────────────

/// OR combinator: stops if either condition fires.
pub struct Or<A, B> {
    a: A,
    b: B,
}

impl<A: StopCondition, B: StopCondition> StopCondition for Or<A, B> {
    fn should_stop(&self, ctx: &StopContext<'_>) -> Option<String> {
        self.a.should_stop(ctx).or_else(|| self.b.should_stop(ctx))
    }
}

/// AND combinator: stops only if both conditions fire.
pub struct And<A, B> {
    a: A,
    b: B,
}

impl<A: StopCondition, B: StopCondition> StopCondition for And<A, B> {
    fn should_stop(&self, ctx: &StopContext<'_>) -> Option<String> {
        match (self.a.should_stop(ctx), self.b.should_stop(ctx)) {
            (Some(desc_a), Some(desc_b)) => Some(format!("{} AND {}", desc_a, desc_b)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use crate::provider::Usage;

    fn make_ctx<'a>(
        iteration: usize,
        usage: &'a Usage,
        steps: &'a [AgentStep],
        elapsed: Duration,
        messages: &'a [Message],
    ) -> StopContext<'a> {
        StopContext {
            iteration,
            total_usage: usage,
            steps,
            elapsed,
            messages,
        }
    }

    #[test]
    fn timeout_fires_when_exceeded() {
        let cond = Timeout::new(Duration::from_secs(10));
        let usage = Usage::default();
        let ctx = make_ctx(0, &usage, &[], Duration::from_secs(15), &[]);
        assert!(cond.should_stop(&ctx).is_some());
    }

    #[test]
    fn timeout_does_not_fire_before_deadline() {
        let cond = Timeout::new(Duration::from_secs(10));
        let usage = Usage::default();
        let ctx = make_ctx(0, &usage, &[], Duration::from_secs(5), &[]);
        assert!(cond.should_stop(&ctx).is_none());
    }

    #[test]
    fn token_budget_fires_when_exceeded() {
        let cond = TokenBudget::new(100);
        let usage = Usage {
            input_tokens: 60,
            output_tokens: 50,
        };
        let ctx = make_ctx(0, &usage, &[], Duration::ZERO, &[]);
        assert!(cond.should_stop(&ctx).is_some());
    }

    #[test]
    fn token_budget_does_not_fire_under_budget() {
        let cond = TokenBudget::new(100);
        let usage = Usage {
            input_tokens: 30,
            output_tokens: 20,
        };
        let ctx = make_ctx(0, &usage, &[], Duration::ZERO, &[]);
        assert!(cond.should_stop(&ctx).is_none());
    }

    #[test]
    fn max_consecutive_errors_fires() {
        let cond = MaxConsecutiveErrors::new(2);
        let steps = vec![
            AgentStep::ToolExecution {
                name: "t1".to_string(),
                input: serde_json::Value::Null,
                output: "ok".to_string(),
                is_error: false,
            },
            AgentStep::ToolExecution {
                name: "t2".to_string(),
                input: serde_json::Value::Null,
                output: "err".to_string(),
                is_error: true,
            },
            AgentStep::ToolExecution {
                name: "t3".to_string(),
                input: serde_json::Value::Null,
                output: "err".to_string(),
                is_error: true,
            },
        ];
        let usage = Usage::default();
        let ctx = make_ctx(0, &usage, &steps, Duration::ZERO, &[]);
        assert!(cond.should_stop(&ctx).is_some());
    }

    #[test]
    fn max_consecutive_errors_does_not_fire_with_success_between() {
        let cond = MaxConsecutiveErrors::new(2);
        let steps = vec![
            AgentStep::ToolExecution {
                name: "t1".to_string(),
                input: serde_json::Value::Null,
                output: "err".to_string(),
                is_error: true,
            },
            AgentStep::ToolExecution {
                name: "t2".to_string(),
                input: serde_json::Value::Null,
                output: "ok".to_string(),
                is_error: false,
            },
            AgentStep::ToolExecution {
                name: "t3".to_string(),
                input: serde_json::Value::Null,
                output: "err".to_string(),
                is_error: true,
            },
        ];
        let usage = Usage::default();
        let ctx = make_ctx(0, &usage, &steps, Duration::ZERO, &[]);
        assert!(cond.should_stop(&ctx).is_none());
    }

    #[test]
    fn custom_stop_fires() {
        let cond = CustomStop::new(|ctx: &StopContext| {
            if ctx.iteration >= 5 {
                Some("custom limit".to_string())
            } else {
                None
            }
        });
        let usage = Usage::default();
        let ctx = make_ctx(5, &usage, &[], Duration::ZERO, &[]);
        assert_eq!(cond.should_stop(&ctx), Some("custom limit".to_string()));
    }

    #[test]
    fn or_combinator_fires_if_either() {
        let cond = Timeout::new(Duration::from_secs(10)).or(TokenBudget::new(100));

        let usage = Usage::default();
        // Timeout fires, budget does not
        let ctx = make_ctx(0, &usage, &[], Duration::from_secs(15), &[]);
        assert!(cond.should_stop(&ctx).is_some());
        assert!(
            cond.should_stop(&ctx)
                .expect("should fire")
                .contains("timeout")
        );
    }

    #[test]
    fn or_combinator_does_not_fire_if_neither() {
        let cond = Timeout::new(Duration::from_secs(10)).or(TokenBudget::new(100));
        let usage = Usage::default();
        let ctx = make_ctx(0, &usage, &[], Duration::from_secs(5), &[]);
        assert!(cond.should_stop(&ctx).is_none());
    }

    #[test]
    fn and_combinator_fires_only_if_both() {
        let cond = Timeout::new(Duration::from_secs(10)).and(TokenBudget::new(100));

        let usage_over = Usage {
            input_tokens: 60,
            output_tokens: 50,
        };

        // Both fire
        let ctx = make_ctx(0, &usage_over, &[], Duration::from_secs(15), &[]);
        let result = cond.should_stop(&ctx);
        assert!(result.is_some());
        assert!(result.expect("should fire").contains("AND"));

        // Only timeout fires
        let usage_under = Usage::default();
        let ctx = make_ctx(0, &usage_under, &[], Duration::from_secs(15), &[]);
        assert!(cond.should_stop(&ctx).is_none());
    }

    #[test]
    fn custom_stop_inspects_messages() {
        let cond = CustomStop::new(|ctx: &StopContext| {
            ctx.messages
                .iter()
                .rev()
                .flat_map(|m| &m.content)
                .find_map(|c| match c {
                    crate::message::Content::ToolResult { output, .. }
                        if output.contains("all_done") =>
                    {
                        Some("all tasks completed".to_string())
                    }
                    _ => None,
                })
        });

        let messages = vec![Message {
            role: crate::message::Role::User,
            content: vec![crate::message::Content::ToolResult {
                id: "call_1".to_string(),
                output: "status: all_done".to_string(),
                is_error: false,
            }],
        }];
        let usage = Usage::default();
        let ctx = make_ctx(0, &usage, &[], Duration::ZERO, &messages);
        assert_eq!(
            cond.should_stop(&ctx),
            Some("all tasks completed".to_string())
        );
    }

    #[test]
    fn boxed_stop_condition_works() {
        let cond: Box<dyn StopCondition> = Box::new(Timeout::new(Duration::from_secs(1)));
        let usage = Usage::default();
        let ctx = make_ctx(0, &usage, &[], Duration::from_secs(5), &[]);
        assert!(cond.should_stop(&ctx).is_some());
    }
}
