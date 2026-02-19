use crate::context::ContextBuilder;
use crate::message::{Message, Role};
use crate::tool::Tool;

/// A plugin bundles tools and context for the agent.
///
/// Plugins follow the skills pattern: each plugin provides a set of tools
/// (deterministic functions the LLM can invoke) and context strings
/// (instructions injected as system messages).
///
/// # Tools
///
/// `tools()` uses `&mut self` so implementations can drain internal storage,
/// transferring ownership of tools to the agent. This is called once when
/// the plugin is registered via [`Agent::with_plugin`](crate::agent::Agent::with_plugin).
///
/// # Context
///
/// `context()` returns instruction strings that are injected as system
/// messages into the conversation. These are inserted after any existing
/// system messages during `build_initial`.
pub trait Plugin: Send + Sync {
    /// The plugin's unique name.
    fn name(&self) -> &str;

    /// A short description of the plugin's purpose.
    fn description(&self) -> &str;

    /// Return tools provided by this plugin, draining internal storage.
    fn tools(&mut self) -> Vec<Box<dyn Tool>>;

    /// Return context strings to inject as system messages.
    fn context(&self) -> Vec<String>;
}

/// Internal context builder wrapper that injects plugin context.
///
/// Wraps an existing `ContextBuilder` and prepends plugin context strings
/// as system messages after any existing system messages from the inner builder.
pub(crate) struct PluginContextBuilder {
    pub(crate) inner: Box<dyn ContextBuilder>,
    pub(crate) plugin_contexts: Vec<String>,
}

impl ContextBuilder for PluginContextBuilder {
    fn build_initial(&self, messages: Vec<Message>) -> Vec<Message> {
        let mut result = self.inner.build_initial(messages);

        // Find the position after existing system messages
        let insert_pos = result
            .iter()
            .position(|m| m.role != Role::System)
            .unwrap_or(result.len());

        for (i, ctx) in self.plugin_contexts.iter().enumerate() {
            result.insert(insert_pos + i, Message::system(ctx));
        }

        result
    }

    fn build_iteration(&self, conversation: &[Message], iteration: usize) -> Vec<Message> {
        self.inner.build_iteration(conversation, iteration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DefaultContextBuilder;

    #[test]
    fn plugin_context_builder_injects_after_system() {
        let inner = DefaultContextBuilder::new("System prompt");
        let builder = PluginContextBuilder {
            inner: Box::new(inner),
            plugin_contexts: vec![
                "Plugin A context".to_string(),
                "Plugin B context".to_string(),
            ],
        };

        let messages = vec![Message::user("Hello")];
        let result = builder.build_initial(messages);

        assert_eq!(result.len(), 4);
        assert_eq!(result[0].role, Role::System); // Original system prompt
        assert_eq!(result[1].role, Role::System); // Plugin A context
        assert_eq!(result[2].role, Role::System); // Plugin B context
        assert_eq!(result[3].role, Role::User); // User message
    }

    #[test]
    fn plugin_context_builder_no_system_prompt() {
        let inner = DefaultContextBuilder::default();
        let builder = PluginContextBuilder {
            inner: Box::new(inner),
            plugin_contexts: vec!["Plugin context".to_string()],
        };

        let messages = vec![Message::user("Hello")];
        let result = builder.build_initial(messages);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].role, Role::System); // Plugin context
        assert_eq!(result[1].role, Role::User); // User message
    }

    #[test]
    fn plugin_context_builder_iteration_delegates() {
        let inner = DefaultContextBuilder::new("System");
        let builder = PluginContextBuilder {
            inner: Box::new(inner),
            plugin_contexts: vec!["Plugin".to_string()],
        };

        let conversation = vec![
            Message::system("System"),
            Message::system("Plugin"),
            Message::user("Hi"),
            Message::assistant("Hello"),
        ];

        let result = builder.build_iteration(&conversation, 1);
        assert_eq!(result.len(), conversation.len());
    }
}
