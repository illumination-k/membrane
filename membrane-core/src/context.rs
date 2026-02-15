use crate::message::Message;

/// Controls how conversation context is built before each LLM call.
///
/// The `ContextBuilder` is responsible for assembling the message sequence
/// that will be sent to the LLM. This includes system prompts, conversation
/// history, and any dynamic context injection.
pub trait ContextBuilder: Send + Sync {
    /// Build the initial conversation context before the first LLM call.
    ///
    /// Called once at the start of the ReAct loop with the user's input messages.
    /// Should return the complete message sequence to send to the LLM,
    /// including any system prompts or preambles.
    fn build_initial(&self, messages: Vec<Message>) -> Vec<Message>;

    /// Build the conversation context before each subsequent LLM call.
    ///
    /// Called at the start of each iteration after the first. Receives the
    /// current full conversation state (including all previous assistant
    /// responses and tool results) and returns the messages to send to the LLM.
    ///
    /// Common use cases:
    /// - Trimming old messages to fit context window
    /// - Compressing conversation history
    /// - Injecting dynamic context based on iteration state
    ///
    /// The default implementation returns the conversation unchanged.
    fn build_iteration(&self, conversation: &[Message], _iteration: usize) -> Vec<Message> {
        conversation.to_vec()
    }
}

/// Simple context builder that prepends a system prompt.
///
/// This is the default implementation that prepends a system message
/// (if provided) to the user's messages and passes conversation through
/// unchanged on subsequent iterations.
#[derive(Debug, Clone, Default)]
pub struct DefaultContextBuilder {
    system_prompt: Option<String>,
}

impl DefaultContextBuilder {
    /// Create a new builder with the given system prompt.
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: Some(system_prompt.into()),
        }
    }
}

impl ContextBuilder for DefaultContextBuilder {
    fn build_initial(&self, messages: Vec<Message>) -> Vec<Message> {
        match &self.system_prompt {
            Some(prompt) => {
                let mut result = Vec::with_capacity(messages.len() + 1);
                result.push(Message::system(prompt));
                result.extend(messages);
                result
            }
            None => messages,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Role;

    #[test]
    fn default_with_system_prompt() {
        let builder = DefaultContextBuilder::new("You are helpful.");
        let messages = vec![Message::user("Hello")];
        let result = builder.build_initial(messages);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].role, Role::User);
    }

    #[test]
    fn default_without_system_prompt() {
        let builder = DefaultContextBuilder::default();
        let messages = vec![Message::user("Hello")];
        let result = builder.build_initial(messages);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, Role::User);
    }

    #[test]
    fn default_build_iteration_returns_unchanged() {
        let builder = DefaultContextBuilder::new("System");
        let conversation = vec![
            Message::system("System"),
            Message::user("Hi"),
            Message::assistant("Hello"),
        ];

        let result = builder.build_iteration(&conversation, 1);
        assert_eq!(result.len(), conversation.len());
    }

    #[test]
    fn custom_trimmer_builder() {
        struct TrimmerBuilder {
            system_prompt: String,
            max_messages: usize,
        }

        impl ContextBuilder for TrimmerBuilder {
            fn build_initial(&self, messages: Vec<Message>) -> Vec<Message> {
                let mut result = vec![Message::system(&self.system_prompt)];
                result.extend(messages);
                result
            }

            fn build_iteration(&self, conversation: &[Message], _iteration: usize) -> Vec<Message> {
                let mut result = Vec::new();

                // Keep system message
                if let Some(sys) = conversation.iter().find(|m| m.role == Role::System) {
                    result.push(sys.clone());
                }

                // Take last max_messages non-system messages
                let non_system: Vec<_> = conversation
                    .iter()
                    .filter(|m| m.role != Role::System)
                    .collect();
                let start = non_system.len().saturating_sub(self.max_messages);
                for msg in &non_system[start..] {
                    result.push((*msg).clone());
                }

                result
            }
        }

        let builder = TrimmerBuilder {
            system_prompt: "System".to_string(),
            max_messages: 2,
        };

        let conversation = vec![
            Message::system("System"),
            Message::user("1"),
            Message::assistant("2"),
            Message::user("3"),
            Message::assistant("4"),
        ];

        let trimmed = builder.build_iteration(&conversation, 2);
        // system + last 2 non-system messages
        assert_eq!(trimmed.len(), 3);
        assert_eq!(trimmed[0].role, Role::System);
    }
}
