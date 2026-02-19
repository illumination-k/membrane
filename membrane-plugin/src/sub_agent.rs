use std::path::Path;
use std::sync::Arc;

use membrane_core::sub_agent::{AgentExecutor, SubAgentEntry};

use crate::frontmatter::Frontmatter;

/// A custom subagent definition following the Claude Code subagent syntax.
///
/// Subagents are defined as markdown files in `.claude/agents/` with YAML
/// frontmatter. The markdown body becomes the subagent's system prompt.
///
/// ```text
/// ---
/// name: researcher
/// description: Research a topic thoroughly using read-only tools
/// tools: Read, Glob, Grep
/// model: haiku
/// maxTurns: 10
/// ---
///
/// You are a research agent. Search the codebase to answer questions.
/// Use Glob and Grep to find relevant files, then Read to examine them.
/// Provide detailed answers with file path references.
/// ```
///
/// # Frontmatter Fields
///
/// | Field | Description |
/// |:------|:------------|
/// | `name` | Unique identifier for this subagent |
/// | `description` | When Claude should delegate to this subagent |
/// | `tools` | Allowlist of tools available to the subagent |
/// | `disallowedTools` | Tools to deny from the subagent |
/// | `model` | Model to use (`sonnet`, `opus`, `haiku`, `inherit`) |
/// | `maxTurns` | Maximum agentic turns before stopping |
/// | `skills` | Skills to preload into the subagent's context |
///
/// # Programmatic Construction
///
/// ```
/// use membrane_plugin::SubAgentDefinition;
///
/// let def = SubAgentDefinition::new("researcher", "Research topics in depth")
///     .with_system_prompt("You are a research agent. Search thoroughly.")
///     .with_tools(vec!["Read", "Glob", "Grep"])
///     .with_model("haiku")
///     .with_max_turns(10);
/// ```
///
/// # Loading from File
///
/// ```no_run
/// use membrane_plugin::SubAgentDefinition;
///
/// let def = SubAgentDefinition::from_md("path/to/researcher.md")
///     .expect("failed to load subagent definition");
/// ```
pub struct SubAgentDefinition {
    name: String,
    description: String,
    tools: Vec<String>,
    disallowed_tools: Vec<String>,
    model: Option<String>,
    max_turns: Option<usize>,
    skills: Vec<String>,
    system_prompt: String,
}

impl SubAgentDefinition {
    /// Create a new subagent definition with the given name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            tools: Vec::new(),
            disallowed_tools: Vec::new(),
            model: None,
            max_turns: None,
            skills: Vec::new(),
            system_prompt: String::new(),
        }
    }

    /// Load a subagent definition from a markdown file.
    ///
    /// The file should contain YAML frontmatter followed by the system prompt
    /// as markdown content.
    pub fn from_md(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        Ok(Self::parse_md(&content, path))
    }

    /// Parse a subagent definition from markdown content with a file path
    /// for fallback name resolution.
    fn parse_md(content: &str, path: &Path) -> Self {
        let (fm, body) = Frontmatter::parse(content);

        let fallback_name = path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let name = fm.get("name").unwrap_or(fallback_name).to_string();
        let description = fm.get("description").unwrap_or("").to_string();

        let mut def = Self::new(name, description);
        def.system_prompt = body.trim_start_matches('\n').to_string();
        def.tools = fm.get_list("tools");
        def.disallowed_tools = fm.get_list("disallowedTools");
        def.model = fm.get("model").map(String::from);
        def.max_turns = fm.get_u64("maxTurns").map(|v| v as usize);
        def.skills = fm.get_list("skills");

        def
    }

    /// Parse a subagent definition from markdown content without a file path.
    pub fn parse(content: &str) -> Self {
        Self::parse_md(content, Path::new("unknown.md"))
    }

    // --- Builder methods ---

    /// Set the system prompt for this subagent.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the tool allowlist for this subagent.
    pub fn with_tools(mut self, tools: Vec<impl Into<String>>) -> Self {
        self.tools = tools.into_iter().map(|t| t.into()).collect();
        self
    }

    /// Set the tool denylist for this subagent.
    pub fn with_disallowed_tools(mut self, tools: Vec<impl Into<String>>) -> Self {
        self.disallowed_tools = tools.into_iter().map(|t| t.into()).collect();
        self
    }

    /// Set the model for this subagent.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the maximum number of agentic turns.
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    /// Set the skills to preload into this subagent's context.
    pub fn with_skills(mut self, skills: Vec<impl Into<String>>) -> Self {
        self.skills = skills.into_iter().map(|s| s.into()).collect();
        self
    }

    // --- Conversion to SubAgentEntry ---

    /// Convert this definition into a `SubAgentEntry` for use with
    /// [`Agent::with_sub_agent`](membrane_core::agent::Agent::with_sub_agent).
    ///
    /// The caller provides an `AgentExecutor` implementation (typically an
    /// `Agent<P>` instance) that has been configured with the appropriate
    /// provider and tools based on this definition's fields.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use membrane_plugin::SubAgentDefinition;
    /// use membrane_core::agent::{Agent, AgentConfig};
    ///
    /// let def = SubAgentDefinition::parse("---\nname: researcher\ndescription: Research topics\n---\nResearch thoroughly.");
    ///
    /// // Create an agent configured according to the definition
    /// // let agent = Agent::with_system_prompt(provider, tools, config, def.system_prompt());
    /// // let entry = def.into_entry(agent);
    /// ```
    pub fn into_entry(self, executor: impl AgentExecutor + 'static) -> SubAgentEntry {
        SubAgentEntry::single(self.name, self.description, Arc::new(executor))
    }

    /// Convert this definition into a parallel (fan-out) `SubAgentEntry`.
    ///
    /// The subagent will accept `{ "tasks": [string] }` input and execute
    /// each task concurrently.
    pub fn into_parallel_entry(self, executor: impl AgentExecutor + 'static) -> SubAgentEntry {
        SubAgentEntry::parallel(self.name, self.description, Arc::new(executor))
    }

    // --- Accessors ---

    /// The subagent's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The subagent's description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The tool allowlist.
    pub fn tools(&self) -> &[String] {
        &self.tools
    }

    /// The tool denylist.
    pub fn disallowed_tools(&self) -> &[String] {
        &self.disallowed_tools
    }

    /// The model override.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The maximum number of agentic turns.
    pub fn max_turns(&self) -> Option<usize> {
        self.max_turns
    }

    /// Skills to preload into this subagent's context.
    pub fn skills(&self) -> &[String] {
        &self.skills
    }

    /// The subagent's system prompt (markdown body).
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Build an `AgentConfig` from this definition's fields.
    ///
    /// Uses the definition's `model` and `max_turns` if set, falling back
    /// to the provided defaults.
    pub fn build_config(
        &self,
        default_model: &str,
        default_max_iterations: usize,
    ) -> membrane_core::agent::AgentConfig {
        membrane_core::agent::AgentConfig {
            model: self
                .model
                .clone()
                .unwrap_or_else(|| default_model.to_string()),
            max_iterations: self.max_turns.unwrap_or(default_max_iterations),
            extra_params: serde_json::Map::new(),
        }
    }

    /// Check if a tool name is allowed by this definition's filter rules.
    ///
    /// - If `tools` is non-empty, only those tools are allowed (allowlist).
    /// - If `disallowed_tools` is non-empty, those tools are denied (denylist).
    /// - If both are empty, all tools are allowed.
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        if !self.tools.is_empty() {
            return self.tools.iter().any(|t| t == tool_name);
        }
        if !self.disallowed_tools.is_empty() {
            return !self.disallowed_tools.iter().any(|t| t == tool_name);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_frontmatter() {
        let content = r#"---
name: researcher
description: Research topics thoroughly
tools: Read, Glob, Grep
disallowedTools: Write, Edit
model: haiku
maxTurns: 15
skills: code-review, testing
---

You are a research agent.
Search the codebase to answer questions."#;

        let def = SubAgentDefinition::parse(content);

        assert_eq!(def.name(), "researcher");
        assert_eq!(def.description(), "Research topics thoroughly");
        assert_eq!(def.tools(), &["Read", "Glob", "Grep"]);
        assert_eq!(def.disallowed_tools(), &["Write", "Edit"]);
        assert_eq!(def.model(), Some("haiku"));
        assert_eq!(def.max_turns(), Some(15));
        assert_eq!(def.skills(), &["code-review", "testing"]);
        assert_eq!(
            def.system_prompt(),
            "You are a research agent.\nSearch the codebase to answer questions."
        );
    }

    #[test]
    fn parse_minimal() {
        let content = "Just a system prompt, no frontmatter.";
        let def = SubAgentDefinition::parse(content);

        assert_eq!(def.name(), "unknown");
        assert_eq!(def.description(), "");
        assert_eq!(def.system_prompt(), "Just a system prompt, no frontmatter.");
        assert!(def.tools().is_empty());
        assert!(def.disallowed_tools().is_empty());
        assert!(def.model().is_none());
        assert!(def.max_turns().is_none());
        assert!(def.skills().is_empty());
    }

    #[test]
    fn parse_name_from_filename() {
        let content = "---\ndescription: A subagent\n---\n\nSystem prompt";
        let def = SubAgentDefinition::parse_md(content, Path::new("/agents/my-agent.md"));

        assert_eq!(def.name(), "my-agent");
    }

    #[test]
    fn builder_methods() {
        let def = SubAgentDefinition::new("test-agent", "Test agent description")
            .with_system_prompt("You are a test agent.")
            .with_tools(vec!["Read", "Grep"])
            .with_disallowed_tools(vec!["Write"])
            .with_model("sonnet")
            .with_max_turns(20)
            .with_skills(vec!["review"]);

        assert_eq!(def.name(), "test-agent");
        assert_eq!(def.description(), "Test agent description");
        assert_eq!(def.system_prompt(), "You are a test agent.");
        assert_eq!(def.tools(), &["Read", "Grep"]);
        assert_eq!(def.disallowed_tools(), &["Write"]);
        assert_eq!(def.model(), Some("sonnet"));
        assert_eq!(def.max_turns(), Some(20));
        assert_eq!(def.skills(), &["review"]);
    }

    #[test]
    fn build_config_with_defaults() {
        let def = SubAgentDefinition::new("test", "test");
        let config = def.build_config("gpt-4", 10);

        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.max_iterations, 10);
    }

    #[test]
    fn build_config_with_overrides() {
        let def = SubAgentDefinition::new("test", "test")
            .with_model("haiku")
            .with_max_turns(5);
        let config = def.build_config("gpt-4", 10);

        assert_eq!(config.model, "haiku");
        assert_eq!(config.max_iterations, 5);
    }

    #[test]
    fn is_tool_allowed_allowlist() {
        let def = SubAgentDefinition::new("test", "test").with_tools(vec!["Read", "Grep"]);

        assert!(def.is_tool_allowed("Read"));
        assert!(def.is_tool_allowed("Grep"));
        assert!(!def.is_tool_allowed("Write"));
    }

    #[test]
    fn is_tool_allowed_denylist() {
        let def =
            SubAgentDefinition::new("test", "test").with_disallowed_tools(vec!["Write", "Edit"]);

        assert!(def.is_tool_allowed("Read"));
        assert!(def.is_tool_allowed("Grep"));
        assert!(!def.is_tool_allowed("Write"));
        assert!(!def.is_tool_allowed("Edit"));
    }

    #[test]
    fn is_tool_allowed_no_filter() {
        let def = SubAgentDefinition::new("test", "test");

        assert!(def.is_tool_allowed("Read"));
        assert!(def.is_tool_allowed("Write"));
        assert!(def.is_tool_allowed("anything"));
    }

    #[test]
    fn parse_yaml_sequence_tools() {
        let content = "---\ntools:\n  - Read\n  - Glob\n  - Grep\n---\nPrompt";
        let def = SubAgentDefinition::parse(content);

        assert_eq!(def.tools(), &["Read", "Glob", "Grep"]);
    }
}
