use std::path::Path;

use membrane_core::tool::Tool;

use crate::frontmatter::Frontmatter;

/// Execution context for a skill.
///
/// When set to `Fork`, the skill runs in an isolated subagent context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillContext {
    /// Run in a forked subagent context.
    Fork,
}

/// A single skill following the Agent Skills open standard.
///
/// Each skill corresponds to a `SKILL.md` file with YAML frontmatter:
///
/// ```text
/// ---
/// name: code-review
/// description: Reviews code for quality and correctness
/// allowed-tools: Read, Grep, Glob
/// user-invocable: true
/// ---
///
/// Review code for correctness, readability, and performance.
/// ```
///
/// # Frontmatter Fields
///
/// | Field | Description |
/// |:------|:------------|
/// | `name` | Skill name (lowercase, hyphens, max 64 chars) |
/// | `description` | What the skill does and when to use it |
/// | `allowed-tools` | Comma-separated tool names pre-approved for this skill |
/// | `user-invocable` | Whether users can invoke this skill (default: true) |
/// | `disable-model-invocation` | Prevent auto-invocation by the model (default: false) |
/// | `argument-hint` | Hint shown during autocomplete (e.g. `[filename]`) |
/// | `model` | Model override when this skill is active |
/// | `context` | Set to `fork` for subagent execution |
/// | `agent` | Subagent type when `context: fork` |
///
/// # Programmatic Construction
///
/// ```
/// use membrane_plugin::Skill;
///
/// let skill = Skill::new("code-review", "Reviews code for quality")
///     .with_instructions("Review code for correctness and style.")
///     .with_allowed_tools(vec!["Read", "Grep"]);
/// ```
///
/// # Loading from SKILL.md
///
/// ```no_run
/// use membrane_plugin::Skill;
///
/// let skill = Skill::from_skill_md("path/to/code-review/SKILL.md")
///     .expect("failed to load skill");
/// ```
pub struct Skill {
    // --- Agent Skills spec fields ---
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) allowed_tools: Vec<String>,

    // --- Claude Code extension fields ---
    pub(crate) user_invocable: bool,
    pub(crate) disable_model_invocation: bool,
    pub(crate) argument_hint: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) context: Option<SkillContext>,
    pub(crate) agent: Option<String>,

    // --- Content ---
    pub(crate) instructions: String,
    pub(crate) tools: Vec<Box<dyn Tool>>,
}

impl Skill {
    /// Create a new skill with the given name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            allowed_tools: Vec::new(),
            user_invocable: true,
            disable_model_invocation: false,
            argument_hint: None,
            model: None,
            context: None,
            agent: None,
            instructions: String::new(),
            tools: Vec::new(),
        }
    }

    /// Load a skill from a `SKILL.md` file.
    ///
    /// Parses YAML frontmatter for metadata and uses the remaining
    /// markdown content as instructions.
    pub fn from_skill_md(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        Ok(Self::parse_skill_md(&content, path))
    }

    /// Parse a SKILL.md content string into a `Skill`.
    ///
    /// If `path` is provided, the parent directory name is used as a
    /// fallback for the skill name when not specified in frontmatter.
    fn parse_skill_md(content: &str, path: &Path) -> Self {
        let (fm, body) = Frontmatter::parse(content);

        let fallback_name = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let name = fm.get("name").unwrap_or(fallback_name).to_string();

        let description = fm.get("description").unwrap_or("").to_string();

        let mut skill = Self::new(name, description);
        skill.instructions = body.trim_start_matches('\n').to_string();
        skill.allowed_tools = fm.get_list("allowed-tools");
        skill.user_invocable = fm.get_bool("user-invocable").unwrap_or(true);
        skill.disable_model_invocation = fm.get_bool("disable-model-invocation").unwrap_or(false);
        skill.argument_hint = fm.get("argument-hint").map(String::from);
        skill.model = fm.get("model").map(String::from);
        skill.agent = fm.get("agent").map(String::from);

        if fm.get("context") == Some("fork") {
            skill.context = Some(SkillContext::Fork);
        }

        skill
    }

    /// Parse a SKILL.md content string without a file path.
    pub fn parse(content: &str) -> Self {
        Self::parse_skill_md(content, Path::new("unknown/SKILL.md"))
    }

    // --- Builder methods ---

    /// Set the instructions (markdown body) for this skill.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Add a tool to this skill.
    pub fn with_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Set the allowed tools for this skill.
    pub fn with_allowed_tools(mut self, tools: Vec<impl Into<String>>) -> Self {
        self.allowed_tools = tools.into_iter().map(|t| t.into()).collect();
        self
    }

    /// Set whether users can invoke this skill.
    pub fn with_user_invocable(mut self, user_invocable: bool) -> Self {
        self.user_invocable = user_invocable;
        self
    }

    /// Set whether the model is prevented from auto-invoking this skill.
    pub fn with_disable_model_invocation(mut self, disable: bool) -> Self {
        self.disable_model_invocation = disable;
        self
    }

    /// Set the argument hint for autocomplete.
    pub fn with_argument_hint(mut self, hint: impl Into<String>) -> Self {
        self.argument_hint = Some(hint.into());
        self
    }

    /// Set the model override for this skill.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the execution context to fork (subagent).
    pub fn with_fork_context(mut self, agent: impl Into<String>) -> Self {
        self.context = Some(SkillContext::Fork);
        self.agent = Some(agent.into());
        self
    }

    // --- Argument substitution ---

    /// Render skill instructions with argument substitution.
    ///
    /// Supports the following substitutions:
    /// - `$ARGUMENTS` — all arguments joined by space
    /// - `$ARGUMENTS[N]` — argument at index N (0-based)
    /// - `$N` — shorthand for `$ARGUMENTS[N]`
    ///
    /// If `$ARGUMENTS` does not appear in the instructions, the arguments
    /// are appended as `ARGUMENTS: <value>` (Claude Code convention).
    pub fn invoke(&self, args: &[&str]) -> String {
        let joined = args.join(" ");
        let mut result = self.instructions.clone();

        // Check for any argument placeholder ($ARGUMENTS, $ARGUMENTS[N], $N)
        let has_placeholder = result.contains("$ARGUMENTS")
            || args
                .iter()
                .enumerate()
                .any(|(i, _)| result.contains(&format!("${}", i)));

        // Replace indexed forms first: $ARGUMENTS[N] and $N
        for (i, arg) in args.iter().enumerate() {
            let indexed = format!("$ARGUMENTS[{}]", i);
            result = result.replace(&indexed, arg);

            let shorthand = format!("${}", i);
            result = result.replace(&shorthand, arg);
        }

        // Replace $ARGUMENTS with all args joined
        result = result.replace("$ARGUMENTS", &joined);

        // If no placeholder was present, append (Claude Code convention)
        if !has_placeholder && !joined.is_empty() {
            result.push_str(&format!("\n\nARGUMENTS: {}", joined));
        }

        result
    }

    // --- Accessors ---

    /// The skill's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The skill's description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The skill's instructions (markdown body).
    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    /// Tool names pre-approved for this skill.
    pub fn allowed_tools(&self) -> &[String] {
        &self.allowed_tools
    }

    /// Whether users can invoke this skill.
    pub fn user_invocable(&self) -> bool {
        self.user_invocable
    }

    /// Whether the model is prevented from auto-invoking this skill.
    pub fn disable_model_invocation(&self) -> bool {
        self.disable_model_invocation
    }

    /// Argument hint for autocomplete.
    pub fn argument_hint(&self) -> Option<&str> {
        self.argument_hint.as_deref()
    }

    /// Model override for this skill.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Execution context (Fork for subagent).
    pub fn context_mode(&self) -> Option<&SkillContext> {
        self.context.as_ref()
    }

    /// Subagent type when context is Fork.
    pub fn agent(&self) -> Option<&str> {
        self.agent.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skill_md_full_frontmatter() {
        let content = "\
---
name: code-review
description: Reviews code for quality
allowed-tools: Read, Grep, Glob
user-invocable: true
disable-model-invocation: false
argument-hint: [filename]
model: claude-sonnet-4-5-20250929
context: fork
agent: Explore
---

Review the code for correctness.";

        let skill = Skill::parse(content);

        assert_eq!(skill.name(), "code-review");
        assert_eq!(skill.description(), "Reviews code for quality");
        assert_eq!(skill.allowed_tools(), &["Read", "Grep", "Glob"]);
        assert!(skill.user_invocable());
        assert!(!skill.disable_model_invocation());
        assert_eq!(skill.argument_hint(), Some("[filename]"));
        assert_eq!(skill.model(), Some("claude-sonnet-4-5-20250929"));
        assert_eq!(skill.context_mode(), Some(&SkillContext::Fork));
        assert_eq!(skill.agent(), Some("Explore"));
        assert_eq!(skill.instructions(), "Review the code for correctness.");
    }

    #[test]
    fn parse_skill_md_minimal() {
        let content = "Just instructions, no frontmatter.";
        let skill = Skill::parse(content);

        assert_eq!(skill.name(), "unknown");
        assert_eq!(skill.description(), "");
        assert_eq!(skill.instructions(), "Just instructions, no frontmatter.");
        assert!(skill.user_invocable());
        assert!(!skill.disable_model_invocation());
    }

    #[test]
    fn parse_skill_md_name_from_directory() {
        let content = "---\ndescription: A skill\n---\n\nInstructions";
        let skill = Skill::parse_skill_md(content, Path::new("/skills/my-skill/SKILL.md"));

        assert_eq!(skill.name(), "my-skill");
    }

    #[test]
    fn invoke_with_arguments_placeholder() {
        let skill = Skill::new("test", "").with_instructions("Review $ARGUMENTS carefully.");
        let result = skill.invoke(&["file.rs", "main.rs"]);
        assert_eq!(result, "Review file.rs main.rs carefully.");
    }

    #[test]
    fn invoke_with_indexed_arguments() {
        let skill =
            Skill::new("test", "").with_instructions("Compare $ARGUMENTS[0] with $ARGUMENTS[1].");
        let result = skill.invoke(&["old.rs", "new.rs"]);
        assert_eq!(result, "Compare old.rs with new.rs.");
    }

    #[test]
    fn invoke_with_shorthand() {
        let skill = Skill::new("test", "").with_instructions("File: $0, Format: $1");
        let result = skill.invoke(&["main.rs", "json"]);
        assert_eq!(result, "File: main.rs, Format: json");
    }

    #[test]
    fn invoke_appends_when_no_placeholder() {
        let skill = Skill::new("test", "").with_instructions("Do the task.");
        let result = skill.invoke(&["arg1", "arg2"]);
        assert_eq!(result, "Do the task.\n\nARGUMENTS: arg1 arg2");
    }

    #[test]
    fn invoke_no_args_no_append() {
        let skill = Skill::new("test", "").with_instructions("Do the task.");
        let result = skill.invoke(&[]);
        assert_eq!(result, "Do the task.");
    }

    #[test]
    fn builder_methods() {
        let skill = Skill::new("test", "desc")
            .with_instructions("instructions")
            .with_allowed_tools(vec!["Read", "Grep"])
            .with_user_invocable(false)
            .with_disable_model_invocation(true)
            .with_argument_hint("[file]")
            .with_model("gpt-4")
            .with_fork_context("Explore");

        assert_eq!(skill.name(), "test");
        assert_eq!(skill.description(), "desc");
        assert_eq!(skill.instructions(), "instructions");
        assert_eq!(skill.allowed_tools(), &["Read", "Grep"]);
        assert!(!skill.user_invocable());
        assert!(skill.disable_model_invocation());
        assert_eq!(skill.argument_hint(), Some("[file]"));
        assert_eq!(skill.model(), Some("gpt-4"));
        assert_eq!(skill.context_mode(), Some(&SkillContext::Fork));
        assert_eq!(skill.agent(), Some("Explore"));
    }
}
