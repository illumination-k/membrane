use membrane_core::tool::Tool;

/// A single skill within a [`SkillsPlugin`](crate::SkillsPlugin).
///
/// Each skill has a name, description, instructions (context injected as
/// system messages), and an optional set of tools. Skills follow the
/// Claude Code skills pattern where each skill bundles related instructions
/// and capabilities.
///
/// # Example
///
/// ```
/// use membrane_plugin::Skill;
///
/// let skill = Skill::new("code_review", "Reviews code for quality and correctness")
///     .with_instructions("You are a code reviewer. Focus on correctness, readability, and performance.");
/// ```
pub struct Skill {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) instructions: String,
    pub(crate) tools: Vec<Box<dyn Tool>>,
}

impl Skill {
    /// Create a new skill with the given name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            instructions: String::new(),
            tools: Vec::new(),
        }
    }

    /// Set the instructions for this skill.
    ///
    /// Instructions are injected as system-level context when the plugin
    /// is registered with an agent.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Add a tool to this skill.
    pub fn with_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// The skill's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The skill's description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The skill's instructions.
    pub fn instructions(&self) -> &str {
        &self.instructions
    }
}
