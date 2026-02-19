use std::path::Path;

use membrane_core::plugin::Plugin;
use membrane_core::tool::Tool;

use crate::skill::Skill;

/// A plugin composed of multiple skills.
///
/// `SkillsPlugin` follows the Claude Code skills pattern: each skill
/// bundles instructions (context) and tools. When registered with an
/// agent via `with_plugin()`, all skill tools are added to the agent
/// and all skill instructions are injected as system messages.
///
/// # Programmatic Construction
///
/// ```
/// use membrane_plugin::{Skill, SkillsPlugin};
///
/// let plugin = SkillsPlugin::new("my_skills", "Custom skills for my agent")
///     .with_skill(
///         Skill::new("code_review", "Reviews code")
///             .with_instructions("Review code for correctness and style.")
///     )
///     .with_skill(
///         Skill::new("testing", "Generates tests")
///             .with_instructions("Generate comprehensive unit tests.")
///     );
/// ```
///
/// # Loading from Directory
///
/// Skills instructions can be loaded from `.md` files in a directory,
/// following the Claude Code convention (`.claude/skills/` directory):
///
/// ```no_run
/// use membrane_plugin::SkillsPlugin;
///
/// let plugin = SkillsPlugin::new("project_skills", "Project-specific skills")
///     .load_skills_dir(".claude/skills")
///     .expect("failed to load skills directory");
/// ```
pub struct SkillsPlugin {
    name: String,
    description: String,
    skills: Vec<Skill>,
}

impl SkillsPlugin {
    /// Create a new empty skills plugin.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            skills: Vec::new(),
        }
    }

    /// Add a skill to this plugin.
    pub fn with_skill(mut self, skill: Skill) -> Self {
        self.skills.push(skill);
        self
    }

    /// Load skill instructions from `.md` files in a directory.
    ///
    /// Each `.md` file becomes a skill whose name is the file stem
    /// (e.g., `code_review.md` → skill name `code_review`).
    /// The file content is used as the skill's instructions.
    ///
    /// Tools cannot be loaded from files; use [`Skill::with_tool`] to
    /// attach tools to specific skills after loading.
    pub fn load_skills_dir(mut self, dir: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let dir = dir.as_ref();

        let mut entries: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .map(|ext| ext == "md")
                    .unwrap_or(false)
            })
            .collect();

        // Sort by filename for deterministic ordering
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            let instructions = std::fs::read_to_string(&path)?;

            self.skills.push(Skill {
                name,
                description: String::new(),
                instructions,
                tools: Vec::new(),
            });
        }

        Ok(self)
    }

    /// Return a reference to the skills in this plugin.
    pub fn skills(&self) -> &[Skill] {
        &self.skills
    }

    /// Return a mutable reference to a skill by name.
    ///
    /// Useful for attaching tools to skills loaded from files:
    ///
    /// ```no_run
    /// use membrane_plugin::SkillsPlugin;
    ///
    /// let mut plugin = SkillsPlugin::new("project", "Project skills")
    ///     .load_skills_dir(".claude/skills")
    ///     .expect("failed to load");
    ///
    /// // Attach tools to a specific skill after loading
    /// // if let Some(skill) = plugin.skill_mut("code_review") {
    /// //     skill.with_tool(my_tool);
    /// // }
    /// ```
    pub fn skill_mut(&mut self, name: &str) -> Option<&mut Skill> {
        self.skills.iter_mut().find(|s| s.name == name)
    }
}

impl Plugin for SkillsPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn tools(&mut self) -> Vec<Box<dyn Tool>> {
        self.skills
            .iter_mut()
            .flat_map(|s| s.tools.drain(..))
            .collect()
    }

    fn context(&self) -> Vec<String> {
        self.skills
            .iter()
            .filter(|s| !s.instructions.is_empty())
            .map(|s| {
                if s.description.is_empty() {
                    format!("## Skill: {}\n\n{}", s.name, s.instructions)
                } else {
                    format!(
                        "## Skill: {} — {}\n\n{}",
                        s.name, s.description, s.instructions
                    )
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrane_core::error::Error;
    use membrane_core::tool::ToolDefinition;
    use std::future::Future;
    use std::pin::Pin;

    struct DummyTool {
        tool_name: String,
    }

    impl DummyTool {
        fn new(name: &str) -> Self {
            Self {
                tool_name: name.to_string(),
            }
        }
    }

    impl Tool for DummyTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.tool_name.clone(),
                description: format!("Dummy tool: {}", self.tool_name),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        fn execute(
            &self,
            _input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
            Box::pin(async { Ok("ok".to_string()) })
        }
    }

    #[test]
    fn skills_plugin_collects_tools() {
        let mut plugin = SkillsPlugin::new("test", "Test plugin")
            .with_skill(
                Skill::new("s1", "Skill 1")
                    .with_tool(DummyTool::new("tool_a"))
                    .with_tool(DummyTool::new("tool_b")),
            )
            .with_skill(Skill::new("s2", "Skill 2").with_tool(DummyTool::new("tool_c")));

        let tools = plugin.tools();
        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        assert_eq!(names, vec!["tool_a", "tool_b", "tool_c"]);
    }

    #[test]
    fn skills_plugin_tools_drain() {
        let mut plugin = SkillsPlugin::new("test", "Test")
            .with_skill(Skill::new("s1", "Skill 1").with_tool(DummyTool::new("tool_a")));

        let tools = plugin.tools();
        assert_eq!(tools.len(), 1);

        // Second call returns empty (drained)
        let tools = plugin.tools();
        assert!(tools.is_empty());
    }

    #[test]
    fn skills_plugin_collects_context() {
        let plugin = SkillsPlugin::new("test", "Test plugin")
            .with_skill(Skill::new("s1", "Skill 1").with_instructions("Instructions for skill 1."))
            .with_skill(Skill::new("s2", "Skill 2"))
            .with_skill(Skill::new("s3", "Skill 3").with_instructions("Instructions for skill 3."));

        let context = plugin.context();
        assert_eq!(context.len(), 2); // s2 has no instructions, excluded

        assert!(context[0].contains("Skill: s1"));
        assert!(context[0].contains("Skill 1"));
        assert!(context[0].contains("Instructions for skill 1."));

        assert!(context[1].contains("Skill: s3"));
        assert!(context[1].contains("Instructions for skill 3."));
    }

    #[test]
    fn skills_plugin_context_without_description() {
        let plugin = SkillsPlugin::new("test", "Test")
            .with_skill(Skill::new("s1", "").with_instructions("Do something."));

        let context = plugin.context();
        assert_eq!(context.len(), 1);
        assert_eq!(context[0], "## Skill: s1\n\nDo something.");
    }

    #[test]
    fn skills_plugin_load_dir() {
        let dir = std::env::temp_dir().join("membrane_test_skills");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create test dir");

        std::fs::write(dir.join("alpha.md"), "Alpha instructions").expect("write");
        std::fs::write(dir.join("beta.md"), "Beta instructions").expect("write");
        std::fs::write(dir.join("not_a_skill.txt"), "Ignored").expect("write");

        let plugin = SkillsPlugin::new("test", "Test")
            .load_skills_dir(&dir)
            .expect("load dir");

        assert_eq!(plugin.skills().len(), 2);

        let names: Vec<&str> = plugin.skills().iter().map(|s| s.name()).collect();
        assert_eq!(names, vec!["alpha", "beta"]); // sorted

        let context = plugin.context();
        assert_eq!(context.len(), 2);
        assert!(context[0].contains("Alpha instructions"));
        assert!(context[1].contains("Beta instructions"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skills_plugin_load_dir_nonexistent() {
        let result =
            SkillsPlugin::new("test", "Test").load_skills_dir("/nonexistent/path/to/skills");
        assert!(result.is_err());
    }

    #[test]
    fn skills_plugin_metadata() {
        let plugin = SkillsPlugin::new("my_plugin", "My plugin description");
        assert_eq!(plugin.name(), "my_plugin");
        assert_eq!(plugin.description(), "My plugin description");
    }

    #[test]
    fn skill_mut_finds_by_name() {
        let mut plugin = SkillsPlugin::new("test", "Test")
            .with_skill(Skill::new("s1", "Skill 1"))
            .with_skill(Skill::new("s2", "Skill 2"));

        assert!(plugin.skill_mut("s1").is_some());
        assert!(plugin.skill_mut("s2").is_some());
        assert!(plugin.skill_mut("s3").is_none());
    }
}
