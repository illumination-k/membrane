use std::path::Path;

use membrane_core::plugin::Plugin;
use membrane_core::tool::Tool;

use crate::skill::Skill;

/// A plugin composed of multiple skills following the Agent Skills open standard.
///
/// `SkillsPlugin` bundles skills that each provide instructions (context)
/// and tools. When registered with an agent via `with_plugin()`:
///
/// - **Tools** from all skills are added to the agent's tool set.
/// - **Context** follows progressive disclosure: skill descriptions are
///   always loaded; full instructions are available via [`Skill::invoke`].
///
/// # Programmatic Construction
///
/// ```
/// use membrane_plugin::{Skill, SkillsPlugin};
///
/// let plugin = SkillsPlugin::new("my_skills", "Custom skills for my agent")
///     .with_skill(
///         Skill::new("code-review", "Reviews code for quality")
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
/// Skills are loaded from `<skill-name>/SKILL.md` subdirectories,
/// following the Agent Skills / Claude Code convention:
///
/// ```text
/// .claude/skills/
/// ├── code-review/
/// │   └── SKILL.md
/// ├── testing/
/// │   └── SKILL.md
/// └── deploy/
///     ├── SKILL.md
///     └── scripts/
///         └── deploy.sh
/// ```
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

    /// Load skills from `<skill-name>/SKILL.md` subdirectories.
    ///
    /// Each subdirectory containing a `SKILL.md` file is loaded as a skill.
    /// The directory name is used as a fallback for the skill name when not
    /// specified in frontmatter (per the Agent Skills spec, `name` must
    /// match the parent directory name).
    ///
    /// Tools cannot be loaded from files; use [`SkillsPlugin::skill_mut`]
    /// to attach tools to specific skills after loading.
    pub fn load_skills_dir(mut self, dir: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let dir = dir.as_ref();

        let mut entries: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let path = entry.path();
                path.is_dir() && path.join("SKILL.md").exists()
            })
            .collect();

        // Sort by directory name for deterministic ordering
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let skill_md_path = entry.path().join("SKILL.md");
            let skill = Skill::from_skill_md(&skill_md_path)?;
            self.skills.push(skill);
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
    /// // if let Some(skill) = plugin.skill_mut("code-review") {
    /// //     // skill = skill.with_tool(my_tool);
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

    /// Returns context for progressive disclosure.
    ///
    /// For skills where `disable_model_invocation` is false, the
    /// description and full instructions are included. Skills with
    /// `disable_model_invocation` set to true are excluded from
    /// automatic context (they require explicit invocation).
    fn context(&self) -> Vec<String> {
        self.skills
            .iter()
            .filter(|s| !s.disable_model_invocation)
            .filter(|s| !s.instructions.is_empty() || !s.description.is_empty())
            .map(|s| {
                let mut parts = Vec::new();

                // Header
                if s.description.is_empty() {
                    parts.push(format!("## Skill: {}", s.name));
                } else {
                    parts.push(format!("## Skill: {} — {}", s.name, s.description));
                }

                // Instructions
                if !s.instructions.is_empty() {
                    parts.push(String::new());
                    parts.push(s.instructions.clone());
                }

                parts.join("\n")
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
        // s2 has description but no instructions, still included for description
        assert_eq!(context.len(), 3);

        assert!(context[0].contains("Skill: s1"));
        assert!(context[0].contains("Instructions for skill 1."));

        assert!(context[1].contains("Skill: s2"));
        assert!(context[1].contains("Skill 2"));

        assert!(context[2].contains("Skill: s3"));
        assert!(context[2].contains("Instructions for skill 3."));
    }

    #[test]
    fn skills_plugin_excludes_disable_model_invocation() {
        let plugin = SkillsPlugin::new("test", "Test")
            .with_skill(Skill::new("visible", "Visible skill").with_instructions("Visible."))
            .with_skill(
                Skill::new("hidden", "Hidden skill")
                    .with_instructions("Hidden.")
                    .with_disable_model_invocation(true),
            );

        let context = plugin.context();
        assert_eq!(context.len(), 1);
        assert!(context[0].contains("visible"));
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
    fn skills_plugin_load_skill_dirs() {
        let dir = std::env::temp_dir().join("membrane_test_skills_v2");
        let _ = std::fs::remove_dir_all(&dir);

        // Create skill directories with SKILL.md
        let alpha_dir = dir.join("alpha");
        std::fs::create_dir_all(&alpha_dir).expect("create alpha dir");
        std::fs::write(
            alpha_dir.join("SKILL.md"),
            "---\nname: alpha\ndescription: Alpha skill\n---\n\nAlpha instructions.",
        )
        .expect("write alpha");

        let beta_dir = dir.join("beta");
        std::fs::create_dir_all(&beta_dir).expect("create beta dir");
        std::fs::write(
            beta_dir.join("SKILL.md"),
            "---\ndescription: Beta skill\nallowed-tools: Read, Grep\n---\n\nBeta instructions.",
        )
        .expect("write beta");

        // Directory without SKILL.md should be ignored
        let ignored_dir = dir.join("no-skill");
        std::fs::create_dir_all(&ignored_dir).expect("create ignored dir");
        std::fs::write(ignored_dir.join("README.md"), "Not a skill").expect("write ignored");

        // File (not dir) should be ignored
        std::fs::write(dir.join("stray.md"), "Not a skill dir").expect("write stray");

        let plugin = SkillsPlugin::new("test", "Test")
            .load_skills_dir(&dir)
            .expect("load dir");

        assert_eq!(plugin.skills().len(), 2);

        let names: Vec<&str> = plugin.skills().iter().map(|s| s.name()).collect();
        assert_eq!(names, vec!["alpha", "beta"]); // sorted by dir name

        // Beta should get name from directory
        assert_eq!(plugin.skills()[1].name(), "beta");
        assert_eq!(plugin.skills()[1].description(), "Beta skill");
        assert_eq!(plugin.skills()[1].allowed_tools(), &["Read", "Grep"]);

        let context = plugin.context();
        assert_eq!(context.len(), 2);
        assert!(context[0].contains("Alpha instructions."));
        assert!(context[1].contains("Beta instructions."));

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
