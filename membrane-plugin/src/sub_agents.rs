use std::path::Path;

use membrane_core::plugin::Plugin;
use membrane_core::tool::Tool;

use crate::sub_agent::SubAgentDefinition;

/// A plugin that loads subagent definitions from `.claude/agents/` directory.
///
/// `SubAgentsPlugin` follows the Claude Code subagent syntax, loading
/// `<name>.md` files from a directory. Each file defines a custom subagent
/// with YAML frontmatter for configuration and markdown body as the system
/// prompt.
///
/// When registered with an agent via `with_plugin()`:
///
/// - **Context** is injected describing available subagents (so the LLM
///   knows when to delegate).
/// - **Definitions** are accessible for the caller to create actual agent
///   instances with appropriate providers and tools.
///
/// # Directory Structure
///
/// ```text
/// .claude/agents/
/// ├── researcher.md     # Read-only research agent
/// ├── planner.md        # Planning agent
/// └── code-writer.md    # Code generation agent
/// ```
///
/// # Example
///
/// ```no_run
/// use membrane_plugin::SubAgentsPlugin;
///
/// let plugin = SubAgentsPlugin::new("project_agents", "Custom project subagents")
///     .load_agents_dir(".claude/agents")
///     .expect("failed to load agents directory");
///
/// // Access definitions to create agent instances
/// for def in plugin.definitions() {
///     println!("Agent: {} - {}", def.name(), def.description());
///     println!("  Model: {:?}", def.model());
///     println!("  Tools: {:?}", def.tools());
///     println!("  Max turns: {:?}", def.max_turns());
/// }
/// ```
///
/// # Integration with Agent
///
/// The plugin provides context describing available subagents. The caller
/// is responsible for creating `Agent` instances from the definitions and
/// registering them as sub-agents:
///
/// ```no_run
/// use membrane_plugin::{SubAgentsPlugin, SubAgentDefinition};
/// use membrane_core::agent::{Agent, AgentConfig};
///
/// let plugin = SubAgentsPlugin::new("agents", "Custom agents")
///     .load_agents_dir(".claude/agents")
///     .expect("failed to load");
///
/// // Build sub-agents from definitions
/// // for def in plugin.into_definitions() {
/// //     let config = def.build_config("default-model", 10);
/// //     let sub_agent = Agent::with_system_prompt(provider, tools, config, def.system_prompt());
/// //     parent = parent.with_sub_agent(def.name(), def.description(), sub_agent);
/// // }
/// ```
pub struct SubAgentsPlugin {
    name: String,
    description: String,
    definitions: Vec<SubAgentDefinition>,
}

impl SubAgentsPlugin {
    /// Create a new empty subagents plugin.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            definitions: Vec::new(),
        }
    }

    /// Add a subagent definition to this plugin.
    pub fn with_definition(mut self, definition: SubAgentDefinition) -> Self {
        self.definitions.push(definition);
        self
    }

    /// Load subagent definitions from `<name>.md` files in a directory.
    ///
    /// Each `.md` file in the directory is parsed as a subagent definition.
    /// The file stem (without `.md`) is used as a fallback name when not
    /// specified in frontmatter.
    ///
    /// Files are sorted by name for deterministic ordering.
    pub fn load_agents_dir(mut self, dir: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let dir = dir.as_ref();

        let mut entries: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let path = entry.path();
                path.is_file()
                    && path
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
            })
            .collect();

        // Sort by filename for deterministic ordering
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let def = SubAgentDefinition::from_md(entry.path())?;
            self.definitions.push(def);
        }

        Ok(self)
    }

    /// Return a reference to the definitions in this plugin.
    pub fn definitions(&self) -> &[SubAgentDefinition] {
        &self.definitions
    }

    /// Return a mutable reference to a definition by name.
    pub fn definition_mut(&mut self, name: &str) -> Option<&mut SubAgentDefinition> {
        self.definitions.iter_mut().find(|d| d.name() == name)
    }

    /// Consume the plugin and return the definitions.
    ///
    /// Use this to take ownership of definitions for creating agent instances.
    pub fn into_definitions(self) -> Vec<SubAgentDefinition> {
        self.definitions
    }

    /// Generate instructions describing available subagents.
    ///
    /// Produces a system-prompt-ready string listing all subagent definitions
    /// with their names, descriptions, and capabilities.
    pub fn build_default_instructions(&self) -> String {
        let mut lines = vec![
            "# Subagents".to_string(),
            String::new(),
            "The following subagents are available for delegation. \
             Use them when the task matches their description."
                .to_string(),
            String::new(),
        ];

        for def in &self.definitions {
            let mut entry = format!("- **{}**", def.name());
            if !def.description().is_empty() {
                entry.push_str(&format!(": {}", def.description()));
            }

            let mut details = Vec::new();
            if let Some(model) = def.model() {
                details.push(format!("model: {}", model));
            }
            if let Some(max_turns) = def.max_turns() {
                details.push(format!("max turns: {}", max_turns));
            }
            if !def.tools().is_empty() {
                details.push(format!("tools: {}", def.tools().join(", ")));
            }
            if !details.is_empty() {
                entry.push_str(&format!(" ({})", details.join(", ")));
            }

            lines.push(entry);
        }

        lines.join("\n")
    }
}

impl Plugin for SubAgentsPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    /// Subagents don't provide tools directly through the plugin interface.
    /// Tools are configured when creating the actual agent instances from
    /// the definitions.
    fn tools(&mut self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }

    /// Returns context describing available subagents.
    ///
    /// This injects system messages so the LLM knows which subagents are
    /// available and when to delegate to them.
    fn context(&self) -> Vec<String> {
        if self.definitions.is_empty() {
            return Vec::new();
        }

        vec![self.build_default_instructions()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_plugin() {
        let plugin = SubAgentsPlugin::new("test", "Test plugin");
        assert_eq!(plugin.name(), "test");
        assert_eq!(plugin.description(), "Test plugin");
        assert!(plugin.definitions().is_empty());
    }

    #[test]
    fn with_definition() {
        let plugin = SubAgentsPlugin::new("test", "Test")
            .with_definition(
                SubAgentDefinition::new("researcher", "Research topics")
                    .with_system_prompt("You are a researcher."),
            )
            .with_definition(
                SubAgentDefinition::new("planner", "Plan tasks")
                    .with_system_prompt("You are a planner."),
            );

        assert_eq!(plugin.definitions().len(), 2);
        assert_eq!(plugin.definitions()[0].name(), "researcher");
        assert_eq!(plugin.definitions()[1].name(), "planner");
    }

    #[test]
    fn definition_mut_finds_by_name() {
        let mut plugin = SubAgentsPlugin::new("test", "Test")
            .with_definition(SubAgentDefinition::new("agent1", "Agent 1"))
            .with_definition(SubAgentDefinition::new("agent2", "Agent 2"));

        assert!(plugin.definition_mut("agent1").is_some());
        assert!(plugin.definition_mut("agent2").is_some());
        assert!(plugin.definition_mut("agent3").is_none());
    }

    #[test]
    fn into_definitions() {
        let plugin = SubAgentsPlugin::new("test", "Test")
            .with_definition(SubAgentDefinition::new("a", "A"))
            .with_definition(SubAgentDefinition::new("b", "B"));

        let defs = plugin.into_definitions();
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name(), "a");
        assert_eq!(defs[1].name(), "b");
    }

    #[test]
    fn plugin_context_empty_definitions() {
        let plugin = SubAgentsPlugin::new("test", "Test");
        let context = plugin.context();
        assert!(context.is_empty());
    }

    #[test]
    fn plugin_context_with_definitions() {
        let plugin = SubAgentsPlugin::new("test", "Test")
            .with_definition(
                SubAgentDefinition::new("researcher", "Research topics")
                    .with_model("haiku")
                    .with_max_turns(10)
                    .with_tools(vec!["Read", "Grep"]),
            )
            .with_definition(SubAgentDefinition::new("planner", "Plan complex tasks"));

        let context = plugin.context();
        assert_eq!(context.len(), 1);

        let ctx = &context[0];
        assert!(ctx.contains("# Subagents"));
        assert!(ctx.contains("**researcher**: Research topics"));
        assert!(ctx.contains("model: haiku"));
        assert!(ctx.contains("max turns: 10"));
        assert!(ctx.contains("tools: Read, Grep"));
        assert!(ctx.contains("**planner**: Plan complex tasks"));
    }

    #[test]
    fn plugin_tools_empty() {
        let mut plugin =
            SubAgentsPlugin::new("test", "Test").with_definition(SubAgentDefinition::new("a", "A"));

        assert!(plugin.tools().is_empty());
    }

    #[test]
    fn build_default_instructions() {
        let plugin = SubAgentsPlugin::new("test", "Test")
            .with_definition(
                SubAgentDefinition::new("researcher", "Research topics")
                    .with_model("haiku")
                    .with_tools(vec!["Read", "Grep"]),
            )
            .with_definition(SubAgentDefinition::new("coder", ""));

        let instructions = plugin.build_default_instructions();

        assert!(instructions.contains("# Subagents"));
        assert!(instructions.contains("**researcher**: Research topics"));
        assert!(instructions.contains("model: haiku"));
        assert!(instructions.contains("tools: Read, Grep"));
        assert!(instructions.contains("**coder**"));
        // No description for coder, so no ": " after name
        assert!(!instructions.contains("**coder**:"));
    }

    #[test]
    fn load_agents_dir() {
        let dir = std::env::temp_dir().join("membrane_test_agents_v1");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");

        // Create agent files
        std::fs::write(
            dir.join("researcher.md"),
            "---\nname: researcher\ndescription: Research topics\ntools: Read, Grep\nmaxTurns: 10\n---\n\nYou are a research agent.",
        )
        .expect("write researcher");

        std::fs::write(
            dir.join("planner.md"),
            "---\ndescription: Plan tasks\nmodel: opus\n---\n\nYou are a planning agent.",
        )
        .expect("write planner");

        // Non-md file should be ignored
        std::fs::write(dir.join("README.txt"), "Not an agent").expect("write readme");

        let plugin = SubAgentsPlugin::new("test", "Test")
            .load_agents_dir(&dir)
            .expect("load dir");

        assert_eq!(plugin.definitions().len(), 2);

        // Sorted by filename
        let names: Vec<&str> = plugin.definitions().iter().map(|d| d.name()).collect();
        assert_eq!(names, vec!["planner", "researcher"]);

        // Verify planner got name from filename
        assert_eq!(plugin.definitions()[0].name(), "planner");
        assert_eq!(plugin.definitions()[0].description(), "Plan tasks");
        assert_eq!(plugin.definitions()[0].model(), Some("opus"));
        assert_eq!(
            plugin.definitions()[0].system_prompt(),
            "You are a planning agent."
        );

        // Verify researcher
        assert_eq!(plugin.definitions()[1].name(), "researcher");
        assert_eq!(plugin.definitions()[1].tools(), &["Read", "Grep"]);
        assert_eq!(plugin.definitions()[1].max_turns(), Some(10));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_agents_dir_nonexistent() {
        let result =
            SubAgentsPlugin::new("test", "Test").load_agents_dir("/nonexistent/path/to/agents");
        assert!(result.is_err());
    }

    #[test]
    fn load_agents_dir_empty() {
        let dir = std::env::temp_dir().join("membrane_test_agents_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");

        let plugin = SubAgentsPlugin::new("test", "Test")
            .load_agents_dir(&dir)
            .expect("load dir");

        assert!(plugin.definitions().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
