/// Parsed YAML frontmatter from a SKILL.md file.
///
/// Frontmatter is delimited by `---` lines at the beginning of a markdown file:
///
/// ```text
/// ---
/// name: my-skill
/// description: What this skill does
/// allowed-tools: Read, Grep, Glob
/// user-invocable: true
/// ---
///
/// Markdown content here...
/// ```
///
/// The YAML block is parsed using `serde_yaml`. Values can be accessed
/// as strings, booleans, or lists via typed accessors.
#[derive(Debug, Clone, Default)]
pub struct Frontmatter {
    mapping: serde_yaml::Mapping,
}

impl Frontmatter {
    /// Parse YAML frontmatter from markdown content.
    ///
    /// Returns `(frontmatter, body)` where `body` is the markdown content
    /// after the closing `---` delimiter. If no frontmatter is found or
    /// the YAML is invalid, returns an empty `Frontmatter` and the original
    /// content.
    pub fn parse(content: &str) -> (Self, &str) {
        let trimmed = content.trim_start();

        // Must start with "---"
        if !trimmed.starts_with("---") {
            return (Self::default(), content);
        }

        // Find the closing "---"
        let after_open = &trimmed[3..];
        let after_open = after_open.strip_prefix('\n').unwrap_or(after_open);

        let Some(close_pos) = after_open.find("\n---") else {
            return (Self::default(), content);
        };

        let yaml_block = &after_open[..close_pos];
        let body_start = &after_open[close_pos + 4..]; // skip "\n---"
        let body = body_start.strip_prefix('\n').unwrap_or(body_start);

        let mapping = serde_yaml::from_str::<serde_yaml::Mapping>(yaml_block).unwrap_or_default();

        (Self { mapping }, body)
    }

    /// Get a field value as a string.
    ///
    /// For non-string YAML values, the raw YAML representation is returned.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.mapping
            .get(serde_yaml::Value::String(key.to_string()))
            .and_then(|v| v.as_str())
    }

    /// Get a boolean field value.
    ///
    /// Returns `None` if the field is not present.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.mapping
            .get(serde_yaml::Value::String(key.to_string()))
            .and_then(|v| v.as_bool())
    }

    /// Get an unsigned integer field value.
    ///
    /// Returns `None` if the field is not present or not a valid u64.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.mapping
            .get(serde_yaml::Value::String(key.to_string()))
            .and_then(|v| v.as_u64())
    }

    /// Get a list field.
    ///
    /// Handles multiple YAML representations:
    /// - YAML sequence: `[Read, Grep, Glob]`
    /// - Comma-separated string: `"Read, Grep, Glob"` (Claude Code convention)
    /// - Space-delimited string: `"Read Grep Glob"` (Agent Skills spec)
    pub fn get_list(&self, key: &str) -> Vec<String> {
        let Some(value) = self.mapping.get(serde_yaml::Value::String(key.to_string())) else {
            return Vec::new();
        };

        match value {
            // YAML sequence: [Read, Grep, Glob]
            serde_yaml::Value::Sequence(seq) => seq
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            // String: comma or space separated
            serde_yaml::Value::String(s) => {
                if s.contains(',') {
                    s.split(',').map(|s| s.trim().to_string()).collect()
                } else {
                    s.split_whitespace().map(|s| s.to_string()).collect()
                }
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_frontmatter() {
        let content = "---\nname: my-skill\ndescription: A test skill\n---\n\nBody content here.";
        let (fm, body) = Frontmatter::parse(content);

        assert_eq!(fm.get("name"), Some("my-skill"));
        assert_eq!(fm.get("description"), Some("A test skill"));
        assert_eq!(body, "\nBody content here.");
    }

    #[test]
    fn parse_no_frontmatter() {
        let content = "Just markdown content.";
        let (fm, body) = Frontmatter::parse(content);

        assert!(fm.get("name").is_none());
        assert_eq!(body, content);
    }

    #[test]
    fn parse_quoted_values() {
        let content = "---\nname: \"my-skill\"\ndescription: 'A test skill'\n---\nBody";
        let (fm, _body) = Frontmatter::parse(content);

        assert_eq!(fm.get("name"), Some("my-skill"));
        assert_eq!(fm.get("description"), Some("A test skill"));
    }

    #[test]
    fn parse_boolean_fields() {
        let content = "---\nuser-invocable: true\ndisable-model-invocation: false\n---\nBody";
        let (fm, _body) = Frontmatter::parse(content);

        assert_eq!(fm.get_bool("user-invocable"), Some(true));
        assert_eq!(fm.get_bool("disable-model-invocation"), Some(false));
        assert_eq!(fm.get_bool("nonexistent"), None);
    }

    #[test]
    fn parse_comma_separated_list() {
        let content = "---\nallowed-tools: Read, Grep, Glob\n---\nBody";
        let (fm, _body) = Frontmatter::parse(content);

        assert_eq!(fm.get_list("allowed-tools"), vec!["Read", "Grep", "Glob"]);
    }

    #[test]
    fn parse_space_separated_list() {
        let content = "---\nallowed-tools: Read Grep Glob\n---\nBody";
        let (fm, _body) = Frontmatter::parse(content);

        assert_eq!(fm.get_list("allowed-tools"), vec!["Read", "Grep", "Glob"]);
    }

    #[test]
    fn parse_yaml_sequence_list() {
        let content = "---\nallowed-tools:\n  - Read\n  - Grep\n  - Glob\n---\nBody";
        let (fm, _body) = Frontmatter::parse(content);

        assert_eq!(fm.get_list("allowed-tools"), vec!["Read", "Grep", "Glob"]);
    }

    #[test]
    fn parse_yaml_inline_sequence() {
        let content = "---\nallowed-tools: [Read, Grep, Glob]\n---\nBody";
        let (fm, _body) = Frontmatter::parse(content);

        assert_eq!(fm.get_list("allowed-tools"), vec!["Read", "Grep", "Glob"]);
    }

    #[test]
    fn parse_empty_list() {
        let content = "---\nname: test\n---\nBody";
        let (fm, _body) = Frontmatter::parse(content);

        assert!(fm.get_list("allowed-tools").is_empty());
    }

    #[test]
    fn parse_all_claude_code_fields() {
        let content = "\
---
name: code-review
description: Reviews code for quality
allowed-tools: Read, Grep, Glob
user-invocable: true
disable-model-invocation: false
argument-hint: \"[filename]\"
model: claude-sonnet-4-5-20250929
context: fork
agent: Explore
---

Review the following code...";
        let (fm, body) = Frontmatter::parse(content);

        assert_eq!(fm.get("name"), Some("code-review"));
        assert_eq!(fm.get("description"), Some("Reviews code for quality"));
        assert_eq!(fm.get_list("allowed-tools"), vec!["Read", "Grep", "Glob"]);
        assert_eq!(fm.get_bool("user-invocable"), Some(true));
        assert_eq!(fm.get_bool("disable-model-invocation"), Some(false));
        assert_eq!(fm.get("argument-hint"), Some("[filename]"));
        assert_eq!(fm.get("model"), Some("claude-sonnet-4-5-20250929"));
        assert_eq!(fm.get("context"), Some("fork"));
        assert_eq!(fm.get("agent"), Some("Explore"));
        assert_eq!(body, "\nReview the following code...");
    }

    #[test]
    fn parse_comments_and_blank_lines() {
        let content = "---\nname: test\n# comment\n\ndescription: desc\n---\nBody";
        let (fm, _body) = Frontmatter::parse(content);

        assert_eq!(fm.get("name"), Some("test"));
        assert_eq!(fm.get("description"), Some("desc"));
    }

    #[test]
    fn parse_u64_fields() {
        let content = "---\nmaxTurns: 25\n---\nBody";
        let (fm, _body) = Frontmatter::parse(content);

        assert_eq!(fm.get_u64("maxTurns"), Some(25));
        assert_eq!(fm.get_u64("nonexistent"), None);
    }

    #[test]
    fn parse_unclosed_frontmatter() {
        let content = "---\nname: test\nNo closing delimiter";
        let (fm, body) = Frontmatter::parse(content);

        assert!(fm.get("name").is_none());
        assert_eq!(body, content);
    }
}
