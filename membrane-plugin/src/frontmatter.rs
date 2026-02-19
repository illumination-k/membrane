use std::collections::BTreeMap;

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
#[derive(Debug, Clone, Default)]
pub struct Frontmatter {
    fields: BTreeMap<String, String>,
}

impl Frontmatter {
    /// Parse YAML frontmatter from markdown content.
    ///
    /// Returns `(frontmatter, body)` where `body` is the markdown content
    /// after the closing `---` delimiter. If no frontmatter is found,
    /// returns an empty `Frontmatter` and the original content.
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

        let mut fields = BTreeMap::new();

        for line in yaml_block.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim().to_string();
                let value = value.trim();
                // Strip surrounding quotes
                let value = value
                    .strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'))
                    .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
                    .unwrap_or(value)
                    .to_string();
                fields.insert(key, value);
            }
        }

        (Self { fields }, body)
    }

    /// Get a string field value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }

    /// Get a boolean field value.
    ///
    /// Returns `None` if the field is not present.
    /// Returns `Some(true)` for "true", `Some(false)` for "false".
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.fields.get(key).map(|v| v.eq_ignore_ascii_case("true"))
    }

    /// Get a comma-separated or space-separated list field.
    ///
    /// The Agent Skills spec uses space-delimited lists, while Claude Code
    /// uses comma-separated. This method handles both.
    pub fn get_list(&self, key: &str) -> Vec<String> {
        self.fields
            .get(key)
            .map(|v| {
                if v.contains(',') {
                    v.split(',').map(|s| s.trim().to_string()).collect()
                } else {
                    v.split_whitespace().map(|s| s.to_string()).collect()
                }
            })
            .unwrap_or_default()
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
argument-hint: [filename]
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
    fn parse_unclosed_frontmatter() {
        let content = "---\nname: test\nNo closing delimiter";
        let (fm, body) = Frontmatter::parse(content);

        assert!(fm.get("name").is_none());
        assert_eq!(body, content);
    }
}
