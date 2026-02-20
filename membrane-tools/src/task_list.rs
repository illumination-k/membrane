use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use membrane_core::error::Error;
use membrane_core::tool::{Tool, ToolDefinition};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Status of a task in the task list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
}

/// A single task item in the task list.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskItem {
    /// Description of what needs to be done.
    pub content: String,
    /// Current status of the task.
    pub status: TaskStatus,
    /// Present continuous form shown during execution (e.g., "Running tests").
    #[serde(rename = "activeForm")]
    pub active_form: String,
}

type SharedTaskList = Arc<Mutex<Vec<TaskItem>>>;

#[derive(Debug, Deserialize, JsonSchema)]
struct TaskListWriteInput {
    /// The updated task list. Replaces the entire list.
    todos: Vec<TaskItem>,
}

/// Tool to create or update a task list.
///
/// Replaces the entire task list with the provided items. Each task has:
/// - `content`: what needs to be done (imperative form)
/// - `status`: one of `pending`, `in_progress`, or `completed`
/// - `activeForm`: present continuous form (e.g., "Running tests")
///
/// This tool shares state with [`TaskListReadTool`]. Create both via
/// [`task_list_tools()`].
pub struct TaskListWriteTool {
    tasks: SharedTaskList,
}

impl Tool for TaskListWriteTool {
    fn definition(&self) -> ToolDefinition {
        let schema = schemars::schema_for!(TaskListWriteInput);
        let input_schema = serde_json::to_value(schema).expect("schema serialization");

        ToolDefinition {
            name: "task_list_write".to_string(),
            description: "Create or update a task list to track progress. Replaces the entire \
                task list with the provided items. Each task has content (what to do), status \
                (pending/in_progress/completed), and activeForm (present continuous form)."
                .to_string(),
            input_schema,
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        Box::pin(async move {
            let input: TaskListWriteInput = serde_json::from_value(input)?;

            let mut tasks = self.tasks.lock().map_err(|e| Error::ToolExecution {
                tool_name: "task_list_write".to_string(),
                message: format!("Failed to acquire lock: {e}"),
            })?;

            *tasks = input.todos;

            let total = tasks.len();
            let completed = tasks
                .iter()
                .filter(|t| t.status == TaskStatus::Completed)
                .count();
            let in_progress = tasks
                .iter()
                .filter(|t| t.status == TaskStatus::InProgress)
                .count();

            Ok(format!(
                "Task list updated. {total} total, {completed} completed, {in_progress} in progress."
            ))
        })
    }
}

/// Tool to read the current task list.
///
/// Returns all tasks with their statuses as JSON. This tool shares state
/// with [`TaskListWriteTool`]. Create both via [`task_list_tools()`].
pub struct TaskListReadTool {
    tasks: SharedTaskList,
}

impl Tool for TaskListReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "task_list_read".to_string(),
            description: "Read the current task list and return all tasks with their statuses."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    fn execute(
        &self,
        _input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        Box::pin(async move {
            let tasks = self.tasks.lock().map_err(|e| Error::ToolExecution {
                tool_name: "task_list_read".to_string(),
                message: format!("Failed to acquire lock: {e}"),
            })?;

            if tasks.is_empty() {
                return Ok("No tasks in the list.".to_string());
            }

            serde_json::to_string_pretty(&*tasks).map_err(|e| Error::ToolExecution {
                tool_name: "task_list_read".to_string(),
                message: format!("Failed to serialize tasks: {e}"),
            })
        })
    }
}

/// Create a pair of task list tools that share state.
///
/// Returns `(TaskListWriteTool, TaskListReadTool)` backed by the same
/// in-memory list. Register both with an [`Agent`](membrane_core::agent::Agent)
/// to give it task tracking capabilities.
///
/// # Example
///
/// ```rust
/// use membrane_tools::task_list_tools;
///
/// let (write_tool, read_tool) = task_list_tools();
/// // Pass both as Box<dyn Tool> to Agent::new(...)
/// ```
pub fn task_list_tools() -> (TaskListWriteTool, TaskListReadTool) {
    let tasks: SharedTaskList = Arc::new(Mutex::new(Vec::new()));
    (
        TaskListWriteTool {
            tasks: tasks.clone(),
        },
        TaskListReadTool { tasks },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn write_and_read_tasks() {
        let (write_tool, read_tool) = task_list_tools();

        // Initially empty
        let result = read_tool
            .execute(serde_json::json!({}))
            .await
            .expect("read should succeed");
        assert_eq!(result, "No tasks in the list.");

        // Write some tasks
        let result = write_tool
            .execute(serde_json::json!({
                "todos": [
                    {"content": "Build project", "status": "pending", "activeForm": "Building project"},
                    {"content": "Run tests", "status": "in_progress", "activeForm": "Running tests"},
                    {"content": "Fix lint", "status": "completed", "activeForm": "Fixing lint"}
                ]
            }))
            .await
            .expect("write should succeed");
        assert!(result.contains("3 total"));
        assert!(result.contains("1 completed"));
        assert!(result.contains("1 in progress"));

        // Read back
        let result = read_tool
            .execute(serde_json::json!({}))
            .await
            .expect("read should succeed");
        let tasks: Vec<TaskItem> = serde_json::from_str(&result).expect("valid JSON");
        assert_eq!(tasks.len(), 3);
        assert_eq!(tasks[0].content, "Build project");
        assert_eq!(tasks[0].status, TaskStatus::Pending);
        assert_eq!(tasks[1].status, TaskStatus::InProgress);
        assert_eq!(tasks[2].status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn write_replaces_entire_list() {
        let (write_tool, read_tool) = task_list_tools();

        // Write initial tasks
        write_tool
            .execute(serde_json::json!({
                "todos": [
                    {"content": "Task A", "status": "pending", "activeForm": "Doing A"},
                    {"content": "Task B", "status": "pending", "activeForm": "Doing B"}
                ]
            }))
            .await
            .expect("write should succeed");

        // Replace with a single task
        write_tool
            .execute(serde_json::json!({
                "todos": [
                    {"content": "Task C", "status": "completed", "activeForm": "Doing C"}
                ]
            }))
            .await
            .expect("write should succeed");

        let result = read_tool
            .execute(serde_json::json!({}))
            .await
            .expect("read should succeed");
        let tasks: Vec<TaskItem> = serde_json::from_str(&result).expect("valid JSON");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].content, "Task C");
        assert_eq!(tasks[0].status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn write_empty_list() {
        let (write_tool, read_tool) = task_list_tools();

        // Write tasks then clear
        write_tool
            .execute(serde_json::json!({
                "todos": [
                    {"content": "Task A", "status": "pending", "activeForm": "Doing A"}
                ]
            }))
            .await
            .expect("write should succeed");

        let result = write_tool
            .execute(serde_json::json!({"todos": []}))
            .await
            .expect("write should succeed");
        assert!(result.contains("0 total"));

        let result = read_tool
            .execute(serde_json::json!({}))
            .await
            .expect("read should succeed");
        assert_eq!(result, "No tasks in the list.");
    }

    #[tokio::test]
    async fn write_invalid_input_returns_error() {
        let (write_tool, _) = task_list_tools();

        let result = write_tool
            .execute(serde_json::json!({"bad_field": 42}))
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn definitions_are_valid() {
        let (write_tool, read_tool) = task_list_tools();

        let write_def = write_tool.definition();
        assert_eq!(write_def.name, "task_list_write");
        assert!(!write_def.description.is_empty());
        assert!(write_def.input_schema.is_object());

        let read_def = read_tool.definition();
        assert_eq!(read_def.name, "task_list_read");
        assert!(!read_def.description.is_empty());
        assert!(read_def.input_schema.is_object());
    }

    #[test]
    fn task_status_serde_roundtrip() {
        let item = TaskItem {
            content: "Test task".to_string(),
            status: TaskStatus::InProgress,
            active_form: "Testing".to_string(),
        };
        let json = serde_json::to_value(&item).expect("serialize");
        assert_eq!(json["status"], "in_progress");
        assert_eq!(json["activeForm"], "Testing");

        let deserialized: TaskItem = serde_json::from_value(json).expect("deserialize");
        assert_eq!(deserialized.status, TaskStatus::InProgress);
        assert_eq!(deserialized.active_form, "Testing");
    }
}
