mod convert;
mod types;

use std::future::Future;
use std::pin::Pin;

use membrane_core::error::Error;
use membrane_core::provider::{ChatRequest, ChatResponse, LlmProvider};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Provides a bearer token for API authentication.
///
/// Implement this trait for dynamic token retrieval (e.g. Azure AD credential refresh).
pub trait TokenProvider: Send + Sync {
    fn token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>>;
}

/// Static API key token provider.
struct StaticToken(String);

impl TokenProvider for StaticToken {
    fn token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        let token = self.0.clone();
        Box::pin(async move { Ok(token) })
    }
}

/// Builder for [`OpenAiProvider`].
///
/// # Examples
///
/// ```ignore
/// // Simple API key
/// let provider = OpenAiProvider::builder()
///     .api_key("sk-...")
///     .build();
///
/// // Azure OpenAI with custom token provider
/// let provider = OpenAiProvider::builder()
///     .token_provider(azure_credential)
///     .base_url("https://my-resource.openai.azure.com/openai")
///     .build();
/// ```
pub struct OpenAiProviderBuilder {
    token_provider: Option<Box<dyn TokenProvider>>,
    base_url: String,
    client: Option<reqwest::Client>,
}

impl OpenAiProviderBuilder {
    /// Set a static API key for authentication.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.token_provider = Some(Box::new(StaticToken(api_key.into())));
        self
    }

    /// Set a custom token provider for dynamic authentication (e.g. Azure AD).
    pub fn token_provider(mut self, provider: impl TokenProvider + 'static) -> Self {
        self.token_provider = Some(Box::new(provider));
        self
    }

    /// Set a custom base URL (default: `https://api.openai.com/v1`).
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set a pre-configured reqwest Client (e.g. for custom timeouts or proxy).
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the provider.
    ///
    /// # Panics
    ///
    /// Panics if neither `api_key` nor `token_provider` was set.
    pub fn build(self) -> OpenAiProvider {
        let token_provider = self
            .token_provider
            .expect("api_key or token_provider must be set");

        OpenAiProvider {
            token_provider,
            base_url: self.base_url,
            client: self.client.unwrap_or_default(),
        }
    }
}

/// OpenAI Chat Completions API provider.
pub struct OpenAiProvider {
    token_provider: Box<dyn TokenProvider>,
    base_url: String,
    client: reqwest::Client,
}

impl OpenAiProvider {
    /// Create a builder for `OpenAiProvider`.
    pub fn builder() -> OpenAiProviderBuilder {
        OpenAiProviderBuilder {
            token_provider: None,
            base_url: DEFAULT_BASE_URL.to_string(),
            client: None,
        }
    }
}

impl LlmProvider for OpenAiProvider {
    #[tracing::instrument(skip_all, fields(model = %request.model))]
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, Error> {
        let openai_request = convert::to_openai_request(request)?;

        let url = format!("{}/chat/completions", self.base_url);

        let token = self.token_provider.token().await?;

        let http_response = self
            .client
            .post(&url)
            .bearer_auth(&token)
            .json(&openai_request)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("HTTP request failed: {e}")))?;

        let status = http_response.status();
        if !status.is_success() {
            let body = http_response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<failed to read body>"));
            let detail = serde_json::from_str::<types::OpenAiErrorResponse>(&body)
                .map(|e| e.error.message)
                .unwrap_or(body);
            return Err(Error::Provider(format!(
                "OpenAI API error (HTTP {status}): {detail}"
            )));
        }

        let openai_response: types::OpenAiResponse = http_response
            .json()
            .await
            .map_err(|e| Error::Provider(format!("Failed to parse OpenAI response: {e}")))?;

        convert::from_openai_response(openai_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrane_core::message::{Content, Message};
    use membrane_core::provider::StopReason;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn ok_response(content: &str) -> serde_json::Value {
        serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": null
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        })
    }

    #[tokio::test]
    async fn simple_chat_request() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_response("Hello!")))
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::builder()
            .api_key("test-key")
            .base_url(mock_server.uri())
            .build();

        let request = ChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![Message::user("Hi")],
            tools: vec![],
            response_format: None,
            max_tokens: None,
            temperature: None,
        };

        let response = provider.chat(request).await.expect("should succeed");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.content.len(), 1);
        assert!(matches!(&response.content[0], Content::Text { text } if text == "Hello!"));
    }

    #[tokio::test]
    async fn tool_call_response() {
        let mock_server = MockServer::start().await;

        let response_body = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{\"query\":\"rust lang\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 12
            }
        });

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::builder()
            .api_key("test-key")
            .base_url(mock_server.uri())
            .build();

        let request = ChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![Message::user("Search for rust")],
            tools: vec![],
            response_format: None,
            max_tokens: None,
            temperature: None,
        };

        let response = provider.chat(request).await.expect("should succeed");
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.content.len(), 1);
        assert!(
            matches!(&response.content[0], Content::ToolUse { id, name, input }
                if id == "call_abc123" && name == "search" && input["query"] == "rust lang")
        );
    }

    #[tokio::test]
    async fn api_error_response() {
        let mock_server = MockServer::start().await;

        let error_body = serde_json::json!({
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error"
            }
        });

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(401).set_body_json(&error_body))
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::builder()
            .api_key("bad-key")
            .base_url(mock_server.uri())
            .build();

        let request = ChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![Message::user("Hi")],
            tools: vec![],
            response_format: None,
            max_tokens: None,
            temperature: None,
        };

        let result = provider.chat(request).await;
        assert!(result.is_err());
        let err_msg = result.expect_err("should be error").to_string();
        assert!(err_msg.contains("Invalid API key"));
        assert!(err_msg.contains("401"));
    }

    #[tokio::test]
    async fn custom_token_provider() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer dynamic-token-123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_response("OK")))
            .mount(&mock_server)
            .await;

        struct DynamicToken;
        impl TokenProvider for DynamicToken {
            fn token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
                Box::pin(async { Ok("dynamic-token-123".to_string()) })
            }
        }

        let provider = OpenAiProvider::builder()
            .token_provider(DynamicToken)
            .base_url(mock_server.uri())
            .build();

        let request = ChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![Message::user("Hi")],
            tools: vec![],
            response_format: None,
            max_tokens: None,
            temperature: None,
        };

        let response = provider.chat(request).await.expect("should succeed");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
    }

    #[test]
    #[should_panic(expected = "api_key or token_provider must be set")]
    fn build_without_auth_panics() {
        let _provider = OpenAiProvider::builder().build();
    }
}
