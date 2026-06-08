// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "openai")]

use std::io;
use std::sync::{Arc, Mutex};

use serial_test::serial;
use sweet_core::{ContentBlock, Message, Model};
use sweet_llm::openai::ReasoningContent;
use sweet_llm::{openai, OpenAIProvider, ProviderError};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn canned_response(content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    })
}

#[tokio::test]
async fn complete_posts_correct_request_and_returns_assistant_message() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .and(body_partial_json(serde_json::json!({
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("hi there")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("test-key")
        .with_base_url(server.uri())
        .with_model("gpt-test");

    let reply = provider
        .complete(&[Message::system("be terse"), Message::user("hello")], &[])
        .await
        .expect("complete should succeed");

    assert_eq!(reply, Message::assistant("hi there"));
}

#[tokio::test]
async fn image_user_message_sends_multimodal_parts_array() {
    // User messages carrying an image must serialize `content` as the
    // OpenAI multimodal parts array (text + image_url with a base64
    // data: URL). Text-only messages keep the plain-string content; that
    // shape is exercised by the test above.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(serde_json::json!({
            "model": "gpt-test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AQID"}
                    }
                ]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("a tiny png")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("test-key")
        .with_base_url(server.uri())
        .with_model("gpt-test");

    let msg = Message::user_blocks(vec![
        ContentBlock::text("describe this"),
        ContentBlock::Image {
            data: vec![1, 2, 3],
            media_type: "image/png".to_string(),
        },
    ]);

    let reply = provider
        .complete(&[msg], &[])
        .await
        .expect("complete should succeed");

    assert_eq!(reply, Message::assistant("a tiny png"));
}

#[tokio::test]
async fn file_user_message_sends_file_content_part() {
    // User messages carrying a non-image file (e.g. PDF) must serialize
    // `content` as the OpenAI multimodal parts array with a `file` type part.
    // The `filename` field is required by the API and must be present.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(serde_json::json!({
            "model": "gpt-test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "review this"},
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
                            "filename": "report.pdf"
                        }
                    }
                ]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("looks good")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("test-key")
        .with_base_url(server.uri())
        .with_model("gpt-test");

    let msg = Message::user_blocks(vec![
        ContentBlock::text("review this"),
        ContentBlock::File {
            data: b"%PDF-1.4".to_vec(),
            media_type: "application/pdf".to_string(),
            filename: "report.pdf".to_string(),
        },
    ]);

    let reply = provider
        .complete(&[msg], &[])
        .await
        .expect("complete should succeed");

    assert_eq!(reply, Message::assistant("looks good"));
}

#[tokio::test]
async fn complete_extracts_reasoning_content_from_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me think..."
                },
                "finish_reason": "stop"
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("test-key").with_base_url(server.uri());

    let reply = provider
        .complete(&[Message::user("what is the meaning of life?")], &[])
        .await
        .expect("complete should succeed");

    assert_eq!(reply.text_content(), "The answer is 42.");
    assert_eq!(reply.reasoning_content(), Some("Let me think..."));
}

#[tokio::test]
async fn complete_sends_reasoning_effort_and_thinking_when_configured() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_partial_json(serde_json::json!({
            "reasoning_effort": "max",
            "thinking": {"type": "enabled", "keep": "all"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("ok")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("test-key")
        .with_base_url(server.uri())
        .with_reasoning_effort("max")
        .with_thinking(sweet_llm::openai::ThinkingMode::PRESERVED);

    provider
        .complete(&[Message::user("hi")], &[])
        .await
        .expect("complete should succeed");
}

#[tokio::test]
async fn complete_echoes_reasoning_content_when_present_even_without_thinking_config() {
    let server = MockServer::start().await;

    // reasoning_content is always echoed back when the model provided it,
    // even if the provider wasn't explicitly configured for thinking.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("ok")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("test-key").with_base_url(server.uri());

    let mut prior = Message::assistant("prev");
    prior.set_reasoning_content("hidden");

    provider
        .complete(&[Message::user("hi"), prior], &[])
        .await
        .expect("complete should succeed");

    let received = server.received_requests().await.unwrap();
    let body_str = std::str::from_utf8(&received[0].body).unwrap();
    assert!(
        body_str.contains("\"reasoning_content\":\"hidden\""),
        "expected outgoing request to echo reasoning_content when present, got: {body_str}"
    );
    assert!(
        !body_str.contains("\"thinking\""),
        "expected outgoing request to omit thinking field, got: {body_str}"
    );
}

#[tokio::test]
async fn complete_echoes_reasoning_content_on_messages_when_thinking_configured() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("ok")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("test-key")
        .with_base_url(server.uri())
        .with_thinking(sweet_llm::openai::ThinkingMode::PRESERVED);

    let mut prior = Message::assistant("prev");
    prior.set_reasoning_content("hidden");

    provider
        .complete(&[Message::user("hi"), prior], &[])
        .await
        .expect("complete should succeed");

    let received = server.received_requests().await.unwrap();
    let body_str = std::str::from_utf8(&received[0].body).unwrap();
    assert!(
        body_str.contains("\"reasoning_content\":\"hidden\""),
        "expected outgoing request to echo reasoning_content, got: {body_str}"
    );
}

#[tokio::test]
async fn complete_echoes_empty_reasoning_content_for_assistant_when_thinking_configured() {
    // When targeting a thinking-aware backend, assistant messages must always
    // carry reasoning_content even if empty (required by DeepSeek/Kimi for
    // tool-call turns).
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("ok")))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("test-key")
        .with_base_url(server.uri())
        .with_thinking(sweet_llm::openai::ThinkingMode::PRESERVED);

    let prior = Message::assistant("prev"); // no reasoning_content set

    provider
        .complete(&[Message::user("hi"), prior], &[])
        .await
        .expect("complete should succeed");

    let received = server.received_requests().await.unwrap();
    let body_str = std::str::from_utf8(&received[0].body).unwrap();
    assert!(
        body_str.contains("\"reasoning_content\":\"\""),
        "expected outgoing assistant message to carry empty reasoning_content, got: {body_str}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn complete_logs_full_request_response_and_conversion_errors_without_api_key() {
    let success_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer secret-key"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response("hi there")))
        .expect(1)
        .mount(&success_server)
        .await;

    let bad_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer secret-key"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": "not-json"
                        }
                    }]
                }
            }]
        })))
        .expect(1)
        .mount(&bad_server)
        .await;

    let success_provider = OpenAIProvider::new("secret-key")
        .with_base_url(success_server.uri())
        .with_model("gpt-test");
    let bad_provider = OpenAIProvider::new("secret-key")
        .with_base_url(bad_server.uri())
        .with_model("gpt-test");

    let ((reply, bad_err), logs) = capture_observability(async {
        let reply = success_provider
            .complete(&[Message::system("be terse"), Message::user("hello")], &[])
            .await
            .unwrap();
        let bad_err = bad_provider
            .complete(&[Message::user("trigger malformed tool args")], &[])
            .await
            .unwrap_err()
            .to_string();
        (reply, bad_err)
    })
    .await;

    assert_eq!(reply, Message::assistant("hi there"));
    assert!(bad_err.contains("provider error"), "{bad_err}");
    assert!(logs.contains("openai.complete.start"), "{logs}");
    assert!(logs.contains("openai.complete"), "{logs}");
    assert!(logs.contains("gpt-test"), "{logs}");
    assert!(logs.contains("be terse"), "{logs}");
    assert!(logs.contains("hello"), "{logs}");
    assert!(logs.contains("hi there"), "{logs}");
    assert!(logs.contains("response_body"), "{logs}");
    assert!(logs.contains("response_message"), "{logs}");
    assert!(logs.contains("not-json"), "{logs}");
    assert!(!logs.contains("secret-key"), "{logs}");
}

#[tokio::test]
async fn non_2xx_response_yields_provider_error_with_body() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .respond_with(ResponseTemplate::new(401).set_body_string("bad key"))
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("nope").with_base_url(server.uri());

    let err = provider
        .complete(&[Message::user("x")], &[])
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("provider error"), "got: {msg}");

    let inner = std::error::Error::source(&err).expect("source");
    let provider_err = inner.downcast_ref::<ProviderError>().expect("downcast");
    match provider_err {
        ProviderError::Http { status, body } => {
            assert_eq!(status.as_u16(), 401);
            assert_eq!(body, "bad key");
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[tokio::test]
async fn empty_choices_yields_empty_response_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header(
            "user-agent",
            format!("sweet/{}", sweet_core::SWEET_VERSION),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"choices": []})))
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("k").with_base_url(server.uri());
    let err = provider
        .complete(&[Message::user("x")], &[])
        .await
        .unwrap_err();
    let inner = std::error::Error::source(&err).unwrap();
    let provider_err = inner.downcast_ref::<ProviderError>().unwrap();
    assert!(matches!(provider_err, ProviderError::EmptyResponse));
}

#[tokio::test]
async fn defaults_match_published_constants() {
    let provider = OpenAIProvider::new("k");
    assert_eq!(provider.base_url(), openai::DEFAULT_BASE_URL);
    assert_eq!(provider.model_name(), openai::DEFAULT_MODEL);
}

#[test]
#[serial]
fn from_env_reads_openai_api_key() {
    // SAFETY: tests in this file using env are gated behind `#[serial]`.
    std::env::set_var(openai::DEFAULT_API_KEY_ENV, "from-env-key");
    let provider = OpenAIProvider::from_env().expect("env var present");
    // Indirect check: defaults survived.
    assert_eq!(provider.base_url(), openai::DEFAULT_BASE_URL);
    std::env::remove_var(openai::DEFAULT_API_KEY_ENV);
}

#[test]
#[serial]
fn from_env_errors_when_unset() {
    std::env::remove_var(openai::DEFAULT_API_KEY_ENV);
    let err = OpenAIProvider::from_env().unwrap_err();
    let inner = std::error::Error::source(&err).unwrap();
    let provider_err = inner.downcast_ref::<ProviderError>().unwrap();
    assert!(matches!(
        provider_err,
        ProviderError::MissingApiKey { var } if *var == openai::DEFAULT_API_KEY_ENV
    ));
}

async fn capture_observability<F, T>(future: F) -> (T, String)
where
    F: std::future::Future<Output = T>,
{
    let writer = SharedWriter::default();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(writer.clone())
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("set test tracing subscriber");
    tracing::callsite::rebuild_interest_cache();
    let result = future.await;
    (result, writer.contents())
}

#[derive(Clone, Default)]
struct SharedWriter {
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl SharedWriter {
    fn contents(&self) -> String {
        String::from_utf8(self.bytes.lock().unwrap().clone()).unwrap()
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedWriter {
    type Writer = SharedWrite;

    fn make_writer(&'a self) -> Self::Writer {
        SharedWrite {
            bytes: self.bytes.clone(),
        }
    }
}

struct SharedWrite {
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl io::Write for SharedWrite {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.bytes.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
