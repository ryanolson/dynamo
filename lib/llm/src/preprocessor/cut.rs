// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The chat and completions preprocessors' closing output after a request is cut short.
//!
//! A request that is killed or passes its deadline ends its response stream cleanly, as if it
//! had finished (see [`dynamo_gdl_adapters::cut_short`]). The preprocessor's response transform
//! decides once, where its input ends, whether the request was cut short, and then writes no
//! usage chunk. The response parsers above it (reasoning, tool calls, the jail of the published
//! `dynamo-parsers`) write on their own input's end and cannot be asked, so the decision reaches
//! them as a [`CutFlag`], and [`UntilCut`] ends their output where it is set.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, ready};

use dynamo_gdl_adapters::CutShort;
use futures::Stream;

/// The payload capture's drop reason for a request cut short.
pub(super) const DROP_CUT_SHORT: &str = "request_cut_short";

/// The decision that a request was cut short, shared by the stage that makes it and the stages
/// that write on end. One allocation per request, made only where such a stage runs.
#[derive(Clone, Debug, Default)]
pub(super) struct CutFlag(Arc<AtomicBool>);

impl CutFlag {
    pub(super) fn set(&self) {
        // Relaxed: it is set and read inside one poll of one task, or read after the task
        // that set it has ended (the payload capture), which orders it.
        self.0.store(true, Ordering::Relaxed);
    }

    pub(super) fn is_set(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// `S` until `cut` is set, then ended: what `S` writes after the stage below it saw a request
/// cut short is closing output, and is dropped.
///
/// Per item it reads the flag once, a load. It runs only on a request with response parsers,
/// which cost far more per item.
pub(super) struct UntilCut<S> {
    stream: Option<S>,
    cut: CutFlag,
}

impl<S> UntilCut<S> {
    pub(super) fn new(stream: S, cut: CutFlag) -> Self {
        Self {
            stream: Some(stream),
            cut,
        }
    }
}

impl<S: Unpin> Unpin for UntilCut<S> {}

impl<S: Stream + Unpin> Stream for UntilCut<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        let Some(stream) = self.stream.as_mut() else {
            return Poll::Ready(None);
        };
        match ready!(Pin::new(stream).poll_next(cx)) {
            Some(item) if !self.cut.is_set() => Poll::Ready(Some(item)),
            _ => {
                self.stream = None;
                Poll::Ready(None)
            }
        }
    }
}

/// A response stream with no request behind it to cut short, for the public response
/// transform that tests and benchmarks drive with streams of their own.
pub(super) struct NeverCut<S>(pub(super) S);

impl<S: Stream + Unpin> Stream for NeverCut<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        Pin::new(&mut self.0).poll_next(cx)
    }
}

impl<S> CutShort for NeverCut<S> {
    fn cut_short(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, ResponseStream};
    use dynamo_runtime::pipeline::{
        AsyncEngineContextProvider, Context, Error, ManyOut, SingleIn, async_trait,
    };
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::{StreamExt, stream};
    use gdl::Around;
    use serde::Serialize;

    use crate::model_card::ModelDeploymentCard;
    use crate::preprocessor::{BelowEngine, OpenAIPreprocessor, PreprocessorNode};
    use crate::protocols::common::llm_backend::{BackendOutput, FinishReason, PreprocessedRequest};
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use crate::protocols::openai::completions::NvCreateCompletionRequest;

    const TINY_LLAMA: &str = "tests/data/sample-models/TinyLlama_v1.1";
    const TOOL_MODEL: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";

    fn output(text: &str, finish_reason: Option<FinishReason>) -> BackendOutput {
        BackendOutput {
            token_ids: vec![1],
            tokens: vec![Some(text.to_string())],
            text: Some(text.to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason,
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
            encoder_result: None,
            worker_trace_link: None,
            engine_data: None,
            routing_data: None,
            jailed_text: None,
        }
    }

    /// An answer the engine finished: some text, then a chunk with its `finish_reason`.
    fn finished_answer() -> Vec<BackendOutput> {
        vec![
            output("Hello", None),
            output(" world", Some(FinishReason::Stop)),
        ]
    }

    /// A tool call the engine had not closed when its stream ended.
    fn open_tool_call() -> Vec<BackendOutput> {
        vec![
            output("<tool_call>\n{\"name\": \"get_weather\", ", None),
            output("\"arguments\": {\"city\": \"Par", None),
        ]
    }

    /// The stages below a preprocessor: sends `outputs`, then does `at_end` to the request's
    /// context, then ends cleanly. A stop or kill that lands before the stream ends.
    struct SignalAtEnd {
        outputs: Vec<BackendOutput>,
        at_end: fn(&dyn AsyncEngineContext),
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>
        for SignalAtEnd
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<BackendOutput>>, Error> {
            let context = request.context();
            let signalled = context.clone();
            let at_end = self.at_end;
            let end = stream::poll_fn(move |_| {
                at_end(signalled.as_ref());
                std::task::Poll::Ready(None)
            });
            let outputs = self.outputs.clone().into_iter().map(Annotated::from_data);
            Ok(ResponseStream::new(
                Box::pin(stream::iter(outputs).chain(end)),
                context,
            ))
        }
    }

    fn kill(context: &dyn AsyncEngineContext) {
        context.kill();
    }

    fn stop(context: &dyn AsyncEngineContext) {
        context.stop_generating();
    }

    fn preprocessor(model: &str, tool_call_parser: Option<&str>) -> Arc<OpenAIPreprocessor> {
        let mut card = ModelDeploymentCard::load_from_disk(model, None).unwrap();
        card.runtime_config.tool_call_parser = tool_call_parser.map(str::to_string);
        OpenAIPreprocessor::new(card).unwrap()
    }

    fn chat_request() -> NvCreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "stream_options": {"include_usage": true},
        }))
        .unwrap()
    }

    fn tool_request() -> NvCreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "stream": true,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}}
                    }
                }
            }],
            "tool_choice": "auto",
        }))
        .unwrap()
    }

    fn completion_request() -> NvCreateCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "prompt": "Hello",
            "stream": true,
            "stream_options": {"include_usage": true},
        }))
        .unwrap()
    }

    /// The completion markers in a response: each choice's `finish_reason`, the usage chunks,
    /// and the tool calls.
    #[derive(Debug, Default, PartialEq)]
    struct Markers {
        finish_reasons: Vec<String>,
        usage_chunks: usize,
        tool_calls: usize,
    }

    fn markers<T: Serialize>(items: &[Annotated<T>]) -> Markers {
        let mut markers = Markers::default();
        for item in items {
            let Some(data) = &item.data else { continue };
            let data = serde_json::to_value(data).unwrap();
            if !data["usage"].is_null() {
                markers.usage_chunks += 1;
            }
            for choice in data["choices"].as_array().into_iter().flatten() {
                if let Some(reason) = choice["finish_reason"].as_str() {
                    markers.finish_reasons.push(reason.to_string());
                }
                if let Some(calls) = choice["delta"]["tool_calls"].as_array() {
                    markers.tool_calls += calls.len();
                }
            }
        }
        markers
    }

    /// The request's response through the preprocessor, as the frontend composes it, over
    /// stages that send `outputs` and then do `at_end`.
    async fn through_pipeline<Req, Resp>(
        preprocessor: Arc<OpenAIPreprocessor>,
        request: Req,
        outputs: Vec<BackendOutput>,
        at_end: fn(&dyn AsyncEngineContext),
    ) -> Vec<Annotated<Resp>>
    where
        Req: dynamo_runtime::engine::Data,
        Resp: dynamo_runtime::engine::Data,
        PreprocessorNode<Req>: gdl::Around<
                UpReq = dynamo_gdl_adapters::Boxed<Req>,
                UpItem = Annotated<Resp>,
                DownReq = dynamo_gdl_adapters::Boxed<PreprocessedRequest>,
                DownItem = Annotated<BackendOutput>,
            >,
    {
        let below = Arc::new(SignalAtEnd { outputs, at_end });
        crate::operator::testing::preprocessor::<Req, Resp>(preprocessor, below)
            .generate(Context::new(request))
            .await
            .expect("the preprocessor accepts the request")
            .collect()
            .await
    }

    /// The request's response from the preprocessor node alone under `cx`, over stages whose
    /// stream sends `outputs` and ends cleanly. No Dynamo caller sets a deadline on a gdl
    /// request, so a deadline test drives the node with a `Cx` that has one.
    async fn through_node<Req, Resp>(
        preprocessor: Arc<OpenAIPreprocessor>,
        request: Req,
        outputs: Vec<BackendOutput>,
        cx: &gdl::Cx,
    ) -> Vec<Annotated<Resp>>
    where
        Req: dynamo_runtime::engine::Data,
        PreprocessorNode<Req>: gdl::Around<
                UpReq = dynamo_gdl_adapters::Boxed<Req>,
                UpItem = Annotated<Resp>,
                DownReq = dynamo_gdl_adapters::Boxed<PreprocessedRequest>,
                DownItem = Annotated<BackendOutput>,
            >,
    {
        let below: BelowEngine = Arc::new(SignalAtEnd {
            outputs: vec![],
            at_end: |_| {},
        });
        let node = PreprocessorNode::<Req>::new(preprocessor, below);
        let (_sent, state) = node
            .forward(cx, Box::new(Context::new(request)))
            .await
            .unwrap_or_else(|_| panic!("the preprocessor accepts the request"));
        let input = gdl::items(stream::iter(
            outputs
                .into_iter()
                .map(|output| Ok(Annotated::from_data(output))),
        ));
        node.backward(cx, state, input)
            .map(|item| item.unwrap_or_else(|_| panic!("the node ends no stream with an `Err`")))
            .collect()
            .await
    }

    fn past_deadline() -> gdl::Cx {
        gdl::Cx::new().with_timeout(Duration::ZERO)
    }

    fn finished(usage_chunks: usize) -> Markers {
        Markers {
            finish_reasons: vec!["stop".to_string()],
            usage_chunks,
            tool_calls: 0,
        }
    }

    /// The finish the engine sent before the kill passes through; the usage chunk the
    /// preprocessor writes when the stream ends does not.
    #[tokio::test]
    async fn chat_writes_no_usage_chunk_after_a_kill() {
        let items = through_pipeline::<
            _,
            crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        >(
            preprocessor(TINY_LLAMA, None),
            chat_request(),
            finished_answer(),
            kill,
        )
        .await;
        assert_eq!(markers(&items), finished(0), "{items:?}");
    }

    #[tokio::test]
    async fn chat_still_writes_its_usage_chunk_after_a_stop() {
        let items = through_pipeline::<
            _,
            crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        >(
            preprocessor(TINY_LLAMA, None),
            chat_request(),
            finished_answer(),
            stop,
        )
        .await;
        assert_eq!(markers(&items), finished(1), "{items:?}");
    }

    #[tokio::test]
    async fn chat_writes_no_usage_chunk_after_the_deadline() {
        let control = through_node::<
            _,
            crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        >(
            preprocessor(TINY_LLAMA, None),
            chat_request(),
            finished_answer(),
            &gdl::Cx::new(),
        )
        .await;
        assert_eq!(markers(&control), finished(1), "{control:?}");

        let items = through_node::<
            _,
            crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        >(
            preprocessor(TINY_LLAMA, None),
            chat_request(),
            finished_answer(),
            &past_deadline(),
        )
        .await;
        assert_eq!(markers(&items), finished(0), "{items:?}");
    }

    #[tokio::test]
    async fn completion_writes_no_usage_chunk_after_a_kill() {
        let items =
            through_pipeline::<_, crate::types::openai::completions::NvCreateCompletionResponse>(
                preprocessor(TINY_LLAMA, None),
                completion_request(),
                finished_answer(),
                kill,
            )
            .await;
        assert_eq!(markers(&items), finished(0), "{items:?}");
    }

    #[tokio::test]
    async fn completion_still_writes_its_usage_chunk_after_a_stop() {
        let items =
            through_pipeline::<_, crate::types::openai::completions::NvCreateCompletionResponse>(
                preprocessor(TINY_LLAMA, None),
                completion_request(),
                finished_answer(),
                stop,
            )
            .await;
        assert_eq!(markers(&items), finished(1), "{items:?}");
    }

    #[tokio::test]
    async fn completion_writes_no_usage_chunk_after_the_deadline() {
        let control =
            through_node::<_, crate::types::openai::completions::NvCreateCompletionResponse>(
                preprocessor(TINY_LLAMA, None),
                completion_request(),
                finished_answer(),
                &gdl::Cx::new(),
            )
            .await;
        assert_eq!(markers(&control), finished(1), "{control:?}");

        let items =
            through_node::<_, crate::types::openai::completions::NvCreateCompletionResponse>(
                preprocessor(TINY_LLAMA, None),
                completion_request(),
                finished_answer(),
                &past_deadline(),
            )
            .await;
        assert_eq!(markers(&items), finished(0), "{items:?}");
    }

    /// What the tool-call jail writes when the stream ends with a call still open.
    async fn jail_after(at_end: fn(&dyn AsyncEngineContext)) -> Markers {
        let items = through_pipeline::<
            _,
            crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        >(
            preprocessor(TOOL_MODEL, Some("hermes")),
            tool_request(),
            open_tool_call(),
            at_end,
        )
        .await;
        markers(&items)
    }

    /// On a clean end the jail closes the open call and finishes the answer. After a kill
    /// it writes nothing of that.
    #[tokio::test]
    async fn the_jail_writes_no_tool_call_or_finish_after_a_kill() {
        let stopped = jail_after(stop).await;
        assert_ne!(stopped, Markers::default(), "the control must write on end");
        assert_eq!(jail_after(kill).await, Markers::default());
    }

    #[tokio::test]
    async fn the_jail_writes_no_tool_call_or_finish_after_the_deadline() {
        let jail = |cx: gdl::Cx| async move {
            let items = through_node::<
                _,
                crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse,
            >(
                preprocessor(TOOL_MODEL, Some("hermes")),
                tool_request(),
                open_tool_call(),
                &cx,
            )
            .await;
            markers(&items)
        };
        assert_ne!(jail(gdl::Cx::new()).await, Markers::default());
        assert_eq!(jail(past_deadline()).await, Markers::default());
    }

    /// The payload record of a chat request whose stream sends [`finished_answer`] and then
    /// does `at_end`: whether the payload is complete, and its drop reason.
    async fn payload_after(
        request_id: &str,
        at_end: fn(&dyn AsyncEngineContext),
    ) -> (bool, Option<String>) {
        use crate::request_trace::{RequestTraceEventType, subscribe};
        let mut records = subscribe();
        let preprocessor = preprocessor(TINY_LLAMA, None);
        let mut request = chat_request();
        let context = Context::with_id_and_metadata((), request_id.to_string(), Default::default());
        let controller = context.context();
        let (_, mut state, _) = preprocessor
            .chat_forward(&mut request, context)
            .await
            .unwrap();
        state.payload_handle = Some(
            crate::request_trace::payload::RequestPayloadHandle::for_test(
                request_id,
                "test-model",
                true,
            ),
        );
        state.payload_parsing_options =
            Some(crate::protocols::openai::ParsingOptions::new(None, None));
        let signalled = controller.clone();
        let end = stream::poll_fn(move |_| {
            at_end(signalled.as_ref());
            std::task::Poll::Ready(None)
        });
        let input = stream::iter(
            finished_answer()
                .into_iter()
                .map(|o| Ok(Annotated::from_data(o))),
        )
        .chain(end);
        let input = dynamo_gdl_adapters::AnnotatedItems::new(
            gdl::items(input),
            gdl::Cx::new(),
            controller.clone(),
        );
        let _: Vec<_> = preprocessor
            .chat_backward(&request, state, None, input, controller, || {
                panic!("this request does not enable speculative prefill")
            })
            .collect()
            .await;
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = records.recv().await.expect("the trace bus is open");
                if record.event_type != RequestTraceEventType::RequestPayload {
                    continue;
                }
                let Some(payload) = record.payload else {
                    continue;
                };
                if payload.request_id == request_id {
                    return (payload.payload_complete, payload.payload_drop_reason);
                }
            }
        })
        .await
        .expect("the payload record arrives")
    }

    /// The payload capture records a request cut short as incomplete, not as a finished one.
    #[tokio::test]
    #[serial_test::serial]
    async fn the_payload_capture_records_a_killed_request_as_cut_short() {
        crate::request_trace::init_bus_for_test(16);
        assert_eq!(payload_after("payload-stopped", stop).await, (true, None));
        assert_eq!(
            payload_after("payload-killed", kill).await,
            (false, Some(super::DROP_CUT_SHORT.to_string()))
        );
    }
}
