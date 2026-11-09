// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The response transform's per-item body and its end, apart from the stream that reads
//! Backend's items: shared by the transform's own stream ([`Postprocessed`]) and the chat
//! preprocessor's default response path ([`ChatResponse`]), which runs the transform and the
//! chat edits in one frame.

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use dynamo_gdl_adapters::CutShort;
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

use super::{
    ANNOTATION_PAYLOAD_USAGE, BackendOutput, DETOKENIZE_TOKEN_COUNT, DETOKENIZE_TOTAL_US,
    DeltaGeneratorExt, MultimodalCounts, attach_llm_metrics, build_llm_metric_annotation, cut,
    drain_router_routing_data,
};
use crate::protocols::common::timing::RequestTracker;
use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;

struct DetokenizeMetricsGuard {
    tracker: Option<Arc<RequestTracker>>,
}

impl DetokenizeMetricsGuard {
    #[inline]
    fn tracker(&self) -> Option<&RequestTracker> {
        self.tracker.as_deref()
    }
}

impl Drop for DetokenizeMetricsGuard {
    fn drop(&mut self) {
        let Some(tracker) = self.tracker() else {
            return;
        };
        if let Some(total) = tracker.detokenize_total_latency() {
            DETOKENIZE_TOTAL_US.inc_by(total.as_micros() as f64);
        }
        DETOKENIZE_TOKEN_COUNT.inc_by(tracker.detokenize_count() as f64);
    }
}

/// The response transform's state for one request: Backend's items become the client's chunks,
/// with the usage chunk at the end.
pub(super) struct ResponseTransform<Resp>
where
    Resp: Send + Sync + Clone + 'static + std::fmt::Debug,
{
    response_generator: Box<dyn DeltaGeneratorExt<Resp>>,
    detokenize_metrics: DetokenizeMetricsGuard,
    context: Arc<dyn AsyncEngineContext>,
    pub(super) cancelled: bool,
    cumulative_output_tokens: usize,
    finish_reason_sent: bool,
    usage_chunk_sent: bool,
    /// Buffered plain usage chunk to send to the client after the payload
    /// chunk (ANNOTATION_PAYLOAD_USAGE). Only Some when is_usage_enabled().
    pending_client_usage: Option<Annotated<Resp>>,
    pub(super) finished: bool,
    emit_payload_usage_chunk: bool,
    trace_tokens_enabled: bool,
    trace_finish_reason_metadata: Option<crate::request_trace::SharedFinishReasonMetadata>,
    mm_counts: MultimodalCounts,
    image_tokens: Option<usize>,
    /// Set when the response stream ended because the request was cut short.
    cut: Option<cut::CutFlag>,
}

impl<Resp> ResponseTransform<Resp>
where
    Resp: Send + Sync + Clone + 'static + std::fmt::Debug,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        generator: Box<dyn DeltaGeneratorExt<Resp>>,
        context: Arc<dyn AsyncEngineContext>,
        emit_payload_usage_chunk: bool,
        trace_tokens_enabled: bool,
        trace_finish_reason_metadata: Option<crate::request_trace::SharedFinishReasonMetadata>,
        mm_counts: MultimodalCounts,
        image_tokens: Option<usize>,
        cut: Option<cut::CutFlag>,
    ) -> Self {
        let tracker = generator.tracker();
        Self {
            response_generator: generator,
            detokenize_metrics: DetokenizeMetricsGuard { tracker },
            context,
            cancelled: false,
            cumulative_output_tokens: 0,
            finish_reason_sent: false,
            usage_chunk_sent: false,
            pending_client_usage: None,
            finished: false,
            emit_payload_usage_chunk,
            trace_tokens_enabled,
            trace_finish_reason_metadata,
            mm_counts,
            image_tokens,
            cut,
        }
    }

    /// Split topology: overlay a standalone router's forwarded routing_data
    /// (timing, query-only token_ids) onto this request's tracker so the
    /// frontend's nvext/timing surfaces populate. Run on every item, before
    /// [`ResponseTransform::item`] and before a cancelled request drops it.
    #[inline(always)]
    pub(super) fn drain(&self, response: &mut Annotated<BackendOutput>) {
        drain_router_routing_data(&mut response.data, self.detokenize_metrics.tracker());
    }

    /// One of Backend's items as the client's chunk. An error converting it marks the request
    /// cancelled: the next item ends the stream.
    // hot path: per-item-response
    #[inline(always)]
    pub(super) fn item(&mut self, response: Annotated<BackendOutput>) -> Annotated<Resp> {
        tracing::trace!(
            request_id = self.context.id(),
            "Processing common response: {:?}",
            response
        );

        // Check if this response has a finish_reason
        let has_finish_reason = response
            .data
            .as_ref()
            .map(|d| d.finish_reason.is_some())
            .unwrap_or(false);

        let (chunk_tokens, isl) = if let Some(ref backend_output) = response.data {
            let chunk_tokens = backend_output.token_ids.len();
            self.cumulative_output_tokens += chunk_tokens;

            let isl = self.response_generator.get_isl().map(|isl| isl as usize);

            crate::request_trace::record_backend_finish_reason_metadata(
                self.trace_finish_reason_metadata.as_ref(),
                backend_output.index,
                backend_output.finish_reason.as_ref(),
                backend_output.stop_reason.as_ref(),
            );

            (chunk_tokens, isl)
        } else {
            (0, None)
        };

        let current_osl = self.cumulative_output_tokens;

        let mut response = response.map_data(|data| {
            self.response_generator
                .choice_from_postprocessor(data)
                .inspect_err(|e| {
                    tracing::error!(
                        request_id = self.context.id(),
                        "Error processing common response: {:?}",
                        e
                    );
                    self.cancelled = true;
                    self.context.stop_generating();
                })
                .map_err(|e| e.to_string())
        });

        // Create LLM metrics annotation with prefill/decode worker info from tracker.
        // Worker types are stored at routing time to avoid expensive MDC lookup.
        let tracker = self.detokenize_metrics.tracker();
        let llm_metrics = build_llm_metric_annotation(
            tracker,
            isl.unwrap_or(0),
            current_osl,
            chunk_tokens,
            None,
            self.mm_counts,
            self.image_tokens,
        );
        if self.trace_tokens_enabled {
            crate::request_trace::record_llm_metric_tokens(tracker, isl, current_osl, None);
        }

        attach_llm_metrics(&mut response, llm_metrics);

        // Mark if we've seen a finish_reason
        if has_finish_reason {
            self.finish_reason_sent = true;
        }

        tracing::trace!(
            request_id = self.context.id(),
            "OpenAI NvCreateChatCompletionStreamResponse: {:?}",
            response
        );

        response
    }

    /// The stream of Backend's items ended (`cut_short`: the request was cut short): the usage
    /// chunk, if one is due. Called once.
    #[cold]
    #[inline(never)]
    pub(super) fn end(&mut self, cut_short: bool) -> Option<Annotated<Resp>> {
        // Stream has ended - must set finished to true to prevent unfold from polling
        // again. The stream is exhausted and will panic if polled after None.
        self.finished = true;

        // A request killed or past its deadline ends its stream cleanly too, and
        // its answer is not whole: write nothing more. Checked once, here.
        if cut_short {
            if let Some(cut) = &self.cut {
                cut.set();
            }
            return None;
        }

        if !(self.finish_reason_sent && !self.usage_chunk_sent) {
            // stream closed
            return None;
        }
        self.usage_chunk_sent = true;

        let usage_chunk = self.response_generator.create_usage_chunk();
        let usage = self.response_generator.get_usage();
        let tracker = self.detokenize_metrics.tracker();
        let cached_tokens = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens.map(|c| c as usize));
        let llm_metrics = build_llm_metric_annotation(
            tracker,
            usage.prompt_tokens as usize,
            usage.completion_tokens as usize,
            0,
            cached_tokens,
            self.mm_counts,
            self.image_tokens,
        );
        if self.trace_tokens_enabled {
            crate::request_trace::record_llm_metric_tokens(
                tracker,
                Some(usage.prompt_tokens as usize),
                usage.completion_tokens as usize,
                cached_tokens,
            );
        }

        let usage_requested = self.response_generator.is_usage_enabled();

        if self.emit_payload_usage_chunk {
            // Payload capture on: emit a dedicated payload-usage chunk that
            // always carries usage for the payload DeltaAggregator
            // (the EventConverter strips it entirely from the
            // client), and buffer the plain client usage chunk only
            // when include_usage was requested.
            if usage_requested {
                self.pending_client_usage = Some(Annotated::<Resp> {
                    id: None,
                    data: Some(usage_chunk.clone()),
                    event: None,
                    comment: None,
                    error: None,
                });
            }

            let annotation = llm_metrics.to_annotation::<()>().unwrap_or_else(|e| {
                tracing::warn!("Failed to serialize metrics: {}", e);
                Annotated::<()>::from_data(())
            });
            Some(Annotated::<Resp> {
                id: None,
                data: Some(usage_chunk),
                event: Some(ANNOTATION_PAYLOAD_USAGE.to_string()),
                comment: annotation.comment,
                error: None,
            })
        } else {
            // Payload capture off: a single usage chunk; data is present only
            // when include_usage was requested. Metrics ride via the
            // typed (serde-skip) llm_metrics field for internal
            // observation, never reaching the client.
            let data = if usage_requested {
                Some(usage_chunk)
            } else {
                None
            };
            let mut annotated_usage = Annotated::<Resp> {
                id: None,
                data,
                event: None,
                comment: None,
                error: None,
            };
            attach_llm_metrics(&mut annotated_usage, llm_metrics);
            Some(annotated_usage)
        }
    }
}

/// The transform over its input stream `S`.
pub(super) struct TransformState<S, Resp>
where
    Resp: Send + Sync + Clone + 'static + std::fmt::Debug,
{
    response_stream: S,
    transform: ResponseTransform<Resp>,
}

impl<S, Resp> TransformState<S, Resp>
where
    S: Stream<Item = Annotated<BackendOutput>> + CutShort + Unpin,
    Resp: Send + Sync + Clone + 'static + std::fmt::Debug,
{
    pub(super) fn new(response_stream: S, transform: ResponseTransform<Resp>) -> Self {
        Self {
            response_stream,
            transform,
        }
    }

    /// The next chunk of the response, or `None` when it has ended.
    #[inline(always)]
    fn poll_step(&mut self, cx: &mut Context<'_>) -> Poll<Option<Annotated<Resp>>> {
        // Drain the buffered client-facing plain usage chunk first.
        // This MUST come before the `finished` guard: the stream-end
        // handler sets self.finished = true before returning the payload
        // chunk, so on the very next iteration the finished guard would
        // terminate before we ever emit the client chunk.
        if let Some(client_chunk) = self.transform.pending_client_usage.take() {
            self.transform.finished = true;
            // Emit unconditionally to match the non-payload path below; the
            // chunk is only buffered after a finish_reason, so payload capture must
            // not alter the client SSE tail.
            return Poll::Ready(Some(client_chunk));
        }

        // If already finished (and no pending client chunk), stop.
        if self.transform.finished {
            return Poll::Ready(None);
        }

        match ready!(self.response_stream.poll_next_unpin(cx)) {
            Some(mut response) => {
                self.transform.drain(&mut response);
                if self.transform.cancelled {
                    tracing::debug!(
                        request_id = self.transform.context.id(),
                        "Cancellation issued last message; closing stream"
                    );
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(self.transform.item(response)))
            }
            None => {
                let cut_short = self.response_stream.cut_short();
                Poll::Ready(self.transform.end(cut_short))
            }
        }
    }
}

/// The response stream: a stream that owns its state, in place of `unfold` and
/// `fuse`, which moved the state (with its buffered usage chunk, the size of an item)
/// and the item through a new future on every item. `None` once the stream has ended,
/// which drops the state, and so flushes the detokenize metrics, when this stream
/// ends rather than when its caller drops it.
pub(super) struct Postprocessed<S, Resp>
where
    Resp: Send + Sync + Clone + 'static + std::fmt::Debug,
{
    pub(super) state: Option<TransformState<S, Resp>>,
}

// Nothing in the state is pinned: the stream below is `Unpin`, and the rest is moved
// and edited freely.
impl<S: Unpin, Resp> Unpin for Postprocessed<S, Resp> where
    Resp: Send + Sync + Clone + 'static + std::fmt::Debug
{
}

impl<S, Resp> Stream for Postprocessed<S, Resp>
where
    S: Stream<Item = Annotated<BackendOutput>> + CutShort + Unpin,
    Resp: Send + Sync + Clone + 'static + std::fmt::Debug,
{
    type Item = Annotated<Resp>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Annotated<Resp>>> {
        let Some(state) = self.state.as_mut() else {
            return Poll::Ready(None);
        };
        // Returned as it is: taking the item out of the `Poll` would move it again.
        let poll = state.poll_step(cx);
        if let Poll::Ready(None) = poll {
            self.state = None;
        }
        poll
    }
}

/// Remove tool calls from a chunk, for a request whose tool calls are not parsed.
#[inline(always)]
pub(super) fn strip_tool_calls(response: &mut Annotated<NvCreateChatCompletionStreamResponse>) {
    if let Some(data) = response.data.as_mut() {
        for choice in &mut data.inner.choices {
            choice.delta.tool_calls = None;
            if choice.finish_reason == Some(dynamo_protocols::types::FinishReason::ToolCalls) {
                choice.finish_reason = Some(dynamo_protocols::types::FinishReason::Stop);
            }
        }
    }
}

/// Give each choice's first delta the assistant role, and no other delta a role.
#[inline(always)]
pub(super) fn normalize_roles(
    role_emitted_choices: &mut HashSet<u32>,
    response: &mut Annotated<NvCreateChatCompletionStreamResponse>,
) {
    if let Some(data) = response.data.as_mut() {
        for choice in &mut data.inner.choices {
            choice.delta.role = role_emitted_choices
                .insert(choice.index)
                .then_some(dynamo_protocols::types::Role::Assistant);
        }
    }
}

/// The chat preprocessor's default response path in one frame: the transform, then the
/// tool-call policy, then the role normalization, on each chunk in place. The same per-item
/// code as the layered streams, without a stream, a box and a call per layer per item.
/// It reads Backend's items through `EndedItems`, which gives an ending below as a last item,
/// so it ends cleanly.
pub(super) struct ChatResponse<S> {
    state: Option<TransformState<S, NvCreateChatCompletionStreamResponse>>,
    strip_tool_calls: bool,
    roles: HashSet<u32>,
}

impl<S> ChatResponse<S> {
    pub(super) fn new(
        state: TransformState<S, NvCreateChatCompletionStreamResponse>,
        strip_tool_calls: bool,
    ) -> Self {
        Self {
            state: Some(state),
            strip_tool_calls,
            roles: HashSet::new(),
        }
    }
}

impl<S: Unpin> Unpin for ChatResponse<S> {}

impl<S> Stream for ChatResponse<S>
where
    S: Stream<Item = Annotated<BackendOutput>> + CutShort + Unpin,
{
    type Item = Annotated<NvCreateChatCompletionStreamResponse>;

    // hot path: per-item-response
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let Some(state) = this.state.as_mut() else {
            return Poll::Ready(None);
        };
        // Edited where the transform built it, and returned as it is.
        let mut poll = state.poll_step(cx);
        match &mut poll {
            Poll::Ready(Some(response)) => {
                if this.strip_tool_calls {
                    strip_tool_calls(response);
                }
                normalize_roles(&mut this.roles, response);
            }
            Poll::Ready(None) => this.state = None,
            Poll::Pending => {}
        }
        poll
    }
}

impl<S> gdl::EndedStream for ChatResponse<S>
where
    S: Stream<Item = Annotated<BackendOutput>> + CutShort + Unpin,
{
    fn ending(self: Pin<&mut Self>) -> Option<gdl::Error> {
        None
    }
}

/// The chat preprocessor's backward half as a gdl inline node, for one request on the default
/// path: gdl calls the same per-item code as [`ChatResponse`] on each of Backend's items, in
/// place of `EndedItems` and the frame, and answers the ending below as `EndedItems` does.
///
/// The item changes type in the conversion, which runs the transform and the two edits. The
/// transform's usage chunk at the end is a chat chunk, so `close` keeps it and sends a
/// placeholder of Backend's type, which the conversion swaps for it.
pub struct ChatRun {
    transform: ResponseTransform<NvCreateChatCompletionStreamResponse>,
    strip_tool_calls: bool,
    roles: HashSet<u32>,
    root: gdl::Cx,
    context: Arc<dyn AsyncEngineContext>,
    // The stream form ended here: an item came after a failed one.
    stopped: bool,
    // The usage chunk, sent in place of the placeholder `close` gave.
    usage: Option<Annotated<NvCreateChatCompletionStreamResponse>>,
}

impl ChatRun {
    pub(super) fn new(
        transform: ResponseTransform<NvCreateChatCompletionStreamResponse>,
        strip_tool_calls: bool,
        root: gdl::Cx,
        context: Arc<dyn AsyncEngineContext>,
    ) -> Self {
        Self {
            transform,
            strip_tool_calls,
            roles: HashSet::new(),
            root,
            context,
            stopped: false,
            usage: None,
        }
    }

    /// An item after a failed one: the stream form ends there, so every later item is dropped
    /// and nothing more is written.
    #[cold]
    #[inline(never)]
    fn stop(&mut self) -> gdl::inline::Verdict {
        if !self.stopped {
            tracing::debug!(
                request_id = self.context.id(),
                "Cancellation issued last message; closing stream"
            );
        }
        self.stopped = true;
        gdl::inline::Verdict::Skip
    }

    #[cold]
    #[inline(never)]
    fn take_usage(&mut self) -> Annotated<NvCreateChatCompletionStreamResponse> {
        self.usage.take().expect("the usage chunk was kept")
    }

    /// The chat chunk for one of Backend's items, or the usage chunk for the placeholder.
    // hot path: per-item-response
    #[inline(always)]
    pub(super) fn convert(
        &mut self,
        item: Annotated<BackendOutput>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        if self.usage.is_some() {
            return self.take_usage();
        }
        let mut response = self.transform.item(item);
        if self.strip_tool_calls {
            strip_tool_calls(&mut response);
        }
        normalize_roles(&mut self.roles, &mut response);
        response
    }
}

impl gdl::inline::Edit<Annotated<BackendOutput>> for ChatRun {
    // hot path: per-item-response
    #[inline(always)]
    fn edit(&mut self, item: &mut Annotated<BackendOutput>) -> gdl::inline::Verdict {
        self.transform.drain(item);
        if self.transform.cancelled || self.stopped {
            return self.stop();
        }
        gdl::inline::Verdict::Keep
    }

    // As `EndedItems`: an ending below is a last item, unless it is the cancel of a killed
    // request, which just ends. After a failed item the stream form drops it and ends.
    fn end(&mut self, ending: Option<gdl::Error>) -> gdl::inline::AtEnd<Annotated<BackendOutput>> {
        let Some(mut item) = ending.and_then(|error| {
            dynamo_gdl_adapters::last_item(error, &self.root, self.context.as_ref())
        }) else {
            return gdl::inline::AtEnd::Pass(None);
        };
        self.transform.drain(&mut item);
        if self.transform.cancelled || self.stopped {
            self.stop();
            return gdl::inline::AtEnd::Pass(None);
        }
        gdl::inline::AtEnd::Item(item)
    }

    // As the transform's end: the usage chunk, unless the request was cut short, through the
    // same two edits.
    fn close(&mut self) -> Option<Annotated<BackendOutput>> {
        if self.stopped || self.transform.finished {
            return None;
        }
        let cut_short = dynamo_gdl_adapters::cut_short(&self.root, self.context.as_ref());
        let mut usage = self.transform.end(cut_short)?;
        if self.strip_tool_calls {
            strip_tool_calls(&mut usage);
        }
        normalize_roles(&mut self.roles, &mut usage);
        self.usage = Some(usage);
        Some(Annotated {
            id: None,
            data: None,
            event: None,
            comment: None,
            error: None,
        })
    }
}

/// The inline half of a node that always declines the inline form: the completions
/// preprocessor, whose requests all run through its stream form.
pub enum Declines {}

impl gdl::inline::Edit<Annotated<BackendOutput>> for Declines {
    fn edit(&mut self, _: &mut Annotated<BackendOutput>) -> gdl::inline::Verdict {
        match *self {}
    }
}
