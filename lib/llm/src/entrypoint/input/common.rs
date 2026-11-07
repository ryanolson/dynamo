// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::time::Duration;

use anyhow::Context as _;
use dynamo_renderer::PromptFormatter;
use futures::future::Either;

use crate::{
    backend::{Backend, ExecutionContext},
    discovery::{ModelManager, ModelWatcher},
    engines::StreamingEngineAdapter,
    entrypoint::EngineConfig,
    http::service::metrics::Metrics,
    kv_router::indexer::{preprocessed_multimodal_cache_keys, try_build_cache_indexer},
    kv_router::{EncoderRouter, KvRouter, PrefillRouter, RoutingHost, RoutingLoadContext},
    migration::Migration,
    model_card::ModelDeploymentCard,
    namespace::NamespaceFilter,
    operator::GdlOperator,
    preprocessor::{
        OpenAIPreprocessor, PreprocessedEmbeddingRequest, PreprocessorNode,
        prompt::prompt_formatter_from_mdc,
    },
    protocols::common::llm_backend::{
        BackendOutput, EmbeddingsEngineOutput, LLMEngineOutput, PreprocessedRequest,
    },
    request_template::RequestTemplate,
    session_affinity::{SessionAffinityMode, create_affinity_coordinator},
    types::{
        Annotated,
        openai::{
            chat_completions::{
                NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
                OpenAIChatCompletionsStreamingEngine,
            },
            embeddings::{
                NvCreateEmbeddingRequest, NvCreateEmbeddingResponse,
                OpenAIEmbeddingsStreamingEngine,
            },
        },
    },
};

use dynamo_gdl_adapters::{Boxed, EngineSink, PipelineEngine};
use dynamo_kv_router::config::min_initial_workers_from_env;
use dynamo_runtime::{
    DistributedRuntime,
    component::Client,
    engine::{AsyncEngine, Data},
    pipeline::{ManyOut, MultimodalCacheKeyExtractor, RouterMode, ServiceEngine, SingleIn},
};
use std::sync::Arc;

#[derive(Clone)]
pub struct PreprocessedRouting {
    backend: RoutingTerminal,
    prefill_router: Arc<PrefillRouter>,
    encoder_router: Arc<EncoderRouter>,
}

/// What a routed pipeline sends its requests to: the worker set's `RoutingHost`, or an engine
/// that a test puts in its place.
#[derive(Clone)]
enum RoutingTerminal {
    Host(Arc<RoutingHost>),
    /// A test's engine, as a gdl sink.
    Engine(TerminalSink),
}

/// A test's engine as the gdl sink of [`RoutingTerminal::Engine`].
type TerminalSink = EngineSink<
    dyn AsyncEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
            anyhow::Error,
        >,
    PreprocessedRequest,
    LLMEngineOutput,
    Boxed<PreprocessedRequest>,
>;

/// The terminal as the sink of a gdl local pipeline: `RoutingHost`'s own sink, or the test's
/// engine through `EngineSink`.
impl gdl::Sink for RoutingTerminal {
    type Req = Boxed<PreprocessedRequest>;
    type Item = Annotated<LLMEngineOutput>;

    fn generate(
        &self,
        cx: gdl::Cx,
        request: Boxed<PreprocessedRequest>,
    ) -> impl Future<Output = gdl::Result<gdl::Items<Annotated<LLMEngineOutput>>>> + Send {
        match self {
            Self::Host(host) => Either::Left(gdl::Sink::generate(host.as_ref(), cx, request)),
            Self::Engine(sink) => Either::Right(gdl::Sink::generate(sink, cx, request)),
        }
    }
}

pub struct PreparedEngine {
    pub service_name: String,
    pub engine: OpenAIChatCompletionsStreamingEngine,
    pub inspect_template: bool,
    pub request_template: Option<RequestTemplate>,
}

async fn wait_for_min_initial_workers(
    client: &Client,
    min_initial_workers: usize,
) -> anyhow::Result<()> {
    if min_initial_workers == 0 {
        return Ok(());
    }

    if min_initial_workers == 1 {
        client.wait_for_instances().await?;
        return Ok(());
    }

    let mut watcher = client.instance_avail_watcher();
    loop {
        let available = watcher.borrow_and_update().len();
        if available >= min_initial_workers {
            return Ok(());
        }

        tokio::time::timeout(Duration::from_secs(120), watcher.changed())
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "timed out waiting for {} initial workers for endpoint {}",
                    min_initial_workers,
                    client.endpoint.id()
                )
            })?
            .map_err(|_| {
                anyhow::anyhow!(
                    "instance watcher closed before {} workers appeared for endpoint {}",
                    min_initial_workers,
                    client.endpoint.id()
                )
            })?;
    }
}

fn router_client(
    client: &Client,
    router_mode: RouterMode,
    chooser: Option<&Arc<KvRouter>>,
) -> anyhow::Result<Client> {
    if router_mode == RouterMode::KV {
        let Some(chooser) = chooser else {
            anyhow::bail!("RouterMode::KV requires KVRouter to not be null");
        };
        Ok(chooser.client().clone())
    } else {
        Ok(client.clone())
    }
}

/// LoRA-aware routing is only implemented for the KV, Random, and RoundRobin modes. `Direct`
/// dispatches to a caller-chosen worker and bypasses both the LoRA filter and non-KV load
/// tracking; the advanced load-based modes (`PowerOfTwoChoices`, `LeastLoaded`,
/// `DeviceAwareWeighted`) have no 2-stage LoRA filtering. Reject those combinations so a
/// misconfiguration fails fast — at startup, before the initial-worker wait — instead of
/// silently mis-routing adapter traffic to a worker without the adapter.
fn validate_router_mode_for_lora(
    router_mode: RouterMode,
    lora_enabled: bool,
    session_affinity_enabled: bool,
) -> anyhow::Result<()> {
    if !lora_enabled {
        return Ok(());
    }
    if session_affinity_enabled
        && matches!(router_mode, RouterMode::Random | RouterMode::RoundRobin)
    {
        anyhow::bail!(
            "session affinity is unsupported with DYN_LORA_ENABLED and {router_mode:?} routing"
        );
    }
    match router_mode {
        RouterMode::KV | RouterMode::Random | RouterMode::RoundRobin => Ok(()),
        RouterMode::Direct
        | RouterMode::PowerOfTwoChoices
        | RouterMode::LeastLoaded
        | RouterMode::DeviceAwareWeighted => anyhow::bail!(
            "LoRA serving (DYN_LORA_ENABLED) is not supported with router mode {router_mode:?}; \
             use KV, Random, or RoundRobin for LoRA-aware routing, or disable LoRA serving."
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn build_preprocessed_routing(
    client: &Client,
    model_manager: Arc<crate::discovery::ModelManager>,
    router_mode: RouterMode,
    load_context: Arc<RoutingLoadContext>,
    chooser: Option<Arc<KvRouter>>,
    prefill_chooser: Option<Arc<PrefillRouter>>,
    encoder_chooser: Option<Arc<EncoderRouter>>,
    enable_multimodal_cache_indexer: bool,
    session_affinity_ttl_secs: Option<u64>,
) -> anyhow::Result<PreprocessedRouting> {
    build_preprocessed_routing_with_session_affinity_mode(
        client,
        model_manager,
        router_mode,
        load_context,
        chooser,
        prefill_chooser,
        encoder_chooser,
        enable_multimodal_cache_indexer,
        session_affinity_ttl_secs,
        SessionAffinityMode::Hard,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_preprocessed_routing_with_session_affinity_mode(
    client: &Client,
    model_manager: Arc<crate::discovery::ModelManager>,
    router_mode: RouterMode,
    load_context: Arc<RoutingLoadContext>,
    chooser: Option<Arc<KvRouter>>,
    prefill_chooser: Option<Arc<PrefillRouter>>,
    encoder_chooser: Option<Arc<EncoderRouter>>,
    enable_multimodal_cache_indexer: bool,
    session_affinity_ttl_secs: Option<u64>,
    session_affinity_mode: SessionAffinityMode,
) -> anyhow::Result<PreprocessedRouting> {
    // Fail fast on an unsupported LoRA + router-mode combination BEFORE waiting for the initial
    // worker set, so a misconfiguration surfaces immediately at startup rather than after the
    // (possibly long) DYN_ROUTER_MIN_INITIAL_WORKERS wait.
    validate_router_mode_for_lora(
        router_mode,
        model_manager.lora_enabled(),
        session_affinity_ttl_secs.is_some(),
    )?;
    let min_initial_workers = min_initial_workers_from_env()?;
    let router_client = router_client(client, router_mode, chooser.as_ref())?;

    wait_for_min_initial_workers(&router_client, min_initial_workers).await?;
    let endpoint_id = router_client.endpoint.id();

    let ttl = session_affinity_ttl_secs.map(Duration::from_secs);
    let affinity = match (ttl, chooser.as_ref()) {
        (Some(ttl), Some(chooser)) => {
            let affinity = chooser.affinity_coordinator(ttl, session_affinity_mode)?;
            affinity.enable_replica_sync(router_client.clone()).await?;
            Some(affinity)
        }
        _ => create_affinity_coordinator(ttl, session_affinity_mode, router_client.clone()).await?,
    };

    let embedding_cache_indexer = if enable_multimodal_cache_indexer
        && matches!(router_mode, RouterMode::DeviceAwareWeighted)
    {
        try_build_cache_indexer(&router_client.endpoint).await
    } else {
        None
    };
    let cache_key_extractor = embedding_cache_indexer.as_ref().map(|_| {
        Arc::new(preprocessed_multimodal_cache_keys)
            as MultimodalCacheKeyExtractor<PreprocessedRequest>
    });

    let prefill_router = prefill_chooser.unwrap_or_else(|| {
        PrefillRouter::disabled_with_session_affinity_mode(
            model_manager.clone(),
            router_mode,
            session_affinity_ttl_secs,
            session_affinity_mode,
        )
    });
    let encoder_router = encoder_chooser.unwrap_or_else(EncoderRouter::disabled);

    // LoRA-aware selection lives in the host only for the built-in modes; under
    // KV routing the KV router applies the adapter filter itself.
    let lora = (router_mode != RouterMode::KV)
        .then(|| {
            model_manager
                .lora_filter_for(&endpoint_id)
                .map(|filter| (filter, model_manager.lora_load_estimator_for(&endpoint_id)))
        })
        .flatten();
    let routing_host = Arc::new(
        RoutingHost::builder(router_client, router_mode)
            .kv_router(chooser)
            .load_context(load_context)
            .affinity_coordinator(affinity)
            .lora(lora)
            .multimodal_cache(embedding_cache_indexer.zip(cache_key_extractor))
            .build()
            .await?,
    );
    if router_mode.is_kv_routing() && prefill_router.conditional_disagg_enabled() {
        prefill_router
            .set_decode_routing_host(routing_host.clone())
            .context("install conditional-disagg decode RoutingHost")?;
    }
    Ok(PreprocessedRouting::with_routing_host(
        routing_host,
        prefill_router,
        encoder_router,
    ))
}

/// Turns an EngineConfig into an OpenAI chat-completions and completions supported StreamingEngine.
pub async fn prepare_engine(
    distributed_runtime: DistributedRuntime,
    engine_config: EngineConfig,
) -> anyhow::Result<PreparedEngine> {
    match engine_config {
        EngineConfig::Dynamic {
            model: local_model,
            prefill_load_estimator,
            ..
        } => {
            let model_manager = Arc::new(ModelManager::new());
            // Create metrics for migration tracking (not exposed via /metrics in Dynamic engine mode)
            let metrics = Arc::new(Metrics::new());
            let mut watcher = ModelWatcher::new(
                distributed_runtime.clone(),
                model_manager.clone(),
                local_model.router_config().clone(),
                local_model.migration_limit(),
                local_model.migration_max_seq_len(),
                None,
                prefill_load_estimator,
                metrics,
            );
            if !local_model.path().as_os_str().is_empty() {
                watcher.set_local_model_path(Some(local_model.path().to_path_buf()));
            }
            watcher.set_tokenizer_backend(local_model.runtime_config().tokenizer_backend);
            watcher.set_tokenizer_fallback_enabled(
                local_model.runtime_config().tokenizer_fallback_enabled,
            );
            let watch_obj = Arc::new(watcher);
            let discovery = distributed_runtime.discovery();
            let discovery_stream = discovery
                .list_and_watch(
                    dynamo_runtime::discovery::DiscoveryQuery::AllModels,
                    Some(distributed_runtime.primary_token().clone()),
                )
                .await?;
            let inner_watch_obj = watch_obj.clone();
            let namespace_filter = NamespaceFilter::from_namespace_and_prefix(
                local_model.namespace(),
                local_model.namespace_prefix(),
            );
            let _watcher_task = tokio::spawn(async move {
                inner_watch_obj
                    .watch(discovery_stream, namespace_filter)
                    .await;
            });
            tracing::info!("Waiting for remote model..");

            // TODO: We use the first model to appear, usually we have only one
            // We should add slash commands to text input `/model <name>` to choose,
            // '/models` to list, and notifications when models are added / removed.

            let model_service_name = watch_obj.wait_for_chat_model().await;
            tracing::info!("Connected to {model_service_name}");
            // In disaggregated deployments the model may be listed before the prefill
            // router is fully activated, causing a transient ModelUnavailable. Retry
            // with a timeout so the startup path doesn't fail during this cold-start
            // window, but also doesn't hang indefinitely on misconfiguration.
            let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
            let engine = loop {
                match model_manager.get_chat_completions_engine(&model_service_name) {
                    Ok(engine) => break engine,
                    Err(crate::discovery::ModelManagerError::ModelUnavailable(_))
                        if tokio::time::Instant::now() < deadline =>
                    {
                        tracing::debug!(
                            model = %model_service_name,
                            "Model listed but not yet servable, waiting for prefill activation"
                        );
                        tokio::time::sleep(Duration::from_millis(500)).await;
                    }
                    Err(e) => return Err(e.into()),
                }
            };
            Ok(PreparedEngine {
                service_name: model_service_name,
                engine,
                inspect_template: false,
                request_template: local_model.request_template(),
            })
        }
        EngineConfig::InProcessText { engine, model, .. } => {
            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with engine pre-processing");
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            Ok(PreparedEngine {
                service_name,
                engine,
                inspect_template: false,
                request_template: model.request_template(),
            })
        }
        EngineConfig::InProcessTokens {
            engine: inner_engine,
            model,
            ..
        } => {
            let pipeline = build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(model.card(), inner_engine, model.card().tokenizer()?)
            .await?;

            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with Dynamo pre-processing");
            Ok(PreparedEngine {
                service_name,
                engine: pipeline,
                inspect_template: true,
                request_template: model.request_template(),
            })
        }
    }
}

/// Build the in-process pipeline: preprocessor, then backend, then `engine`. The HTTP and gRPC
/// inputs build it over an in-process token engine.
///
/// One hop to the engine, with the preprocessor and Backend as its around nodes.
pub async fn build_pipeline<Req, Resp>(
    card: &ModelDeploymentCard,
    engine: ExecutionContext,
    tokenizer: crate::tokenizers::Tokenizer,
) -> anyhow::Result<ServiceEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>>>
where
    Req: Data,
    Resp: Data,
    PreprocessorNode<Req>: gdl::Around<
            UpReq = Boxed<Req>,
            UpItem = Annotated<Resp>,
            DownReq = Boxed<PreprocessedRequest>,
            DownItem = Annotated<BackendOutput>,
        >,
{
    use gdl::{Hop, SinkStage};
    type Pre = PreprocessedRequest;

    let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(card)?;
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone())?;
    let backend = Backend::from_tokenizer(tokenizer);

    let engine = SinkStage::new(Arc::new(EngineSink::<_, Pre, _, Boxed<Pre>>::new(engine)));
    let below = Arc::new(
        PipelineEngine::<_, Pre, BackendOutput, Boxed<Pre>>::for_dynamo_stages(
            around::<_, Pre, _>(backend.clone(), engine.clone()),
        ),
    );
    let (preprocessor_forward, preprocessor_backward) =
        gdl::around(Arc::new(PreprocessorNode::<Req>::new(preprocessor, below)));
    let (backend_forward, backend_backward) =
        gdl::around(Arc::new(GdlOperator::<_, Pre>::new(backend)));
    let hop = Hop::new(
        (preprocessor_forward, (backend_forward, ())),
        engine,
        (backend_backward, (preprocessor_backward, ())),
    );
    Ok(Arc::new(
        PipelineEngine::<_, Req, Resp, Boxed<Req>>::for_dynamo_stages(gdl::Source::new(hop)),
    ))
}

/// Build the embeddings pipeline: the preprocessor, a native gdl operator, over `engine`.
/// Backend passes an embeddings response through unchanged, so the pipeline has none.
///
/// The caller builds `preprocessor` (`OpenAIPreprocessor::new_for_embeddings`), so a card it
/// cannot serve fails before the caller creates `engine`.
pub fn build_embeddings_pipeline(
    preprocessor: Arc<OpenAIPreprocessor>,
    engine: ServiceEngine<
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
    >,
) -> OpenAIEmbeddingsStreamingEngine {
    use gdl::{Hop, OperatorStage, SinkStage};
    type Req = NvCreateEmbeddingRequest;
    type Resp = NvCreateEmbeddingResponse;

    let router = SinkStage::new(Arc::new(EngineSink::new(engine)));
    let preprocessor = OperatorStage::new(
        Arc::new(GdlOperator::<_, Req>::new(preprocessor)),
        Hop::new((), router, ()),
    );
    Arc::new(PipelineEngine::<_, Req, Resp>::for_dynamo_stages(
        gdl::Source::new(Hop::new((), preprocessor, ())),
    ))
}

impl PreprocessedRouting {
    /// Route through `routing_host` behind the given prefill and encoder routers.
    /// [`build_preprocessed_routing`] builds the host from a discovery client.
    pub(crate) fn with_routing_host(
        routing_host: Arc<RoutingHost>,
        prefill_router: Arc<PrefillRouter>,
        encoder_router: Arc<EncoderRouter>,
    ) -> Self {
        Self::with_terminal(
            RoutingTerminal::Host(routing_host),
            prefill_router,
            encoder_router,
        )
    }

    /// Route through `backend_engine`, in place of a `RoutingHost`, behind the given prefill
    /// and encoder routers: a test stands in for the workers.
    pub fn new(
        backend_engine: ServiceEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
        >,
        prefill_router: Arc<PrefillRouter>,
        encoder_router: Arc<EncoderRouter>,
    ) -> Self {
        Self::with_terminal(
            RoutingTerminal::Engine(TerminalSink::new(backend_engine)),
            prefill_router,
            encoder_router,
        )
    }

    fn with_terminal(
        backend: RoutingTerminal,
        prefill_router: Arc<PrefillRouter>,
        encoder_router: Arc<EncoderRouter>,
    ) -> Self {
        Self {
            backend,
            prefill_router,
            encoder_router,
        }
    }

    #[cfg(test)]
    pub(crate) fn prefill_router(&self) -> &Arc<PrefillRouter> {
        &self.prefill_router
    }

    /// The normal way to build an inference pipeline. Connect this directly to HTTP layer.
    ///
    /// Each operator calls the hop below it directly, with no response map per stage.
    pub fn build_pipeline<Req, Resp>(
        &self,
        card: &ModelDeploymentCard,
        preprocessor: Arc<OpenAIPreprocessor>,
        tokenizer: crate::tokenizers::Tokenizer,
        migration_limit: u32,
        migration_max_seq_len: Option<u32>,
        metrics: Arc<Metrics>,
    ) -> anyhow::Result<ServiceEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>>>
    where
        Req: Data,
        Resp: Data,
        PreprocessorNode<Req>: gdl::Around<
                UpReq = Boxed<Req>,
                UpItem = Annotated<Resp>,
                DownReq = Boxed<PreprocessedRequest>,
                DownItem = Annotated<BackendOutput>,
            >,
    {
        let token_backend = Backend::from_tokenizer(tokenizer);
        let migration = Migration::from_mdc(card, migration_limit, migration_max_seq_len, metrics);
        Ok(Arc::new(
            PipelineEngine::<_, Req, Resp, Boxed<Req>>::for_dynamo_stages(self.gdl_pipeline(
                preprocessor,
                migration,
                token_backend,
            )),
        ))
    }

    /// The stages of [`build_pipeline`](Self::build_pipeline), each a native gdl operator,
    /// built from the terminal up, since each stage owns the hop it calls.
    fn gdl_pipeline<Req, Resp>(
        &self,
        preprocessor: Arc<OpenAIPreprocessor>,
        migration: Arc<Migration>,
        token_backend: Arc<Backend>,
    ) -> gdl::Source<impl gdl::Next<Boxed<Req>, Annotated<Resp>>>
    where
        Req: Data,
        Resp: Data,
        PreprocessorNode<Req>: gdl::Around<
                UpReq = Boxed<Req>,
                UpItem = Annotated<Resp>,
                DownReq = Boxed<PreprocessedRequest>,
                DownItem = Annotated<BackendOutput>,
            >,
    {
        type Pre = PreprocessedRequest;
        // Migration over Backend's hop to the routers: the inner end of the preprocessor's hop,
        // and, as an engine of its own, where a speculative prefill is sent.
        let migration = gdl::OperatorStage::new(
            Arc::new(GdlOperator::<_, BackendOutput>::new(migration)),
            around::<_, Pre, _>(token_backend, self.gdl_routers()),
        );
        let below =
            Arc::new(
                PipelineEngine::<_, Pre, BackendOutput, Boxed<Pre>>::for_dynamo_stages(
                    gdl::Hop::new((), migration.clone(), ()),
                ),
            );
        let (forward, backward) =
            gdl::around(Arc::new(PreprocessorNode::<Req>::new(preprocessor, below)));
        gdl::Source::new(gdl::Hop::new((forward, ()), migration, (backward, ())))
    }

    /// The encoder router, the prefill router and the terminal as native gdl stages: the tail
    /// of [`gdl_pipeline`](Self::gdl_pipeline) and
    /// [`gdl_preprocessed_pipeline`](Self::gdl_preprocessed_pipeline), as the inner end of the
    /// hop that calls the encoder router.
    fn gdl_routers(
        &self,
    ) -> impl gdl::Inner<Boxed<PreprocessedRequest>, Item = Annotated<LLMEngineOutput>> {
        use gdl::{Hop, OperatorStage, SinkStage};
        type Pre = PreprocessedRequest;

        let terminal = Hop::new((), SinkStage::new(Arc::new(self.backend.clone())), ());
        let prefill = native::<_, Pre, _>(self.prefill_router.clone(), terminal);
        OperatorStage::new(
            Arc::new(GdlOperator::<_, Pre>::new(self.encoder_router.clone())),
            prefill,
        )
    }

    /// Bring your own pre/post processor. Used when frontend has `--dyn-chat-processor
    /// vllm|sglang`, and by the Generate engine. Built as [`build_pipeline`](Self::build_pipeline)
    /// builds its own.
    pub fn build_preprocessed_pipeline(
        &self,
        card: &ModelDeploymentCard,
        migration_limit: u32,
        migration_max_seq_len: Option<u32>,
        metrics: Arc<Metrics>,
    ) -> anyhow::Result<
        ServiceEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>>,
    > {
        let migration = Migration::from_mdc(card, migration_limit, migration_max_seq_len, metrics);
        Ok(Arc::new(PipelineEngine::<
            _,
            PreprocessedRequest,
            LLMEngineOutput,
            Boxed<PreprocessedRequest>,
        >::for_dynamo_stages(
            self.gdl_preprocessed_pipeline(migration),
        )))
    }

    /// The stages of [`build_preprocessed_pipeline`](Self::build_preprocessed_pipeline).
    fn gdl_preprocessed_pipeline(
        &self,
        migration: Arc<Migration>,
    ) -> gdl::Source<impl gdl::Next<Boxed<PreprocessedRequest>, Annotated<LLMEngineOutput>>> {
        gdl::Source::new(native::<_, LLMEngineOutput, _>(
            migration,
            gdl::Hop::new((), self.gdl_routers(), ()),
        ))
    }
}

/// A hop to `I` with `GdlOperator<O, K>` as its one around node.
type AroundHop<O, K, I> = gdl::Hop<
    (gdl::ForwardHalf<GdlOperator<O, K>>, ()),
    I,
    (gdl::BackwardHalf<GdlOperator<O, K>>, ()),
>;

/// `op` as an around node of the hop to `inner`, keyed by `K` (see [`GdlOperator`]).
fn around<O, K, I>(op: Arc<O>, inner: I) -> AroundHop<O, K, I>
where
    GdlOperator<O, K>: gdl::Around,
{
    let (forward, backward) = gdl::around(Arc::new(GdlOperator::new(op)));
    gdl::Hop::new((forward, ()), inner, (backward, ()))
}

/// `op` as a native gdl operator, keyed by `K` (see [`GdlOperator`]), over the hop `next`.
fn native<O, K, H>(
    op: Arc<O>,
    next: H,
) -> gdl::Hop<(), gdl::OperatorStage<GdlOperator<O, K>, H>, ()> {
    gdl::Hop::new(
        (),
        gdl::OperatorStage::new(Arc::new(GdlOperator::new(op)), next),
        (),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gdl nodes of pipelines A, B and C carry their requests boxed, so gdl's chain moves a
    /// pointer through each node and future, not the request: about 2 KB for a chat request and
    /// 1 KB for a preprocessed one, a `memcpy` per move.
    #[test]
    fn gdl_nodes_carry_their_requests_boxed() {
        use crate::protocols::openai::completions::NvCreateCompletionRequest;
        use std::mem::size_of;

        type Chat = PreprocessorNode<NvCreateChatCompletionRequest>;
        type Completion = PreprocessorNode<NvCreateCompletionRequest>;
        type Pre = PreprocessedRequest;
        type Op<O, K> = GdlOperator<O, K>;
        let sizes = [
            (
                "chat preprocessor in",
                size_of::<<Chat as gdl::Around>::UpReq>(),
            ),
            (
                "chat preprocessor out",
                size_of::<<Chat as gdl::Around>::DownReq>(),
            ),
            (
                "completion preprocessor in",
                size_of::<<Completion as gdl::Around>::UpReq>(),
            ),
            (
                "completion preprocessor out",
                size_of::<<Completion as gdl::Around>::DownReq>(),
            ),
            (
                "backend in",
                size_of::<<Op<Backend, Pre> as gdl::Around>::UpReq>(),
            ),
            (
                "backend out",
                size_of::<<Op<Backend, Pre> as gdl::Around>::DownReq>(),
            ),
            (
                "migration in",
                size_of::<<Op<Migration, BackendOutput> as gdl::Operator>::UpReq>(),
            ),
            (
                "migration out",
                size_of::<<Op<Migration, LLMEngineOutput> as gdl::Operator>::DownReq>(),
            ),
            (
                "encoder router in",
                size_of::<<Op<EncoderRouter, Pre> as gdl::Operator>::UpReq>(),
            ),
            (
                "encoder router out",
                size_of::<<Op<EncoderRouter, Pre> as gdl::Operator>::DownReq>(),
            ),
            (
                "prefill router in",
                size_of::<<Op<PrefillRouter, Pre> as gdl::Operator>::UpReq>(),
            ),
            (
                "prefill router out",
                size_of::<<Op<PrefillRouter, Pre> as gdl::Operator>::DownReq>(),
            ),
            (
                "routing terminal",
                size_of::<<RoutingTerminal as gdl::Sink>::Req>(),
            ),
            ("routing host", size_of::<<RoutingHost as gdl::Sink>::Req>()),
        ];
        let large: Vec<_> = sizes
            .iter()
            .filter(|(_, size)| *size != size_of::<usize>())
            .collect();
        assert!(
            large.is_empty(),
            "requests carried inline, in bytes: {large:?}"
        );
    }

    /// Pipelines B and C on gdl are native gdl stages from the source to the terminal: no
    /// Dynamo engine at a boundary inside them. The test's engine sits behind the terminal.
    #[test]
    fn routed_gdl_pipelines_are_native_from_source_to_terminal() {
        use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

        let card =
            ModelDeploymentCard::load_from_disk("tests/data/sample-models/TinyLlama_v1.1", None)
                .unwrap();
        let tokenizer = card.tokenizer().unwrap();
        let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(&card).unwrap();
        let preprocessor =
            OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone()).unwrap();
        struct Silent;
        #[dynamo_runtime::pipeline::async_trait]
        impl
            AsyncEngine<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<LLMEngineOutput>>,
                anyhow::Error,
            > for Silent
        {
            async fn generate(
                &self,
                _request: SingleIn<PreprocessedRequest>,
            ) -> anyhow::Result<ManyOut<Annotated<LLMEngineOutput>>> {
                anyhow::bail!("not called")
            }
        }
        let routing = PreprocessedRouting::new(
            Arc::new(Silent),
            PrefillRouter::disabled(
                Arc::new(crate::discovery::ModelManager::new()),
                RouterMode::KV,
                None,
            ),
            EncoderRouter::disabled(),
        );
        let migration = || Migration::new(0, None, String::new(), Arc::new(Metrics::new()));
        let b = routing
            .gdl_pipeline::<NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse>(
                preprocessor,
                migration(),
                Backend::from_tokenizer(tokenizer),
            );
        let c = routing.gdl_preprocessed_pipeline(migration());
        for (name, pipeline) in [
            ("b", std::any::type_name_of_val(&b)),
            ("c", std::any::type_name_of_val(&c)),
        ] {
            for adapter in ["EngineSink", "PipelineEngine"] {
                assert!(
                    !pipeline.contains(adapter),
                    "{name} holds {adapter}: {pipeline}"
                );
            }
            assert!(pipeline.contains("RoutingTerminal"), "{name}: {pipeline}");
        }
    }

    #[test]
    fn test_validate_router_mode_for_lora() {
        use RouterMode::*;
        let all = [
            Direct,
            KV,
            Random,
            RoundRobin,
            PowerOfTwoChoices,
            LeastLoaded,
            DeviceAwareWeighted,
        ];

        // LoRA disabled: every mode is allowed (the unmodified routing path).
        for m in all {
            assert!(
                validate_router_mode_for_lora(m, false, false).is_ok(),
                "{m:?} must be allowed when LoRA serving is disabled"
            );
        }

        // LoRA enabled: only the LoRA-aware modes are accepted.
        for m in [KV, Random, RoundRobin] {
            assert!(
                validate_router_mode_for_lora(m, true, false).is_ok(),
                "{m:?} must be supported for LoRA-aware routing"
            );
        }

        // LoRA enabled: Direct + the advanced load-based modes are rejected with a clear error.
        for m in [Direct, PowerOfTwoChoices, LeastLoaded, DeviceAwareWeighted] {
            let err = validate_router_mode_for_lora(m, true, false)
                .expect_err("must reject unsupported LoRA router mode")
                .to_string();
            assert!(
                err.contains("not supported with router mode"),
                "{m:?} rejection must explain the unsupported mode, got: {err}"
            );
        }

        let err = validate_router_mode_for_lora(Random, true, true)
            .expect_err("must reject session affinity with LoRA-aware non-KV routing");
        assert!(err.to_string().contains("session affinity"));
    }
}
