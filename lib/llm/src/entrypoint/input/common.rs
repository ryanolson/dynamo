// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::Context as _;
use dynamo_renderer::PromptFormatter;

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
    preprocessor::{
        OpenAIPreprocessor, PreprocessedEmbeddingRequest, prompt::prompt_formatter_from_mdc,
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

use dynamo_gdl_adapters::{DynOperatorStage, EngineSink, PipelineEngine};
use dynamo_kv_router::config::min_initial_workers_from_env;
use dynamo_runtime::{
    DistributedRuntime,
    component::Client,
    config::environment_names::llm::DYN_LOCAL_PIPELINE,
    engine::{AsyncEngineStream, Data},
    pipeline::{
        Context, ManyOut, MultimodalCacheKeyExtractor, Operator, RouterMode, SegmentSource,
        ServiceBackend, ServiceEngine, ServiceFrontend, SingleIn, Source,
    },
};
use std::sync::Arc;

#[derive(Clone)]
pub struct PreprocessedRouting {
    backend_engine:
        ServiceEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>>,
    prefill_router: Arc<PrefillRouter>,
    encoder_router: Arc<EncoderRouter>,
    local_pipeline: LocalPipeline,
}

/// How a local pipeline composes its stages: [`PreprocessedRouting::build_pipeline`],
/// [`PreprocessedRouting::build_preprocessed_pipeline`], [`build_pipeline`] and
/// [`build_embeddings_pipeline`]. The stages and what they do are the same either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LocalPipeline {
    /// Dynamo's `link` chain.
    #[default]
    Dynamo,
    /// A gdl local pipeline of nested calls, with Dynamo's operators and engine
    /// wrapped by `dynamo-gdl-adapters`.
    Gdl,
}

impl LocalPipeline {
    /// The name `DYN_LOCAL_PIPELINE` takes, and the frontend logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dynamo => "dynamo",
            Self::Gdl => "gdl",
        }
    }

    /// The pipeline `DYN_LOCAL_PIPELINE` selects. The variable is read once per process.
    pub fn configured() -> anyhow::Result<Self> {
        static CONFIGURED: LazyLock<Result<LocalPipeline, String>> = LazyLock::new(|| {
            let value = match std::env::var(DYN_LOCAL_PIPELINE) {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(error) => return Err(format!("invalid {DYN_LOCAL_PIPELINE}: {error}")),
            };
            LocalPipeline::parse(value.as_deref()).map_err(|error| format!("{error:#}"))
        });
        CONFIGURED.clone().map_err(anyhow::Error::msg)
    }

    fn parse(value: Option<&str>) -> anyhow::Result<Self> {
        match value.map(str::trim) {
            None | Some("") => Ok(Self::Dynamo),
            Some(value) if value.eq_ignore_ascii_case("dynamo") => Ok(Self::Dynamo),
            Some(value) if value.eq_ignore_ascii_case("gdl") => Ok(Self::Gdl),
            Some(value) => {
                anyhow::bail!("invalid {DYN_LOCAL_PIPELINE}={value:?}: expected `dynamo` or `gdl`")
            }
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
    Ok(
        PreprocessedRouting::new(routing_host, prefill_router, encoder_router)
            .with_local_pipeline(LocalPipeline::configured()?),
    )
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
            >(
                model.card(),
                inner_engine,
                model.card().tokenizer()?,
                LocalPipeline::configured()?,
            )
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

/// Build the in-process pipeline: preprocessor, then backend, then `engine`, composed on
/// `local_pipeline`. The HTTP and gRPC inputs build it over an in-process token engine.
pub async fn build_pipeline<Req, Resp>(
    card: &ModelDeploymentCard,
    engine: ExecutionContext,
    tokenizer: crate::tokenizers::Tokenizer,
    local_pipeline: LocalPipeline,
) -> anyhow::Result<ServiceEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>>>
where
    Req: Data,
    Resp: Data,
    OpenAIPreprocessor: Operator<
            Context<Req>,
            Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
            Context<PreprocessedRequest>,
            Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>,
        >,
{
    let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(card)?;
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone())?;
    let backend = Backend::from_tokenizer(tokenizer);
    if local_pipeline == LocalPipeline::Gdl {
        return Ok(compose_gdl_in_process_pipeline(
            preprocessor,
            backend,
            engine,
        ));
    }

    let frontend = ServiceFrontend::<SingleIn<Req>, ManyOut<Annotated<Resp>>>::new();
    let preprocessor = preprocessor.into_operator();
    let backend = backend.into_operator();
    let engine = ServiceBackend::from_engine(engine);

    Ok(frontend
        .link(preprocessor.forward_edge())?
        .link(backend.forward_edge())?
        .link(engine)?
        .link(backend.backward_edge())?
        .link(preprocessor.backward_edge())?
        .link_terminal(frontend)?)
}

/// [`build_pipeline`]'s chain as a gdl local pipeline, built from the terminal up as
/// [`PreprocessedRouting::build_pipeline`]'s is.
fn compose_gdl_in_process_pipeline<Req, Resp>(
    preprocessor: Arc<OpenAIPreprocessor>,
    backend: Arc<Backend>,
    engine: ExecutionContext,
) -> ServiceEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>>
where
    Req: Data,
    Resp: Data,
    OpenAIPreprocessor: Operator<
            Context<Req>,
            Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
            Context<PreprocessedRequest>,
            Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>,
        >,
{
    use gdl::{Hop, SinkStage};
    type Pre = PreprocessedRequest;

    let engine = SinkStage::new(Arc::new(EngineSink::new(engine)));
    let backend = DynOperatorStage::<_, Pre, BackendOutput, Pre, LLMEngineOutput>::new(
        backend,
        Hop::new((), engine, ()),
    );
    let preprocessor = DynOperatorStage::<_, Req, Resp, Pre, BackendOutput>::new(
        preprocessor,
        Hop::new((), backend, ()),
    );
    Arc::new(PipelineEngine::<_, Req, Resp>::for_dynamo_stages(
        gdl::Source::new(Hop::new((), preprocessor, ())),
    ))
}

/// Build the embeddings pipeline: preprocessor, then backend, then `engine`, composed on
/// `local_pipeline`.
///
/// The caller builds `preprocessor` (`OpenAIPreprocessor::new_for_embeddings`), so a card it
/// cannot serve fails before the caller creates `engine`.
pub fn build_embeddings_pipeline(
    card: &ModelDeploymentCard,
    preprocessor: Arc<OpenAIPreprocessor>,
    engine: ServiceEngine<
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
    >,
    local_pipeline: LocalPipeline,
) -> anyhow::Result<OpenAIEmbeddingsStreamingEngine> {
    let backend = Backend::from_mdc(card);
    if local_pipeline == LocalPipeline::Gdl {
        return Ok(compose_gdl_embeddings_pipeline(
            preprocessor,
            backend,
            engine,
        ));
    }

    let frontend = SegmentSource::<
        SingleIn<NvCreateEmbeddingRequest>,
        ManyOut<Annotated<NvCreateEmbeddingResponse>>,
    >::new();
    let preprocessor = preprocessor.into_operator();
    let backend = backend.into_operator();
    let service_backend = ServiceBackend::from_engine(engine);

    Ok(frontend
        .link(preprocessor.forward_edge())?
        .link(backend.forward_edge())?
        .link(service_backend)?
        .link(backend.backward_edge())?
        .link(preprocessor.backward_edge())?
        .link_terminal(frontend)?)
}

/// [`build_embeddings_pipeline`]'s chain as a gdl local pipeline, built from the terminal up
/// as [`PreprocessedRouting::build_pipeline`]'s is.
fn compose_gdl_embeddings_pipeline(
    preprocessor: Arc<OpenAIPreprocessor>,
    backend: Arc<Backend>,
    engine: ServiceEngine<
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
    >,
) -> OpenAIEmbeddingsStreamingEngine {
    use gdl::{Hop, SinkStage};
    type Pre = PreprocessedEmbeddingRequest;
    type Out = EmbeddingsEngineOutput;
    type Req = NvCreateEmbeddingRequest;
    type Resp = NvCreateEmbeddingResponse;

    let router = SinkStage::new(Arc::new(EngineSink::new(engine)));
    let backend = DynOperatorStage::<_, Pre, Out, Pre, Out>::new(backend, Hop::new((), router, ()));
    let preprocessor =
        DynOperatorStage::<_, Req, Resp, Pre, Out>::new(preprocessor, Hop::new((), backend, ()));
    Arc::new(PipelineEngine::<_, Req, Resp>::for_dynamo_stages(
        gdl::Source::new(Hop::new((), preprocessor, ())),
    ))
}

impl PreprocessedRouting {
    /// Route through `backend_engine` behind the given prefill and encoder routers, on
    /// Dynamo's local pipeline.
    ///
    /// [`build_preprocessed_routing`] builds the backend engine (a `RoutingHost`) from a
    /// discovery client and selects the local pipeline from `DYN_LOCAL_PIPELINE`; this takes
    /// any engine, so a test can stand in for the workers.
    pub fn new(
        backend_engine: ServiceEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
        >,
        prefill_router: Arc<PrefillRouter>,
        encoder_router: Arc<EncoderRouter>,
    ) -> Self {
        Self {
            backend_engine,
            prefill_router,
            encoder_router,
            local_pipeline: LocalPipeline::Dynamo,
        }
    }

    /// Compose [`build_pipeline`](Self::build_pipeline) and
    /// [`build_preprocessed_pipeline`](Self::build_preprocessed_pipeline) on `local_pipeline`.
    pub fn with_local_pipeline(mut self, local_pipeline: LocalPipeline) -> Self {
        self.local_pipeline = local_pipeline;
        self
    }

    /// The local pipeline [`build_pipeline`](Self::build_pipeline) and
    /// [`build_preprocessed_pipeline`](Self::build_preprocessed_pipeline) compose on.
    pub fn local_pipeline(&self) -> LocalPipeline {
        self.local_pipeline
    }

    /// The normal way to build an inference pipeline. Connect this directly to HTTP layer.
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
        OpenAIPreprocessor: Operator<
                Context<Req>,
                Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
                Context<PreprocessedRequest>,
                Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>,
            >,
    {
        let token_backend = Backend::from_tokenizer(tokenizer);
        let migration = Migration::from_mdc(card, migration_limit, migration_max_seq_len, metrics);
        if self.local_pipeline == LocalPipeline::Gdl {
            return Ok(self.compose_gdl_pipeline(preprocessor, migration, token_backend));
        }

        let frontend = SegmentSource::<SingleIn<Req>, ManyOut<Annotated<Resp>>>::new();
        let preprocessor_op = preprocessor.into_operator();
        let token_backend = token_backend.into_operator();
        let migration = migration.into_operator_for::<BackendOutput>();
        let prefill_op = self.prefill_router.into_operator();
        let encoder_op = self.encoder_router.into_operator();
        let backend = ServiceBackend::from_engine(self.backend_engine.clone());

        let engine = frontend
            .link(preprocessor_op.forward_edge())?
            .link(migration.forward_edge())?
            .link(token_backend.forward_edge())?
            .link(encoder_op.forward_edge())?
            .link(prefill_op.forward_edge())?
            .link(backend)?
            .link(prefill_op.backward_edge())?
            .link(encoder_op.backward_edge())?
            .link(token_backend.backward_edge())?
            .link(migration.backward_edge())?
            .link(preprocessor_op.backward_edge())?
            .link_terminal(frontend)?;

        Ok(engine)
    }

    /// [`build_pipeline`](Self::build_pipeline)'s chain as a gdl local pipeline: each
    /// operator calls the hop below it directly, with no response map per stage.
    ///
    /// Built from the terminal up, since each stage owns the hop it calls. Each operator
    /// is a `DynOperatorStage`, so it still gets its downstream as an `AsyncEngine`; the
    /// types are spelled out because Backend and the preprocessor implement `Operator`
    /// more than once.
    fn compose_gdl_pipeline<Req, Resp>(
        &self,
        preprocessor: Arc<OpenAIPreprocessor>,
        migration: Arc<Migration>,
        token_backend: Arc<Backend>,
    ) -> ServiceEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>>
    where
        Req: Data,
        Resp: Data,
        OpenAIPreprocessor: Operator<
                Context<Req>,
                Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
                Context<PreprocessedRequest>,
                Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>,
            >,
    {
        use gdl::Hop;
        type Pre = PreprocessedRequest;
        type Out = LLMEngineOutput;

        let token_backend = DynOperatorStage::<_, Pre, BackendOutput, Pre, Out>::new(
            token_backend,
            Hop::new((), self.gdl_routers(), ()),
        );
        let migration = DynOperatorStage::<_, Pre, BackendOutput, Pre, BackendOutput>::new(
            migration,
            Hop::new((), token_backend, ()),
        );
        let preprocessor = DynOperatorStage::<_, Req, Resp, Pre, BackendOutput>::new(
            preprocessor,
            Hop::new((), migration, ()),
        );
        Arc::new(PipelineEngine::<_, Req, Resp>::for_dynamo_stages(
            gdl::Source::new(Hop::new((), preprocessor, ())),
        ))
    }

    /// The encoder router, the prefill router and the backend engine as gdl stages: the tail
    /// that [`compose_gdl_pipeline`](Self::compose_gdl_pipeline) and
    /// [`compose_gdl_preprocessed_pipeline`](Self::compose_gdl_preprocessed_pipeline) share.
    fn gdl_routers(
        &self,
    ) -> DynOperatorStage<
        EncoderRouter,
        PreprocessedRequest,
        LLMEngineOutput,
        PreprocessedRequest,
        LLMEngineOutput,
    > {
        use gdl::{Hop, SinkStage};
        type Pre = PreprocessedRequest;
        type Out = LLMEngineOutput;

        let routing = SinkStage::new(Arc::new(EngineSink::new(self.backend_engine.clone())));
        let prefill = DynOperatorStage::<_, Pre, Out, Pre, Out>::new(
            self.prefill_router.clone(),
            Hop::new((), routing, ()),
        );
        DynOperatorStage::new(self.encoder_router.clone(), Hop::new((), prefill, ()))
    }

    /// Bring your own pre/post processor. Used when frontend has `--dyn-chat-processor
    /// vllm|sglang`, and by the Generate engine.
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
        if self.local_pipeline == LocalPipeline::Gdl {
            return Ok(self.compose_gdl_preprocessed_pipeline(migration));
        }

        let frontend = SegmentSource::<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
        >::new();
        let migration = migration.into_operator_for::<LLMEngineOutput>();
        let prefill_op = self.prefill_router.into_operator();
        let encoder_op = self.encoder_router.into_operator();
        let backend = ServiceBackend::from_engine(self.backend_engine.clone());

        let engine = frontend
            .link(migration.forward_edge())?
            .link(encoder_op.forward_edge())?
            .link(prefill_op.forward_edge())?
            .link(backend)?
            .link(prefill_op.backward_edge())?
            .link(encoder_op.backward_edge())?
            .link(migration.backward_edge())?
            .link_terminal(frontend)?;

        Ok(engine)
    }

    /// [`build_preprocessed_pipeline`](Self::build_preprocessed_pipeline)'s chain as a gdl
    /// local pipeline, built as [`compose_gdl_pipeline`](Self::compose_gdl_pipeline) builds
    /// its own.
    fn compose_gdl_preprocessed_pipeline(
        &self,
        migration: Arc<Migration>,
    ) -> ServiceEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>> {
        use gdl::Hop;
        type Pre = PreprocessedRequest;
        type Out = LLMEngineOutput;

        let migration = DynOperatorStage::<_, Pre, Out, Pre, Out>::new(
            migration,
            Hop::new((), self.gdl_routers(), ()),
        );
        Arc::new(PipelineEngine::<_, Pre, Out>::for_dynamo_stages(
            gdl::Source::new(Hop::new((), migration, ())),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_pipeline_is_dynamo_unless_gdl_is_named() {
        for value in [None, Some(""), Some("dynamo"), Some(" Dynamo ")] {
            assert_eq!(
                LocalPipeline::parse(value).unwrap(),
                LocalPipeline::Dynamo,
                "{value:?}"
            );
        }
        for value in [Some("gdl"), Some("GDL"), Some(" gdl\n")] {
            assert_eq!(
                LocalPipeline::parse(value).unwrap(),
                LocalPipeline::Gdl,
                "{value:?}"
            );
        }
        // A misspelled value must not fall back to Dynamo's pipeline: the run
        // would look like a gdl run and measure Dynamo's.
        let err = LocalPipeline::parse(Some("gld")).unwrap_err().to_string();
        assert!(
            err.contains("DYN_LOCAL_PIPELINE") && err.contains("gld"),
            "{err}"
        );
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
