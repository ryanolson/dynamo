# Hot paths

The code paths that carry the load in the frontend's local pipeline, found by profiling. Each
entry names its entry point, the profile that showed it, and the benchmark that guards it. The
code marks each entry point with `// hot path: <entry>`. A change to a marked function, or to a
file an entry lists, runs the entry's benchmark before and after on the same host and build.

Per response item is the cost that scales: a stream sends one item per token or chunk, so a cost
per item grows with every token of every stream. A cost per request is paid once.

## per-item-response

The per-item path of a chat response: Migration's response stream (pipelines B and C), Backend's
decoder stream, then the preprocessor's response transform, through the stream layers between
the stages.

- **Entry points:** `DecoderStream::poll_next` in `lib/llm/src/backend.rs` (Backend's stream over
  the engine's stream, built by `Backend::generate_tokens`), and
  `transform_postprocessor_stream_with_image_tokens` in `lib/llm/src/preprocessor.rs`, with the
  in-place edits of `lib/llm/src/preprocessor/edit.rs` after the parsers, and
  `Migrated::poll_next` in `lib/llm/src/migration.rs` (Migration's stream over the active
  attempt, built by `Migration::migrate`).
- **Guard:** `lib/llm/benches/hotpath.sh`, rows A, B and C in both forms: the `memcpy/item` and
  `memcpy@16` columns, with no tolerance (the count is the same in every run of one build), and
  the `instr/item` column, tolerance 2% per item, at least 50 instructions. Two builds of the
  same tree agree within 0.25% per row (aarch64). One more move of an item costs about 100 to
  200 instructions, inside the instruction tolerance, so the `memcpy` columns are what catch it.
- **Rule:** a stage edits a stream item through `&mut` in a stream that owns its state. It does
  not take the item by value and return it (`map(|mut item| { ...; item })`, `unfold`), because
  each such step moves the 1 KB item two or three times. Box the fields that are `None` on the
  default path, so that `None` costs a pointer (`engine_and_backend_outputs_stay_small` keeps the
  sizes).

  Worked example, Migration's response. It was

  ```rust
  stream::unfold(retry_manager, |mut retry_manager| async move {
      retry_manager.next().await.map(|response| (response, retry_manager))
  })
  .fuse()
  ```

  Every item moved the whole `RetryManager`, with its `PreprocessedRequest`, into a new future
  and out again, and moved the item through the future's output tuple, `Option` and `Fuse`. Now
  `Migrated` (`lib/llm/src/migration.rs`) owns the manager. It polls the attempt's stream,
  checks the item through `&mut` (`RetryManager::accept`), and returns the same `Poll` value it
  got, so the item is not taken out and wrapped again. Only a migration, which must await the
  next dispatch, runs as a future, and that future owns the manager until it ends. Per extra
  item (aarch64, 16 items, before and after this change alone):

  | Row | `memcpy` calls | Instructions |
  |---|---:|---:|
  | `c` | 13 -> 5 | 2,826 -> 1,109 |
  | `c-gdl` | 22 -> 13 | 4,264 -> 2,627 |
  | `b` | 55.2 -> 43.2 | 23,270 -> 20,943 |
  | `b-gdl` | 70.2 -> 59.2 | 25,492 -> 23,636 |

  Wrapping an item as `Ok(item)` for gdl costs no `memcpy` (`Result<Annotated<T>, gdl::Error>`
  and `Annotated<T>` have the same layout), but taking it out of a `Result` does: about 3 calls
  per item at the `PipelineEngine` boundary.

  The same pattern is still on per-item paths that no row measures: the reasoning parser
  (`parse_reasoning_content_from_stream_inner` and `strip_leading_reasoning_start_from_stream`,
  both `unfold`; `annotate_reasoning_usage`, a `map`), the input `map` of
  `apply_tool_calling_jail`, the `map` in `apply_unified_response_policies`, and the response
  `map`s of the HTTP completions handler (`http/service/openai.rs`).
- **Profile:** callgrind over the measured window (`lib/llm/benches/callgrind.sh`) of rows `a` and
  `a-gdl`, at 1 and 16 items, 200 and 400 requests. Per extra item = (16 items - 1 item) / 15.
  Self instructions, grouped by component. Commit 845f5a9993, x86_64 bench host
  (`-C target-cpu=x86-64-v3`), one pinned core, 2026-11-03.

  | Component, self Ir per extra item | `a` (dynamo) | share | `a-gdl` (native gdl) | share |
  |---|---:|---:|---:|---:|
  | `memcpy`: items moved by value through `map`, `fuse` and `unfold` | 6,868 | 28.7% | 7,957 | 32.0% |
  | `malloc` and `free`: 34.2 allocations per item | 3,467 | 14.5% | 2,998 | 12.1% |
  | detokenize (`DecodeStream`) | 2,989 | 12.5% | 2,991 | 12.0% |
  | delta generation | 570 | 2.4% | 570 | 2.3% |
  | stream layers (`futures` combinators, gdl boundaries) | 420 | 1.8% | 714 | 2.9% |
  | Backend's decoder stream, own code | 356 | 1.5% | 361 | 1.5% |
  | the preprocessor's response transform, own code | 264 | 1.1% | 286 | 1.2% |
  | not grouped above: | 8,960 | 37.5% | 8,970 | 36.1% |
  | - oniguruma regex, from the tokenizer's `Replace` decoder (part of detokenizing) | 3,919 | 16.4% | | |
  | - hashing (`RandomState`, SipHash) | 1,119 | 4.7% | | |
  | - `Vec` growth (`RawVec::reserve`), mostly the tokenizer's `Strip` decoder | 1,084 | 4.5% | | |
  | - many small functions | 2,838 | 11.9% | | |
  | **total** | **23,895** | 100% | **24,846** | 100% |

  The profile depends on the tokenizer. The harness uses TinyLlama, whose decoder is
  `Replace("▁" -> " ")`, `ByteFallback`, `Fuse`, `Strip`; the `Replace` step runs oniguruma on
  every decode step, so detokenizing is about 12.5% + 16.4% of the per-item cost here. ByteLevel
  tokenizers (the Llama 3.1 and DeepSeek sample models) have no regex in their decoder.

  The 34.2 allocations per item are not attributed yet: the harness pauses callgrind inside its
  counting allocator, so callgrind does not see their callers. The tokenizer's C regex also calls
  `malloc` about 14 times per item, which the harness does not count.

  The items are large: `Annotated<LLMEngineOutput>` is 1,112 bytes, `Annotated<BackendOutput>`
  1,008 and `Annotated<NvCreateChatCompletionStreamResponse>` 768. gdl's own boundaries copy no
  item; the copies are the stages' own transforms, and both forms pay them.

- **After the in-place stages, the boxed fields and gdl items in their `Result`** (aarch64, `hotpath.sh --base` with
  `HOTPATH_RUNS=3`, 2026-11-03; memcpy calls and instructions per extra item):

  | Row | memcpy, before | memcpy, after | instructions, before | instructions, after |
  |---|---:|---:|---:|---:|
  | `a` | 70.2 | 41.2 | 24,100 | 19,869 |
  | `a-gdl` | 79.2 | 51.2 | 25,703 | 21,210 |
  | `b` | 79.2 | 42.2 | 25,971 | 20,055 |
  | `b-gdl` | 100.2 | 53.2 | 29,731 | 21,560 |
  | `c` | 13.0 | 5.0 | 2,823 | 797 |
  | `c-gdl` | 22.0 | 10.0 | 4,263 | 1,532 |

  The copies left on `a` are Backend's decoder stream (5 per item), the preprocessor's transform
  (5), the in-place edits (2), and string copies inside the tokenizer's decode. The gdl rows pay
  about 5 (`c-gdl`) to 11 (`b-gdl`) more: taking items out of gdl's `Result` where a reader needs
  Dynamo items (the `PipelineEngine` boundary, and the preprocessor reading Backend), the box
  `EngineSink` puts around a Dynamo engine's stream, and gdl's own `Bounded` layer (2). Backend
  and Migration read gdl items in their `Result` (`CheckedItems`), so they take no item out.

- **Not covered:** the window starts where the HTTP handler calls the engine. HTTP parsing, the
  handler's request metrics and SSE encoding are outside it, and the reasoning and tool-call
  parsers are not on these rows.
- **Second view:** this entry has instruction counts only. A sampling profile (`perf`) of the same
  rows confirms or drops it once timing runs are allowed again.
