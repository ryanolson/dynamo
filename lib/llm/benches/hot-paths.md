# Hot paths

The code paths that carry the load in the frontend's local pipeline, found by profiling. Each
entry names its entry point, the profile that showed it, and the benchmark that guards it. The
code marks each entry point with `// hot path: <entry>`. A change to a marked function, or to a
file an entry lists, runs the entry's benchmark before and after on the same host and build.

Per response item is the cost that scales: a stream sends one item per token or chunk, so a cost
per item grows with every token of every stream. A cost per request is paid once.

## per-item-response

The per-item path of a chat response: Backend's decoder stream, then the preprocessor's response
transform, through the stream layers between the stages.

- **Entry points:** `Backend::generate_tokens` in `lib/llm/src/backend.rs` (the `unfold` over the
  engine's stream), and `transform_postprocessor_stream_with_image_tokens` in
  `lib/llm/src/preprocessor.rs`.
- **Guard:** `lib/llm/benches/hotpath.sh`, rows `A dynamo` and `A gdl`, the `instr/item` column.
  Tolerance 2% per item, at least 50 instructions. Two builds of the same tree agree within 0.25%
  per row (aarch64).
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

- **Not covered:** the window starts where the HTTP handler calls the engine. HTTP parsing, the
  handler's request metrics and SSE encoding are outside it, and the reasoning and tool-call
  parsers are not on these rows.
- **Second view:** this entry has instruction counts only. A sampling profile (`perf`) of the same
  rows confirms or drops it once timing runs are allowed again.
