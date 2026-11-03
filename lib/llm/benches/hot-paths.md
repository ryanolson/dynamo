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

- **Entry points:** `DecoderStream::poll_next` in `lib/llm/src/backend.rs` (Backend's stream over
  the engine's stream, built by `Backend::generate_tokens`), and
  `transform_postprocessor_stream_with_image_tokens` in `lib/llm/src/preprocessor.rs`, with the
  in-place edits of `lib/llm/src/preprocessor/edit.rs` after the parsers.
- **Guard:** `lib/llm/benches/hotpath.sh`, rows `A dynamo` and `A gdl`: the `memcpy/item` and
  `memcpy@16` columns, with no tolerance (the count is the same in every run of one build), and
  the `instr/item` column, tolerance 2% per item, at least 50 instructions. Two builds of the
  same tree agree within 0.25% per row (aarch64). One more move of an item costs about 100 to
  200 instructions, inside the instruction tolerance, so the `memcpy` columns are what catch it.
- **Rule:** a stage edits a stream item through `&mut` in a stream that owns its state. It does
  not take the item by value and return it (`map(|mut item| { ...; item })`, `unfold`), because
  each such step moves the 1 KB item two or three times.
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
  | not attributed (tokenizer and serde internals, hashing, other futures) | 8,960 | 37.5% | 8,970 | 36.1% |
  | **total** | **23,895** | 100% | **24,846** | 100% |

  The items are large: `Annotated<LLMEngineOutput>` is 1,112 bytes, `Annotated<BackendOutput>`
  1,008 and `Annotated<NvCreateChatCompletionStreamResponse>` 768. gdl's own boundaries copy no
  item; the copies are the stages' own transforms, and both forms pay them.

- **Not covered:** the window starts where the HTTP handler calls the engine. HTTP parsing, the
  handler's request metrics and SSE encoding are outside it, and the reasoning and tool-call
  parsers are not on these rows.
- **Second view:** this entry has instruction counts only. A sampling profile (`perf`) of the same
  rows confirms or drops it once timing runs are allowed again.
