# Hot paths

The code paths that carry the load in the frontend's local pipeline, found by profiling. Each
entry names its entry point, the profile that showed it, and the benchmark that guards it. The
code marks each entry point with `// hot path: <entry>`. A change to a marked function, or to a
file an entry lists, runs the entry's benchmark before and after on the same host and build.

Per response item is the cost that scales: a stream sends one item per token or chunk, so a cost
per item grows with every token of every stream. A cost per request is paid once.

## per-item-response

The per-item path of a chat response: Migration's response stream (pipelines B and C), Backend's
detokenizing, then the preprocessor's response transform, through the stream layers between
the stages.

- **Entry points:** `BackendRun::edit` in `lib/llm/src/backend.rs` (Backend's backward half as a
  gdl inline node, which pipelines A and B run) and `DecoderStream::poll_next` (the same per-item
  body, `DecoderState::process`, as a stream node, which Backend's tests also run), and
  `transform_postprocessor_stream_with_image_tokens` in `lib/llm/src/preprocessor.rs`, with the
  in-place edits of `lib/llm/src/preprocessor/edit.rs` after the parsers, and
  `Migrated::poll_next` in `lib/llm/src/migration.rs` (Migration's stream over the active
  attempt, built by `Migration::migrate`).
- **Guard:** `lib/llm/benches/hotpath.sh`, rows A, B and C (form `gdl`): the `memcpy/item` and
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

  The pipelines run in gdl's ended form, so no item is wrapped in a `Result` or taken out of
  one. In gdl's `Result` form, wrapping an item as `Ok(item)` cost no `memcpy`
  (`Result<Annotated<T>, gdl::Error>` and `Annotated<T>` have the same layout), but taking it out
  of a `Result` did: about 3 calls per item at the `PipelineEngine` boundary.

  A stream that reads its input and returns the input's poll matches the end without the item,
  and consumes that poll before it calls anything:

  ```rust
  let polled = items.as_mut().poll_next(task);
  match polled {
      Poll::Ready(Some(_)) | Poll::Pending => {}
      Poll::Ready(None) => {
          drop(polled);
          return Poll::Ready(self.end());
      }
  }
  polled
  ```

  Matching the end with `if let Poll::Ready(None) = poll { return Poll::Ready(self.end()); }`
  and returning `poll` after it copies every item into the caller's slot: one `memcpy` of 632
  bytes per item at each reader. Measured on `EndedItems` (`lib/gdl-adapters/src/engine.rs`) at
  the exit and at Backend's, the preprocessor's and Migration's inputs, and reproduced in a
  standalone crate with a 632-byte item behind a `dyn` stream.

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

- **After the in-place stages and the boxed fields, in gdl's `Result` form** (aarch64,
  `hotpath.sh --base` with `HOTPATH_RUNS=3`, 2026-11-03; memcpy calls and instructions per extra
  item):

  | Row | memcpy, before | memcpy, after | instructions, before | instructions, after |
  |---|---:|---:|---:|---:|
  | `a` | 70.2 | 41.2 | 24,100 | 19,869 |
  | `a-gdl` | 79.2 | 51.2 | 25,703 | 21,210 |
  | `b` | 79.2 | 42.2 | 25,971 | 20,055 |
  | `b-gdl` | 100.2 | 53.2 | 29,731 | 21,560 |
  | `c` | 13.0 | 5.0 | 2,823 | 797 |
  | `c-gdl` | 22.0 | 10.0 | 4,263 | 1,532 |

  The copies left on `a` are Backend's decoder stream (5 per item), the preprocessor's transform
  (5), the in-place edits (2), and string copies inside the tokenizer's decode. The gdl rows then
  paid about 5 (`c-gdl`) to 11 (`b-gdl`) more: taking items out of gdl's `Result` where a reader
  needed Dynamo items (the `PipelineEngine` boundary, and the preprocessor reading Backend), the
  box `EngineSink` put around a Dynamo engine's stream, and gdl's own `Bounded` layer (2). Reading
  items in their `Result` saved the copy only where a stage edited an item and passed it on, not
  where it converted it to a new type. The ended form, below, removed all of these copies.

- **Per request, a future copied with its gap:** when a coroutine's `Unresumed` layout puts a
  large gap before a field that is filled by `memcpy`, LLVM copies the gap with the field. The
  gap is the space the coroutine keeps for a later state, so the copy carries no data. rustc
  picks which field comes after a gap, and a small change to a type can change its choice.

  `Migration::migrate`'s future is the one to watch: its `RetryManager::build` state is about
  9.6 KB, and rustc puts `migrate`'s arguments before or after that space. On this tree they
  sit before it: on `b-gdl` and `c-gdl` the `Unresumed` variant puts `request` at offset 0, so
  the layout has no gap before the first argument and no sign of the copy (debug build,
  aarch64).

  A later dynamo (ai-dynamo main at 53015f9b05) showed the copy. There, a change of 24 bytes
  in `PreprocessedRequest` made rustc put `migrate`'s arguments after 9,727 bytes of padding
  on `b-gdl` (debug build), and `gdl::compose::build` copied 9,808 bytes per request into the
  box of Migration's future: the gap and the 80-byte `gdl::Cx` after it. The numbers below
  were measured on that tree (aarch64), not on this one.

  To find a copy like this:

  1. Find the function that calls `memcpy` with the large size: the `memcpy` size attribution
     of the bench host, or callgrind's callers of `memcpy` in the window.
  2. Read its disassembly. On the later dynamo the stack frame of the `build` instance was
     9,808 bytes, and `mov w2, #0x2650` (9,808) came before `bl memcpy`.
  3. Print the layout:
     `RUSTC_BOOTSTRAP=1 cargo rustc -p dynamo-llm --lib -- -Zprint-type-sizes`. The sign is a
     `padding` line before the first `upvar` of the `Unresumed` variant.

  Changes to `migrate` that were tried on the later dynamo did not remove the cost:

  - `migrate` as a plain function that returns `RetryManager::build`'s future: the future
    fell from 12,272 to 9,872 bytes on B, and the 9,808-byte copy went. But the code that ran
    in the first poll then ran inside `build`, and it copied the 2 KB request three more times.
    `b-gdl` lost 1 `memcpy` and 1.3k instructions per request; `c-gdl` gained 1 `memcpy` and
    1.5k instructions.
  - The same, with the context read before the request leaves its box: a new copy of 7,400
    bytes, from a gap of 7,376 bytes before `session_affinity` in `RetryManager::build`'s
    future.
  - The same, with the context read before `transfer`: the same 4 request copies.

  So a new shape of `migrate` moves the copy and does not remove it. Two changes can remove
  it: keep the request in its box until `RetryManager` takes it, so that no 2 KB request
  moves when the future is built, or change how gdl's `compose::build` makes the future.

- **The switch to one local pipeline:** the last table with both forms, before Dynamo's `link`
  chain stopped composing the frontend's pipelines. Rows of form `dynamo` for A, B, C and
  embeddings are Dynamo's local pipeline; from the switch on, `hotpath.sh` measures the gdl
  form alone and skips those rows in a base. Measured on the x86_64 bench host, `hotpath.sh
  --base` with `HOTPATH_RUNS=3`, head `fab21e5fac` (the same tree as the gdl port at
  `220310d4e4`) against `cecbba9cdc`, 2026-11-05, unchanged:

  ```text
  # hotpath format=3 arch=x86_64 gdl=83319c8fb235 commit=fab21e5facc6
  # base format=3 arch=x86_64 gdl=cbe60da85546 commit=accda4b4e3ff
  pipeline     form    allocs allocs/item     instr instr/item memcpy memcpy/item allocs@16   instr@16 memcpy@16  range                change allocs@16    instr instr/item memcpy memcpy@16  mark
  engine       -            7        1.00      8540        607     23        4.00        22      17650        83  7..7,22..22               0         0       -6         +0      0         0
  A            dynamo     346       34.20    168580      19779    362       41.20       859     465259       980  346..346,859..859         0         0      +30       -206      0         0
  A            gdl        328       34.20    162056      21118    338       51.20       841     478828      1106  328..328,841..841         0         0     -424        -24     -2        -2
  B            dynamo     399       34.20    271438      20091    455       42.20       912     572803      1088  399..399,912..912         0         0     +136        +73      0         0
  B            gdl        343       34.20    186412      21392    378       53.20       856     507289      1176  343..343,856..856         0         0      +28         +2      0         0
  C            dynamo      71        1.00    117923        783    134        5.00        86     129671       209  71..71,86..86             0         0      +61         -5      0         0
  C            gdl         38        1.00     27173       1389     74       10.00        53      48002       224  38..38,53..53             0         0      +14         +0      0         0
  embed-engine -            8        1.00      5425        768      6        3.00        23      16943        51  8..8,23..23               0         0      +12         -0      0         0
  embeddings   dynamo     199      138.93    103571      79941    168       90.80      2283    1302689      1530  199..199,2283..2284       0         0      -26        -49      0         0
  embeddings   gdl        177      139.00     97903      79933    123       90.80      2262    1296892      1485  177..177,2261..2262       0         0      -45       -128      0         0
  ingress      dynamo      49        3.00     49770       2755    139        7.00        94      91091       244  49..49,94..94             0         0      +36         -2      0         0
  ```

  Native gdl against Dynamo's chain in this table, per request at 1 item: A −3.9%, B −31%,
  C −77%, embeddings −5.5% in instructions, and fewer allocations on every pipeline. Per extra
  item: A +6.8%, B +6.5%, C +606 instructions, embeddings equal; most of the per-item gap is
  `memcpy` (A and B 10 to 11 more calls per item).

- **Against Dynamo with the same fixes, in gdl's `Result` form:** the reference is ai-dynamo
  `7c195fd1a7` with the fixes this port carries that help Dynamo on its own (the in-place stages, the boxed fields,
  Migration's stream, Ingress calling its engine, the behavior fixes), measured with this
  harness's dynamo rows. aarch64 (Neoverse-V2), Rust 1.99.0, both trees built with the same
  `RUSTFLAGS`, valgrind 3.28, `callgrind.sh` with 3 runs per row, lowest instructions kept:

  | Row | Instructions per request, 1 item | Per extra item | memcpy per item |
  |---|---:|---:|---:|
  | A | 169,772 -> 155,876 (-8.2%) | 19,906 -> 21,138 (+6.2%) | 41.2 -> 51.2 |
  | B | 214,798 -> 170,405 (-20.7%) | 20,157 -> 21,379 (+6.1%) | 42.2 -> 53.2 |
  | C | 62,503 -> 30,715 (-50.9%) | 795 -> 1,398 | 5 -> 10 |
  | embeddings | 99,579 -> 95,474 (-4.1%) | 81,684 -> 81,500 | 90.8 -> 90.8 |

  Where the extra copies per item were, on A (B the same, plus 2 in Backend reading through its
  `Result`-form input wrapper, since removed, and 1 fewer in Migration): 4 where the engine's
  stream entered gdl through `EngineSink` (`map(Ok)` and its `Fuse`), 2 at gdl's `Fuse` between Backend and the
  preprocessor, 2 taking items out of their `Result` at the `PipelineEngine` exit, 1 in gdl's
  `Bounded`, and 1 in the roles edit after the parsers. Taking the item out at the exit in one
  `map`, with the endings out of line, left every count unchanged.

  End to end, with the frontend (pipeline B) on 2 cores and mocker workers, 512-token prompts,
  5 runs per case: at 2,048 output tokens and concurrency 512, where the frontend is the
  bottleneck, the `Result` form gave 5.9% fewer output tokens per second and used 6.9% more
  frontend CPU per token; at 16 output tokens it gave 5 to 7% more, as the per-request saving
  won.
- **In gdl's ended form:** items cross each node as they are, and a stream's ending is asked
  once, at its end (`EndedItems` at the exit and at each native node's input). Against the same
  reference (its later tree `fcf74a821e`, which differs only in a test), both arms rebuilt on the
  aarch64 bench host with Rust 1.99.0 and the same `RUSTFLAGS`, valgrind 3.28. Per extra item,
  the median of 6 per-run slopes (C: of 3); per request at 1 item, the median:

  | Row | Instructions per request, 1 item | Per extra item | memcpy per item |
  |---|---:|---:|---:|
  | A | 169,800 -> 154,656 (-8.9%) | 19,894 -> 20,096 (+202, +1.0%) | 41.2 -> 41.2 |
  | B | 214,813 -> 169,051 (-21.3%) | 20,108 -> 20,228 (+121, +0.6%) | 42.2 -> 42.2 |
  | C | 62,503 -> 30,263 (-51.6%) | 795 -> 882 (+87) | 5 -> 5 |

  The copies per item equal the reference's. Per run, A's and B's slopes spread over about 100 to
  190 instructions, most of it in the allocator. Counted per instruction (callgrind
  `--dump-instr=yes`, each instruction charged to the symbol whose address range holds it), A's
  remaining instructions per item are gdl's: its node boundary (`EndedFuse`, 29 at each of 2)
  and its attempt boundary (`EndedBounded`, 39). The adapters add a net 15: the exit (22) and the
  box around a Dynamo engine's stream (7), less the `ResponseStream` the exit no longer needs
  (-14). The stages' own code is within 3 of the reference's. Self costs summed from a dump's
  `fn=` lines, without `--dump-instr`, overstated these by two to three times.

  End to end, the same setup as above, 5 interleaved runs per arm at every output length (16,
  256, 2,048) and concurrency (1, 64, 512), `gdl` at `b4a87784c0` against the reference: at
  2,048 output tokens and concurrency 512 the port gives 1.9% fewer output tokens per second and
  uses 1.1% more frontend CPU per token (9.82 -> 9.93 µs), both inside the reference's range; at
  16 output tokens it gives 1 to 8% more at every concurrency. One point fell outside the
  reference's range, 2,048 output tokens at concurrency 64 (4.2% fewer tokens per second, 5.0%
  more CPU per token). Five more runs per arm of that point alone, with `perf stat` on the
  frontend, put it inside: 0.9% fewer tokens per second and 0.3% more CPU per token, cycles and
  instructions per token both +0.6%, IPC unchanged (1.250 and 1.248), context switches 2% fewer.
  The reference itself moved between those two sessions, from 9.82 to 10.10 µs per token at that
  point, so compare arms run in one session only.
- **Backend as an inline node:** Backend's backward half is a gdl inline node
  (`gdl::inline::AroundInline`, used through `InlineHalf`). gdl calls its `edit` on each item in
  the stream that already holds the hop's boundaries, so the box, node boundary and stream of its
  own that a stream node has are gone, with the `EndedItems` it read through. Same setup as
  above, gdl `3cf13c1a`, against the same reference; the median of 6 per-run slopes (C: of 3):

  | Row | Instructions per request, 1 item | Per extra item | memcpy per item |
  |---|---:|---:|---:|
  | A | 169,768 -> 154,177 | 19,906 -> 19,624 (-282) | 41.2 -> 37.2 |
  | B | 214,865 -> 168,942 | 20,103 -> 19,833 (-270) | 42.2 -> 38.2 |
  | C | 62,446 -> 30,246 | 798 -> 882 (+84) | 5 -> 5 |

  Against Backend as a stream node in the ended form (the table above), that is 472 fewer
  instructions per item on A and 395 on B, and 4 fewer copies on each. C has no Backend. On A,
  the inline half and the engine's node boundary share one box: 2 allocations fewer per request.
- **Base for aarch64:** the table a change to this path compares against on the aarch64 bench
  host (Neoverse-V2), with gdl from `NVIDIA-dev/dynamo-gdl` at `3cf13c1a`, the ended form, and
  Backend's backward half as an inline node. Rust 1.99.0, `RUSTFLAGS="-C target-cpu=neoverse-n1 -C force-frame-pointers=yes --cfg tokio_unstable
  --cfg rustix_use_libc"`, valgrind 3.28, `hotpath.sh` with `HOTPATH_RUNS=3`, which keeps the
  lowest instructions of 3 runs. Save the block as a file and pass it to `hotpath.sh --base`:

  ```text
  # hotpath format=3 arch=aarch64 gdl=3cf13c1a0572 commit=a54f04d8e5f6
  pipeline     form    allocs allocs/item     instr instr/item memcpy memcpy/item allocs@16   instr@16 memcpy@16  range
  engine       -            7        1.00      9568        606     24        4.00        22      18654        84  7..7,22..22
  A            gdl        326       34.20    152196      19779    319       37.20       839     448876       877  326..326,839..839
  B            gdl        343       34.20    167368      19907    359       38.20       856     465968       932  343..343,856..856
  C            gdl         38        1.00     30077        893     72        5.00        53      43474       147  38..38,53..53
  embed-engine -            8        1.00      5975        961      7        3.00        23      20395        52  8..8,23..23
  embeddings   gdl        177      139.00     95327      81032    125       90.80      2262    1310804      1487  177..177,2261..2262
  ingress      dynamo      49        3.00     48915       2808    139        7.00        94      91042       244  49..49,94..94
  ```

  The ended form with Backend as a stream node (gdl `43629be7`), against the `Result` form's base
  on the same host (gdl `3686ff4d`): per item, `memcpy` 51.2 -> 41.2 on A, 53.2 -> 42.2 on B and
  10 -> 5 on C, and instructions -1,015 on A, -1,268 on B and -516 on C; per request, one
  allocation fewer on every gdl row. Backend as an inline node then took per item 4 more `memcpy`
  calls off A and B and 271 and 233 instructions, and per request 2 allocations off A. The `embeddings` row moves by
  several percent per item from run to run: its 16 inputs are tokenized on rayon's pool, and the
  instructions of rayon's work-stealing loop depend on thread timing. Read that row's per-item
  column with that spread in mind.
- **Not covered:** the window starts where the HTTP handler calls the engine. HTTP parsing, the
  handler's request metrics and SSE encoding are outside it, and the reasoning and tool-call
  parsers are not on these rows.
- **Second view:** a sampling profile (`perf record`, cycles) of the frontend at 2,048 output
  tokens and concurrency 64 has the same top symbols, in the same order, for the port and the
  reference: `memcpy` about 7%, then the tokenizer, `malloc` and `free`, and `serde_json`. No gdl
  boundary reaches 1.5% of the samples, and cycles per token follow instructions per token.
