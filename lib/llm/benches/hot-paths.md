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
  the engine's stream, built by the backward half of Backend's around node), and
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

  The preprocessor does not: measured with its response transform reading Backend's gdl items
  in their `Result`, `a-gdl` and `b-gdl` each paid one more memcpy per item than through
  `AnnotatedItems`, as the transform takes each item out of the `Result` to convert it, and the
  same read with the item's fields taken in place paid two more per item on `a` and `b`. Reading
  in place pays where a stage edits an item and passes it on, not where it converts it to a new
  type.

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

- **Against Dynamo with the same fixes:** the reference is ai-dynamo `7c195fd1a7` with the
  fixes this port carries that help Dynamo on its own (the in-place stages, the boxed fields,
  Migration's stream, Ingress calling its engine, the behavior fixes), measured with this
  harness's dynamo rows. aarch64 (Neoverse-V2), Rust 1.99.0, both trees built with the same
  `RUSTFLAGS`, valgrind 3.28, `callgrind.sh` with 3 runs per row, lowest instructions kept:

  | Row | Instructions per request, 1 item | Per extra item | memcpy per item |
  |---|---:|---:|---:|
  | A | 169,772 -> 155,876 (-8.2%) | 19,906 -> 21,138 (+6.2%) | 41.2 -> 51.2 |
  | B | 214,798 -> 170,405 (-20.7%) | 20,157 -> 21,379 (+6.1%) | 42.2 -> 53.2 |
  | C | 62,503 -> 30,715 (-50.9%) | 795 -> 1,398 | 5 -> 10 |
  | embeddings | 99,579 -> 95,474 (-4.1%) | 81,684 -> 81,500 | 90.8 -> 90.8 |

  Where the extra copies per item are, on A (B is the same, plus 2 in Backend reading through
  `CheckedItems` and 1 fewer in Migration): 4 where the engine's stream enters gdl through
  `EngineSink` (`map(Ok)` and its `Fuse`), 2 at gdl's `Fuse` between Backend and the
  preprocessor, 2 taking items out of their `Result` at the `PipelineEngine` exit, 1 in gdl's
  `Bounded`, and 1 in the roles edit after the parsers. Taking the item out at the exit in one
  `map`, with the endings out of line, left every count unchanged.

  End to end, with the frontend (pipeline B) on 2 cores and mocker workers, 512-token prompts,
  5 runs per case: at 2,048 output tokens and concurrency 512, where the frontend is the
  bottleneck, gdl gives 5.9% fewer output tokens per second and uses 6.9% more frontend CPU per
  token; at 16 output tokens it gives 5 to 7% more, as the per-request saving wins.
- **Base for aarch64:** the table a change to this path compares against on the aarch64 bench
  host (Neoverse-V2), with gdl from `NVIDIA-dev/dynamo-gdl` at `3686ff4d`. Rust 1.99.0,
  `RUSTFLAGS="-C target-cpu=neoverse-n1 -C force-frame-pointers=yes --cfg tokio_unstable --cfg
  rustix_use_libc"`, valgrind 3.28, `hotpath.sh` with `HOTPATH_RUNS=3`. Save the block as a file
  and pass it to `hotpath.sh --base`:

  ```text
  # hotpath format=3 arch=aarch64 gdl=3686ff4d927d commit=6e9970c2e937
  pipeline     form    allocs allocs/item     instr instr/item memcpy memcpy/item allocs@16   instr@16 memcpy@16  range
  engine       -            7        1.00      9530        617     24        4.00        22      18787        84  7..7,22..22
  A            gdl        329       34.20    155923      21065    336       51.20       842     471893      1104  329..329,842..842
  B            gdl        344       34.20    170455      21408    376       53.20       857     491573      1174  344..344,857..857
  C            gdl         39        1.00     30712       1398     76       10.00        54      51682       226  39..39,54..54
  embed-engine -            8        1.00      6141        960      7        3.00        23      20544        52  8..8,23..23
  embeddings   gdl        178      139.00     95380      76675    125       90.80      2263    1245508      1487  178..178,2262..2263
  ingress      dynamo      49        3.00     49215       2846    139        7.00        94      91904       244  49..49,94..94
  ```

  Against gdl `83319c8f`, the same tree otherwise: allocations and `memcpy` equal on every row,
  and instructions per item within 75 (A −73, B +29, C 0). The `embeddings` row moves by several
  percent per item from run to run: 76,675 in this table, 81,763 on a later run of the same
  binary, and 81,210 on one run with gdl `83319c8f`. Its 16 inputs are tokenized on rayon's
  pool, and the instructions of rayon's work-stealing loop depend on thread timing; the whole
  difference between those two runs is in that loop. Read that row's per-item column with that
  spread in mind.
- **Not covered:** the window starts where the HTTP handler calls the engine. HTTP parsing, the
  handler's request metrics and SSE encoding are outside it, and the reasoning and tool-call
  parsers are not on these rows.
- **Second view:** this entry has instruction counts only. A sampling profile (`perf`) of the same
  rows confirms or drops it once timing runs are allowed again.
