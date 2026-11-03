# LLM Benchmarks

## tokenizer_simple

Uses Criterion with local test data. Runs automatically via `cargo bench`.

```bash
cargo bench --bench tokenizer_simple -p dynamo-llm
```

## tokenizer_dataset

Downloads a real dataset from HuggingFace Hub (LongBench-v2, ~500 samples) and
measures encode throughput, comparing `HuggingFaceTokenizer` against `FastTokenizer`.

This benchmark is **opt-in**: it exits immediately unless `RUN_BENCH=1` is set.
This prevents it from running during `cargo test --all-targets` in CI, since it
takes several minutes to complete.

### Basic run (default: Qwen/Qwen3-0.6B, LongBench-v2, 503 samples)

```bash
RUN_BENCH=1 cargo bench --bench tokenizer_dataset -p dynamo-llm
```

### Override tokenizer

```bash
RUN_BENCH=1 TOKENIZER_PATH=deepseek-ai/DeepSeek-V3 \
  cargo bench --bench tokenizer_dataset -p dynamo-llm
```

Use a local `tokenizer.json` file:

```bash
RUN_BENCH=1 TOKENIZER_PATH=/path/to/tokenizer.json \
  cargo bench --bench tokenizer_dataset -p dynamo-llm
```

### Override dataset and sample count

```bash
RUN_BENCH=1 DATASET=RyokoAI/ShareGPT52K MAX_SAMPLES=50 \
  cargo bench --bench tokenizer_dataset -p dynamo-llm
```

### Batched mode

```bash
RUN_BENCH=1 BATCH_SIZE=64 cargo bench --bench tokenizer_dataset -p dynamo-llm
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `RUN_BENCH` | unset | Must be set to any value to run the benchmark |
| `TOKENIZER_PATH` | `Qwen/Qwen3-0.6B` | HuggingFace model name or local path to `tokenizer.json` |
| `DATASET` | `zai-org/LongBench-v2` | HuggingFace dataset name (`zai-org/LongBench-v2` or `RyokoAI/ShareGPT52K`) |
| `MAX_SAMPLES` | `503` | Maximum number of samples to process |
| `BATCH_SIZE` | unset | If set, runs batched mode instead of sequential |

## image_decode

Compares Rust `image::ImageReader` with the default libjpeg-turbo path for JPEG
decoding. The benchmark measures a synthetic 2400x1080 RGB JPEG:

```bash
cargo bench --bench image_decode -p dynamo-llm
```

For the reproducible 3840x2160, 100-image concurrency sweep at C1, C8, and C32,
use the [media decode benchmark runner](../../../benchmarks/multimodal/media_decode/README.md).
The sweep is opt-in so `cargo test --all-targets` does not execute the workload
in CI.

## video_decode

Measures the `video-rs` decoder. The default benchmark uses the checked-in
320x240 H.264 fixture and samples 30 of its 100 frames:

```bash
cargo bench -p dynamo-llm --features media-ffmpeg --bench video_decode -- \
  video_decode_h264_320x240_100_to_30
```

The C1, C8, and C32 sweep is opt-in so `cargo test --all-targets` executes only
the small benchmark. Generate or provide a larger input, then run:

```bash
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i testsrc2=size=1280x720:rate=24:duration=10 \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -g 48 -an \
  /tmp/dynamo-video-720p24-10s.mp4

RUN_VIDEO_DECODE_SWEEP=1 \
VIDEO_DECODE_BENCH_INPUT=/tmp/dynamo-video-720p24-10s.mp4 \
VIDEO_DECODE_BENCH_NUM_FRAMES=30 \
  cargo bench -p dynamo-llm --features media-ffmpeg --bench video_decode -- \
    video_decode_concurrent
```

Each concurrent iteration decodes one video per worker. Override
`VIDEO_DECODE_BENCH_NUM_FRAMES` to change the number of uniformly sampled output
frames. If `VIDEO_DECODE_BENCH_INPUT` is unset, the sweep uses the small
checked-in fixture.

## request_trace_finish_metadata

Measures request finish metadata overhead with tracing disabled/enabled, plus
tool-call metadata recording across increasing numbers of calls.

```bash
cargo bench --bench request_trace_finish_metadata -p dynamo-llm --features request-trace-bench
```

## callgrind.sh and hotpath.sh

The paths these scripts guard, and the profiles that found them, are in
[hot-paths.md](hot-paths.md).

`hotpath.sh` prints the hot-path table: allocations and instructions per request, for each
pipeline (A, B, C, embeddings, and the worker's `Ingress`) in both forms. The dynamo form is the
default path (`DYN_LOCAL_PIPELINE` unset); the gdl form is `DYN_LOCAL_PIPELINE=gdl`. The engine
rows are the test engines alone, so a pipeline's own cost is its row minus its engine's row.
Needs `valgrind` and `jq`.

```bash
lib/llm/benches/hotpath.sh > table.txt              # the table
lib/llm/benches/hotpath.sh --base table.txt         # the table, and each row's change
```

Per request, `allocs` and `instr` are at 1 item, and `allocs/item` and `instr/item` are the
cost of each item after the first: `(count at 16 items - count at 1 item) / 15`. `allocs@16`
and `instr@16` are the counts at 16 items, and `range` is the lowest and highest allocations
of one request, at 1 and at 16 items. For the embeddings rows, an item is an input.

A dynamo row's allocations are its typical request's. A gdl row's are its highest request's:
at gdl `79bcc11d`, under CPU load, some gdl requests make 2 allocations fewer in their window,
so the typical count moves between runs. `range` shows how far.

With `--base`, the script marks a row `RISE` when the default path got worse and `WORSE` when a
gdl row did, and exits 1 if it marked any row. A row gets worse when its allocations at 1 or at
16 items rise at all, or when its instructions rise by more than the tolerance (see below). It
compares the raw counts, not the rounded per-item figures.

The table's first line is its provenance: `# hotpath format=2 arch=x86_64 gdl=<rev>
commit=<rev>`. One strict parser reads the base and the new table, and fails the run before
any valgrind run on any line it cannot read in full: a stray line, a non-number in a count, a
missing, extra or repeated row, or a base with another table format or arch. A base it could
not read in full could let a rise pass. A table written with `--base` is a valid base. The
commit and gdl rev are printed, not checked.

`callgrind.sh` measures single rows, one run each, and prints raw counts:

```bash
lib/llm/benches/callgrind.sh            # every row at 1 and 16 items
lib/llm/benches/callgrind.sh b 1 b 16   # selected rows
```

### How it counts

The pipeline rows run the `tests/allocations.rs` binary; the `ingress` row runs
`lib/runtime/tests/ingress_allocations.rs`. Each builds its path with a test engine at the end
and runs warm-up requests, then N measured requests (`CALLGRIND_REQUESTS`, default 200). The
binary asks callgrind for a dump before and after the measured requests. The script reads the
second dump, the window, and divides its instructions by N. Setup and teardown are not in it.
Allocations are the binary's own count, from the same run.

The window leaves out work whose cost changed from run to run but is not the code under test:

- **The system allocator.** Glibc malloc's cost follows its heap state, which follows how the
  threads interleave. The allocations column counts its calls instead.
- **The runtime thread while it is parked.** How often it parks and wakes depends on when the
  other threads run. The harness's wait at the end of each window is left out too.
- **The stand-in frontend's thread** in the `ingress` row.

The window also reads no clock and sets no timer. Callgrind's `--toggle-collect=<function>` and
its zero request are not used: on this code the first gave counts that changed by 10x between
runs, and the second left about 4 million instructions in a window that did not grow with N.

The same binaries check allocations without valgrind:
`cargo test -p dynamo-llm --test allocations` and
`cargo test -p dynamo-runtime --test ingress_allocations`.

### Spread and tolerance

Allocations do not vary: every run gives the same count. Instructions vary a little, and only
upwards: a thread that spins or parks once more adds instructions. Two builds of unchanged
request-path code (one with an extra log line in setup code, built from another path) give the
same lowest count within 0.25% for every row.

Instructions per request over 5 runs of one build, N = 200, on a 20-core aarch64 host with
other valgrind runs beside it: highest minus lowest, the larger of two builds.

| Rows | 1 item | 16 items |
|---|---:|---:|
| `c`, `c-gdl` | 49 (0.07%) | 43 (0.04%) |
| `a`, `a-gdl`, `b`, `b-gdl` | 496 (0.27%) | 3,438 (0.59%) |
| `engine`, `embed-engine` | 77 | 120 |
| `ingress` | 283 (0.58%) | 243 (0.27%) |
| `embed` | 549 (0.56%) | 5,900 (0.48%) |
| `embeddings-gdl` | 1,282 (1.19%) | 22,408 (1.81%) |

The embeddings rows vary most: their batch encode runs on rayon, whose idle threads spin a
varying number of times.

So `hotpath.sh` counts an instruction rise only above the row's tolerance, the larger of a
share of the base and a floor:

| Rows | Per request | Per item |
|---|---|---|
| all but embeddings | 1%, at least 300 | 2%, at least 50 |
| `embed`, `embeddings-gdl` | 3%, at least 300 | 6%, at least 50 |

`HOTPATH_FLOOR` and `HOTPATH_ITEM_FLOOR` set the floors. For a merge gate, run with
`HOTPATH_RUNS=3`: the script keeps each row's lowest count. A rise smaller than the tolerance
is shown but not marked.

### Running it on the bench host

Build with the `profiling` profile (the default; release with symbols). Debug counts mean
little.

Under valgrind, the gdl rows die with SIGSEGV unless built with `--cfg rustix_use_libc`: gdl's
`Cx` reads the clock through rustix's vDSO path (`rustix::backend::param::auxv::check_elf_base`),
and valgrind does not map the vDSO. `RUSTFLAGS` replaces the flags in `.cargo/config.toml`,
including `--cfg tokio_unstable`, so copy your target's list from that file and add the cfg.
Changing `RUSTFLAGS` rebuilds every dependency, so give the valgrind build its own
`CARGO_TARGET_DIR`, and one per checkout: cargo can take a binary built from another checkout
in the same target directory as up to date, and the table then measures that checkout's code.

On x86_64:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3 -C force-frame-pointers=yes --cfg tokio_unstable --cfg rustix_use_libc" \
  CARGO_TARGET_DIR=$PWD/target/valgrind HOTPATH_RUNS=3 lib/llm/benches/hotpath.sh > table.txt
```

On aarch64 (valgrind 3.28 or newer; 3.25 cannot decode `neoverse-n1` code):

```bash
RUSTFLAGS="-C target-cpu=neoverse-n1 -C force-frame-pointers=yes --cfg tokio_unstable --cfg rustix_use_libc" \
  CARGO_TARGET_DIR=$PWD/target/valgrind HOTPATH_RUNS=3 lib/llm/benches/hotpath.sh > table.txt
```

The first build takes about 30 minutes. The rows run in parallel, `CALLGRIND_JOBS` at a time
(default: the number of CPUs); each takes 2 to 6 minutes under valgrind.

The scripts fail, naming the row and the valgrind log, when valgrind exits non-zero, its log
says the program died by a signal, or the run has no window dump (for example, when the client
requests did nothing). A run that dies still writes counts, and a run with no window writes the
count of the whole run, so a number from either would be wrong. `callgrind-test.sh` tests both
scripts with stand-ins for `cargo` and `valgrind`, so it needs neither a build nor valgrind:

```bash
lib/llm/benches/callgrind-test.sh
```
