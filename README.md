# microgpt (Rust)

A port of Andrej Karpathy's [microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95):
the most atomic way to train and run inference for a GPT, in pure, dependency-free Rust (std only).

Same model, same data, same hyperparameters:

- GPT-2 style transformer: 1 layer, 16-dim embeddings, 4 heads, context 16, rmsnorm, no biases, ReLU MLP
- character-level tokenizer over the makemore `names.txt` dataset (auto-downloaded to `input.txt`)
- Adam with bias correction and linear learning-rate decay
- temperature-0.5 sampling of 20 new names at the end

Two things differ from the gist by choice: training runs for 20000 steps instead of 1000, and each
step is a minibatch of 16 documents (the gist uses one) processed in parallel across 4 threads.

## Run

```bash
cargo run --release                    # 20000 steps, batch 16, 4 threads (fewer on small machines)
cargo run --release -- 1000 1 1        # the gist's recipe: 1000 steps, one document per step
cargo run --release -- 5000 128 8      # [num_steps] [batch_size] [threads]
cargo test --release                   # gradient check (f64), KV-cache equivalence, loss-decreases, parallel == serial
```

## How training works

Each step takes the next `batch_size` documents from the shuffled dataset (wrapping around), so
20000 x 16 is about 10 passes over the 31033 training names. Every document is tokenized as
`[BOS] name [BOS]`, forwarded through the model in one shot with causal attention, and its loss is
the mean cross-entropy over its next-character predictions. The batch gradient is the mean over
documents of those per-document gradients, followed by one Adam step with the learning rate decayed
linearly to zero over the run.

Parallelism is data parallelism over the documents of a batch. A pool of persistent worker threads
(the main thread is one of them) claims documents from an atomic counter, so fast cores naturally
take more work than slow ones, and each worker accumulates into its own gradient buffer. After the
batch, the main thread sums the worker buffers, scales by `1/batch_size`, runs Adam, and publishes
the next step by bumping an atomic epoch counter. Workers spin briefly then yield while waiting;
a step is tens of microseconds, so parking threads on channels or condvars would cost more than
the compute. Nothing allocates in steady state. Because which worker sums which document varies,
multi-threaded results are deterministic only up to floating-point summation order.

## Loss curves

Set `MICROGPT_LOG=<file.csv>` to record `step,train_time_s,train_loss,eval_loss` every
`MICROGPT_EVAL_EVERY` steps (default 100). The last 1000 names of the shuffled dataset are held
out from training and used for `eval_loss`, so runs with different batch sizes are compared on
the same names. Parameters are snapshotted during training and evaluated afterwards, so logging
does not disturb the timings. The final held-out loss is always printed.

[docs/loss-curves.html](docs/loss-curves.html) charts held-out loss against training wall-clock
time for the gist's recipe (`20000 1 1`), the same recipe run for the batched runs' time budget
(`156000 1 1`), and minibatches of 32 on 8 threads, 16 on 4 and 32 on 4 (`20000 32 8`,
`20000 16 4`, `20000 32 4`); the logs are in [docs/loss-logs](docs/loss-logs). On this laptop the
batched runs reach a held-out loss of 2.30 about 4-6x sooner than the single-name recipe
and all end near 2.11 versus 2.22 for the single-name recipe given the same time.
16 x 4 finishes in 89% of the 32 x 8 time and 32 x 4 in 151%, and 4 threads are far less
sensitive to other processes using cores, which is why 16 x 4 is the default.

## What was optimised

The Python original spends essentially all of its time in a scalar autograd engine: every add and
multiply allocates a `Value` node and the backward pass walks that graph. This port keeps the maths
identical but removes the graph entirely:

| | |
|---|---|
| **Hand-derived vector backward** | Each op (embedding, rmsnorm, fused QKV, causal softmax attention, output projection, ReLU MLP, LM head, cross-entropy) has an explicit backward kernel. No graph, no per-node allocation. |
| **Flat buffers** | Parameters, gradients and both Adam moments are four contiguous `Vec<f32>`; every matrix is a `(offset, rows, cols)` view. Adam is a single vectorisable loop. |
| **Fixed-size activations** | Activations and backward scratch are inline arrays inside one boxed struct, sized at compile time. This is not cosmetic: with separate heap `Vec`s reached through a reference, LLVM cannot prove the buffers are disjoint and emits ~30% slower forward/backward code (less vectorised, more spills). |
| **KV cache for free** | The stored K/V rows of the activation buffer *are* the cache: generation runs `forward(pos, pos+1)` against it instead of recomputing the prefix. |
| **Vector FMA kernels** | Every accumulate uses `mul_add`, and the dot product keeps four independent lanes, so the reductions compile to one vector fused multiply-add per four elements instead of separate multiplies and adds in a serial chain (LLVM neither contracts `a*b+c` nor reassociates float sums on its own). Measured 13% faster per document. |
| **Fused QKV projection** | `attn_wq`, `attn_wk`, `attn_wv` are one `48 x 16` matrix, so one matvec and one outer-product accumulate instead of three. |
| **Spin-synchronised worker pool** | See above. Main thread works too, so `threads` is the number of computing threads. |
| **Build flags** | `opt-level=3`, fat LTO, one codegen unit, `panic=abort`, `target-cpu=native` (see `.cargo/config.toml`; drop that file for a portable binary). |

Measured on an Apple M-series laptop with 8 performance + 2 efficiency cores:

| run | wall time |
|---|---|
| `python3 microgpt.py` (1000 steps x 1 doc) | ~109 s |
| `microgpt 1000 1 1` (same recipe) | ~15 ms |
| `microgpt 20000 1 1` | ~290 ms (14.5 µs/step) |
| `microgpt 20000 32 1` | ~7.0 s (350 µs/step, ~11 µs/doc) |
| `microgpt 20000 16 4` (default) | ~1.15 s (56 µs/step, 3x over one thread) |
| `microgpt 20000 32 8` | ~1.6 s (80 µs/step, 5x over one thread) |
| `microgpt 5000 128 8` | ~1.3 s (250 µs/step, 8x over one thread) |

Scaling is bounded by synchronisation: a 32-document step is under 40 µs of compute on 8 cores,
and the barrier plus reduce-and-Adam costs another ~30 µs. Bigger batches amortise that better;
more threads than physical performance cores make it worse. The default of 16 x 4 is the
configuration that reached the best held-out loss per second of training in the loss-curve
comparison below.

The RNG is xoshiro256** with Box-Muller gaussians. It is deterministic (seed 42) but does not
reproduce Python's `random` stream, so the exact loss curve and sampled names differ from the gist.

## Layout

Everything is in [src/main.rs](src/main.rs), top to bottom: RNG, linear-algebra kernels, model shapes,
activation buffers, forward, backward, Adam, dataset, threaded trainer, main, tests.
