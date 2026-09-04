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

The model shape is fixed at build time (the activation buffers are fixed-size arrays), with the
gist's values as defaults. Override with environment variables when building:

```bash
MICROGPT_N_EMBD=32 MICROGPT_N_LAYER=2 cargo build --release   # also MICROGPT_N_HEAD, MICROGPT_BLOCK_SIZE
```

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
take more work than slow ones, and each worker accumulates into its own gradient buffer. Then, in
a second phase behind a barrier, each worker owns a contiguous slice of the parameters: it sums
that slice across all gradient buffers, applies Adam to it, and mirrors it into the transposed
layout the forward pass reads. The main thread then publishes the next step by bumping an atomic
epoch counter. The parameter, moment and gradient buffers are shared without locks; the two
barriers per step are what make that sound (see `SharedBuf`). Workers spin briefly then yield
while waiting; a step is tens of microseconds, so parking threads on channels or condvars would
cost more than the compute. Nothing allocates in steady state. Because which worker sums which
document varies, multi-threaded results are deterministic only up to floating-point summation
order.

## Report

[docs/report.html](docs/report.html) is a self-contained write-up of the port and every experiment
below: what each optimisation bought, thread scaling, batch size at equal data, the learning-rate
sweep, seed noise, and model size at matched time budgets, with charts.

## Loss curves and experiments

Knobs, all environment variables so the positional arguments stay `[num_steps] [batch_size] [threads]`:

| variable | default | effect |
|---|---|---|
| `MICROGPT_LOG` | unset | write `step,train_time_s,train_loss,eval_loss` to this CSV |
| `MICROGPT_EVAL_EVERY` | 100 | log interval in steps |
| `MICROGPT_LR` | 0.01 | peak learning rate (decays linearly to zero) |
| `MICROGPT_SEED` | unset | reseed before parameter init; the data shuffle and held-out split always use seed 42, so losses stay comparable across seeds |

Set `MICROGPT_LOG=<file.csv>` to record `step,train_time_s,train_loss,eval_loss` every
`MICROGPT_EVAL_EVERY` steps (default 100). The last 1000 names of the shuffled dataset are held
out from training and used for `eval_loss`, so runs with different batch sizes are compared on
the same names. Parameters are snapshotted during training and evaluated afterwards, so logging
does not disturb the timings. The final held-out loss is always printed.

[docs/loss-curves.html](docs/loss-curves.html) charts held-out loss against training wall-clock
time for the gist's recipe (`20000 1 1`), the same recipe run for the batched runs' time budget
(`207000 1 1`), and minibatches of 32 on 8 threads and of 16, 32 and 64 on 4 threads
(`20000 32 8`, `20000 16 4`, `20000 32 4`, `20000 64 4`); the logs are in
[docs/loss-logs](docs/loss-logs). On this laptop the batched runs reach a held-out loss of 2.30
about 4-7x sooner than the single-name recipe and end between 2.10 (64 x 4, 2.7 s) and
2.13 (16 x 4, 0.87 s), versus 2.20 for the single-name recipe given 2.7 s. Bigger
batches keep buying a little loss for proportionally more time; 16 x 4 is the default because it
has the best loss per second, and 4 threads are far less sensitive to other processes using
cores than 8.

### Model size

Held-out loss for four model shapes, 4 threads, `steps x batch` chosen to land on the same time
budgets (set at 1.1, 3.6 and 11 s with an earlier build; the final kernels bring the same recipes
in at about 0.8, 2.6 and 8 s). Seed noise is about 0.005, so every gap below is real.

| model | params | ~0.8 s | ~2.6 s | ~8 s |
|---|---|---|---|---|
| 1 layer, 16-dim (gist) | 4,192 | 2.123 (20000 x 16) | 2.103 (20000 x 64) | |
| 1 layer, 32-dim | 14,528 | 2.107 (6700 x 16) | **2.060** (6600 x 64) | 2.053 (20000 x 64) |
| 2 layers, 16-dim | 7,264 | 2.105 (11500 x 16) | 2.091 (20000 x 32) | 2.064 (20000 x 128, 9.2 s) |
| 2 layers, 32-dim | 26,816 | 2.108 (3500 x 16) | 2.083 (11500 x 16) | **2.026** (10500 x 64) |

Every larger shape beats the gist's model at every budget. The 32-dim single layer is the best
buy up to a few seconds and saturates near 2.055; the 2-layer 32-dim model keeps improving and
wins once you can afford eight seconds or more. Per-step costs at 16 x 4: 43 us (16-dim),
74 us (2 x 16), 120 us (32-dim), 211 us (2 x 32). The build defaults stay at the gist's shape so
the numbers above remain reproducible; pick a larger one with the build variables.

### Findings from the sweeps (held-out loss, 4 threads unless noted)

- **Seed noise:** four seeds of `20000 16 4` end at 2.120-2.130, so differences under ~0.01 are noise.
- **Learning rate:** 0.01 is already at the optimum for batched Adam here (`16 x 4`: 0.0025 -> 2.135,
  0.005 -> 2.119, 0.01 -> 2.122, 0.02 -> 2.162; `64 x 4`: 0.005 -> 2.107, 0.01 -> 2.105, 0.02 -> 2.124).
  Larger batches do not want a larger rate. The single-name recipe prefers a lower one
  (`20000 1 1` at 0.005 -> 2.203 versus 2.223 at 0.01).
- **Batch size at equal data (640k names):** 8 -> 2.139, 16 -> 2.130, 32 -> 2.119, 64 -> 2.108,
  128 -> 2.109. Larger batches are better per name up to 64 and flat after; with this tiny model
  the noise of small batches costs more than the extra updates gain.
- **Threads:** on a quiet machine 6 threads are ~20% faster than 4 (`20000 32 x`: 1.48 s -> 1.15 s;
  `20000 16 x`: 0.88 s -> 0.76 s), but any background load (a browser, Gatekeeper scanning new
  binaries) turned the same 6-thread runs into 3-8 s while 4-thread runs stayed near 1.1 s. The
  default stays at 4; pass 6 explicitly on a quiet machine.

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
| **Position-batched linear layers** | Forward and backward apply each linear layer to all positions of a document at once. The weight gradient is `dW += dY^T X` summed over positions, so `dW` is touched once per document instead of once per position. Only attention and rmsnorm still loop per position. 15% faster per document. |
| **Row-accumulated forward against transposed weights** | Profiling showed the forward's one-dot-product-per-output form taking 45% of the time despite being a third of the FLOPs: a 16-element dot is mostly horizontal-reduction overhead. The forward now reads a transposed copy of the weights (refreshed after each Adam step, ~1 µs) and computes each output row as register-accumulated `y += x[j] * Wt[j]` updates, the same shape as the backward matmuls. Another 1.25x per document. |
| **Vector FMA kernels** | Every accumulate uses `mul_add`, and the dot product keeps four independent lanes, so the reductions compile to one vector fused multiply-add per four elements instead of separate multiplies and adds in a serial chain (LLVM neither contracts `a*b+c` nor reassociates float sums on its own). Measured 13% faster per document. |
| **Fused QKV projection** | `attn_wq`, `attn_wk`, `attn_wv` are one `48 x 16` matrix, so one matvec and one outer-product accumulate instead of three. |
| **Spin-synchronised worker pool** | See above. Main thread works too, so `threads` is the number of computing threads. |
| **Reduce-and-Adam fused into the workers** | The per-step tail (sum the gradient buffers, Adam, transpose) used to run on the main thread while the others spun. Each worker now does it for its own parameter slice behind a second barrier. The loops have to be flat: a first version with a chunked reduce and an index-table scatter was 3x slower per element because the vector sqrt/div ran latency-bound and the scatter defeated the store pipeline. 6-8% per step at 4-8 threads, 2% slower at 1 thread. |
| **Build flags** | `opt-level=3`, fat LTO, one codegen unit, `panic=abort`, `target-cpu=native` (see `.cargo/config.toml`; drop that file for a portable binary). |

Measured on an Apple M-series laptop with 8 performance + 2 efficiency cores:

| run | wall time |
|---|---|
| `python3 microgpt.py` (1000 steps x 1 doc) | ~109 s |
| `microgpt 1000 1 1` (same recipe) | ~14 ms |
| `microgpt 20000 1 1` | ~230 ms (11.5 µs/step) |
| `microgpt 20000 32 1` | ~4.8 s (240 µs/step, ~7.5 µs/doc) |
| `microgpt 20000 16 4` (default) | ~0.88 s (44 µs/step, 2.8x over one thread) |
| `microgpt 20000 32 8` | ~1.1 s (56 µs/step, 4.3x over one thread) |
| `microgpt 20000 32 6` | ~1.15 s (58 µs/step, 4.1x over one thread) |

Scaling is bounded by synchronisation: a 32-document step is about 30 µs of compute on 8 cores,
and the two barriers, the parallel update and the load imbalance between documents of different
lengths cost another ~25 µs. Bigger batches amortise that better;
more threads than physical performance cores make it worse. The default of 16 x 4 is the
configuration that reached the best held-out loss per second of training in the loss-curve
comparison below.

The RNG is xoshiro256** with Box-Muller gaussians. It is deterministic (seed 42) but does not
reproduce Python's `random` stream, so the exact loss curve and sampled names differ from the gist.

## Layout

Everything is in [src/main.rs](src/main.rs), top to bottom: RNG, linear-algebra kernels, model shapes,
activation buffers, forward, backward, Adam, dataset, threaded trainer, main, tests.
