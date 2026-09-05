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
MICROGPT_N_EMBD=32 MICROGPT_N_LAYER=2 cargo build --release            # also MICROGPT_N_HEAD, MICROGPT_BLOCK_SIZE
MICROGPT_N_EXPERTS=8 MICROGPT_TOP_K=2 MICROGPT_HIDDEN=32 cargo build --release   # mixture of experts
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

Published at [kodart.github.io/microgpt](https://kodart.github.io/microgpt/) (the interactive
loss curves are at [kodart.github.io/microgpt/loss-curves.html](https://kodart.github.io/microgpt/loss-curves.html)).
[docs/report.html](docs/report.html) is a self-contained write-up of the port and every experiment
below: what each optimisation bought, thread scaling, batch size at equal data, the learning-rate
sweep, seed noise, and model size at matched time budgets, with charts.

## Training on a pre-tokenized corpus

Set `MICROGPT_TOKENS=<path>/tokens.bin` to train on token sequences instead of `input.txt`.
The file holds little-endian `u16` tokens; `<path>/tokens.idx.bin` beside it starts with the
magic `MGPT`, a `u32` vocabulary size and a `u32` document count, then one 13-byte record per
document: `u64` token offset, `u32` length, `u8` split (0 train, 1 held out). Every document
must begin with token 1 (BOS) and end with token 2 (EOS); sampling starts from 1 and stops at 2
and prints token ids. Documents longer than the block are truncated. This is the format the
melody pipeline in the `wow` project writes; a 515k-melody, 207M-token corpus loads in about
a second and the gist-sized model trains on it at ~270k tokens/s on 4 threads.

## Loss curves and experiments

Knobs, all environment variables so the positional arguments stay `[num_steps] [batch_size] [threads]`:

| variable | default | effect |
|---|---|---|
| `MICROGPT_LOG` | unset | write `step,train_time_s,train_loss,eval_loss` to this CSV |
| `MICROGPT_EVAL_EVERY` | 100 | log interval in steps |
| `MICROGPT_LR` | 0.01 | peak learning rate |
| `MICROGPT_SCHEDULE` | linear | decay of the learning rate to zero over the run: `linear` (the gist), `cosine`, or `constant` (no decay) |
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

### Mixture of experts

The MLP of each block can be a mixture of experts, as in GLM, DeepSeek and Mixtral: `N_EXPERTS`
MLPs of hidden width `N_HIDDEN`, and a router (a `N_EXPERTS x n_embd` matrix) that scores them
per position. The `TOP_K` most probable experts run, their outputs are mixed with the
renormalised router probabilities as gates, and the result is added to the residual stream.
Parameters grow with the number of experts; compute per token only with `TOP_K x N_HIDDEN`.
The dense model is the one-expert, top-1 case and is bit-for-bit unchanged.

How it is implemented: the (position, slot) pairs of a document are grouped by expert, each
expert's inputs are gathered into a contiguous buffer, and every expert runs as one
position-batched matmul (the same kernels as the dense path). The backward pass mirrors that:
gradients reach the chosen experts scaled by their gate, the gate gradient `dx . expert_output`
flows through the renormalisation and the softmax to the router, and unchosen experts get
nothing. Because unchosen experts never learn, a Switch-Transformer-style balancing loss
`0.01 * N * sum_e f_e * P_e` is added per document (`f_e` = fraction of pairs routed to `e`,
`P_e` = mean router probability); its gradient pushes the router away from experts it overuses.
The training loss includes it; the reported held-out loss is plain cross-entropy. Runs print
the held-out expert usage per layer. The f64 gradient check covers the router and experts,
skipping parameters whose perturbation flips a top-k choice (the loss is not differentiable there).

Held-out loss, 1 layer, 16-dim, 4 threads, 20000 steps:

| MLP | params | 16 x 4 | 64 x 4 | µs/step at 16 x 4 |
|---|---|---|---|---|
| dense, width 64 | 4,192 | 2.123 | 2.104 | 43 |
| 4 experts of 32, top-1 | 6,304 | 2.156 | 2.136 | 43 |
| 4 experts of 32, top-2 | 6,304 | 2.114 | 2.089 | 50 |
| 8 experts of 32, top-2 | 10,496 | 2.106 | 2.073 | 60 |
| 4 experts of 64, top-2 | 10,400 | 2.096 | 2.077 | 71 |
| 8 experts of 64, top-2 | 18,688 | 2.100 | **2.049** | 84 |

Top-2 with the same per-token compute as the dense MLP (4 experts of 32) already beats it, and
8 experts of 64 at 64 x 4 reaches 2.049 in 5.6 s, better and faster than the 32-dim dense model
(2.053 in 7.9 s). Top-1 is worse than dense: each token then gets a 32-wide expert instead of a
64-wide MLP, and with one gate there is no gate gradient to teach the router. Usage stays within
a few percent of even with the balancing loss on. Without it (coefficient 0) routing goes lopsided
rather than collapsing: 44/9/8/40% for 4 experts, and with 8 experts two of them end up at 0% and
3%; the held-out loss is the same at 4 experts (2.121 vs 2.114 at 16 x 4, 2.087 vs 2.089 at
64 x 4) and slightly worse at 8 (2.080 vs 2.073).

What the experts learn: set `MICROGPT_EXPERT_STATS=1` to print, per layer, which expert each
input character and each position is routed to on the held-out names. For the 8-experts-of-64
model the split is by character class. One expert takes every name start (100% of BOS, 83% of
position 1) and most initial consonants; one takes `a` (94%); one takes `e`, `i`, `y` (78-88%);
one takes `o`; one takes `n`, `r`, `x`; one takes `h`, `l`, `s`; one takes `v`, `d`, `g`, `t`;
and one mostly handles second positions and `u`. In a character model the current character
largely fixes the shape of the next-character distribution (after a vowel a consonant is likely,
after `q` a `u`), so an MLP specialised per character group is a better use of the same compute
than one MLP that must serve all of them. The stacked 2-layer, 32-dim, 8-expert model
(`MICROGPT_N_LAYER=2 MICROGPT_N_EMBD=32 MICROGPT_N_EXPERTS=8 MICROGPT_TOP_K=2 MICROGPT_HIDDEN=64`,
76,480 params) reaches 2.024 at 64 x 4 in 17.8 s, only marginally better than the dense 2-layer
32-dim model's 2.026 in 7.5 s: at 640k names the two capacity gains barely stack.

### Pruning with compensation

`MICROGPT_PRUNE=1` (or a list such as `MICROGPT_PRUNE=8,16,32`) runs a pruning study after
training: hidden neurons are removed from every expert of every layer and the held-out loss is
measured at each level, for three ways of choosing and removing them.

- **magnitude:** drop the neurons with the smallest `||fc1[j]|| * ||fc2[:, j]||`; change nothing else.
- **activation:** drop the smallest `||fc2[:, j]||^2 * E[r_j^2]` (output weight times how much the
  neuron fires); change nothing else.
- **OBS + compensation:** Optimal Brain Surgeon on the layer's own reconstruction problem. With
  `R` the hidden activations on 4000 training names, choose the pruned `fc2'` minimising
  `||fc2 R - fc2' R||^2`. That quadratic's Hessian is `G = R R^T`, so removing neuron `j` costs
  `||fc2[:, j]||^2 / [G^-1]_jj` and the surviving columns absorb it through
  `fc2[i, :] -= fc2[i, j] / [G^-1]_jj * G^-1[j, :]`; `G^-1` then drops row and column `j` by its
  Schur complement, and the next cheapest neuron goes. This is the layer-wise form of the
  second-order pruning theory (Hassibi & Stork 1993) that SparseGPT and Optimal Brain
  Compression apply to large language models.
- **OBS + fine-tune:** the OBS-pruned model trained for `MICROGPT_PRUNE_FINETUNE` more steps
  (default 2000) at a fifth of the learning rate. A pruned neuron has its `fc1` row and `fc2`
  column zeroed, and `relu(0) = 0` keeps it dead under training.

Held-out loss after removing `k` of 64 hidden neurons (per expert), 4 threads:

| model | base | k | magnitude | activation | OBS | OBS + fine-tune |
|---|---|---|---|---|---|---|
| dense, `20000 16 4` (12 neurons never activate) | 2.121 | 16 | 2.131 | 2.122 | **2.121** | 2.124 |
| | | 32 | 2.211 | 2.147 | 2.135 | **2.132** |
| | | 48 | 2.270 | 2.194 | 2.180 | **2.153** |
| dense, `20000 64 4` (no dead neurons) | 2.100 | 16 | 2.237 | 2.145 | 2.127 | **2.112** |
| | | 32 | 2.401 | 2.236 | 2.169 | **2.129** |
| | | 48 | 2.510 | 2.341 | 2.254 | **2.156** |
| 8 experts of 64, `20000 64 4` | 2.058 | 16 | 2.240 | 2.103 | 2.086 | **2.068** |
| | | 32 | 2.371 | 2.176 | 2.141 | **2.091** |
| | | 48 | 2.428 | 2.286 | 2.216 | **2.123** |

What the numbers say. Magnitude pruning is the wrong criterion at every level: a neuron with
small weights can still carry most of the layer's variance. Ranking by activation fixes most of
that, and compensation on top is worth a further 0.01-0.09. The default-trained model has 12
neurons that never fire, so the first 16 come off for free with any Hessian-aware method; the
better-trained 64 x 4 model has no dead neurons and every removal costs. A short fine-tune
recovers about half of what pruning costs beyond the free region (removing half of the
64 x 4 model's neurons ends 0.03 above the unpruned model instead of 0.07), but at 16 removed it
slightly *hurts* the default model, because a fresh Adam at a fifth of the rate perturbs an
already converged network more than it repairs. The experts prune about as gracefully as the
dense MLP per neuron, and since they hold 8x the neurons, the mixture loses far less per
parameter removed.

### BLAS and the GPU

Both were measured rather than assumed. Routing the three matmul kernels through Apple's
Accelerate `cblas_sgemm` (which uses the AMX matrix coprocessor) makes every model *slower* on
whole training steps, even though Accelerate wins an isolated per-call benchmark on the 32-dim
shapes by 3-7x:

| model | config | our kernels | Accelerate |
|---|---|---|---|
| gist (16-dim) | 20000 x 16 x 4 | 43 µs/step | 62 µs/step |
| 2 layers, 32-dim, hidden 128 | 20000 x 16 x 4 | 218 | 243 |
| 2 layers, 32-dim, hidden 128 | 5000 x 64 x 4 | 729 | 810 |
| 2 layers, 32-dim, 8 experts of 64 | 20000 x 16 x 4 | 281 | 564 |

Per document the matrices have only 16 rows (one per position), so each call is a few hundred
nanoseconds of arithmetic and BLAS's per-call overhead dominates; our kernels also get their
dimensions constant-folded and fully unrolled in the real build, which the isolated benchmark
hides. The expert groups (about 4 rows each) are hopeless for BLAS. Where Accelerate does win is
on the *same* shapes with hundreds of rows: at 256 rows it runs the 32-dim layers at ~700 GFLOP/s
against ~45 for our kernels, which do no cache blocking. So BLAS pays only after a structural
change, batching positions across all the documents of a worker's share so each layer stage is
one large matmul; that is the route to a several-fold gain on the 2-layer models, not a drop-in.

The GPU (Metal, via wgpu) is further off: a synchronous kernel round trip measures 1.3 ms on
this M1 Pro and each queued dispatch about 2.6 µs, so even a fully resident training loop with
its 10-15 dispatches per step would start near the CPU's 44 µs step before doing any arithmetic.
It pays only for batches of thousands of names, which train worse per name at this data size,
or for models far larger than any here.

### Findings from the sweeps (held-out loss, 4 threads unless noted)

- **Seed noise:** four seeds of `20000 16 4` end at 2.120-2.130, so differences under ~0.01 are noise.
- **Learning rate:** 0.01 is already at the optimum for batched Adam here (`16 x 4`: 0.0025 -> 2.135,
  0.005 -> 2.119, 0.01 -> 2.122, 0.02 -> 2.162; `64 x 4`: 0.005 -> 2.107, 0.01 -> 2.105, 0.02 -> 2.124).
  Larger batches do not want a larger rate. The single-name recipe prefers a lower one
  (`20000 1 1` at 0.005 -> 2.203 versus 2.223 at 0.01).
- **Learning-rate schedule:** linear and cosine decay are indistinguishable at the default run
  (three seeds each: 2.121-2.130 linear, 2.127-2.132 cosine) and cosine is ahead by 0.003-0.006
  in single-seed 64 x 4 runs, dense and 8-expert alike, which is at the noise floor. No decay at
  all costs 0.07 (2.195-2.208). The decay to zero is what matters, not its shape; linear stays
  the default.
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
