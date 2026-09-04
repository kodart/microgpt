//! microgpt in Rust.
//!
//! A port of Andrej Karpathy's `microgpt.py` gist: the most atomic way to train and run
//! inference for a GPT, in pure, dependency-free Rust (std only).
//!
//! The Python original builds a scalar autograd graph (one heap object per arithmetic op) and
//! walks it backwards. That is the whole cost of the program. Here the maths is identical but
//! the backward pass is derived by hand at the vector level, so the model is just a handful of
//! flat `f32` buffers and tight loops:
//!
//!  * parameters, gradients and Adam moments live in four flat arrays (Adam is one loop)
//!  * activations are allocated once and reused; the stored K/V rows double as the KV cache
//!  * Q, K and V projections are fused into a single (3*n_embd x n_embd) matrix
//!  * all hot loops are plain indexed loops over contiguous slices, so LLVM vectorises them
//!
//! Model: GPT-2 flavoured, with rmsnorm instead of layernorm, no biases, ReLU instead of GeLU.

// The numeric kernels are written as plain indexed loops on purpose: they read like the maths
// and LLVM vectorises them just as well as iterator chains.
#![allow(clippy::needless_range_loop)]

use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::Instant;

/// Training/inference runs in f32; the gradient-check test runs the same code in f64.
#[cfg(not(test))]
type Float = f32;
#[cfg(test)]
type Float = f64;

// ---------------------------------------------------------------------------------------------
// Hyperparameters (same values as the gist)
// ---------------------------------------------------------------------------------------------

// N_LAYER, N_EMBD, BLOCK_SIZE and N_HEAD come from build.rs: the gist's values (1, 16, 16, 4)
// unless overridden with MICROGPT_N_LAYER / MICROGPT_N_EMBD / ... at build time.
include!(concat!(env!("OUT_DIR"), "/model_config.rs"));
const HEAD_DIM: usize = N_EMBD / N_HEAD; // dimension of each head
const N_HIDDEN: usize = 4 * N_EMBD; // MLP hidden width
const INIT_STD: Float = 0.08; // std of the gaussian parameter init
const RMS_EPS: Float = 1e-5; // rmsnorm epsilon

const LEARNING_RATE: Float = 0.01;
const BETA1: Float = 0.85;
const BETA2: Float = 0.99;
const EPS_ADAM: Float = 1e-8;
const DEFAULT_NUM_STEPS: usize = 20000;
const DEFAULT_BATCH_SIZE: usize = 16; // documents per optimizer step
const DEFAULT_THREADS: usize = 4; // upper bound on the default thread count

const TEMPERATURE: Float = 0.5; // in (0, 1], controls the "creativity" of generated text
const NUM_SAMPLES: usize = 20;

const N_EVAL_DOCS: usize = 1000; // held-out names used only to measure the loss

const NAMES_URL: &str = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt";

// ---------------------------------------------------------------------------------------------
// RNG: xoshiro256** seeded through splitmix64. Deterministic, dependency-free.
// ---------------------------------------------------------------------------------------------

struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> Float {
        ((self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)) as Float
    }

    /// Gaussian sample via Box-Muller.
    fn gauss(&mut self, mean: Float, std: Float) -> Float {
        let u1 = 1.0 - self.uniform(); // (0, 1], safe for ln
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI as Float * u2;
        mean + std * r * theta.cos()
    }

    /// In-place Fisher-Yates shuffle.
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }

    /// Sample an index proportionally to `weights` (need not be normalised).
    fn choice_weighted(&mut self, weights: &[Float]) -> usize {
        let total: Float = weights.iter().sum();
        let mut r = self.uniform() * total;
        for (i, &w) in weights.iter().enumerate() {
            if r < w {
                return i;
            }
            r -= w;
        }
        weights.len() - 1 // floating point slack: fall through to the last index
    }
}

// ---------------------------------------------------------------------------------------------
// Small dense-linear-algebra kernels over contiguous slices. Matrices are row-major [rows][cols].
// ---------------------------------------------------------------------------------------------

#[inline]
fn row(buf: &[Float], i: usize, width: usize) -> &[Float] {
    &buf[i * width..(i + 1) * width]
}

#[inline]
fn row_mut(buf: &mut [Float], i: usize, width: usize) -> &mut [Float] {
    &mut buf[i * width..(i + 1) * width]
}

#[inline]
fn dot(a: &[Float], b: &[Float]) -> Float {
    // Four independent lanes so the reduction maps to one vector FMA per 4 elements instead of
    // a serial chain of scalar FMAs (LLVM may not reassociate floating-point sums on its own).
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0.0 as Float; 4];
    for (x, y) in ca.zip(cb) {
        for l in 0..4 {
            acc[l] = x[l].mul_add(y[l], acc[l]);
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        s = x.mul_add(*y, s);
    }
    s
}

/// Width of the register-resident accumulator used by the accumulating matmuls: 16 floats is
/// four NEON registers, so a chunk of an output row stays in registers across the whole
/// reduction instead of being re-read and re-written from L1 for every term.
const ACC: usize = 16;

/// C[n x m] = A[n x k] . B[m x k]^T  -- rows of A against rows of a weight matrix laid out
/// [out][in], i.e. the forward pass of a linear layer over `n` positions at once.
#[inline]
fn matmul_abt(a: &[Float], n: usize, k: usize, b: &[Float], m: usize, c: &mut [Float]) {
    debug_assert!(a.len() >= n * k && b.len() == m * k && c.len() >= n * m);
    for i in 0..n {
        let ai = &a[i * k..(i + 1) * k];
        let ci = &mut c[i * m..(i + 1) * m];
        for o in 0..m {
            ci[o] = dot(ai, &b[o * k..(o + 1) * k]);
        }
    }
}

/// C[n x k] += A[n x m] . B[m x k]  -- dX += dY . W for a weight matrix W laid out [out][in]
/// (m = out, k = in). Each 16-wide chunk of an output row is accumulated in registers over all
/// `m` terms before being written back once.
#[inline]
fn matmul_ab_acc(a: &[Float], n: usize, m: usize, b: &[Float], k: usize, c: &mut [Float]) {
    debug_assert!(a.len() >= n * m && b.len() == m * k && c.len() >= n * k);
    for i in 0..n {
        let ai = &a[i * m..(i + 1) * m];
        let ci = &mut c[i * k..(i + 1) * k];
        let mut cs = 0;
        while cs + ACC <= k {
            let mut acc = [0.0 as Float; ACC];
            acc.copy_from_slice(&ci[cs..cs + ACC]);
            for o in 0..m {
                let g = ai[o];
                let bo = &b[o * k + cs..o * k + cs + ACC];
                for j in 0..ACC {
                    acc[j] = g.mul_add(bo[j], acc[j]);
                }
            }
            ci[cs..cs + ACC].copy_from_slice(&acc);
            cs += ACC;
        }
        for j in cs..k {
            let mut v = ci[j];
            for o in 0..m {
                v = ai[o].mul_add(b[o * k + j], v);
            }
            ci[j] = v;
        }
    }
}

/// C[m x k] += A[n x m]^T . B[n x k]  -- dW += dY^T . X summed over the `n` positions of a
/// document, so the weight gradient is touched once per document instead of once per position.
#[inline]
fn matmul_atb_acc(a: &[Float], n: usize, m: usize, b: &[Float], k: usize, c: &mut [Float]) {
    debug_assert!(a.len() >= n * m && b.len() >= n * k && c.len() == m * k);
    for o in 0..m {
        let co = &mut c[o * k..(o + 1) * k];
        let mut cs = 0;
        while cs + ACC <= k {
            let mut acc = [0.0 as Float; ACC];
            acc.copy_from_slice(&co[cs..cs + ACC]);
            for i in 0..n {
                let g = a[i * m + o];
                let bi = &b[i * k + cs..i * k + cs + ACC];
                for j in 0..ACC {
                    acc[j] = g.mul_add(bi[j], acc[j]);
                }
            }
            co[cs..cs + ACC].copy_from_slice(&acc);
            cs += ACC;
        }
        for j in cs..k {
            let mut v = co[j];
            for i in 0..n {
                v = a[i * m + o].mul_add(b[i * k + j], v);
            }
            co[j] = v;
        }
    }
}

/// x[i] += y[i]
#[inline]
fn add_assign(x: &mut [Float], y: &[Float]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += *b;
    }
}

/// y = x * (mean(x^2) + eps)^-0.5
#[inline]
fn rmsnorm(x: &[Float], y: &mut [Float]) {
    let ms = dot(x, x) / x.len() as Float;
    let scale = 1.0 / (ms + RMS_EPS).sqrt();
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi = xi * scale;
    }
}

/// dx += d rmsnorm(x) / dx applied to dy.
/// With s = (ms + eps)^-1/2:  dx_k = s*dy_k - (s^3 / D) * (dy . x) * x_k
#[inline]
fn rmsnorm_bwd(x: &[Float], dy: &[Float], dx: &mut [Float]) {
    let d = x.len() as Float;
    let ms = dot(x, x) / d;
    let s = 1.0 / (ms + RMS_EPS).sqrt();
    let c = s * s * s / d * dot(dy, x);
    for k in 0..x.len() {
        dx[k] = s.mul_add(dy[k], (-c).mul_add(x[k], dx[k]));
    }
}

/// In-place softmax over a slice.
#[inline]
fn softmax_inplace(v: &mut [Float]) {
    let max = v.iter().copied().fold(Float::NEG_INFINITY, Float::max);
    let mut total = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        total += *x;
    }
    let inv = 1.0 / total;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

// ---------------------------------------------------------------------------------------------
// Model: a description of where each weight matrix lives inside one flat parameter buffer.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Tensor {
    off: usize,
    rows: usize,
    cols: usize,
}

impl Tensor {
    #[inline]
    fn len(&self) -> usize {
        self.rows * self.cols
    }
    #[inline]
    fn view<'a>(&self, buf: &'a [Float]) -> &'a [Float] {
        &buf[self.off..self.off + self.len()]
    }
    #[inline]
    fn view_mut<'a>(&self, buf: &'a mut [Float]) -> &'a mut [Float] {
        &mut buf[self.off..self.off + self.len()]
    }
    #[inline]
    fn row<'a>(&self, buf: &'a [Float], r: usize) -> &'a [Float] {
        row(self.view(buf), r, self.cols)
    }
}

struct LayerShapes {
    wqkv: Tensor, // [3*n_embd][n_embd]: fused attn_wq / attn_wk / attn_wv
    wo: Tensor,   // [n_embd][n_embd]
    fc1: Tensor,  // [4*n_embd][n_embd]
    fc2: Tensor,  // [n_embd][4*n_embd]
}

struct Model {
    vocab_size: usize,
    wte: Tensor,     // [vocab_size][n_embd] token embeddings
    wpe: Tensor,     // [block_size][n_embd] position embeddings
    lm_head: Tensor, // [vocab_size][n_embd]
    layers: Vec<LayerShapes>,
    n_params: usize,
}

impl Model {
    fn new(vocab_size: usize) -> Self {
        let mut n = 0usize;
        let mut alloc = |rows: usize, cols: usize| {
            let t = Tensor { off: n, rows, cols };
            n += rows * cols;
            t
        };
        let wte = alloc(vocab_size, N_EMBD);
        let wpe = alloc(BLOCK_SIZE, N_EMBD);
        let lm_head = alloc(vocab_size, N_EMBD);
        let layers = (0..N_LAYER)
            .map(|_| LayerShapes {
                wqkv: alloc(3 * N_EMBD, N_EMBD),
                wo: alloc(N_EMBD, N_EMBD),
                fc1: alloc(N_HIDDEN, N_EMBD),
                fc2: alloc(N_EMBD, N_HIDDEN),
            })
            .collect();
        Model { vocab_size, wte, wpe, lm_head, layers, n_params: n }
    }

    fn init_params(&self, rng: &mut Rng) -> Vec<Float> {
        (0..self.n_params).map(|_| rng.gauss(0.0, INIT_STD)).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Activations: allocated once for BLOCK_SIZE positions and reused for every step.
// The stored K/V rows in `qkv` are exactly the KV cache needed for incremental inference.
//
// Everything with compile-time dimensions is a fixed-size inline array inside one boxed struct,
// not a `Vec`. This matters a lot for codegen: with separate heap buffers reached through
// `&mut Acts`, LLVM cannot prove the buffers are disjoint and emits noticeably worse (less
// vectorised, more spilled) forward/backward code unless the buffers happen to be locals of the
// calling function. Inline arrays at fixed offsets of a single `noalias` struct give it both
// disjointness and constant lengths in every calling context. Only vocab-sized buffers stay Vecs.
// ---------------------------------------------------------------------------------------------

const T: usize = BLOCK_SIZE; // short alias for the sequence-length dimension of the buffers

struct LayerActs {
    h: [Float; T * N_EMBD],            // rmsnorm(x_in), input to qkv projection
    qkv: [Float; T * 3 * N_EMBD],      // q | k | v
    attn_w: [Float; N_HEAD * T * T],   // softmax attention weights (causal, lower triangle used)
    attn_out: [Float; T * N_EMBD],     // concatenated head outputs
    x2: [Float; T * N_EMBD],           // x_in + wo(attn_out)      (residual stream after attention)
    h2: [Float; T * N_EMBD],           // rmsnorm(x2)
    r: [Float; T * N_HIDDEN],          // relu(fc1(h2))
}

impl LayerActs {
    fn zeroed() -> Self {
        LayerActs {
            h: [0.0; T * N_EMBD],
            qkv: [0.0; T * 3 * N_EMBD],
            attn_w: [0.0; N_HEAD * T * T],
            attn_out: [0.0; T * N_EMBD],
            x2: [0.0; T * N_EMBD],
            h2: [0.0; T * N_EMBD],
            r: [0.0; T * N_HIDDEN],
        }
    }
}

struct Acts {
    x0: [Float; T * N_EMBD],                  // wte[tok] + wpe[pos]
    xs: [[Float; T * N_EMBD]; N_LAYER + 1],   // xs[l] is the input of layer l, xs[N_LAYER] the final
    layers: [LayerActs; N_LAYER],
    logits: Vec<Float>, // [T][V]
    probs: Vec<Float>,  // [T][V]
}

impl Acts {
    fn new(vocab_size: usize) -> Box<Self> {
        Box::new(Acts {
            x0: [0.0; T * N_EMBD],
            xs: [[0.0; T * N_EMBD]; N_LAYER + 1],
            layers: std::array::from_fn(|_| LayerActs::zeroed()),
            logits: vec![0.0; T * vocab_size],
            probs: vec![0.0; T * vocab_size],
        })
    }
}

/// Scratch buffers for the backward pass.
struct Grads {
    dx: [Float; T * N_EMBD],
    dx2: [Float; T * N_EMBD],
    datt: [Float; T * N_EMBD],
    dh: [Float; T * N_EMBD],
    dr: [Float; T * N_HIDDEN],
    dqkv: [Float; T * 3 * N_EMBD],
    dlogits: Vec<Float>, // [T][V]
    dx0: [Float; N_EMBD],
}

impl Grads {
    fn new(vocab_size: usize) -> Box<Self> {
        Box::new(Grads {
            dx: [0.0; T * N_EMBD],
            dx2: [0.0; T * N_EMBD],
            datt: [0.0; T * N_EMBD],
            dh: [0.0; T * N_EMBD],
            dr: [0.0; T * N_HIDDEN],
            dqkv: [0.0; T * 3 * N_EMBD],
            dlogits: vec![0.0; T * vocab_size],
            dx0: [0.0; N_EMBD],
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------------------------

impl Model {
    /// Run positions `from..to` of `tokens` through the model, filling `acts` (including logits).
    /// Positions `< from` must already be present in `acts` (they act as the KV cache), so
    /// training calls this with `from = 0` and generation with `from = pos, to = pos + 1`.
    ///
    /// Each layer is applied stage by stage over all `from..to` positions: the linear layers are
    /// one small matmul over the positions, and only attention and rmsnorm loop per position.
    fn forward(&self, p: &[Float], tokens: &[usize], from: usize, to: usize, a: &mut Acts) {
        const E: usize = N_EMBD;
        let n = to - from;
        let rows = from * E..to * E;
        let scale = 1.0 / (HEAD_DIM as Float).sqrt();

        // token + position embedding, then rmsnorm (not redundant: gradients flow via residual)
        for i in from..to {
            let wte = self.wte.row(p, tokens[i]);
            let wpe = self.wpe.row(p, i);
            let x0 = row_mut(&mut a.x0, i, E);
            for d in 0..E {
                x0[d] = wte[d] + wpe[d];
            }
            rmsnorm(x0, row_mut(&mut a.xs[0], i, E));
        }

        for l in 0..N_LAYER {
            let (lo, hi) = a.xs[..].split_at_mut(l + 1);
            let (xs_in, xs_out) = (&lo[l], &mut hi[0]);
            let la = &mut a.layers[l];
            let sh = &self.layers[l];

            // 1) Multi-head attention block
            for i in from..to {
                rmsnorm(row(xs_in, i, E), row_mut(&mut la.h, i, E));
            }
            matmul_abt(&la.h[rows.clone()], n, E, sh.wqkv.view(p), 3 * E, &mut la.qkv[from * 3 * E..to * 3 * E]);
            for i in from..to {
                for hd in 0..N_HEAD {
                    let hs = hd * HEAD_DIM;
                    let q = &la.qkv[i * 3 * E + hs..][..HEAD_DIM];
                    let w = &mut la.attn_w[(hd * T + i) * T..][..=i];
                    for j in 0..=i {
                        let k = &la.qkv[j * 3 * E + E + hs..][..HEAD_DIM];
                        w[j] = dot(q, k) * scale;
                    }
                    softmax_inplace(w);
                    let o = &mut la.attn_out[i * E + hs..][..HEAD_DIM];
                    o.fill(0.0);
                    for j in 0..=i {
                        let v = &la.qkv[j * 3 * E + 2 * E + hs..][..HEAD_DIM];
                        let wj = w[j];
                        for d in 0..HEAD_DIM {
                            o[d] = wj.mul_add(v[d], o[d]);
                        }
                    }
                }
            }
            matmul_abt(&la.attn_out[rows.clone()], n, E, sh.wo.view(p), E, &mut la.x2[rows.clone()]);
            add_assign(&mut la.x2[rows.clone()], &xs_in[rows.clone()]);

            // 2) MLP block
            for i in from..to {
                rmsnorm(row(&la.x2, i, E), row_mut(&mut la.h2, i, E));
            }
            let hrows = from * N_HIDDEN..to * N_HIDDEN;
            matmul_abt(&la.h2[rows.clone()], n, E, sh.fc1.view(p), N_HIDDEN, &mut la.r[hrows.clone()]);
            for v in la.r[hrows.clone()].iter_mut() {
                *v = v.max(0.0);
            }
            matmul_abt(&la.r[hrows], n, N_HIDDEN, sh.fc2.view(p), E, &mut xs_out[rows.clone()]);
            add_assign(&mut xs_out[rows.clone()], &la.x2[rows.clone()]);
        }

        let v = self.vocab_size;
        matmul_abt(&a.xs[N_LAYER][rows], n, E, self.lm_head.view(p), v, &mut a.logits[from * v..to * v]);
    }

    /// Softmax the first `n` logit rows into `probs` and return the mean cross-entropy loss
    /// against `targets[i] = tokens[i + 1]`.
    fn loss(&self, tokens: &[usize], n: usize, a: &mut Acts) -> Float {
        let v = self.vocab_size;
        let mut loss = 0.0;
        for i in 0..n {
            let probs = row_mut(&mut a.probs, i, v);
            probs.copy_from_slice(row(&a.logits, i, v));
            softmax_inplace(probs);
            loss -= probs[tokens[i + 1]].ln();
        }
        loss / n as Float
    }

    // -----------------------------------------------------------------------------------------
    // Backward pass: accumulates d(loss)/d(params) into `g` for the first `n` positions.
    // Mirrors the forward stage by stage: every linear layer contributes one dW += dY^T X over
    // all positions and one dX += dY W, then rmsnorm and attention are walked per position.
    // -----------------------------------------------------------------------------------------
    fn backward(&self, p: &[Float], g: &mut [Float], tokens: &[usize], n: usize, a: &Acts, s: &mut Grads) {
        const E: usize = N_EMBD;
        const H: usize = N_HIDDEN;
        let v = self.vocab_size;
        let scale = 1.0 / (HEAD_DIM as Float).sqrt();
        let inv_n = 1.0 / n as Float;

        // loss = mean_i( -log softmax(logits_i)[target_i] )  =>  dlogits = (probs - onehot) / n
        let dl = &mut s.dlogits[..n * v];
        dl.copy_from_slice(&a.probs[..n * v]);
        for i in 0..n {
            dl[i * v + tokens[i + 1]] -= 1.0;
        }
        for x in dl.iter_mut() {
            *x *= inv_n;
        }
        let x_final = &a.xs[N_LAYER][..n * E];
        matmul_atb_acc(dl, n, v, x_final, E, self.lm_head.view_mut(g));
        s.dx[..n * E].fill(0.0);
        matmul_ab_acc(dl, n, v, self.lm_head.view(p), E, &mut s.dx[..n * E]);

        for l in (0..N_LAYER).rev() {
            let la = &a.layers[l];
            let sh = &self.layers[l];
            let x_in = &a.xs[l][..n * E];
            let dx = &mut s.dx[..n * E];

            // MLP block: x_out = x2 + fc2(relu(fc1(rmsnorm(x2))));  dx holds d/dx_out
            let r = &la.r[..n * H];
            matmul_atb_acc(dx, n, E, r, H, sh.fc2.view_mut(g));
            let dr = &mut s.dr[..n * H];
            dr.fill(0.0);
            matmul_ab_acc(dx, n, E, sh.fc2.view(p), H, dr);
            for (d, &rv) in dr.iter_mut().zip(r) {
                if rv <= 0.0 {
                    *d = 0.0;
                }
            }
            matmul_atb_acc(dr, n, H, &la.h2[..n * E], E, sh.fc1.view_mut(g));
            let dh = &mut s.dh[..n * E];
            dh.fill(0.0);
            matmul_ab_acc(dr, n, H, sh.fc1.view(p), E, dh);
            let dx2 = &mut s.dx2[..n * E];
            dx2.copy_from_slice(dx);
            for i in 0..n {
                rmsnorm_bwd(row(&la.x2, i, E), row(dh, i, E), row_mut(dx2, i, E));
            }

            // attention output projection: x2 = x_in + wo(attn_out)
            matmul_atb_acc(dx2, n, E, &la.attn_out[..n * E], E, sh.wo.view_mut(g));
            let datt = &mut s.datt[..n * E];
            datt.fill(0.0);
            matmul_ab_acc(dx2, n, E, sh.wo.view(p), E, datt);

            // attention core (mixes positions)
            s.dqkv[..n * 3 * E].fill(0.0);
            let mut dw = [0.0 as Float; T];
            for hd in 0..N_HEAD {
                let hs = hd * HEAD_DIM;
                for i in 0..n {
                    let d_o = &s.datt[i * E + hs..][..HEAD_DIM];
                    let w = &la.attn_w[(hd * T + i) * T..][..=i];
                    // out_i = sum_j w_ij v_j
                    let mut wdot = 0.0;
                    for j in 0..=i {
                        let vj = &la.qkv[j * 3 * E + 2 * E + hs..][..HEAD_DIM];
                        dw[j] = dot(d_o, vj);
                        wdot += w[j] * dw[j];
                        let dv = &mut s.dqkv[j * 3 * E + 2 * E + hs..][..HEAD_DIM];
                        for d in 0..HEAD_DIM {
                            dv[d] = w[j].mul_add(d_o[d], dv[d]);
                        }
                    }
                    // softmax backward, then logits_ij = (q_i . k_j) * scale
                    let q = &la.qkv[i * 3 * E + hs..][..HEAD_DIM];
                    for j in 0..=i {
                        let ds = w[j] * (dw[j] - wdot) * scale;
                        let k = &la.qkv[j * 3 * E + E + hs..][..HEAD_DIM];
                        let dq = i * 3 * E + hs; // q part of row i
                        let dk = j * 3 * E + E + hs; // k part of row j
                        for d in 0..HEAD_DIM {
                            s.dqkv[dq + d] = ds.mul_add(k[d], s.dqkv[dq + d]);
                            s.dqkv[dk + d] = ds.mul_add(q[d], s.dqkv[dk + d]);
                        }
                    }
                }
            }

            // qkv projection and the pre-attention rmsnorm: h = rmsnorm(x_in)
            let dqkv = &s.dqkv[..n * 3 * E];
            matmul_atb_acc(dqkv, n, 3 * E, &la.h[..n * E], E, sh.wqkv.view_mut(g));
            let dh = &mut s.dh[..n * E];
            dh.fill(0.0);
            matmul_ab_acc(dqkv, n, 3 * E, sh.wqkv.view(p), E, dh);
            let dx = &mut s.dx[..n * E];
            dx.copy_from_slice(&s.dx2[..n * E]);
            for i in 0..n {
                rmsnorm_bwd(row(x_in, i, E), row(dh, i, E), row_mut(dx, i, E));
            }
        }

        // embeddings: xs[0] = rmsnorm(x0), x0 = wte[tok] + wpe[pos]
        for i in 0..n {
            s.dx0.fill(0.0);
            rmsnorm_bwd(row(&a.x0, i, E), row(&s.dx, i, E), &mut s.dx0);
            let t = tokens[i];
            let dwte = &mut g[self.wte.off + t * E..][..E];
            for d in 0..E {
                dwte[d] += s.dx0[d];
            }
            let dwpe = &mut g[self.wpe.off + i * E..][..E];
            for d in 0..E {
                dwpe[d] += s.dx0[d];
            }
        }
    }
}

impl Model {
    /// Mean per-document loss over `docs` (forward only, no gradients).
    fn eval_loss(&self, p: &[Float], data: &Dataset, docs: &[String], acts: &mut Acts, tokens: &mut Vec<usize>) -> Float {
        if docs.is_empty() {
            return Float::NAN;
        }
        let mut total = 0.0;
        for doc in docs {
            data.tokenize(doc, tokens);
            let n = (tokens.len() - 1).min(BLOCK_SIZE);
            self.forward(p, tokens, 0, n, acts);
            total += self.loss(tokens, n, acts);
        }
        total / docs.len() as Float
    }
}

// ---------------------------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------------------------

struct Adam {
    m: Vec<Float>,
    v: Vec<Float>,
}

impl Adam {
    fn new(n: usize) -> Self {
        Adam { m: vec![0.0; n], v: vec![0.0; n] }
    }

    /// One Adam step with bias correction; zeroes the gradient buffer afterwards.
    fn step(&mut self, p: &mut [Float], g: &mut [Float], lr: Float, t: usize) {
        let c1 = 1.0 / (1.0 - BETA1.powi(t as i32));
        let c2 = 1.0 / (1.0 - BETA2.powi(t as i32));
        for i in 0..p.len() {
            let gi = g[i];
            let m = BETA1.mul_add(self.m[i], (1.0 - BETA1) * gi);
            let v = BETA2.mul_add(self.v[i], (1.0 - BETA2) * gi * gi);
            self.m[i] = m;
            self.v[i] = v;
            p[i] -= lr * (m * c1) / ((v * c2).sqrt() + EPS_ADAM);
            g[i] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------------------------

struct Dataset {
    docs: Vec<String>, // shuffled; docs[..n_train] are trained on, docs[n_train..] are held out
    n_train: usize,
    uchars: Vec<char>, // sorted unique characters; index = token id
    bos: usize,        // beginning-of-sequence token id
}

impl Dataset {
    fn load(path: &str, rng: &mut Rng) -> io::Result<Self> {
        if !Path::new(path).exists() {
            eprintln!("downloading {NAMES_URL} -> {path}");
            let ok = Command::new("curl")
                .args(["-sSL", NAMES_URL, "-o", path])
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !ok {
                return Err(io::Error::other(format!(
                    "could not download the dataset; please save {NAMES_URL} as {path}"
                )));
            }
        }
        let text = std::fs::read_to_string(path)?;
        let mut docs: Vec<String> =
            text.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect();
        rng.shuffle(&mut docs);
        Ok(Self::new(docs, N_EVAL_DOCS))
    }

    /// The last `n_eval` documents are held out from training (at least one document is kept).
    fn new(docs: Vec<String>, n_eval: usize) -> Self {
        let uchars: Vec<char> = docs.iter().flat_map(|d| d.chars()).collect::<BTreeSet<_>>().into_iter().collect();
        let bos = uchars.len();
        let n_train = docs.len().saturating_sub(n_eval).max(1);
        Dataset { docs, n_train, uchars, bos }
    }

    fn eval_docs(&self) -> &[String] {
        &self.docs[self.n_train..]
    }

    fn vocab_size(&self) -> usize {
        self.uchars.len() + 1
    }

    /// [BOS] + chars + [BOS]
    fn tokenize(&self, doc: &str, out: &mut Vec<usize>) {
        out.clear();
        out.push(self.bos);
        for ch in doc.chars() {
            out.push(self.uchars.binary_search(&ch).expect("character not in vocab"));
        }
        out.push(self.bos);
    }
}

// ---------------------------------------------------------------------------------------------
// Training: minibatches spread over a pool of persistent worker threads.
//
// Every step, workers claim documents of the batch one at a time from a shared atomic counter
// (so fast cores naturally take more work than slow ones, and no worker is a straggler). Each
// worker forwards/backwards its documents into its own gradient buffer (sum of the per-document,
// per-token-averaged gradients). The main thread reduces the buffers, scales by 1/batch_size and
// takes one Adam step. Which worker sums which document varies between runs, so multi-threaded
// results are deterministic only up to floating-point summation order.
//
// A step is only tens of microseconds of compute, so parking/unparking threads through channels
// or condvars would dominate. Workers instead spin (briefly, then yield) on an atomic epoch
// counter that the main thread bumps to publish each step, and count themselves done on another
// atomic. Nothing allocates in steady state.
// ---------------------------------------------------------------------------------------------

struct TrainConfig {
    num_steps: usize,
    batch_size: usize,
    n_threads: usize,
    /// Optional loss log: every `eval_every` steps a row `step,train_time_s,train_loss,eval_loss`
    /// is written to this file. `train_loss` is the mean batch loss over the interval and
    /// `eval_loss` the mean loss on the held-out split. Parameters are snapshotted during training
    /// (a 16 KB copy) and evaluated after it ends, so logging does not perturb the timings.
    log_path: Option<String>,
    eval_every: usize,
    /// Peak learning rate; decays linearly to zero over `num_steps`.
    lr: Float,
}

/// Per-worker mailbox. Locks are only ever taken uncontended (main and worker alternate strictly).
struct WorkerSlot {
    grad: Mutex<Vec<Float>>, // accumulated gradient for the current step
    loss: Mutex<Float>,      // sum of per-document mean losses for the current step
}

struct Shared {
    epoch: AtomicUsize,    // step counter published by main; QUIT tells workers to exit
    next_doc: AtomicUsize, // next document (0..batch_size) of the current step to be claimed
    done: AtomicUsize,     // workers that have finished the current epoch
    batch_size: usize,
    params: RwLock<Vec<Float>>,
    slots: Vec<WorkerSlot>,
}

const QUIT: usize = usize::MAX;

/// Spin until `flag` differs from `seen`; back off to yielding so we behave when oversubscribed.
fn wait_until_changed(flag: &AtomicUsize, seen: usize) -> usize {
    let mut spins = 0u32;
    loop {
        let v = flag.load(Ordering::Acquire);
        if v != seen {
            return v;
        }
        if spins < 4096 {
            spins += 1;
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

fn wait_until_equal(flag: &AtomicUsize, target: usize) {
    let mut spins = 0u32;
    while flag.load(Ordering::Acquire) != target {
        if spins < 4096 {
            spins += 1;
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

/// Everything one thread needs to process documents; the main thread is worker 0.
struct Worker<'a> {
    model: &'a Model,
    data: &'a Dataset,
    shared: &'a Shared,
    slot: &'a WorkerSlot,
    acts: Box<Acts>,
    scratch: Box<Grads>,
    tokens: Vec<usize>,
}

impl<'a> Worker<'a> {
    fn new(model: &'a Model, data: &'a Dataset, shared: &'a Shared, w: usize) -> Self {
        Worker {
            model,
            data,
            shared,
            slot: &shared.slots[w],
            acts: Acts::new(model.vocab_size),
            scratch: Grads::new(model.vocab_size),
            tokens: Vec::with_capacity(BLOCK_SIZE + 2),
        }
    }

    /// Claim and process documents of step `epoch - 1` until the batch is exhausted.
    fn run_step(&mut self, epoch: usize) {
        let shared = self.shared;
        let base = (epoch - 1) * shared.batch_size;
        let p = shared.params.read().unwrap();
        let mut grad = self.slot.grad.lock().unwrap();
        let mut loss_sum = 0.0;
        loop {
            let k = shared.next_doc.fetch_add(1, Ordering::AcqRel);
            if k >= shared.batch_size {
                break;
            }
            let doc = &self.data.docs[(base + k) % self.data.n_train];
            self.data.tokenize(doc, &mut self.tokens);
            let n = (self.tokens.len() - 1).min(BLOCK_SIZE);
            self.model.forward(&p, &self.tokens, 0, n, &mut self.acts);
            loss_sum += self.model.loss(&self.tokens, n, &mut self.acts);
            self.model.backward(&p, &mut grad, &self.tokens, n, &self.acts, &mut self.scratch);
        }
        *self.slot.loss.lock().unwrap() = loss_sum;
    }
}

/// Train `params` in place; returns the loss of the last minibatch.
fn train(model: &Model, data: &Dataset, params: &mut Vec<Float>, cfg: &TrainConfig, out: &mut impl Write) -> io::Result<Float> {
    let batch_size = cfg.batch_size.max(1);
    let n_threads = cfg.n_threads.clamp(1, batch_size);
    let num_steps = cfg.num_steps;
    let n_params = model.n_params;

    let shared = Shared {
        epoch: AtomicUsize::new(0),
        next_doc: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
        batch_size,
        params: RwLock::new(std::mem::take(params)),
        slots: (0..n_threads).map(|_| WorkerSlot { grad: Mutex::new(vec![0.0 as Float; n_params]), loss: Mutex::new(0.0) }).collect(),
    };
    let mut grads = vec![0.0 as Float; n_params];
    let mut adam = Adam::new(n_params);
    let mut loss = 0.0;

    // (step, training seconds so far, mean train loss over the interval, parameter snapshot)
    let mut snapshots: Vec<(usize, f64, Float, Vec<Float>)> = Vec::new();
    let logging = cfg.log_path.is_some();
    let eval_every = cfg.eval_every.max(1);
    let mut interval_loss = 0.0;
    let mut interval_steps = 0usize;
    let start = Instant::now();

    std::thread::scope(|scope| -> io::Result<()> {
        // workers 1.. are threads; worker 0 is the main thread
        for w in 1..n_threads {
            let shared = &shared;
            scope.spawn(move || {
                let mut worker = Worker::new(model, data, shared, w);
                let mut seen = 0;
                loop {
                    let epoch = wait_until_changed(&shared.epoch, seen);
                    if epoch == QUIT {
                        break;
                    }
                    seen = epoch;
                    worker.run_step(epoch);
                    shared.done.fetch_add(1, Ordering::AcqRel);
                }
            });
        }

        let mut main_worker = Worker::new(model, data, &shared, 0);
        let inv_b = 1.0 / batch_size as Float;
        if logging {
            snapshots.push((0, 0.0, Float::NAN, shared.params.read().unwrap().clone()));
        }
        for step in 0..num_steps {
            // publish the step; every worker (this thread included) claims documents via next_doc
            shared.next_doc.store(0, Ordering::Release);
            shared.done.store(0, Ordering::Release);
            shared.epoch.store(step + 1, Ordering::Release);
            main_worker.run_step(step + 1);
            wait_until_equal(&shared.done, n_threads - 1);

            // reduce the worker buffers, zeroing them for reuse
            let mut loss_sum = 0.0;
            for slot in &shared.slots {
                loss_sum += *slot.loss.lock().unwrap();
                let mut wg = slot.grad.lock().unwrap();
                for (g, x) in grads.iter_mut().zip(wg.iter_mut()) {
                    *g = x.mul_add(inv_b, *g);
                    *x = 0.0;
                }
            }
            loss = loss_sum * inv_b;

            let lr_t = cfg.lr * (1.0 - step as Float / num_steps as Float); // linear decay
            adam.step(&mut shared.params.write().unwrap(), &mut grads, lr_t, step + 1);

            if logging {
                interval_loss += loss;
                interval_steps += 1;
                if (step + 1) % eval_every == 0 || step + 1 == num_steps {
                    let t = start.elapsed().as_secs_f64();
                    snapshots.push((step + 1, t, interval_loss / interval_steps as Float, shared.params.read().unwrap().clone()));
                    interval_loss = 0.0;
                    interval_steps = 0;
                }
            }

            if (step + 1) % 50 == 0 || step + 1 == num_steps {
                write!(out, "step {:5} / {:5} | loss {:.4}\r", step + 1, num_steps, loss)?;
                out.flush()?;
            }
        }
        shared.epoch.store(QUIT, Ordering::Release);
        Ok(())
    })?;

    *params = shared.params.into_inner().unwrap();

    if let Some(path) = &cfg.log_path {
        let mut log = io::BufWriter::new(std::fs::File::create(path)?);
        let mut acts = Acts::new(model.vocab_size);
        let mut tokens = Vec::with_capacity(BLOCK_SIZE + 2);
        writeln!(log, "step,train_time_s,train_loss,eval_loss")?;
        for (step, t, train_loss, snap) in &snapshots {
            let ev = model.eval_loss(snap, data, data.eval_docs(), &mut acts, &mut tokens);
            let train_loss = if train_loss.is_nan() { ev } else { *train_loss };
            writeln!(log, "{step},{t:.6},{train_loss},{ev}")?;
        }
    }
    Ok(loss)
}

// ---------------------------------------------------------------------------------------------
// Main: train + sample
// ---------------------------------------------------------------------------------------------

/// A step of this tiny model is only tens of microseconds of compute, so the per-step barrier
/// and reduction are a large fixed cost and thread scaling flattens out early. Measured on a
/// 10-core laptop (see docs/loss-curves.html), 16 documents on 4 threads reached the same
/// held-out loss as 32 on 8 in about half the time, and using every core hurts as soon as the
/// OS needs one. So the default is 4 threads, or fewer on smaller machines.
fn default_threads() -> usize {
    let n = std::thread::available_parallelism().map_or(1, |n| n.get());
    n.min(DEFAULT_THREADS)
}

fn main() -> io::Result<()> {
    let usage = "usage: microgpt [num_steps] [batch_size] [threads]";
    let mut args = std::env::args().skip(1).map(|s| s.parse::<usize>().expect(usage));
    let cfg = TrainConfig {
        num_steps: args.next().unwrap_or(DEFAULT_NUM_STEPS),
        batch_size: args.next().unwrap_or(DEFAULT_BATCH_SIZE),
        n_threads: args.next().unwrap_or_else(default_threads),
        log_path: std::env::var("MICROGPT_LOG").ok(),
        eval_every: std::env::var("MICROGPT_EVAL_EVERY").ok().and_then(|s| s.parse().ok()).unwrap_or(100),
        lr: std::env::var("MICROGPT_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(LEARNING_RATE),
    };
    let mut rng = Rng::new(42); // let there be order among chaos

    // The dataset is always shuffled with seed 42, so the held-out split and the training order
    // are the same for every run. MICROGPT_SEED reseeds the generator after that, i.e. it changes
    // the parameter initialisation and the sampling, which keeps held-out losses comparable
    // between seeds. Unset, the single seed-42 stream continues as in the gist.
    let data = Dataset::load("input.txt", &mut rng)?;
    let seed: Option<u64> = std::env::var("MICROGPT_SEED").ok().map(|s| s.parse().expect("MICROGPT_SEED must be an integer"));
    if let Some(seed) = seed {
        rng = Rng::new(seed);
    }
    println!("num docs: {} ({} train, {} held out)", data.docs.len(), data.n_train, data.eval_docs().len());
    println!("vocab size: {}", data.vocab_size());

    let model = Model::new(data.vocab_size());
    let mut params = model.init_params(&mut rng);
    println!("model: {N_LAYER} layer(s), n_embd {N_EMBD}, {N_HEAD} heads, block size {BLOCK_SIZE} | num params: {}", model.n_params);
    println!(
        "steps: {} | batch size: {} | threads: {} | lr: {} | seed: {}",
        cfg.num_steps,
        cfg.batch_size,
        cfg.n_threads.clamp(1, cfg.batch_size.max(1)),
        cfg.lr,
        seed.map_or("42 (default)".to_string(), |s| s.to_string())
    );

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let start = Instant::now();
    let loss = train(&model, &data, &mut params, &cfg, &mut out)?;
    let elapsed = start.elapsed();
    writeln!(out)?;
    writeln!(
        out,
        "final loss {loss:.4} | trained {} steps x {} docs in {elapsed:.2?} ({:.1} us/step)",
        cfg.num_steps,
        cfg.batch_size,
        elapsed.as_secs_f64() * 1e6 / cfg.num_steps as f64
    )?;
    let v = model.vocab_size;
    let mut acts = Acts::new(v);
    let mut tokens = Vec::with_capacity(BLOCK_SIZE + 2);
    let held_out = model.eval_loss(&params, &data, data.eval_docs(), &mut acts, &mut tokens);
    writeln!(out, "held-out loss {held_out:.4} (mean over {} names never trained on)", data.eval_docs().len())?;

    // Inference: may the model babble back to us
    writeln!(out, "--- inference (new, hallucinated names) ---")?;
    let mut probs = vec![0.0 as Float; v];
    for sample_idx in 0..NUM_SAMPLES {
        tokens.clear();
        tokens.push(data.bos);
        let mut sample = String::new();
        for pos in 0..BLOCK_SIZE {
            model.forward(&params, &tokens, pos, pos + 1, &mut acts); // earlier positions = KV cache
            let logits = row(&acts.logits, pos, v);
            for (pr, &lg) in probs.iter_mut().zip(logits) {
                *pr = lg / TEMPERATURE;
            }
            softmax_inplace(&mut probs);
            let next = rng.choice_weighted(&probs);
            if next == data.bos {
                break;
            }
            sample.push(data.uchars[next]);
            tokens.push(next);
        }
        writeln!(out, "sample {:2}: {}", sample_idx + 1, sample)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Tests (run in f64 so finite differences are meaningful)
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_dataset() -> Dataset {
        let docs: Vec<String> = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "charlotte"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        Dataset::new(docs, 0)
    }

    fn full_loss(model: &Model, p: &[Float], tokens: &[usize], n: usize, acts: &mut Acts) -> Float {
        model.forward(p, tokens, 0, n, acts);
        model.loss(tokens, n, acts)
    }

    #[test]
    fn manual_backward_matches_finite_differences() {
        let data = tiny_dataset();
        let mut rng = Rng::new(7);
        let model = Model::new(data.vocab_size());
        let mut params = model.init_params(&mut rng);
        // larger weights so that no gradient is vanishingly small
        for x in params.iter_mut() {
            *x *= 4.0;
        }
        let mut grads = vec![0.0; model.n_params];
        let mut acts = Acts::new(model.vocab_size);
        let mut scratch = Grads::new(model.vocab_size);

        let mut tokens = Vec::new();
        data.tokenize("charlotte", &mut tokens);
        let n = (tokens.len() - 1).min(BLOCK_SIZE);

        full_loss(&model, &params, &tokens, n, &mut acts);
        model.backward(&params, &mut grads, &tokens, n, &acts, &mut scratch);

        let h = 1e-5;
        let mut checked = 0;
        let mut max_rel = 0.0f64;
        for idx in (0..model.n_params).step_by(7) {
            let orig = params[idx];
            params[idx] = orig + h;
            let lp = full_loss(&model, &params, &tokens, n, &mut acts);
            params[idx] = orig - h;
            let lm = full_loss(&model, &params, &tokens, n, &mut acts);
            params[idx] = orig;
            let numeric = (lp - lm) / (2.0 * h);
            let analytic = grads[idx];
            let denom = numeric.abs().max(analytic.abs()).max(1e-6);
            let rel = (numeric - analytic).abs() / denom;
            if numeric.abs() > 1e-7 || analytic.abs() > 1e-7 {
                max_rel = max_rel.max(rel);
                checked += 1;
                // central differences with h = 1e-5 resolve gradients to roughly 1e-10 absolute;
                // for the wider/deeper models some gradients are ~1e-6, so allow either bound
                let abs = (numeric - analytic).abs();
                assert!(rel < 1e-4 || abs < 1e-8, "param {idx}: analytic {analytic} vs numeric {numeric} (rel {rel}, abs {abs})");
            }
        }
        assert!(checked > 300, "too few non-zero gradients checked: {checked}");
        eprintln!("checked {checked} params, max relative error {max_rel:.2e}");
    }

    #[test]
    fn incremental_forward_matches_full_forward() {
        let data = tiny_dataset();
        let mut rng = Rng::new(3);
        let model = Model::new(data.vocab_size());
        let params = model.init_params(&mut rng);
        let mut tokens = Vec::new();
        data.tokenize("isabella", &mut tokens);
        let n = tokens.len() - 1;

        let mut full = Acts::new(model.vocab_size);
        model.forward(&params, &tokens, 0, n, &mut full);
        let mut inc = Acts::new(model.vocab_size);
        for pos in 0..n {
            model.forward(&params, &tokens, pos, pos + 1, &mut inc);
        }
        for (a, b) in full.logits[..n * model.vocab_size].iter().zip(&inc.logits) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn loss_decreases_with_training() {
        let data = tiny_dataset();
        let mut rng = Rng::new(42);
        let model = Model::new(data.vocab_size());
        let mut params = model.init_params(&mut rng);
        let mut grads = vec![0.0; model.n_params];
        let mut adam = Adam::new(model.n_params);
        let mut acts = Acts::new(model.vocab_size);
        let mut scratch = Grads::new(model.vocab_size);
        let mut tokens = Vec::new();

        let mut first = 0.0;
        let mut last = 0.0;
        let steps = 200;
        for step in 0..steps {
            data.tokenize(&data.docs[step % data.n_train], &mut tokens);
            let n = (tokens.len() - 1).min(BLOCK_SIZE);
            let loss = full_loss(&model, &params, &tokens, n, &mut acts);
            model.backward(&params, &mut grads, &tokens, n, &acts, &mut scratch);
            adam.step(&mut params, &mut grads, LEARNING_RATE, step + 1);
            if step == 0 {
                first = loss;
            }
            last = loss;
        }
        assert!(last < first * 0.5, "loss did not drop: {first} -> {last}");
    }

    #[test]
    fn parallel_training_matches_serial() {
        let data = tiny_dataset();
        let mut rng = Rng::new(42);
        let model = Model::new(data.vocab_size());
        let init = model.init_params(&mut rng);
        let run = |n_threads: usize| {
            let mut params = init.clone();
            let cfg = TrainConfig { num_steps: 30, batch_size: 5, n_threads, log_path: None, eval_every: 100, lr: LEARNING_RATE };
            let loss = train(&model, &data, &mut params, &cfg, &mut io::sink()).unwrap();
            (loss, params)
        };
        let (l1, p1) = run(1);
        let (l3, p3) = run(3);
        assert!((l1 - l3).abs() < 1e-9, "{l1} vs {l3}");
        for (a, b) in p1.iter().zip(&p3) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }
}
