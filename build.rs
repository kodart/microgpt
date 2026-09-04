//! Model dimensions are compile-time constants (the activation buffers are fixed-size arrays),
//! so they are chosen at build time through environment variables:
//!
//!     MICROGPT_N_EMBD=32 MICROGPT_N_LAYER=2 cargo build --release
//!     MICROGPT_N_EXPERTS=4 MICROGPT_TOP_K=2 MICROGPT_HIDDEN=32 cargo build --release
//!
//! Unset variables keep the gist's values: a dense MLP is the one-expert, top-1 case.
use std::{env, fs, path::Path};

fn knob(name: &str, default: usize) -> usize {
    println!("cargo:rerun-if-env-changed={name}");
    match env::var(name) {
        Ok(v) => v.parse().unwrap_or_else(|_| panic!("{name} must be a positive integer, got {v:?}")),
        Err(_) => default,
    }
}

fn main() {
    let n_layer = knob("MICROGPT_N_LAYER", 1);
    let n_embd = knob("MICROGPT_N_EMBD", 16);
    let n_head = knob("MICROGPT_N_HEAD", 4);
    let block_size = knob("MICROGPT_BLOCK_SIZE", 16);
    let n_experts = knob("MICROGPT_N_EXPERTS", 1);
    let top_k = knob("MICROGPT_TOP_K", 1);
    let hidden = knob("MICROGPT_HIDDEN", 4 * n_embd);
    assert!(n_layer >= 1 && n_embd >= 1 && n_head >= 1 && block_size >= 1, "model dimensions must be >= 1");
    assert!(n_embd.is_multiple_of(n_head), "MICROGPT_N_EMBD ({n_embd}) must be a multiple of MICROGPT_N_HEAD ({n_head})");
    assert!(n_experts >= 1 && top_k >= 1 && top_k <= n_experts, "need 1 <= MICROGPT_TOP_K <= MICROGPT_N_EXPERTS");
    assert!(hidden >= 1, "MICROGPT_HIDDEN must be >= 1");
    let out = Path::new(&env::var("OUT_DIR").unwrap()).join("model_config.rs");
    fs::write(
        out,
        format!(
            "const N_LAYER: usize = {n_layer}; // depth of the transformer (number of layers)\n\
             const N_EMBD: usize = {n_embd}; // width of the network (embedding dimension)\n\
             const BLOCK_SIZE: usize = {block_size}; // maximum context length\n\
             const N_HEAD: usize = {n_head}; // number of attention heads\n\
             const N_EXPERTS: usize = {n_experts}; // MLP experts per layer (1 = dense MLP)\n\
             const TOP_K: usize = {top_k}; // experts used per token\n\
             const N_HIDDEN: usize = {hidden}; // hidden width of each expert\n"
        ),
    )
    .unwrap();
}
