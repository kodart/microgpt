//! Model dimensions are compile-time constants (the activation buffers are fixed-size arrays),
//! so they are chosen at build time through environment variables:
//!
//!     MICROGPT_N_EMBD=32 MICROGPT_N_LAYER=2 cargo build --release
//!
//! Unset variables keep the gist's values.
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
    assert!(n_layer >= 1 && n_embd >= 1 && n_head >= 1 && block_size >= 1, "model dimensions must be >= 1");
    assert!(n_embd % n_head == 0, "MICROGPT_N_EMBD ({n_embd}) must be a multiple of MICROGPT_N_HEAD ({n_head})");
    let out = Path::new(&env::var("OUT_DIR").unwrap()).join("model_config.rs");
    fs::write(
        out,
        format!(
            "const N_LAYER: usize = {n_layer}; // depth of the transformer (number of layers)\n\
             const N_EMBD: usize = {n_embd}; // width of the network (embedding dimension)\n\
             const BLOCK_SIZE: usize = {block_size}; // maximum context length\n\
             const N_HEAD: usize = {n_head}; // number of attention heads\n"
        ),
    )
    .unwrap();
}
