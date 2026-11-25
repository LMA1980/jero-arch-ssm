---
name: jero-arch-ssm
description: State-space and hybrid recurrent-transformer architecture recipes (`Mamba`, `Mamba2`, `RWKV6/7`, `Jamba`, `Zamba2`, `Granite`, `LFM2`, `LFM2.5-MoE`) for the SpacemiT K3 platform.
---

# `jero-arch-ssm` Architecture Crate

Provides state-space and hybrid recurrent-transformer inference recipes (`Mamba2Recipe`, `Rwkv7Recipe`, `HybridSsmTransformerRecipe`, `GraniteRecipe`, `Lfm2MoeRecipe`) and O(1) ping-pong recurrent arena scheduling for `ModelWeights` checkpoints.

## Exported Surface

- **Recipes (`ArchRecipe`)**:
  - `Mamba2Recipe` (`Mamba2Recipe::new`, `Mamba2Recipe::new_mamba`)
  - `Rwkv7Recipe` (`Rwkv7Recipe::new`, `Rwkv7Recipe::new_rwkv6`)
  - `HybridSsmTransformerRecipe` (`Jamba`, `Zamba2`, `FalconH1`, `Hymba`, `Qwen3Next`, `NemotronH`, `LFM2` hybrid)
  - `GraniteRecipe` (`Granite` dense and `Granite-4` hybrid Mamba2+GQA)
  - `Lfm2MoeRecipe` (`LFM2.5-MoE` with `lfm2_shortconv_step`, `KV_BLOCK_SIZE = 16`, and top-k routed experts)
- **Hybrid Schedule & Arena Helpers**:
  - `HybridLayerKind` (`Attention`, `Mamba2`, `Rwkv`, `Lfm2ShortConv`)
  - `resolve_hybrid_layer_schedule`, `resolve_hybrid_layer_schedule_with_metadata`, `resolve_hybrid_layer_schedule_for_metadata`
  - `hybrid_recurrent_slot_floats`, `count_hybrid_schedule_layers`, `estimate_hybrid_recurrent_bytes_per_layer`, `estimate_hybrid_ssm_arena_bytes`
  - `RECURRENT_HEADER_FLOATS`, `inspect_ping_pong_header`
- **Tensor & Routing Utilities**:
  - `matvec_any`, `select_top_k_experts`, `lfm2_shortconv_step`

## ADR-045 & Storage Contract

- **Flat Crate Boundary**: Depends only on `jero-core`, `jero-kernels-rvv`, and `rayon` (`with-kernels` feature enabled by default). Never depend on `jero-engine`.
- **Fail-Closed Weight Hydration**: Mandatory projection, norm, recurrence (`ssm_a`, `ssm_d`), router (`ffn_gate_inp`), and expert matrices resolve through `ModelWeights::tensor(...)` and return `K3Error::TensorBoundsError` (or `K3Error::ShapeMismatch` / `K3Error::InvalidWeightData` on corrupt/non-finite data) rather than falling back to zero or deterministic synthetic slices.
- **Dual Checkpoint Support**: Resolves GGUF (`blk.<L>.*`, `[in, out]`) and Hugging Face SafeTensors (`model.layers.<L>.*`, `[out, in]`) tensor layouts uniformly across `GgufFile` and `SafeTensorsFile`.
- **Atomic Ping-Pong Commit**: `RECURRENT_HEADER_FLOATS` metadata is only initialized/flipped on successful step completion so failed steps leave prior recurrent state untouched.
- **License & Lints**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`, `#![deny(unsafe_op_in_unsafe_fn)]`, `clippy::pedantic`, and `clippy::undocumented_unsafe_blocks`.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo check --target riscv64gc-unknown-linux-gnu --all-targets
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
cargo agents plugin validate Symposium.toml --no-check-crates
```
