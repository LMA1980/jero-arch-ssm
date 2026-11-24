---
name: jero-arch-ssm
description: State-space and hybrid recurrent-transformer architecture recipes (`Mamba`, `Mamba2`, `RWKV6/7`, `Jamba`, `Zamba2`, `Granite`, `LFM2`) for the SpacemiT K3 platform.
---

# `jero-arch-ssm` Architecture Crate

Provides State-space and hybrid recurrent-transformer recipes (`Mamba2Recipe`, `Rwkv7Recipe`, `HybridSsmTransformerRecipe`, `GraniteRecipe`, `Lfm2MoeRecipe`) for K3 inference.

## Crate Contract & Dependencies

- **License**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`
- **Upstream Dependencies**: `jero-core` (zero-copy `ModelWeights`, `ExecutionContext`, `ArchRecipe`, `K3Error`) and optional `jero-kernels-rvv` (enabled by default via `with-kernels`).
- **Safety & Lints**: `#![deny(unsafe_op_in_unsafe_fn)]`, `clippy::pedantic`, and `clippy::undocumented_unsafe_blocks`.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
```
