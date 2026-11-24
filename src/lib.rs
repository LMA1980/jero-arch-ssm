//! `jero-arch-ssm`: State-space and hybrid recurrent-transformer recipes (`Mamba`, `Mamba2`,
//! `RWKV6/7`, `Jamba`, `Zamba2`, `Granite`, `LFM2`, `LFM2.5-MoE`) for the `SpacemiT` K3 platform.
//!
//! Enforces strict ADR-045 fail-closed weight hydration across both `GgufFile` and `SafeTensorsFile`
//! checkpoints with O(1) double-buffered ping-pong recurrent state arenas.

#![deny(unsafe_op_in_unsafe_fn)]

mod arena;
mod dense;
pub mod granite;
pub mod hybrid;
pub mod mamba2;
pub mod rwkv7;
pub mod schedule;
mod weights;

#[cfg(test)]
mod tests;

pub use arena::{RECURRENT_HEADER_FLOATS, inspect_ping_pong_header};
pub use dense::select_top_k_experts;
pub use granite::GraniteRecipe;
pub use hybrid::{HybridSsmTransformerRecipe, Lfm2MoeRecipe, lfm2_shortconv_step};
pub use mamba2::Mamba2Recipe;
pub use rwkv7::Rwkv7Recipe;
pub use schedule::{
    HybridLayerKind, KV_BLOCK_SIZE, count_hybrid_schedule_layers,
    estimate_hybrid_recurrent_bytes_per_layer, estimate_hybrid_ssm_arena_bytes,
    hybrid_recurrent_slot_floats, resolve_hybrid_layer_schedule,
    resolve_hybrid_layer_schedule_for_metadata, resolve_hybrid_layer_schedule_with_metadata,
};
pub use weights::matvec_any;

/// Canonical crate identifier for this architecture recipe package.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Canonical SPDX license identifier for this architecture recipe package.
pub const CRATE_LICENSE: &str = "LGPL-3.0-only WITH LGPL-3.0-linking-exception";

/// Returns the crate description configured in `Cargo.toml`.
#[must_use]
pub const fn crate_description() -> &'static str {
    env!("CARGO_PKG_DESCRIPTION")
}
