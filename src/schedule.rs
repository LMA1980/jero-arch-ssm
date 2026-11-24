//! Hybrid layer schedule classification, recurrent slot sizing, and arena budgeting.

use core::mem::size_of;

use jero_core::{K3Error, ModelMetadata, ModelWeights, RecurrentStateSlotHeader};

/// Number of token positions per paged KV cache block.
pub const KV_BLOCK_SIZE: usize = 16;

/// Disambiguates per-layer state backing across hybrid SSM, short-conv, and attention architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLayerKind {
    /// Standard transformer GQA attention layer backed by paged KV cache blocks.
    GqaAttention {
        /// Compact attention-only KV layer slot index (`0..num_attention_layers`).
        kv_layer_slot: usize,
    },
    /// Mamba-2 Selective State-Space Duality (SSD) layer backed by O(1) ping-pong recurrent slots.
    Mamba2Ssd {
        /// Compact recurrent layer slot index (`0..num_recurrent_layers`).
        recurrent_layer_slot: usize,
    },
    /// `LiquidAI` LFM2 gated causal short-convolution layer backed by O(1) ping-pong recurrent slots.
    Lfm2ShortConv {
        /// Compact recurrent layer slot index (`0..num_recurrent_layers`).
        recurrent_layer_slot: usize,
    },
}

/// Probes layers `0..num_layers` using a tensor-presence predicate supporting both GGUF (`blk.{i}.*`)
/// and Hugging Face `SafeTensors` (`model.layers.{i}.*`, `backbone.layers.{i}.*`) naming.
pub(crate) fn resolve_hybrid_schedule_by_probe(
    num_layers: usize,
    mut contains: impl FnMut(&str) -> bool,
) -> (Vec<HybridLayerKind>, bool) {
    let mut schedule = Vec::with_capacity(num_layers);
    let mut next_kv_slot = 0_usize;
    let mut next_recurrent_slot = 0_usize;
    let mut saw_known_tensor = false;

    for layer in 0..num_layers {
        let is_attn = contains(&format!("blk.{layer}.attn_q.weight"))
            || contains(&format!("blk.{layer}.attn_k.weight"))
            || contains(&format!("blk.{layer}.attn_qkv.weight"))
            || contains(&format!("model.layers.{layer}.self_attn.q_proj.weight"))
            || contains(&format!("model.layers.{layer}.self_attn.k_proj.weight"))
            || contains(&format!("model.layers.{layer}.self_attn.qkv_proj.weight"));
        if is_attn {
            saw_known_tensor = true;
            schedule.push(HybridLayerKind::GqaAttention {
                kv_layer_slot: next_kv_slot,
            });
            next_kv_slot += 1;
            continue;
        }

        let is_shortconv = contains(&format!("blk.{layer}.shortconv.conv.weight"))
            || contains(&format!("blk.{layer}.shortconv.in_proj.weight"))
            || contains(&format!("blk.{layer}.shortconv.out_proj.weight"))
            || contains(&format!("blk.{layer}.shortconv_conv.weight"))
            || contains(&format!("model.layers.{layer}.conv.conv.weight"))
            || contains(&format!("model.layers.{layer}.conv.in_proj.weight"))
            || contains(&format!("model.layers.{layer}.conv.out_proj.weight"));
        if is_shortconv {
            saw_known_tensor = true;
            schedule.push(HybridLayerKind::Lfm2ShortConv {
                recurrent_layer_slot: next_recurrent_slot,
            });
            next_recurrent_slot += 1;
            continue;
        }

        let is_ssm = contains(&format!("blk.{layer}.ssm_in.weight"))
            || contains(&format!("blk.{layer}.ssm_out.weight"))
            || contains(&format!("blk.{layer}.ssm_conv1d.weight"))
            || contains(&format!("backbone.layers.{layer}.mixer.in_proj.weight"))
            || contains(&format!("backbone.layers.{layer}.mixer.out_proj.weight"))
            || contains(&format!("backbone.layers.{layer}.mixer.conv1d.weight"))
            || contains(&format!("model.layers.{layer}.mamba.in_proj.weight"))
            || contains(&format!("model.layers.{layer}.mamba.out_proj.weight"))
            || contains(&format!("model.layers.{layer}.mamba.conv1d.weight"));
        if is_ssm {
            saw_known_tensor = true;
            schedule.push(HybridLayerKind::Mamba2Ssd {
                recurrent_layer_slot: next_recurrent_slot,
            });
            next_recurrent_slot += 1;
            continue;
        }

        schedule.push(HybridLayerKind::GqaAttention {
            kv_layer_slot: next_kv_slot,
        });
        next_kv_slot += 1;
    }

    (schedule, saw_known_tensor)
}

#[must_use]
pub(crate) fn default_hybrid_layer_schedule(arch: &str, num_layers: usize) -> Vec<HybridLayerKind> {
    let mut schedule = Vec::with_capacity(num_layers);
    let mut kv_slot = 0_usize;
    let mut rec_slot = 0_usize;
    for idx in 0..num_layers {
        let is_attn = match arch {
            "jamba" | "zamba2" => idx % 8 == 4,
            "granitehybrid" => idx % 8 == 7,
            "lfm2" | "lfm2moe" => matches!(idx % 16, 2 | 5 | 8 | 10 | 12 | 14),
            _ => true,
        };
        if is_attn {
            schedule.push(HybridLayerKind::GqaAttention {
                kv_layer_slot: kv_slot,
            });
            kv_slot += 1;
        } else {
            let kind = if matches!(arch, "lfm2" | "lfm2moe") {
                HybridLayerKind::Lfm2ShortConv {
                    recurrent_layer_slot: rec_slot,
                }
            } else {
                HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: rec_slot,
                }
            };
            schedule.push(kind);
            rec_slot += 1;
        }
    }
    schedule
}

/// Classifies layers `0..num_layers` into compact [`HybridLayerKind`] execution slots by inspecting
/// layer tensor presence on `weights`.
///
/// Inspects GGUF and `SafeTensors` naming for GQA attention, `LFM2` `ShortConv`, and `Mamba-2`
/// SSD layers. Layers lacking distinguishing tensors default to [`HybridLayerKind::GqaAttention`]
/// with monotonically assigned compact slot indices.
#[must_use]
pub fn resolve_hybrid_layer_schedule(
    num_layers: usize,
    weights: &dyn ModelWeights,
) -> Vec<HybridLayerKind> {
    resolve_hybrid_schedule_by_probe(num_layers, |name| weights.contains_tensor(name)).0
}

/// Resolves a hybrid layer schedule purely from [`ModelMetadata`] (inspecting tensor descriptors,
/// `head_count_kv_per_layer`, and canonical per-architecture layer templates).
#[must_use]
pub fn resolve_hybrid_layer_schedule_for_metadata(
    arch: &str,
    num_layers: usize,
    metadata: Option<&ModelMetadata>,
) -> Vec<HybridLayerKind> {
    if let Some(meta) = metadata {
        if !meta.tensors().is_empty() {
            let (by_meta_tensors, saw_known) =
                resolve_hybrid_schedule_by_probe(num_layers, |name| {
                    meta.find_tensor(name).is_some()
                });
            if saw_known
                && by_meta_tensors
                    .iter()
                    .any(|k| !matches!(k, HybridLayerKind::GqaAttention { .. }))
            {
                return by_meta_tensors;
            }
        }

        let per_layer_kv = meta.head_count_kv_per_layer();
        if !per_layer_kv.is_empty() {
            let mut schedule = Vec::with_capacity(num_layers);
            let mut next_kv_slot = 0_usize;
            let mut next_recurrent_slot = 0_usize;
            let use_shortconv = matches!(arch, "lfm2" | "lfm2moe");
            for idx in 0..num_layers {
                let kv_heads = per_layer_kv.get(idx).copied().unwrap_or(0);
                if kv_heads > 0 {
                    schedule.push(HybridLayerKind::GqaAttention {
                        kv_layer_slot: next_kv_slot,
                    });
                    next_kv_slot += 1;
                } else {
                    let kind = if use_shortconv {
                        HybridLayerKind::Lfm2ShortConv {
                            recurrent_layer_slot: next_recurrent_slot,
                        }
                    } else {
                        HybridLayerKind::Mamba2Ssd {
                            recurrent_layer_slot: next_recurrent_slot,
                        }
                    };
                    schedule.push(kind);
                    next_recurrent_slot += 1;
                }
            }
            return schedule;
        }
    }

    default_hybrid_layer_schedule(arch, num_layers)
}

/// Resolves a hybrid layer schedule combining tensor inspection with GGUF `head_count_kv_per_layer`
/// metadata hints (`kv_heads == 0` dispatches to recurrent Mamba-2 or `LFM2` `ShortConv`).
#[must_use]
pub fn resolve_hybrid_layer_schedule_with_metadata(
    arch: &str,
    num_layers: usize,
    metadata: Option<&ModelMetadata>,
    weights: &dyn ModelWeights,
) -> Vec<HybridLayerKind> {
    if weights.tensor_count() > 0 {
        let (by_weights, saw_known) =
            resolve_hybrid_schedule_by_probe(num_layers, |name| weights.contains_tensor(name));
        if saw_known
            && by_weights
                .iter()
                .any(|k| !matches!(k, HybridLayerKind::GqaAttention { .. }))
        {
            return by_weights;
        }
    }

    resolve_hybrid_layer_schedule_for_metadata(arch, num_layers, metadata)
}

/// Counts `(attention_layers, recurrent_layers)` across a hybrid checkpoint metadata record.
#[must_use]
pub fn count_hybrid_schedule_layers(metadata: &ModelMetadata) -> (usize, usize) {
    let total_layers = metadata.layer_count().max(1);
    let schedule = resolve_hybrid_layer_schedule_for_metadata(
        metadata.architecture(),
        total_layers,
        Some(metadata),
    );
    let attn = schedule
        .iter()
        .filter(|k| matches!(k, HybridLayerKind::GqaAttention { .. }))
        .count();
    let rec = schedule.len().saturating_sub(attn);
    (attn, rec)
}

/// Computes the per-layer O(1) recurrent state element count (`f32` elements) shared by
/// pre-flight arena sizing and hybrid recipe execution.
#[must_use]
pub fn hybrid_recurrent_slot_floats(
    arch: &str,
    hidden_dim: usize,
    d_inner: usize,
    d_conv: usize,
    d_state: usize,
    l_cache: usize,
) -> Option<usize> {
    if matches!(arch, "lfm2" | "lfm2moe") {
        let dim = hidden_dim.max(2);
        let conv_hist = l_cache.max(2) - 1;
        dim.checked_mul(conv_hist)
    } else {
        let inner = d_inner.max(2);
        let conv_hist = d_conv.max(2) - 1;
        let state = d_state.max(1);
        inner.checked_mul(conv_hist.checked_add(state)?)
    }
}

/// Estimates raw per-layer recurrent state bytes for a hybrid model's SSM or shortconv layers.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if slot byte sizing overflows `usize`.
pub fn estimate_hybrid_recurrent_bytes_per_layer(
    metadata: &ModelMetadata,
) -> Result<usize, K3Error> {
    let overflow = |reason: &str| K3Error::TensorBoundsError {
        tensor_name: "hybrid_recurrent_state".to_owned(),
        offset: 0,
        byte_len: 0,
        buffer_len: usize::MAX,
        reason: reason.to_owned(),
    };
    let arch = metadata.architecture();
    let default_d_state = if arch == "granitehybrid" { 128 } else { 16 };
    let hidden_dim = metadata.embedding_dim().max(2);
    let d_inner = metadata
        .ssm_inner_size()
        .unwrap_or_else(|| metadata.embedding_dim().saturating_mul(2))
        .max(2);
    let d_conv = metadata.ssm_conv_kernel().unwrap_or(4).max(2);
    let d_state = metadata.ssm_state_size().unwrap_or(default_d_state).max(1);
    let l_cache = metadata.shortconv_l_cache().unwrap_or(3).max(2);
    let err_msg = if matches!(arch, "lfm2" | "lfm2moe") {
        "lfm2 shortconv state bytes overflowed usize"
    } else {
        "hybrid ssm state bytes overflowed usize"
    };
    hybrid_recurrent_slot_floats(arch, hidden_dim, d_inner, d_conv, d_state, l_cache)
        .and_then(|floats| floats.checked_mul(size_of::<f32>()))
        .ok_or_else(|| overflow(err_msg))
}

/// Computes the combined paged-KV + O(1) double-buffered ping-pong recurrent arena byte budget
/// for a hybrid [`HybridLayerKind`] schedule.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `n_ctx == 0`, or [`K3Error::TensorBoundsError`] if
/// byte accumulation overflows `usize`.
pub fn estimate_hybrid_ssm_arena_bytes(
    schedule: &[HybridLayerKind],
    n_ctx: usize,
    kv_bytes_per_attn_layer: usize,
    state_bytes_per_ssm_layer: usize,
) -> Result<usize, K3Error> {
    if n_ctx == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "hybrid".to_owned(),
            reason: "requested context length n_ctx must be greater than zero".to_owned(),
        });
    }

    let mut attn_layers = 0_usize;
    let mut recurrent_layers = 0_usize;
    for kind in schedule {
        match kind {
            HybridLayerKind::GqaAttention { .. } => attn_layers += 1,
            HybridLayerKind::Mamba2Ssd { .. } | HybridLayerKind::Lfm2ShortConv { .. } => {
                recurrent_layers += 1;
            }
        }
    }

    let overflow = |reason: &str| K3Error::TensorBoundsError {
        tensor_name: "hybrid_ssm_arena".to_owned(),
        offset: 0,
        byte_len: 0,
        buffer_len: usize::MAX,
        reason: reason.to_owned(),
    };

    let kv_total = attn_layers
        .checked_mul(kv_bytes_per_attn_layer)
        .ok_or_else(|| overflow("hybrid KV bytes overflowed usize"))?;

    let recurrent_total = if recurrent_layers == 0 {
        0
    } else {
        let header_bytes = size_of::<RecurrentStateSlotHeader>();
        let banks_bytes = recurrent_layers
            .checked_mul(2)
            .and_then(|slots| slots.checked_mul(state_bytes_per_ssm_layer))
            .ok_or_else(|| overflow("hybrid dual-bank recurrent bytes overflowed usize"))?;
        header_bytes
            .checked_add(banks_bytes)
            .ok_or_else(|| overflow("hybrid recurrent region bytes overflowed usize"))?
    };

    kv_total
        .checked_add(recurrent_total)
        .ok_or_else(|| overflow("combined hybrid arena bytes overflowed usize"))
}
