//! Dense IBM `Granite` (`granite`) execution recipe with calibrated scaling multipliers.

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};

use crate::dense::{
    DenseAttnLayerSpec, even_head_dim, finalize_scaled_logits, rmsnorm_standard, run_decode_with,
    run_dense_attention_sublayer_scaled, run_dense_swiglu_ffn, run_prefill_with,
};
use crate::weights::{
    add_bias_if_present_strict, embed_token_strict, has_any_tensor, is_supported_quant,
    read_required_f32,
};

/// Execution recipe for dense IBM `Granite` (`granite` — Granite 3 / `Granite-4.1-Guardian-8B`)
/// transformers supporting GGUF `embedding_scale`, `residual_scale`, `attention.scale`, and
/// `logit_scale` multipliers.
#[derive(Debug, Clone)]
pub struct GraniteRecipe {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Dense `SwiGLU` intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value attention head count (`GQA`).
    pub n_heads_kv: usize,
    /// Scalar dimension per attention head.
    pub head_dim: usize,
    /// Output vocabulary size.
    pub vocab_size: usize,
    /// `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency.
    pub rope_theta: f32,
    /// Multiplier applied to input token embeddings (`granite.embedding_scale`).
    pub embedding_scale: f32,
    /// Multiplier applied to residual branch deltas (`granite.residual_scale`).
    pub residual_scale: f32,
    /// Explicit QK score scale overriding `1 / sqrt(head_dim)` (`granite.attention.scale`).
    pub attention_scale: Option<f32>,
    /// Final logit divisor (`granite.logit_scale`, scaling output logits by `1.0 / logit_scale`).
    pub logit_scale: f32,
}

impl Default for GraniteRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl GraniteRecipe {
    /// Creates a `Granite` dense recipe with 8B-class defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 40,
            hidden_dim: 4096,
            ffn_dim: 12_800,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            vocab_size: 49_155,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            embedding_scale: 1.0,
            residual_scale: 1.0,
            attention_scale: None,
            logit_scale: 1.0,
        }
    }

    fn execute_single_token(
        &self,
        tok: u32,
        pos: usize,
        seq_len: usize,
        logits: &mut [f32],
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        if !self.embedding_scale.is_finite()
            || !self.residual_scale.is_finite()
            || !self.logit_scale.is_finite()
            || self.attention_scale.is_some_and(|s| !s.is_finite())
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "granite".to_owned(),
                reason: "Granite scaling multipliers must be finite".to_owned(),
            });
        }
        let hidden_dim = self.hidden_dim.max(2);
        let head_d_even = even_head_dim(self.head_dim);
        let n_kv = self.n_heads_kv.max(1);
        let num_layers = self.num_layers.max(1);

        let attn_spec = DenseAttnLayerSpec {
            ffn_dim: self.ffn_dim.max(4),
            n_heads_q: self.n_heads_q.max(1),
            n_heads_kv: n_kv,
            head_dim: head_d_even,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            use_qkv_bias: false,
            use_qk_norm: false,
        };

        let mut hidden = embed_token_strict(tok, hidden_dim, weights)?;
        if (self.embedding_scale - 1.0).abs() > f32::EPSILON {
            for h in &mut hidden {
                *h *= self.embedding_scale;
            }
        }

        let layer_stride = kv_cache.len() / num_layers;
        for layer in 0..num_layers {
            let mut layer_spec = attn_spec;
            let qb_g = format!("blk.{layer}.attn_q.bias");
            let kb_g = format!("blk.{layer}.attn_k.bias");
            let vb_g = format!("blk.{layer}.attn_v.bias");
            let qb_h = format!("model.layers.{layer}.self_attn.q_proj.bias");
            let kb_h = format!("model.layers.{layer}.self_attn.k_proj.bias");
            let vb_h = format!("model.layers.{layer}.self_attn.v_proj.bias");
            layer_spec.use_qkv_bias =
                has_any_tensor(weights, &[&qb_g, &kb_g, &vb_g, &qb_h, &kb_h, &vb_h]);
            let layer_kv = if layer_stride > 0 && (layer + 1) * layer_stride <= kv_cache.len() {
                &mut kv_cache[layer * layer_stride..(layer + 1) * layer_stride]
            } else {
                &mut kv_cache[..0]
            };
            run_dense_attention_sublayer_scaled(
                layer_spec,
                layer,
                pos,
                seq_len,
                &mut hidden,
                layer_kv,
                weights,
                self.attention_scale,
                self.residual_scale,
            )?;

            let ffn_norm_g = format!("blk.{layer}.ffn_norm.weight");
            let ffn_norm_hf = format!("model.layers.{layer}.post_attention_layernorm.weight");
            let ffn_norm_w = read_required_f32(weights, &[&ffn_norm_g, &ffn_norm_hf], hidden_dim)?;
            let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
            rmsnorm_standard(&mut norm_ffn_in, &hidden, &ffn_norm_w, self.rms_eps)?;
            let mut ffn_out =
                run_dense_swiglu_ffn(layer, hidden_dim, attn_spec.ffn_dim, &norm_ffn_in, weights)?;
            let down_b_g = format!("blk.{layer}.ffn_down.bias");
            let down_b_hf = format!("model.layers.{layer}.mlp.down_proj.bias");
            add_bias_if_present_strict(&mut ffn_out, weights, &[&down_b_g, &down_b_hf])?;
            for (h, &f) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h += f * self.residual_scale;
            }
        }

        if !logits.is_empty() {
            finalize_scaled_logits(&hidden, self.rms_eps, self.logit_scale, logits, weights)?;
        }
        Ok(())
    }
}

impl ArchRecipe for GraniteRecipe {
    fn architecture_id(&self) -> &'static str {
        "granite"
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_prefill_with(ctx, "granite", |tok, pos, seq_len, kv, logits| {
            self.execute_single_token(tok, pos, seq_len, logits, kv, weights)
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_decode_with(ctx, "granite", |tok, pos, kv, logits| {
            self.execute_single_token(tok, pos, pos + 1, logits, kv, weights)
        })
    }
}
