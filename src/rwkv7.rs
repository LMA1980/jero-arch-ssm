//! `RWKV-6` (`Finch`) and `RWKV-7` (`Goose`) linear-RNN execution recipe.

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};
use jero_kernels_rvv::{rwkv_token_shift_lerp_f32, rwkv7_wkv_delta_rule_step_f32, softplus_f32};

use crate::arena::PingPongRecurrentArena;
use crate::dense::{finalize_scaled_logits, rmsnorm_standard, run_decode_with, run_prefill_with};
use crate::weights::{
    embed_token_strict, is_supported_quant, project_linear_strict, read_required_f32,
};

/// Execution recipe for `RWKV-6` (`Finch`) and `RWKV-7` (`Goose`) linear-RNN architectures.
///
/// Combines causal token-shift interpolation (`rwkv_token_shift_lerp_f32`), matrix-valued delta-rule
/// WKV state updates (`rwkv7_wkv_delta_rule_step_f32`), channel-mix feed-forward blocks, and
/// double-buffered ping-pong `wkv_state` slots (`O(1)` in `n_ctx`).
#[derive(Debug, Clone)]
pub struct Rwkv7Recipe {
    /// Architecture identifier (`"rwkv7"` or `"rwkv6"`).
    pub architecture: &'static str,
    /// Number of recurrent blocks.
    pub num_layers: usize,
    /// Model embedding dimension.
    pub hidden_dim: usize,
    /// Channel-mixing feed-forward dimension.
    pub ffn_dim: usize,
    /// Per-head WKV state matrix dimension (`head_size`).
    pub head_size: usize,
    /// Number of WKV heads (`hidden_dim / head_size`).
    pub n_heads: usize,
    /// Vocabulary size for final LM-head projection.
    pub vocab_size: usize,
    /// Layer/RMSNorm epsilon.
    pub rms_eps: f32,
}

impl Default for Rwkv7Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Rwkv7Recipe {
    /// Creates an `RWKV-7` (`Goose`) recipe matching 1.5B-scale geometry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "rwkv7",
            num_layers: 24,
            hidden_dim: 2048,
            ffn_dim: 7168,
            head_size: 64,
            n_heads: 32,
            vocab_size: 65_536,
            rms_eps: 1e-5,
        }
    }

    /// Creates an `RWKV-6` (`Finch`) recipe targeting `"rwkv6"`.
    #[must_use]
    pub const fn new_rwkv6() -> Self {
        Self {
            architecture: "rwkv6",
            num_layers: 24,
            hidden_dim: 2048,
            ffn_dim: 7168,
            head_size: 64,
            n_heads: 32,
            vocab_size: 65_536,
            rms_eps: 1e-5,
        }
    }

    fn effective_dims(&self) -> (usize, usize, usize, usize) {
        let hidden_dim = self.hidden_dim.max(2);
        let ffn_dim = self.ffn_dim.max(hidden_dim);
        let head_size = self.head_size.clamp(1, hidden_dim);
        let mut h = self.n_heads.clamp(1, hidden_dim);
        while h * head_size != hidden_dim && h > 1 {
            h -= 1;
        }
        let eff_head_size = if h * head_size == hidden_dim {
            head_size
        } else {
            hidden_dim / h.max(1)
        }
        .max(1);
        (hidden_dim, ffn_dim, h, eff_head_size)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "RWKV time-mix step projects r/w/k/v/a/g and updates dual-bank WKV state"
    )]
    fn execute_time_mix(
        &self,
        layer: usize,
        hidden: &mut [f32],
        prev_time_shift: &mut [f32],
        wkv_state: &mut [f32],
        n_heads: usize,
        head_size: usize,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = hidden.len();
        let vec_dim = n_heads * head_size;
        let ln_g = format!("blk.{layer}.attn_norm.weight");
        let ln_g2 = format!("blk.{layer}.ln1.weight");
        let ln_hf = format!("rwkv.blocks.{layer}.ln1.weight");
        let attn_norm_w = read_required_f32(weights, &[&ln_g, &ln_g2, &ln_hf], hidden_dim)?;
        let mut x_norm = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut x_norm, hidden, &attn_norm_w, self.rms_eps)?;

        let lerp_g = format!("blk.{layer}.time_mix_x_r");
        let lerp_hf = format!("rwkv.blocks.{layer}.att.x_r");
        let lerp_time = read_required_f32(weights, &[&lerp_g, &lerp_hf], hidden_dim)?;
        let mut shifted_time = vec![0.0_f32; hidden_dim];
        rwkv_token_shift_lerp_f32(prev_time_shift, &x_norm, &lerp_time, &mut shifted_time)?;
        prev_time_shift.copy_from_slice(&x_norm);

        let r_g = format!("blk.{layer}.time_mix_r.weight");
        let r_hf = format!("rwkv.blocks.{layer}.att.receptance.weight");
        let r_vec = project_linear_strict(weights, &[&r_g, &r_hf], &shifted_time, vec_dim)?;

        let w_g = format!("blk.{layer}.time_mix_w.weight");
        let w_hf = format!("rwkv.blocks.{layer}.att.w.weight");
        let mut w_decay = project_linear_strict(weights, &[&w_g, &w_hf], &shifted_time, vec_dim)?;
        for w in &mut w_decay {
            *w = (-softplus_f32(w.clamp(-8.0, 8.0))).exp().clamp(0.0, 0.999);
        }

        let k_g = format!("blk.{layer}.time_mix_k.weight");
        let k_hf = format!("rwkv.blocks.{layer}.att.key.weight");
        let k_vec = project_linear_strict(weights, &[&k_g, &k_hf], &shifted_time, vec_dim)?;

        let v_g = format!("blk.{layer}.time_mix_v.weight");
        let v_hf = format!("rwkv.blocks.{layer}.att.value.weight");
        let v_vec = project_linear_strict(weights, &[&v_g, &v_hf], &shifted_time, vec_dim)?;

        let a_g = format!("blk.{layer}.time_mix_a.weight");
        let a_hf = format!("rwkv.blocks.{layer}.att.a.weight");
        let a_vec = project_linear_strict(weights, &[&a_g, &a_hf], &shifted_time, vec_dim)?;

        let g_g = format!("blk.{layer}.time_mix_g.weight");
        let g_hf = format!("rwkv.blocks.{layer}.att.gate.weight");
        let mut b_vec = project_linear_strict(weights, &[&g_g, &g_hf], &shifted_time, vec_dim)?;
        for b in &mut b_vec {
            *b = -b.tanh() * 0.25;
        }

        let mut wkv_out = vec![0.0_f32; vec_dim];
        rwkv7_wkv_delta_rule_step_f32(
            wkv_state,
            &mut wkv_out,
            &r_vec,
            &w_decay,
            &k_vec,
            &v_vec,
            &a_vec,
            &b_vec,
            n_heads,
            head_size,
        )?;

        let o_g = format!("blk.{layer}.time_mix_output.weight");
        let o_hf = format!("rwkv.blocks.{layer}.att.output.weight");
        let tm_out = project_linear_strict(weights, &[&o_g, &o_hf], &wkv_out, hidden_dim)?;
        for (h, &o) in hidden.iter_mut().zip(tm_out.iter()) {
            *h += o;
        }
        Ok(())
    }

    fn execute_channel_mix(
        &self,
        layer: usize,
        hidden: &mut [f32],
        prev_chan_shift: &mut [f32],
        ffn_dim: usize,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = hidden.len();
        let ln_g = format!("blk.{layer}.ffn_norm.weight");
        let ln_g2 = format!("blk.{layer}.ln2.weight");
        let ln_hf = format!("rwkv.blocks.{layer}.ln2.weight");
        let ffn_norm_w = read_required_f32(weights, &[&ln_g, &ln_g2, &ln_hf], hidden_dim)?;
        let mut x_ffn_in = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut x_ffn_in, hidden, &ffn_norm_w, self.rms_eps)?;

        let lerp_g = format!("blk.{layer}.channel_mix_lerp_k");
        let lerp_hf = format!("rwkv.blocks.{layer}.ffn.x_k");
        let lerp_chan = read_required_f32(weights, &[&lerp_g, &lerp_hf], hidden_dim)?;
        let mut shifted_chan = vec![0.0_f32; hidden_dim];
        rwkv_token_shift_lerp_f32(prev_chan_shift, &x_ffn_in, &lerp_chan, &mut shifted_chan)?;
        prev_chan_shift.copy_from_slice(&x_ffn_in);

        let key_g1 = format!("blk.{layer}.channel_mix_key.weight");
        let key_g2 = format!("blk.{layer}.ffn_key.weight");
        let key_hf = format!("rwkv.blocks.{layer}.ffn.key.weight");
        let val_g1 = format!("blk.{layer}.channel_mix_value.weight");
        let val_g2 = format!("blk.{layer}.ffn_value.weight");
        let val_hf = format!("rwkv.blocks.{layer}.ffn.value.weight");

        let mut cm_k = project_linear_strict(
            weights,
            &[&key_g1, &key_g2, &key_hf],
            &shifted_chan,
            ffn_dim,
        )?;
        for k in &mut cm_k {
            let r = k.max(0.0);
            *k = r * r;
        }
        let cm_v = project_linear_strict(weights, &[&val_g1, &val_g2, &val_hf], &cm_k, hidden_dim)?;
        for (h, &v) in hidden.iter_mut().zip(cm_v.iter()) {
            *h += v;
        }
        Ok(())
    }

    fn execute_single_token(
        &self,
        tok: u32,
        pos: usize,
        logits: &mut [f32],
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let (hidden_dim, ffn_dim, n_heads, head_size) = self.effective_dims();
        let wkv_elems = n_heads * head_size * head_size;
        let slot_floats = 2 * hidden_dim + wkv_elems;
        let num_layers = self.num_layers.max(1);

        let mut hidden = embed_token_strict(tok, hidden_dim, weights)?;
        let mut slot_buf = vec![0.0_f32; slot_floats];
        let mut arena = PingPongRecurrentArena::open(kv_cache, num_layers, slot_floats);

        for layer in 0..num_layers {
            arena.read_slot(layer, &mut slot_buf);
            let (prev_both, wkv_state) = slot_buf.split_at_mut(2 * hidden_dim);
            let (prev_time_shift, prev_chan_shift) = prev_both.split_at_mut(hidden_dim);

            self.execute_time_mix(
                layer,
                &mut hidden,
                prev_time_shift,
                wkv_state,
                n_heads,
                head_size,
                weights,
            )?;
            self.execute_channel_mix(layer, &mut hidden, prev_chan_shift, ffn_dim, weights)?;
            arena.write_staging_slot(layer, &slot_buf);
        }

        if !logits.is_empty() {
            finalize_scaled_logits(&hidden, self.rms_eps, 1.0, logits, weights)?;
        }

        let step_num = u32::try_from(pos + 1).unwrap_or(u32::MAX);
        arena.commit(step_num);
        Ok(())
    }
}

impl ArchRecipe for Rwkv7Recipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_prefill_with(ctx, self.architecture, |tok, pos, _seq_len, kv, logits| {
            self.execute_single_token(tok, pos, logits, kv, weights)
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_decode_with(ctx, self.architecture, |tok, pos, kv, logits| {
            self.execute_single_token(tok, pos, logits, kv, weights)
        })
    }
}
