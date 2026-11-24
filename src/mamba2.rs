//! `Mamba` and `Mamba-2` selective state-space recipes and shared single-step SSM layer logic.

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};
use jero_kernels_rvv::{
    mamba2_conv1d_silu_chunk_f32, mamba2_conv1d_silu_step, mamba2_ssd_chunk_scan_f32,
    mamba2_ssd_intra_chunk_ime_i8, mamba2_ssd_state_step_f32,
};

use crate::arena::PingPongRecurrentArena;
use crate::dense::{finalize_scaled_logits, rmsnorm_standard, run_decode_with};
use crate::weights::{
    embed_token_strict, is_supported_quant, project_linear_strict, read_optional_f32,
    read_required_f32,
};

#[inline]
pub(crate) fn resolve_divisible_ssm_geometry(
    d_inner: usize,
    requested_heads: usize,
    requested_groups: usize,
) -> (usize, usize, usize) {
    let inner = d_inner.max(1);
    let mut n_heads = requested_heads.clamp(1, inner);
    while n_heads > 1 && inner % n_heads != 0 {
        n_heads -= 1;
    }
    let head_dim = (inner / n_heads).max(1);
    let mut n_groups = requested_groups.clamp(1, n_heads);
    while n_groups > 1 && n_heads % n_groups != 0 {
        n_groups -= 1;
    }
    (n_heads, head_dim, n_groups)
}

pub(crate) struct Mamba2LayerParams {
    pub(crate) conv_weight: Vec<f32>,
    pub(crate) conv_bias: Option<Vec<f32>>,
    pub(crate) dt_bias: Option<Vec<f32>>,
    pub(crate) a_log_or_neg: Vec<f32>,
    pub(crate) d_skip: Vec<f32>,
}

pub(crate) fn load_mamba2_layer_params(
    weights: &dyn ModelWeights,
    layer: usize,
    d_inner: usize,
    d_conv: usize,
    n_heads: usize,
) -> Result<Mamba2LayerParams, K3Error> {
    let conv_weight_gguf = format!("blk.{layer}.ssm_conv1d.weight");
    let conv_weight_bb = format!("backbone.layers.{layer}.mixer.conv1d.weight");
    let conv_weight_hf = format!("model.layers.{layer}.mamba.conv1d.weight");
    let conv_weight = read_required_f32(
        weights,
        &[&conv_weight_gguf, &conv_weight_bb, &conv_weight_hf],
        d_inner * d_conv,
    )?;

    let conv_bias_gguf = format!("blk.{layer}.ssm_conv1d.bias");
    let conv_bias_bb = format!("backbone.layers.{layer}.mixer.conv1d.bias");
    let conv_bias_hf = format!("model.layers.{layer}.mamba.conv1d.bias");
    let conv_bias = read_optional_f32(
        weights,
        &[&conv_bias_gguf, &conv_bias_bb, &conv_bias_hf],
        d_inner,
    )?;

    let dt_bias_gguf = format!("blk.{layer}.ssm_dt.bias");
    let dt_bias_bb = format!("backbone.layers.{layer}.mixer.dt_bias");
    let dt_bias_hf = format!("model.layers.{layer}.mamba.dt_bias");
    let dt_bias = read_optional_f32(weights, &[&dt_bias_gguf, &dt_bias_bb, &dt_bias_hf], n_heads)?;

    let a_log_gguf = format!("blk.{layer}.ssm_a");
    let a_log_bb = format!("backbone.layers.{layer}.mixer.A_log");
    let a_log_hf = format!("model.layers.{layer}.mamba.A_log");
    let a_log_or_neg = read_required_f32(weights, &[&a_log_gguf, &a_log_bb, &a_log_hf], n_heads)?;

    let skip_d_gguf = format!("blk.{layer}.ssm_d");
    let skip_d_bb = format!("backbone.layers.{layer}.mixer.D");
    let skip_d_hf = format!("model.layers.{layer}.mamba.D");
    let d_skip = read_required_f32(weights, &[&skip_d_gguf, &skip_d_bb, &skip_d_hf], n_heads)?;

    Ok(Mamba2LayerParams {
        conv_weight,
        conv_bias,
        dt_bias,
        a_log_or_neg,
        d_skip,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "shared Mamba2 single-step SSM update reads/writes ping-pong recurrent slices"
)]
pub(crate) fn run_mamba2_layer_step(
    layer: usize,
    x_norm: &[f32],
    hidden: &mut [f32],
    conv_state: &mut [f32],
    ssm_state: &mut [f32],
    d_inner: usize,
    d_conv: usize,
    d_state: usize,
    n_heads: usize,
    head_dim: usize,
    n_groups: usize,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    let proj_dim = 2 * d_inner + 2 * n_groups * d_state + n_heads;
    let in_g = format!("blk.{layer}.ssm_in.weight");
    let in_b = format!("backbone.layers.{layer}.mixer.in_proj.weight");
    let in_m = format!("model.layers.{layer}.mamba.in_proj.weight");
    let in_proj = project_linear_strict(weights, &[&in_g, &in_b, &in_m], x_norm, proj_dim)?;

    let z_gate = &in_proj[0..d_inner];
    let x_in = &in_proj[d_inner..2 * d_inner];
    let b_off = 2 * d_inner;
    let c_off = b_off + n_groups * d_state;
    let dt_off = c_off + n_groups * d_state;
    let b_vec = &in_proj[b_off..c_off];
    let c_vec = &in_proj[c_off..dt_off];
    let dt_raw = &in_proj[dt_off..dt_off + n_heads];

    let params = load_mamba2_layer_params(weights, layer, d_inner, d_conv, n_heads)?;

    let mut conv_out = vec![0.0_f32; d_inner];
    mamba2_conv1d_silu_step(
        conv_state,
        x_in,
        &params.conv_weight,
        params.conv_bias.as_deref(),
        &mut conv_out,
        d_inner,
        d_conv,
    )?;

    let mut ssm_y = vec![0.0_f32; d_inner];
    mamba2_ssd_state_step_f32(
        ssm_state,
        &mut ssm_y,
        &conv_out,
        dt_raw,
        params.dt_bias.as_deref(),
        &params.a_log_or_neg,
        b_vec,
        c_vec,
        Some(&params.d_skip),
        Some(z_gate),
        n_heads,
        head_dim,
        d_state,
        n_groups,
    )?;

    let out_g = format!("blk.{layer}.ssm_out.weight");
    let out_b = format!("backbone.layers.{layer}.mixer.out_proj.weight");
    let out_m = format!("model.layers.{layer}.mamba.out_proj.weight");
    let out = project_linear_strict(weights, &[&out_g, &out_b, &out_m], &ssm_y, hidden_dim)?;
    for (h, &o) in hidden.iter_mut().zip(out.iter()) {
        *h += o;
    }
    Ok(())
}

/// Execution recipe for pure Selective State-Space (`Mamba` and `Mamba-2`) checkpoints.
///
/// Uses chunked `conv1d + SiLU` and `SSD` scans during multi-token prefill (plus intra-chunk
/// quantized matrix scoring), single-step kernels during autoregressive decode, and double-buffered
/// ping-pong (`slot_0` / `slot_1`) `O(1)` recurrent state banks independent of `n_ctx`.
#[derive(Debug, Clone)]
pub struct Mamba2Recipe {
    /// Architecture identifier (`"mamba2"` or `"mamba"`).
    pub architecture: &'static str,
    /// Number of recurrent SSM layers.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Expanded inner SSM channel dimension (`d_inner`, defaults to `2 * hidden_dim` when 0).
    pub d_inner: usize,
    /// Causal depthwise conv kernel width (`d_conv`).
    pub d_conv: usize,
    /// Recurrent SSD state dimension per head (`d_state`).
    pub d_state: usize,
    /// Number of SSD heads (`n_heads`, divides `d_inner`).
    pub n_heads: usize,
    /// Number of B/C parameter groups (`n_groups`, divides `n_heads`).
    pub n_groups: usize,
    /// Vocabulary size for final LM-head logit projection.
    pub vocab_size: usize,
    /// `RMSNorm` epsilon.
    pub rms_eps: f32,
}

struct MambaChunkScratch {
    z_seq: Vec<f32>,
    x_seq: Vec<f32>,
    b_seq: Vec<f32>,
    c_seq: Vec<f32>,
    dt_seq: Vec<f32>,
    conv_out_seq: Vec<f32>,
    ssm_out_seq: Vec<f32>,
    ime_scores: Vec<f32>,
    slot_buf: Vec<f32>,
}

impl MambaChunkScratch {
    fn new(
        seq_len: usize,
        d_inner: usize,
        n_heads: usize,
        n_groups: usize,
        group_step: usize,
        slot_floats: usize,
    ) -> Self {
        Self {
            z_seq: vec![0.0_f32; seq_len * d_inner],
            x_seq: vec![0.0_f32; seq_len * d_inner],
            b_seq: vec![0.0_f32; seq_len * group_step],
            c_seq: vec![0.0_f32; seq_len * group_step],
            dt_seq: vec![0.0_f32; seq_len * n_heads],
            conv_out_seq: vec![0.0_f32; seq_len * d_inner],
            ssm_out_seq: vec![0.0_f32; seq_len * d_inner],
            ime_scores: vec![0.0_f32; n_groups * seq_len * seq_len],
            slot_buf: vec![0.0_f32; slot_floats],
        }
    }
}

impl Default for Mamba2Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Mamba2Recipe {
    /// Creates a `Mamba-2` recipe with defaults matching 2.7B production geometry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "mamba2",
            num_layers: 64,
            hidden_dim: 2560,
            d_inner: 5120,
            d_conv: 4,
            d_state: 128,
            n_heads: 80,
            n_groups: 1,
            vocab_size: 50_288,
            rms_eps: 1e-5,
        }
    }

    /// Creates a `Mamba-1` compatibility recipe targeting `"mamba"`.
    #[must_use]
    pub const fn new_mamba() -> Self {
        Self {
            architecture: "mamba",
            num_layers: 48,
            hidden_dim: 2048,
            d_inner: 4096,
            d_conv: 4,
            d_state: 16,
            n_heads: 64,
            n_groups: 1,
            vocab_size: 50_280,
            rms_eps: 1e-5,
        }
    }

    fn effective_dims(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        let hidden_dim = self.hidden_dim.max(2);
        let d_inner = if self.d_inner == 0 {
            hidden_dim.saturating_mul(2)
        } else {
            self.d_inner
        }
        .max(1);
        let d_conv = self.d_conv.max(2);
        let d_state = self.d_state.max(1);
        let (n_heads, head_dim, n_groups) =
            resolve_divisible_ssm_geometry(d_inner, self.n_heads, self.n_groups);
        (
            hidden_dim, d_inner, d_conv, d_state, n_heads, head_dim, n_groups,
        )
    }

    fn read_layer_norm(
        weights: &dyn ModelWeights,
        layer: usize,
        hidden_dim: usize,
    ) -> Result<Vec<f32>, K3Error> {
        let n_attn = format!("blk.{layer}.attn_norm.weight");
        let n_ssm = format!("blk.{layer}.ssm_norm.weight");
        let n_b = format!("backbone.layers.{layer}.norm.weight");
        let n_hf = format!("model.layers.{layer}.input_layernorm.weight");
        read_required_f32(weights, &[&n_attn, &n_ssm, &n_b, &n_hf], hidden_dim)
    }

    fn project_prefill_in_seq(
        &self,
        layer: usize,
        seq_len: usize,
        h_seq: &[f32],
        scratch: &mut MambaChunkScratch,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let (hidden_dim, d_inner, _d_conv, d_state, n_heads, _head_dim, n_groups) =
            self.effective_dims();
        let group_step = n_groups * d_state;
        let norm_w = Self::read_layer_norm(weights, layer, hidden_dim)?;
        let proj_dim = 2 * d_inner + 2 * group_step + n_heads;
        let in_g = format!("blk.{layer}.ssm_in.weight");
        let in_b = format!("backbone.layers.{layer}.mixer.in_proj.weight");
        let in_m = format!("model.layers.{layer}.mamba.in_proj.weight");
        let mut normed = vec![0.0_f32; hidden_dim];
        for t in 0..seq_len {
            let h_tok = &h_seq[t * hidden_dim..(t + 1) * hidden_dim];
            rmsnorm_standard(&mut normed, h_tok, &norm_w, self.rms_eps)?;
            let proj = project_linear_strict(weights, &[&in_g, &in_b, &in_m], &normed, proj_dim)?;
            scratch.z_seq[t * d_inner..(t + 1) * d_inner].copy_from_slice(&proj[0..d_inner]);
            scratch.x_seq[t * d_inner..(t + 1) * d_inner]
                .copy_from_slice(&proj[d_inner..2 * d_inner]);
            let b_off = 2 * d_inner;
            let c_off = b_off + group_step;
            let dt_off = c_off + group_step;
            scratch.b_seq[t * group_step..(t + 1) * group_step]
                .copy_from_slice(&proj[b_off..c_off]);
            scratch.c_seq[t * group_step..(t + 1) * group_step]
                .copy_from_slice(&proj[c_off..dt_off]);
            scratch.dt_seq[t * n_heads..(t + 1) * n_heads]
                .copy_from_slice(&proj[dt_off..dt_off + n_heads]);
        }
        Ok(())
    }

    fn run_prefill_layer(
        &self,
        layer: usize,
        seq_len: usize,
        h_seq: &mut [f32],
        scratch: &mut MambaChunkScratch,
        arena: &mut PingPongRecurrentArena<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let (hidden_dim, d_inner, d_conv, d_state, n_heads, head_dim, n_groups) =
            self.effective_dims();
        let conv_state_elems = d_inner * (d_conv - 1);
        self.project_prefill_in_seq(layer, seq_len, h_seq, scratch, weights)?;

        if seq_len > 1 {
            mamba2_ssd_intra_chunk_ime_i8(
                &scratch.c_seq,
                &scratch.b_seq,
                &mut scratch.ime_scores,
                seq_len,
                n_groups,
                d_state,
            )?;
        }

        let params = load_mamba2_layer_params(weights, layer, d_inner, d_conv, n_heads)?;

        arena.read_slot(layer, &mut scratch.slot_buf);
        let (conv_state, ssm_state) = scratch.slot_buf.split_at_mut(conv_state_elems);
        mamba2_conv1d_silu_chunk_f32(
            conv_state,
            &scratch.x_seq,
            &params.conv_weight,
            params.conv_bias.as_deref(),
            &mut scratch.conv_out_seq,
            seq_len,
            d_inner,
            d_conv,
        )?;
        mamba2_ssd_chunk_scan_f32(
            ssm_state,
            &mut scratch.ssm_out_seq,
            &scratch.conv_out_seq,
            &scratch.dt_seq,
            params.dt_bias.as_deref(),
            &params.a_log_or_neg,
            &scratch.b_seq,
            &scratch.c_seq,
            Some(&params.d_skip),
            Some(&scratch.z_seq),
            seq_len,
            n_heads,
            head_dim,
            d_state,
            n_groups,
        )?;

        let out_g = format!("blk.{layer}.ssm_out.weight");
        let out_b = format!("backbone.layers.{layer}.mixer.out_proj.weight");
        let out_m = format!("model.layers.{layer}.mamba.out_proj.weight");
        for t in 0..seq_len {
            let ssm_tok = &scratch.ssm_out_seq[t * d_inner..(t + 1) * d_inner];
            let out_proj =
                project_linear_strict(weights, &[&out_g, &out_b, &out_m], ssm_tok, hidden_dim)?;
            let h_tok = &mut h_seq[t * hidden_dim..(t + 1) * hidden_dim];
            for (h, &o) in h_tok.iter_mut().zip(out_proj.iter()) {
                *h += o;
            }
        }
        arena.write_staging_slot(layer, &scratch.slot_buf);
        Ok(())
    }

    fn run_chunked_prefill(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt = ctx
            .jump_forward_tokens()
            .unwrap_or_else(|| ctx.prompt_tokens())
            .to_vec();
        if prompt.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "prefill prompt_tokens cannot be empty".to_owned(),
            });
        }
        let start_pos = ctx.position();
        let seq_len = prompt.len();
        let end_pos = start_pos
            .checked_add(seq_len)
            .ok_or(K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: ctx.n_ctx(),
            })?;
        if end_pos > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: end_pos,
                max_context_tokens: ctx.n_ctx(),
            });
        }

        let (hidden_dim, d_inner, d_conv, d_state, n_heads, _head_dim, n_groups) =
            self.effective_dims();
        let num_layers = self.num_layers.max(1);

        let mut h_seq = Vec::with_capacity(seq_len * hidden_dim);
        for &tok in &prompt {
            h_seq.extend_from_slice(&embed_token_strict(tok, hidden_dim, weights)?);
        }

        let compute_logits = ctx.compute_logits();
        let logits_len = ctx.logits().len();
        let slot_floats = d_inner * (d_conv - 1) + d_inner * d_state;
        let mut scratch = MambaChunkScratch::new(
            seq_len,
            d_inner,
            n_heads,
            n_groups,
            n_groups * d_state,
            slot_floats,
        );
        let mut arena = PingPongRecurrentArena::open(ctx.kv_cache_mut(), num_layers, slot_floats);

        for layer in 0..num_layers {
            self.run_prefill_layer(
                layer,
                seq_len,
                &mut h_seq,
                &mut scratch,
                &mut arena,
                weights,
            )?;
        }

        let mut final_logits = if compute_logits {
            vec![0.0_f32; logits_len]
        } else {
            Vec::new()
        };
        if !final_logits.is_empty() {
            let last_hidden = &h_seq[(seq_len - 1) * hidden_dim..];
            finalize_scaled_logits(last_hidden, self.rms_eps, 1.0, &mut final_logits, weights)?;
        }

        let step_num = u32::try_from(end_pos).unwrap_or(u32::MAX);
        arena.commit(step_num);
        if !final_logits.is_empty() {
            ctx.logits_mut().copy_from_slice(&final_logits);
        }
        let _ = ctx.advance_position(seq_len)?;
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
        let (hidden_dim, d_inner, d_conv, d_state, n_heads, head_dim, n_groups) =
            self.effective_dims();
        let num_layers = self.num_layers.max(1);
        let conv_state_elems = d_inner * (d_conv - 1);
        let slot_floats = conv_state_elems + d_inner * d_state;

        let mut hidden = embed_token_strict(tok, hidden_dim, weights)?;
        let mut slot_buf = vec![0.0_f32; slot_floats];
        let mut arena = PingPongRecurrentArena::open(kv_cache, num_layers, slot_floats);
        let mut normed = vec![0.0_f32; hidden_dim];

        for layer in 0..num_layers {
            let norm_w = Self::read_layer_norm(weights, layer, hidden_dim)?;
            rmsnorm_standard(&mut normed, &hidden, &norm_w, self.rms_eps)?;

            arena.read_slot(layer, &mut slot_buf);
            let (conv_state, ssm_state) = slot_buf.split_at_mut(conv_state_elems);
            run_mamba2_layer_step(
                layer,
                &normed,
                &mut hidden,
                conv_state,
                ssm_state,
                d_inner,
                d_conv,
                d_state,
                n_heads,
                head_dim,
                n_groups,
                weights,
            )?;
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

impl ArchRecipe for Mamba2Recipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.run_chunked_prefill(ctx, weights)
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
