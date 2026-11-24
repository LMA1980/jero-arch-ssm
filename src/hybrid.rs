//! Hybrid attention + SSM/short-conv recipes (`jamba`, `zamba2`, `granitehybrid`, `lfm2`, `lfm2moe`).

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};
use jero_kernels_rvv::lfm2_shortconv_step_f32;

use crate::arena::{PingPongRecurrentArena, RECURRENT_HEADER_FLOATS};
use crate::dense::{
    DenseAttnLayerSpec, dispatch_moe_ffn_strict, even_head_dim, finalize_scaled_logits,
    rmsnorm_standard, router_present, run_decode_with, run_dense_attention_sublayer_scaled,
    run_dense_swiglu_ffn, run_prefill_with,
};
use crate::mamba2::{resolve_divisible_ssm_geometry, run_mamba2_layer_step};
use crate::schedule::{
    HybridLayerKind, KV_BLOCK_SIZE, hybrid_recurrent_slot_floats, resolve_hybrid_layer_schedule,
};
use crate::weights::{
    embed_token_strict, has_any_tensor, is_supported_quant, project_linear_strict,
    read_optional_f32, read_required_f32,
};

/// Executes one `LFM2` gated causal depthwise short-convolution operator step, delegating
/// directly to [`lfm2_shortconv_step_f32`] over `[dim, l_cache - 1]` history and `[dim, l_cache]`
/// filter weights.
///
/// # Errors
///
/// Returns [`K3Error`] if dimensions or buffer lengths fail kernel validation.
#[expect(
    clippy::too_many_arguments,
    reason = "matches lfm2_shortconv_step_f32 gate, projection, state, and kernel arguments"
)]
pub fn lfm2_shortconv_step(
    conv_state: &mut [f32],
    b_gate: &[f32],
    c_gate: &[f32],
    x_proj: &[f32],
    conv_weight: &[f32],
    conv_bias: Option<&[f32]>,
    out: &mut [f32],
    dim: usize,
    l_cache: usize,
) -> Result<(), K3Error> {
    lfm2_shortconv_step_f32(
        conv_state,
        b_gate,
        c_gate,
        x_proj,
        conv_weight,
        conv_bias,
        out,
        dim,
        l_cache,
    )
}

/// Hybrid attention + SSM/short-convolution execution recipe covering `jamba`, `zamba2`,
/// `granitehybrid`, and `LiquidAI` `lfm2`.
///
/// Dispatches each layer via [`HybridLayerKind`]: `GqaAttention` layers update paged KV cache
/// planes through their compact `kv_layer_slot`, while `Mamba2Ssd` and `Lfm2ShortConv` layers
/// execute linear-time state updates over trailing double-buffered ping-pong recurrent banks.
#[derive(Debug, Clone)]
pub struct HybridSsmTransformerRecipe {
    /// Exact architecture identifier (`"jamba"`, `"zamba2"`, `"granitehybrid"`, or `"lfm2"`).
    pub architecture_id: &'static str,
    /// Per-layer dispatch schedule built at load time or via [`resolve_hybrid_layer_schedule`].
    pub layer_schedule: Vec<HybridLayerKind>,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Attention query head count.
    pub n_heads_q: usize,
    /// Attention key/value head count.
    pub n_heads_kv: usize,
    /// Scalar dimension per attention head.
    pub head_dim: usize,
    /// Inner SSM expansion width (`ssm.inner_size`).
    pub d_inner: usize,
    /// Causal conv kernel width (`ssm.conv_kernel`).
    pub d_conv: usize,
    /// Mamba-2 state width (`ssm.state_size`).
    pub d_state: usize,
    /// Mamba-2 B/C group count (`ssm.group_count`).
    pub n_groups: usize,
    /// LFM2 short-convolution cache length (`shortconv.l_cache`).
    pub shortconv_l_cache: usize,
    /// Output vocabulary size.
    pub vocab_size: usize,
    /// `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency (`rope.freq_base`).
    pub rope_theta: f32,
    /// Optional `GraniteHybrid` embedding multiplier (`1.0` default).
    pub embedding_scale: f32,
    /// Optional `GraniteHybrid` residual branch multiplier (`1.0` default).
    pub residual_scale: f32,
    /// Optional `GraniteHybrid` explicit attention score scale override.
    pub attention_scale: Option<f32>,
    /// Optional `GraniteHybrid` output logit divisor (`1.0` default).
    pub logit_scale: f32,
}

impl Default for HybridSsmTransformerRecipe {
    fn default() -> Self {
        Self::new_lfm2()
    }
}

impl HybridSsmTransformerRecipe {
    /// Creates an alternating 1:7 `Jamba` hybrid attention + Mamba-2 recipe (`"jamba"`).
    #[must_use]
    pub fn new_jamba() -> Self {
        let mut schedule = Vec::with_capacity(32);
        let mut kv_slot = 0_usize;
        let mut ssm_slot = 0_usize;
        for idx in 0..32_usize {
            if idx % 8 == 4 {
                schedule.push(HybridLayerKind::GqaAttention {
                    kv_layer_slot: kv_slot,
                });
                kv_slot += 1;
            } else {
                schedule.push(HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: ssm_slot,
                });
                ssm_slot += 1;
            }
        }
        Self {
            architecture_id: "jamba",
            layer_schedule: schedule,
            hidden_dim: 4096,
            ffn_dim: 14_336,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            d_inner: 8192,
            d_conv: 4,
            d_state: 16,
            n_groups: 1,
            shortconv_l_cache: 3,
            vocab_size: 65_536,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            embedding_scale: 1.0,
            residual_scale: 1.0,
            attention_scale: None,
            logit_scale: 1.0,
        }
    }

    /// Creates a `Zamba2` hybrid attention + Mamba-2 recipe (`"zamba2"`).
    #[must_use]
    pub fn new_zamba2() -> Self {
        let mut recipe = Self::new_jamba();
        recipe.architecture_id = "zamba2";
        recipe
    }

    /// Creates a `GraniteHybrid` (`Granite 4.0` / `Bamba`) recipe (`"granitehybrid"`).
    #[must_use]
    pub fn new_granitehybrid() -> Self {
        let mut schedule = Vec::with_capacity(32);
        let mut kv_slot = 0_usize;
        let mut ssm_slot = 0_usize;
        for idx in 0..32_usize {
            if idx % 8 == 7 {
                schedule.push(HybridLayerKind::GqaAttention {
                    kv_layer_slot: kv_slot,
                });
                kv_slot += 1;
            } else {
                schedule.push(HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: ssm_slot,
                });
                ssm_slot += 1;
            }
        }
        Self {
            architecture_id: "granitehybrid",
            layer_schedule: schedule,
            hidden_dim: 4096,
            ffn_dim: 11_008,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            d_inner: 8192,
            d_conv: 4,
            d_state: 128,
            n_groups: 1,
            shortconv_l_cache: 3,
            vocab_size: 49_152,
            rms_eps: 1e-5,
            rope_theta: 1_000_000.0,
            embedding_scale: 1.0,
            residual_scale: 1.0,
            attention_scale: None,
            logit_scale: 1.0,
        }
    }

    /// Creates a `LiquidAI` `LFM2` hybrid GQA + `ShortConv` recipe (`"lfm2"`).
    #[must_use]
    pub fn new_lfm2() -> Self {
        let mut schedule = Vec::with_capacity(16);
        let mut kv_slot = 0_usize;
        let mut sc_slot = 0_usize;
        for idx in 0..16_usize {
            if matches!(idx, 2 | 5 | 8 | 10 | 12 | 14) {
                schedule.push(HybridLayerKind::GqaAttention {
                    kv_layer_slot: kv_slot,
                });
                kv_slot += 1;
            } else {
                schedule.push(HybridLayerKind::Lfm2ShortConv {
                    recurrent_layer_slot: sc_slot,
                });
                sc_slot += 1;
            }
        }
        Self {
            architecture_id: "lfm2",
            layer_schedule: schedule,
            hidden_dim: 2048,
            ffn_dim: 8192,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 64,
            d_inner: 2048,
            d_conv: 3,
            d_state: 16,
            n_groups: 1,
            shortconv_l_cache: 3,
            vocab_size: 65_536,
            rms_eps: 1e-5,
            rope_theta: 1_000_000.0,
            embedding_scale: 1.0,
            residual_scale: 1.0,
            attention_scale: None,
            logit_scale: 1.0,
        }
    }

    /// Constructs a compact hybrid recipe with an explicit [`HybridLayerKind`] schedule.
    #[must_use]
    pub fn from_schedule(
        architecture_id: &'static str,
        layer_schedule: Vec<HybridLayerKind>,
    ) -> Self {
        let mut base = match architecture_id {
            "jamba" => Self::new_jamba(),
            "zamba2" => Self::new_zamba2(),
            "granitehybrid" => Self::new_granitehybrid(),
            _ => Self::new_lfm2(),
        };
        base.architecture_id = architecture_id;
        base.layer_schedule = layer_schedule;
        base
    }

    fn effective_schedule(&self, weights: &dyn ModelWeights) -> Vec<HybridLayerKind> {
        if weights.tensor_count() > 0 {
            let inferred = resolve_hybrid_layer_schedule(self.layer_schedule.len().max(1), weights);
            if inferred
                .iter()
                .any(|k| !matches!(k, HybridLayerKind::GqaAttention { .. }))
            {
                return inferred;
            }
        }
        if self.layer_schedule.is_empty() {
            vec![HybridLayerKind::GqaAttention { kv_layer_slot: 0 }]
        } else {
            self.layer_schedule.clone()
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "dispatches single-layer Mamba2 SSD update inside hybrid loop"
    )]
    fn run_hybrid_ssm_layer(
        &self,
        layer_idx: usize,
        recurrent_layer_slot: usize,
        hidden: &mut [f32],
        slot_buf: &mut [f32],
        rec_arena: &mut PingPongRecurrentArena<'_>,
        ssm_geom: (usize, usize, usize, usize, usize, usize),
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = hidden.len();
        let (d_inner, d_conv, d_state, ssm_heads, ssm_head_dim, n_groups) = ssm_geom;
        let n_attn = format!("blk.{layer_idx}.attn_norm.weight");
        let n_op = format!("blk.{layer_idx}.operator_norm.weight");
        let n_ssm = format!("blk.{layer_idx}.ssm_norm.weight");
        let n_b = format!("backbone.layers.{layer_idx}.norm.weight");
        let n_hf = format!("model.layers.{layer_idx}.input_layernorm.weight");
        let norm_w =
            read_required_f32(weights, &[&n_attn, &n_op, &n_ssm, &n_b, &n_hf], hidden_dim)?;
        let mut normed = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut normed, hidden, &norm_w, self.rms_eps)?;

        rec_arena.read_slot(recurrent_layer_slot, slot_buf);
        let conv_elems = d_inner * (d_conv - 1);
        let (conv_state, ssm_rest) = slot_buf.split_at_mut(conv_elems);
        let ssm_state = &mut ssm_rest[..d_inner * d_state];
        let mut ssm_delta = hidden.to_vec();
        run_mamba2_layer_step(
            layer_idx,
            &normed,
            &mut ssm_delta,
            conv_state,
            ssm_state,
            d_inner,
            d_conv,
            d_state,
            ssm_heads,
            ssm_head_dim,
            n_groups,
            weights,
        )?;
        for (h, d) in hidden.iter_mut().zip(ssm_delta.iter()) {
            let branch = *d - *h;
            *h += branch * self.residual_scale;
        }
        rec_arena.write_staging_slot(recurrent_layer_slot, slot_buf);
        Ok(())
    }

    fn run_hybrid_shortconv_layer(
        &self,
        layer_idx: usize,
        recurrent_layer_slot: usize,
        hidden: &mut [f32],
        slot_buf: &mut [f32],
        rec_arena: &mut PingPongRecurrentArena<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = hidden.len();
        let l_cache = self.shortconv_l_cache.max(2);
        let n_op = format!("blk.{layer_idx}.operator_norm.weight");
        let n_attn = format!("blk.{layer_idx}.attn_norm.weight");
        let n_hf_op = format!("model.layers.{layer_idx}.operator_norm.weight");
        let n_hf_in = format!("model.layers.{layer_idx}.input_layernorm.weight");
        let norm_w = read_required_f32(weights, &[&n_op, &n_attn, &n_hf_op, &n_hf_in], hidden_dim)?;
        let mut normed = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut normed, hidden, &norm_w, self.rms_eps)?;

        let in_g = format!("blk.{layer_idx}.shortconv.in_proj.weight");
        let in_hf = format!("model.layers.{layer_idx}.conv.in_proj.weight");
        let proj3 = project_linear_strict(weights, &[&in_g, &in_hf], &normed, 3 * hidden_dim)?;
        let b_gate = &proj3[0..hidden_dim];
        let c_gate = &proj3[hidden_dim..2 * hidden_dim];
        let x_proj = &proj3[2 * hidden_dim..3 * hidden_dim];

        let conv_weight_gguf = format!("blk.{layer_idx}.shortconv.conv.weight");
        let conv_weight_alt = format!("blk.{layer_idx}.shortconv_conv.weight");
        let conv_weight_hf = format!("model.layers.{layer_idx}.conv.conv.weight");
        let conv_weight = read_required_f32(
            weights,
            &[&conv_weight_gguf, &conv_weight_alt, &conv_weight_hf],
            hidden_dim * l_cache,
        )?;

        let conv_bias_gguf = format!("blk.{layer_idx}.shortconv.conv.bias");
        let conv_bias_hf = format!("model.layers.{layer_idx}.conv.conv.bias");
        let conv_bias = read_optional_f32(weights, &[&conv_bias_gguf, &conv_bias_hf], hidden_dim)?;

        let conv_elems = hidden_dim * (l_cache - 1);
        rec_arena.read_slot(recurrent_layer_slot, slot_buf);
        let conv_state = &mut slot_buf[..conv_elems];
        let mut sc_out = vec![0.0_f32; hidden_dim];
        lfm2_shortconv_step(
            conv_state,
            b_gate,
            c_gate,
            x_proj,
            &conv_weight,
            conv_bias.as_deref(),
            &mut sc_out,
            hidden_dim,
            l_cache,
        )?;

        let out_g = format!("blk.{layer_idx}.shortconv.out_proj.weight");
        let out_hf = format!("model.layers.{layer_idx}.conv.out_proj.weight");
        let out_proj = project_linear_strict(weights, &[&out_g, &out_hf], &sc_out, hidden_dim)?;
        rec_arena.write_staging_slot(recurrent_layer_slot, slot_buf);
        for (h, &o) in hidden.iter_mut().zip(out_proj.iter()) {
            *h += o * self.residual_scale;
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "dispatches GqaAttention, Mamba2Ssd, or Lfm2ShortConv across disjoint arena planes"
    )]
    fn run_hybrid_layer_step(
        &self,
        layer_idx: usize,
        kind: HybridLayerKind,
        step: (usize, usize),
        hidden: &mut [f32],
        attn_spec: DenseAttnLayerSpec,
        kv_region: &mut [f32],
        rec_arena: &mut PingPongRecurrentArena<'_>,
        slot_buf: &mut [f32],
        ssm_geom: (usize, usize, usize, usize, usize, usize),
        num_kv_layers: usize,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let (pos, seq_len) = step;
        let layer_stride = kv_region.len().checked_div(num_kv_layers).unwrap_or(0);
        match kind {
            HybridLayerKind::GqaAttention { kv_layer_slot } => {
                let layer_kv = if kv_layer_slot < num_kv_layers
                    && layer_stride > 0
                    && (kv_layer_slot + 1) * layer_stride <= kv_region.len()
                {
                    &mut kv_region[kv_layer_slot * layer_stride..(kv_layer_slot + 1) * layer_stride]
                } else {
                    &mut kv_region[..0]
                };
                let mut layer_spec = attn_spec;
                let q_norm_gguf = format!("blk.{layer_idx}.attn_q_norm.weight");
                let k_norm_gguf = format!("blk.{layer_idx}.attn_k_norm.weight");
                let q_norm_hf = format!("model.layers.{layer_idx}.self_attn.q_layernorm.weight");
                let k_norm_hf = format!("model.layers.{layer_idx}.self_attn.k_layernorm.weight");
                layer_spec.use_qk_norm = has_any_tensor(
                    weights,
                    &[&q_norm_gguf, &k_norm_gguf, &q_norm_hf, &k_norm_hf],
                );
                let q_bias_gguf = format!("blk.{layer_idx}.attn_q.bias");
                let k_bias_gguf = format!("blk.{layer_idx}.attn_k.bias");
                let v_bias_gguf = format!("blk.{layer_idx}.attn_v.bias");
                let q_bias_hf = format!("model.layers.{layer_idx}.self_attn.q_proj.bias");
                let k_bias_hf = format!("model.layers.{layer_idx}.self_attn.k_proj.bias");
                let v_bias_hf = format!("model.layers.{layer_idx}.self_attn.v_proj.bias");
                layer_spec.use_qkv_bias = has_any_tensor(
                    weights,
                    &[
                        &q_bias_gguf,
                        &k_bias_gguf,
                        &v_bias_gguf,
                        &q_bias_hf,
                        &k_bias_hf,
                        &v_bias_hf,
                    ],
                );
                run_dense_attention_sublayer_scaled(
                    layer_spec,
                    layer_idx,
                    pos,
                    seq_len,
                    hidden,
                    layer_kv,
                    weights,
                    self.attention_scale,
                    self.residual_scale,
                )?;
            }
            HybridLayerKind::Mamba2Ssd {
                recurrent_layer_slot,
            } => {
                self.run_hybrid_ssm_layer(
                    layer_idx,
                    recurrent_layer_slot,
                    hidden,
                    slot_buf,
                    rec_arena,
                    ssm_geom,
                    weights,
                )?;
            }
            HybridLayerKind::Lfm2ShortConv {
                recurrent_layer_slot,
            } => {
                self.run_hybrid_shortconv_layer(
                    layer_idx,
                    recurrent_layer_slot,
                    hidden,
                    slot_buf,
                    rec_arena,
                    weights,
                )?;
            }
        }
        Ok(())
    }

    fn resolve_rec_slot_floats(
        &self,
        schedule: &[HybridLayerKind],
        hidden_dim: usize,
        dims: (usize, usize, usize, usize),
    ) -> usize {
        let (d_inner, d_conv, d_state, l_cache) = dims;
        let mut slot_floats = hybrid_recurrent_slot_floats(
            self.architecture_id,
            hidden_dim,
            d_inner,
            d_conv,
            d_state,
            l_cache,
        )
        .unwrap_or(hidden_dim * (l_cache - 1));
        if schedule
            .iter()
            .any(|k| matches!(k, HybridLayerKind::Mamba2Ssd { .. }))
        {
            slot_floats = slot_floats.max(d_inner * (d_conv - 1) + d_inner * d_state);
        }
        if schedule
            .iter()
            .any(|k| matches!(k, HybridLayerKind::Lfm2ShortConv { .. }))
        {
            slot_floats = slot_floats.max(hidden_dim * (l_cache - 1));
        }
        slot_floats
    }

    fn validate_scales(&self) -> Result<(), K3Error> {
        if !self.embedding_scale.is_finite()
            || !self.residual_scale.is_finite()
            || !self.logit_scale.is_finite()
            || self.attention_scale.is_some_and(|s| !s.is_finite())
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture_id.to_owned(),
                reason: "hybrid recipe multipliers must be finite".to_owned(),
            });
        }
        Ok(())
    }

    fn execute_hybrid_token<Ffn>(
        &self,
        step: (u32, usize, usize, usize),
        logits: &mut [f32],
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
        mut run_ffn: Ffn,
    ) -> Result<(), K3Error>
    where
        Ffn: FnMut(usize, &[f32], &dyn ModelWeights) -> Result<Vec<f32>, K3Error>,
    {
        self.validate_scales()?;
        let (tok, pos, seq_len, n_ctx) = step;
        let schedule = self.effective_schedule(weights);
        let hidden_dim = self.hidden_dim.max(2);
        let head_d_even = even_head_dim(self.head_dim);
        let n_kv = self.n_heads_kv.max(1);
        let num_kv_layers = schedule
            .iter()
            .filter(|k| matches!(k, HybridLayerKind::GqaAttention { .. }))
            .count();
        let num_rec_layers = schedule.len().saturating_sub(num_kv_layers);

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

        let d_inner = if self.d_inner == 0 {
            hidden_dim
        } else {
            self.d_inner
        }
        .max(1);
        let d_conv = self.d_conv.max(2);
        let d_state = self.d_state.max(1);
        let l_cache = self.shortconv_l_cache.max(2);
        let (ssm_heads, ssm_head_dim, n_groups) =
            resolve_divisible_ssm_geometry(d_inner, self.n_heads_q, self.n_groups);

        let rec_slot_floats = self.resolve_rec_slot_floats(
            &schedule,
            hidden_dim,
            (d_inner, d_conv, d_state, l_cache),
        );
        let rec_needed = if num_rec_layers > 0 {
            RECURRENT_HEADER_FLOATS.saturating_add(2 * num_rec_layers * rec_slot_floats)
        } else {
            0
        };

        let mut hidden = embed_token_strict(tok, hidden_dim, weights)?;
        if (self.embedding_scale - 1.0).abs() > f32::EPSILON {
            for h in &mut hidden {
                *h *= self.embedding_scale;
            }
        }

        let kv_dim = n_kv * head_d_even;
        let (kv_region_len, rec_end) = partition_hybrid_arena(
            kv_cache.len(),
            num_kv_layers,
            num_rec_layers,
            kv_dim,
            n_ctx,
            rec_needed,
        );
        let (kv_region, rest) = kv_cache.split_at_mut(kv_region_len);
        let rec_len = rec_end.saturating_sub(kv_region_len).min(rest.len());
        let rec_region = &mut rest[..rec_len];

        let mut rec_arena =
            PingPongRecurrentArena::open(rec_region, num_rec_layers, rec_slot_floats);
        let mut slot_buf = vec![0.0_f32; rec_slot_floats];
        let residual_scale = self.residual_scale;
        let ssm_geom = (d_inner, d_conv, d_state, ssm_heads, ssm_head_dim, n_groups);

        for (layer_idx, &kind) in schedule.iter().enumerate() {
            self.run_hybrid_layer_step(
                layer_idx,
                kind,
                (pos, seq_len),
                &mut hidden,
                attn_spec,
                kv_region,
                &mut rec_arena,
                &mut slot_buf,
                ssm_geom,
                num_kv_layers,
                weights,
            )?;

            let ffn_norm_g = format!("blk.{layer_idx}.ffn_norm.weight");
            let ffn_norm_hf = format!("model.layers.{layer_idx}.post_attention_layernorm.weight");
            let ffn_norm_w = read_required_f32(weights, &[&ffn_norm_g, &ffn_norm_hf], hidden_dim)?;
            let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
            rmsnorm_standard(&mut norm_ffn_in, &hidden, &ffn_norm_w, self.rms_eps)?;
            let ffn_out = run_ffn(layer_idx, &norm_ffn_in, weights)?;
            for (h, &f) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h += f * residual_scale;
            }
        }

        if !logits.is_empty() {
            finalize_scaled_logits(&hidden, self.rms_eps, self.logit_scale, logits, weights)?;
        }

        let step_num = u32::try_from(pos + 1).unwrap_or(u32::MAX);
        rec_arena.commit(step_num);
        Ok(())
    }
}

/// Partitions a flat hybrid workspace slice into step-invariant paged-KV and ping-pong
/// recurrent sub-slices whose boundary matches single-slot arena allocation.
fn partition_hybrid_arena(
    total_floats: usize,
    num_kv_layers: usize,
    num_rec_layers: usize,
    kv_dim: usize,
    n_ctx: usize,
    rec_needed: usize,
) -> (usize, usize) {
    if num_kv_layers == 0 {
        return (0, rec_needed.min(total_floats));
    }
    if num_rec_layers == 0 {
        return (total_floats, total_floats);
    }
    let kv_tokens_padded = n_ctx.max(1).div_ceil(KV_BLOCK_SIZE) * KV_BLOCK_SIZE;
    let arena_kv_floats = num_kv_layers
        .saturating_mul(2)
        .saturating_mul(kv_dim)
        .saturating_mul(kv_tokens_padded);
    if arena_kv_floats > 0 && total_floats >= arena_kv_floats.saturating_add(rec_needed) {
        return (arena_kv_floats, arena_kv_floats + rec_needed);
    }
    if total_floats > rec_needed && kv_dim > 0 {
        let kv_avail = total_floats - rec_needed;
        let kv_slot_unit = num_kv_layers.saturating_mul(2).saturating_mul(kv_dim);
        let per_layer_kv = if kv_slot_unit > 0 && kv_avail >= kv_slot_unit {
            (kv_avail / kv_slot_unit) * 2 * kv_dim
        } else {
            kv_avail / num_kv_layers
        };
        let kv_region_len = per_layer_kv * num_kv_layers;
        return (
            kv_region_len,
            (kv_region_len + rec_needed).min(total_floats),
        );
    }
    (0, total_floats)
}

impl ArchRecipe for HybridSsmTransformerRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture_id
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = self.hidden_dim.max(2);
        let ffn_dim = self.ffn_dim.max(4);
        let n_ctx = ctx.n_ctx();
        run_prefill_with(
            ctx,
            self.architecture_id,
            |tok, pos, seq_len, kv, logits| {
                self.execute_hybrid_token(
                    (tok, pos, seq_len, n_ctx),
                    logits,
                    kv,
                    weights,
                    |layer, norm_in, w| {
                        run_dense_swiglu_ffn(layer, hidden_dim, ffn_dim, norm_in, w)
                    },
                )
            },
        )
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = self.hidden_dim.max(2);
        let ffn_dim = self.ffn_dim.max(4);
        let n_ctx = ctx.n_ctx();
        run_decode_with(ctx, self.architecture_id, |tok, pos, kv, logits| {
            self.execute_hybrid_token(
                (tok, pos, pos + 1, n_ctx),
                logits,
                kv,
                weights,
                |layer, norm_in, w| run_dense_swiglu_ffn(layer, hidden_dim, ffn_dim, norm_in, w),
            )
        })
    }
}

/// Execution recipe for `LiquidAI` `LFM2.5-MoE` (`lfm2moe`) combining gated causal
/// short-convolution + GQA hybrid layers with sparse top-`k` `MoE` FFN blocks.
#[derive(Debug, Clone)]
pub struct Lfm2MoeRecipe {
    /// Underlying hybrid `ShortConv` + GQA backbone configuration.
    pub base: HybridSsmTransformerRecipe,
    /// Number of routed sparse experts per `MoE` layer.
    pub num_experts: usize,
    /// Number of active experts selected per token (`top_k`).
    pub num_experts_per_tok: usize,
    /// Per-expert feed-forward intermediate dimension.
    pub expert_ffn_dim: usize,
    /// Number of leading dense `SwiGLU` layers before sparse `MoE` layers begin.
    pub num_dense_layers: usize,
}

impl Default for Lfm2MoeRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Lfm2MoeRecipe {
    /// Creates an `LFM2.5-MoE` recipe targeting `"lfm2moe"`.
    #[must_use]
    pub fn new() -> Self {
        let mut base = HybridSsmTransformerRecipe::new_lfm2();
        base.architecture_id = "lfm2moe";
        Self {
            base,
            num_experts: 32,
            num_experts_per_tok: 4,
            expert_ffn_dim: 1792,
            num_dense_layers: 2,
        }
    }

    fn execute_layer_ffn(
        &self,
        layer: usize,
        norm_in: &[f32],
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        let hidden_dim = self.base.hidden_dim.max(2);
        let dense_dim = self.base.ffn_dim.max(4);
        let is_moe = layer >= self.num_dense_layers || router_present(weights, layer);
        if !is_moe {
            return run_dense_swiglu_ffn(layer, hidden_dim, dense_dim, norm_in, weights);
        }
        let mut out = vec![0.0_f32; hidden_dim];
        let _ = dispatch_moe_ffn_strict(
            &mut out,
            norm_in,
            layer,
            self.num_experts,
            self.num_experts_per_tok,
            self.expert_ffn_dim,
            weights,
        )?;
        Ok(out)
    }
}

impl ArchRecipe for Lfm2MoeRecipe {
    fn architecture_id(&self) -> &'static str {
        "lfm2moe"
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let n_ctx = ctx.n_ctx();
        run_prefill_with(ctx, "lfm2moe", |tok, pos, seq_len, kv, logits| {
            self.base.execute_hybrid_token(
                (tok, pos, seq_len, n_ctx),
                logits,
                kv,
                weights,
                |l, n, w| self.execute_layer_ffn(l, n, w),
            )
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let n_ctx = ctx.n_ctx();
        run_decode_with(ctx, "lfm2moe", |tok, pos, kv, logits| {
            self.base.execute_hybrid_token(
                (tok, pos, pos + 1, n_ctx),
                logits,
                kv,
                weights,
                |l, n, w| self.execute_layer_ffn(l, n, w),
            )
        })
    }
}
