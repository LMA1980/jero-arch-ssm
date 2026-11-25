//! Self-contained normalization, attention, `SwiGLU` FFN, `MoE` routing, and step helpers.

use jero_core::{ExecutionContext, K3Error, ModelWeights};
use jero_kernels_rvv::{
    BaselineKernel, VectorKernel, apply_rope_inplace, gqa_causal_attention_f32, swiglu_f32,
};

use crate::weights::{
    OUTPUT_NORM_CANDIDATES, add_bias_if_present_strict, has_any_tensor, matvec_any,
    project_linear_strict, project_lm_head_strict, read_optional_f32, read_required_f32,
    resolve_expert_matrix_strict, resolve_required_tensor, tensor_bounds_err,
};

#[must_use]
pub(crate) const fn even_head_dim(raw: usize) -> usize {
    let clamped = if raw < 2 { 2 } else { raw };
    if clamped % 2 == 0 {
        clamped
    } else {
        clamped - 1
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseAttnLayerSpec {
    pub(crate) ffn_dim: usize,
    pub(crate) n_heads_q: usize,
    pub(crate) n_heads_kv: usize,
    pub(crate) head_dim: usize,
    pub(crate) rms_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) use_qkv_bias: bool,
    pub(crate) use_qk_norm: bool,
}

#[derive(Debug, Clone, Copy)]
struct GqaHeadDims {
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
}

pub(crate) fn rmsnorm_standard(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    BaselineKernel::new().rmsnorm(output, input, weight, eps)
}

fn apply_per_head_standard_rmsnorm(
    heads: &mut [f32],
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    if head_dim == 0 || weight.len() != head_dim || heads.len() % head_dim != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "qk_norm".to_owned(),
            offset: 0,
            byte_len: size_of_val(heads),
            buffer_len: size_of_val(weight),
            reason: format!(
                "invalid per-head rmsnorm dimensions: heads_len={}, head_dim={head_dim}, weight_len={}",
                heads.len(),
                weight.len()
            ),
        });
    }
    let kernel = BaselineKernel::new();
    let mut tmp = vec![0.0_f32; head_dim];
    for head in heads.chunks_exact_mut(head_dim) {
        kernel.rmsnorm(&mut tmp, head, weight, eps)?;
        head.copy_from_slice(&tmp);
    }
    Ok(())
}

fn run_gqa_causal_with_cache_scaled(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    layer_kv: &mut [f32],
    pos: usize,
    dims: GqaHeadDims,
    attn_scale_override: Option<f32>,
) -> Result<Vec<f32>, K3Error> {
    let q_dim = q.len();
    let kv_dim = k.len();
    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim fits comfortably in f32"
    )]
    let scale = attn_scale_override.unwrap_or_else(|| 1.0_f32 / (dims.head_dim as f32).sqrt());
    let mut attn_out = vec![0.0_f32; q_dim];

    let capacity_slots = layer_kv.len().checked_div(2 * kv_dim.max(1)).unwrap_or(0);
    if capacity_slots > 0 {
        let slot = pos % capacity_slots;
        let active_len = (pos + 1).min(capacity_slots);
        let ring_elems = capacity_slots * kv_dim;
        let (k_ring, v_ring) = layer_kv[..2 * ring_elems].split_at_mut(ring_elems);
        k_ring[slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(k);
        v_ring[slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(v);
        gqa_causal_attention_f32(
            &mut attn_out,
            q,
            &k_ring[..active_len * kv_dim],
            &v_ring[..active_len * kv_dim],
            dims.n_heads_q,
            dims.n_heads_kv,
            dims.head_dim,
            active_len,
            scale,
        )?;
    } else {
        gqa_causal_attention_f32(
            &mut attn_out,
            q,
            k,
            v,
            dims.n_heads_q,
            dims.n_heads_kv,
            dims.head_dim,
            1,
            scale,
        )?;
    }
    Ok(attn_out)
}

#[expect(
    clippy::too_many_arguments,
    reason = "shared GQA attention helper supports Granite explicit attention_scale and residual_scale multipliers"
)]
pub(crate) fn run_dense_attention_sublayer_scaled(
    spec: DenseAttnLayerSpec,
    layer: usize,
    pos: usize,
    seq_len: usize,
    hidden: &mut [f32],
    layer_kv: &mut [f32],
    weights: &dyn ModelWeights,
    attn_scale_override: Option<f32>,
    residual_scale: f32,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    let attn_norm_gguf = format!("blk.{layer}.attn_norm.weight");
    let attn_norm_hf = format!("model.layers.{layer}.input_layernorm.weight");
    let attn_norm_w = read_required_f32(weights, &[&attn_norm_gguf, &attn_norm_hf], hidden_dim)?;
    let mut norm_hidden = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut norm_hidden, hidden, &attn_norm_w, spec.rms_eps)?;

    let head_d_even = even_head_dim(spec.head_dim);
    let n_q = spec.n_heads_q.max(1);
    let n_kv = spec.n_heads_kv.max(1);
    let q_dim = n_q * head_d_even;
    let kv_dim = n_kv * head_d_even;

    let q_gguf = format!("blk.{layer}.attn_q.weight");
    let q_hf = format!("model.layers.{layer}.self_attn.q_proj.weight");
    let k_gguf = format!("blk.{layer}.attn_k.weight");
    let k_hf = format!("model.layers.{layer}.self_attn.k_proj.weight");
    let v_gguf = format!("blk.{layer}.attn_v.weight");
    let v_hf = format!("model.layers.{layer}.self_attn.v_proj.weight");
    let o_gguf = format!("blk.{layer}.attn_output.weight");
    let o_hf = format!("model.layers.{layer}.self_attn.o_proj.weight");

    let mut q = project_linear_strict(weights, &[&q_gguf, &q_hf], &norm_hidden, q_dim)?;
    let mut k = project_linear_strict(weights, &[&k_gguf, &k_hf], &norm_hidden, kv_dim)?;
    let mut v = project_linear_strict(weights, &[&v_gguf, &v_hf], &norm_hidden, kv_dim)?;

    if spec.use_qkv_bias {
        let qb_g = format!("blk.{layer}.attn_q.bias");
        let qb_h = format!("model.layers.{layer}.self_attn.q_proj.bias");
        let kb_g = format!("blk.{layer}.attn_k.bias");
        let kb_h = format!("model.layers.{layer}.self_attn.k_proj.bias");
        let vb_g = format!("blk.{layer}.attn_v.bias");
        let vb_h = format!("model.layers.{layer}.self_attn.v_proj.bias");
        add_bias_if_present_strict(&mut q, weights, &[&qb_g, &qb_h])?;
        add_bias_if_present_strict(&mut k, weights, &[&kb_g, &kb_h])?;
        add_bias_if_present_strict(&mut v, weights, &[&vb_g, &vb_h])?;
    }

    if spec.use_qk_norm {
        let qn_g = format!("blk.{layer}.attn_q_norm.weight");
        let qn_h = format!("model.layers.{layer}.self_attn.q_layernorm.weight");
        let kn_g = format!("blk.{layer}.attn_k_norm.weight");
        let kn_h = format!("model.layers.{layer}.self_attn.k_layernorm.weight");
        let q_norm_w = read_required_f32(weights, &[&qn_g, &qn_h], head_d_even)?;
        let k_norm_w = read_required_f32(weights, &[&kn_g, &kn_h], head_d_even)?;
        apply_per_head_standard_rmsnorm(&mut q, head_d_even, &q_norm_w, spec.rms_eps)?;
        apply_per_head_standard_rmsnorm(&mut k, head_d_even, &k_norm_w, spec.rms_eps)?;
    }

    apply_rope_inplace(
        &mut q,
        head_d_even,
        pos,
        seq_len.max(pos + 1),
        spec.rope_theta,
        None,
    )?;
    apply_rope_inplace(
        &mut k,
        head_d_even,
        pos,
        seq_len.max(pos + 1),
        spec.rope_theta,
        None,
    )?;

    let attn_out = run_gqa_causal_with_cache_scaled(
        &q,
        &k,
        &v,
        layer_kv,
        pos,
        GqaHeadDims {
            n_heads_q: n_q,
            n_heads_kv: n_kv,
            head_dim: head_d_even,
        },
        attn_scale_override,
    )?;
    let proj_attn = project_linear_strict(weights, &[&o_gguf, &o_hf], &attn_out, hidden_dim)?;
    for (h, &r) in hidden.iter_mut().zip(proj_attn.iter()) {
        *h += r * residual_scale;
    }
    Ok(())
}

pub(crate) fn run_dense_swiglu_ffn(
    layer: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    norm_ffn_in: &[f32],
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let w_dim = ffn_dim.max(4);
    let gate_g = format!("blk.{layer}.ffn_gate.weight");
    let gate_h = format!("model.layers.{layer}.mlp.gate_proj.weight");
    let gate_h2 = format!("model.layers.{layer}.feed_forward.w1.weight");
    let up_g = format!("blk.{layer}.ffn_up.weight");
    let up_h = format!("model.layers.{layer}.mlp.up_proj.weight");
    let up_h2 = format!("model.layers.{layer}.feed_forward.w3.weight");
    let down_g = format!("blk.{layer}.ffn_down.weight");
    let down_h = format!("model.layers.{layer}.mlp.down_proj.weight");
    let down_h2 = format!("model.layers.{layer}.feed_forward.w2.weight");

    let gate = project_linear_strict(weights, &[&gate_g, &gate_h, &gate_h2], norm_ffn_in, w_dim)?;
    let up = project_linear_strict(weights, &[&up_g, &up_h, &up_h2], norm_ffn_in, w_dim)?;
    let mut act = vec![0.0_f32; w_dim];
    swiglu_f32(&mut act, &gate, &up)?;
    project_linear_strict(weights, &[&down_g, &down_h, &down_h2], &act, hidden_dim)
}

/// Selects the top-`k` experts deterministically from `router_logits` with ascending index
/// tie-breaking, returning `(expert_idx, gating_weight)` pairs.
#[must_use]
pub fn select_top_k_experts(
    router_logits: &[f32],
    top_k: usize,
    normalize: bool,
) -> Vec<(usize, f32)> {
    if router_logits.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let max_logit = router_logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let mut probs = Vec::with_capacity(router_logits.len());
    let mut sum_exp = 0.0_f32;
    for &logit in router_logits {
        let e = if logit.is_finite() && max_logit.is_finite() {
            (logit - max_logit).exp()
        } else {
            0.0_f32
        };
        probs.push(e);
        sum_exp += e;
    }

    if sum_exp > 0.0_f32 && sum_exp.is_finite() {
        for p in &mut probs {
            *p /= sum_exp;
        }
    } else {
        #[expect(
            clippy::cast_precision_loss,
            reason = "expert count fits comfortably in f32"
        )]
        let uniform = 1.0_f32 / (router_logits.len() as f32);
        probs.fill(uniform);
    }

    let mut ranked: Vec<(usize, f32, f32)> = router_logits
        .iter()
        .copied()
        .zip(probs)
        .enumerate()
        .map(|(idx, (logit, prob))| {
            let finite_logit = if logit.is_finite() {
                logit
            } else {
                f32::NEG_INFINITY
            };
            (idx, finite_logit, prob)
        })
        .collect();

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let k = top_k.min(ranked.len());
    let mut selected: Vec<(usize, f32)> = ranked
        .into_iter()
        .take(k)
        .map(|(idx, _, prob)| (idx, prob))
        .collect();

    if normalize {
        let selected_sum: f32 = selected.iter().map(|&(_, w)| w).sum();
        if selected_sum > 0.0_f32 && selected_sum.is_finite() {
            for (_, w) in &mut selected {
                *w /= selected_sum;
            }
        } else {
            #[expect(clippy::cast_precision_loss, reason = "top_k fits comfortably in f32")]
            let uniform = 1.0_f32 / (k as f32);
            for (_, w) in &mut selected {
                *w = uniform;
            }
        }
    }

    selected
}

fn select_experts_with_optional_bias(
    router_logits: &[f32],
    router_bias: Option<&[f32]>,
    top_k: usize,
) -> Vec<(usize, f32)> {
    let Some(bias) = router_bias else {
        return select_top_k_experts(router_logits, top_k, true);
    };
    let n = router_logits.len();
    if n == 0 || top_k == 0 {
        return Vec::new();
    }
    let biased: Vec<f32> = router_logits
        .iter()
        .enumerate()
        .map(|(idx, &l)| l + bias.get(idx).copied().unwrap_or(0.0))
        .collect();
    let chosen = select_top_k_experts(&biased, top_k, false);
    let max_logit = router_logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = router_logits
        .iter()
        .map(|&l| {
            if l.is_finite() && max_logit.is_finite() {
                (l - max_logit).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum_probs: f32 = probs.iter().sum();
    let mut out: Vec<(usize, f32)> = chosen
        .into_iter()
        .map(|(idx, _)| {
            let p = if sum_probs > 0.0 && sum_probs.is_finite() {
                probs[idx] / sum_probs
            } else {
                0.0
            };
            (idx, p)
        })
        .collect();
    let sel_sum: f32 = out.iter().map(|&(_, w)| w).sum();
    if sel_sum > 0.0 && sel_sum.is_finite() {
        for (_, w) in &mut out {
            *w /= sel_sum;
        }
    } else {
        #[expect(clippy::cast_precision_loss, reason = "top_k fits comfortably in f32")]
        let uniform = 1.0_f32 / (out.len().max(1) as f32);
        for (_, w) in &mut out {
            *w = uniform;
        }
    }
    out
}

fn run_single_swiglu_expert(
    expert_gate: &jero_core::TensorView<'_>,
    expert_up: &jero_core::TensorView<'_>,
    expert_down: &jero_core::TensorView<'_>,
    norm_ffn_in: &[f32],
    expert_ffn_dim: usize,
) -> Result<Vec<f32>, K3Error> {
    let hidden_dim = norm_ffn_in.len();
    let gate_elems = expert_gate.num_elements();
    let eff_ffn = if gate_elems > 0 && hidden_dim > 0 && gate_elems % hidden_dim == 0 {
        gate_elems / hidden_dim
    } else {
        return Err(tensor_bounds_err(
            expert_gate,
            format!(
                "expert gate elements {gate_elems} incompatible with hidden_dim {hidden_dim} (expected expert_ffn_dim={expert_ffn_dim})"
            ),
        ));
    };
    let mut gate = vec![0.0_f32; eff_ffn];
    let mut up = vec![0.0_f32; eff_ffn];
    matvec_any(&mut gate, expert_gate, norm_ffn_in)?;
    matvec_any(&mut up, expert_up, norm_ffn_in)?;
    let mut act = vec![0.0_f32; eff_ffn];
    swiglu_f32(&mut act, &gate, &up)?;
    let mut out = vec![0.0_f32; hidden_dim];
    matvec_any(&mut out, expert_down, &act)?;
    Ok(out)
}

/// Strictly evaluates a sparse top-`k` `MoE` FFN layer, failing closed on missing router or
/// selected expert weight tensors rather than falling back to dense stubs.
pub(crate) fn dispatch_moe_ffn_strict(
    out: &mut [f32],
    norm_ffn_in: &[f32],
    layer: usize,
    num_experts: usize,
    top_k: usize,
    expert_ffn_dim: usize,
    weights: &dyn ModelWeights,
) -> Result<Vec<(usize, f32)>, K3Error> {
    let hidden_dim = norm_ffn_in.len();
    if out.len() != hidden_dim || hidden_dim == 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: format!("blk.{layer}.ffn_gate_inp.weight"),
            offset: 0,
            byte_len: out.len(),
            buffer_len: hidden_dim,
            reason: "MoE FFN output slice length mismatch".to_owned(),
        });
    }
    if num_experts == 0 || top_k == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "lfm2moe".to_owned(),
            reason: format!(
                "MoE dispatch requires num_experts > 0 and top_k > 0 (got {num_experts}, {top_k})"
            ),
        });
    }
    out.fill(0.0);

    let router_g = format!("blk.{layer}.ffn_gate_inp.weight");
    let router_h1 = format!("model.layers.{layer}.block_sparse_moe.gate.weight");
    let router_h2 = format!("model.layers.{layer}.feed_forward.gate.weight");
    let router_view = resolve_required_tensor(weights, &[&router_g, &router_h1, &router_h2])?;
    let router_elems = router_view.num_elements();
    if router_elems == 0 || router_elems % hidden_dim != 0 {
        return Err(tensor_bounds_err(
            &router_view,
            format!("MoE router elements {router_elems} not divisible by hidden_dim {hidden_dim}"),
        ));
    }
    let eff_experts = router_elems / hidden_dim;
    if eff_experts == 0 || eff_experts > num_experts {
        return Err(tensor_bounds_err(
            &router_view,
            format!(
                "MoE router row count {eff_experts} outside configured num_experts {num_experts}"
            ),
        ));
    }

    let mut router_logits = vec![0.0_f32; eff_experts];
    matvec_any(&mut router_logits, &router_view, norm_ffn_in)?;

    let bias_g = format!("blk.{layer}.exp_probs_b.bias");
    let bias_h = format!("model.layers.{layer}.feed_forward.expert_bias");
    let bias_opt = read_optional_f32(weights, &[&bias_g, &bias_h], eff_experts)?;
    let selected = select_experts_with_optional_bias(
        &router_logits,
        bias_opt.as_deref(),
        top_k.min(eff_experts),
    );

    let g_stack = format!("blk.{layer}.ffn_gate_exps.weight");
    let u_stack = format!("blk.{layer}.ffn_up_exps.weight");
    let d_stack = format!("blk.{layer}.ffn_down_exps.weight");

    for &(expert_idx, gate_weight) in &selected {
        let g_idx = format!("blk.{layer}.ffn_gate_exps.{expert_idx}.weight");
        let g_h1 = format!("model.layers.{layer}.block_sparse_moe.experts.{expert_idx}.w1.weight");
        let g_h2 = format!("model.layers.{layer}.feed_forward.experts.{expert_idx}.w1.weight");
        let u_idx = format!("blk.{layer}.ffn_up_exps.{expert_idx}.weight");
        let u_h1 = format!("model.layers.{layer}.block_sparse_moe.experts.{expert_idx}.w3.weight");
        let u_h2 = format!("model.layers.{layer}.feed_forward.experts.{expert_idx}.w3.weight");
        let d_idx = format!("blk.{layer}.ffn_down_exps.{expert_idx}.weight");
        let d_h1 = format!("model.layers.{layer}.block_sparse_moe.experts.{expert_idx}.w2.weight");
        let d_h2 = format!("model.layers.{layer}.feed_forward.experts.{expert_idx}.w2.weight");

        let g_view = resolve_expert_matrix_strict(
            weights,
            &[&g_stack],
            &[&g_idx, &g_h1, &g_h2],
            expert_idx,
            eff_experts,
        )?;
        let u_view = resolve_expert_matrix_strict(
            weights,
            &[&u_stack],
            &[&u_idx, &u_h1, &u_h2],
            expert_idx,
            eff_experts,
        )?;
        let d_view = resolve_expert_matrix_strict(
            weights,
            &[&d_stack],
            &[&d_idx, &d_h1, &d_h2],
            expert_idx,
            eff_experts,
        )?;
        let exp_out =
            run_single_swiglu_expert(&g_view, &u_view, &d_view, norm_ffn_in, expert_ffn_dim)?;
        for (dst, &src) in out.iter_mut().zip(exp_out.iter()) {
            *dst += gate_weight * src;
        }
    }

    Ok(selected)
}

pub(crate) fn router_present(weights: &dyn ModelWeights, layer: usize) -> bool {
    let r_g = format!("blk.{layer}.ffn_gate_inp.weight");
    let r_h1 = format!("model.layers.{layer}.block_sparse_moe.gate.weight");
    let r_h2 = format!("model.layers.{layer}.feed_forward.gate.weight");
    has_any_tensor(weights, &[&r_g, &r_h1, &r_h2])
}

pub(crate) fn finalize_scaled_logits(
    hidden: &[f32],
    rms_eps: f32,
    logit_scale: f32,
    logits: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    if !logit_scale.is_finite() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "ssm".to_owned(),
            reason: format!("logit_scale must be finite, got {logit_scale}"),
        });
    }
    let hidden_dim = hidden.len();
    let norm_out_w = read_required_f32(weights, OUTPUT_NORM_CANDIDATES, hidden_dim)?;
    let mut normed_final = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut normed_final, hidden, &norm_out_w, rms_eps)?;
    project_lm_head_strict(weights, &normed_final, logits)?;
    if logit_scale > 0.0 && (logit_scale - 1.0).abs() > f32::EPSILON {
        let inv_scale = 1.0_f32 / logit_scale;
        for val in logits.iter_mut() {
            *val *= inv_scale;
        }
    }
    Ok(())
}

pub(crate) fn run_prefill_with<F>(
    ctx: &mut ExecutionContext<'_>,
    architecture: &str,
    mut step_fn: F,
) -> Result<(), K3Error>
where
    F: FnMut(u32, usize, usize, &mut [f32], &mut [f32]) -> Result<(), K3Error>,
{
    let prompt = ctx
        .jump_forward_tokens()
        .unwrap_or_else(|| ctx.prompt_tokens())
        .to_vec();
    if prompt.is_empty() {
        return Err(K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: "prefill prompt_tokens cannot be empty".to_owned(),
        });
    }
    let seq_len = prompt.len();
    let start_pos = ctx.position();
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

    let logits_len = ctx.logits().len();
    let compute_logits = ctx.compute_logits();
    for (offset, &tok) in prompt.iter().enumerate() {
        let need_logits = compute_logits && offset + 1 == seq_len;
        let mut step_logits = if need_logits {
            vec![0.0_f32; logits_len]
        } else {
            Vec::new()
        };
        step_fn(
            tok,
            start_pos + offset,
            end_pos,
            ctx.kv_cache_mut(),
            &mut step_logits,
        )?;
        if need_logits {
            ctx.logits_mut().copy_from_slice(&step_logits);
        }
    }
    let _ = ctx.advance_position(seq_len)?;
    Ok(())
}

pub(crate) fn run_decode_with<F>(
    ctx: &mut ExecutionContext<'_>,
    architecture: &str,
    mut step_fn: F,
) -> Result<(), K3Error>
where
    F: FnMut(u32, usize, &mut [f32], &mut [f32]) -> Result<(), K3Error>,
{
    let tok = ctx
        .decode_token()
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: "decode_step called without a staged decode token".to_owned(),
        })?;
    let pos = ctx.position();
    if pos >= ctx.n_ctx() {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: pos.saturating_add(1),
            max_context_tokens: ctx.n_ctx(),
        });
    }
    let mut step_logits = vec![0.0_f32; ctx.logits().len()];
    step_fn(tok, pos, ctx.kv_cache_mut(), &mut step_logits)?;
    ctx.logits_mut().copy_from_slice(&step_logits);
    let _ = ctx.advance_position(1)?;
    Ok(())
}
