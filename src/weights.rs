//! Fail-closed weight lookup, quantized matrix-vector projection, and SafeTensors/GGUF tensor resolution.

use jero_core::{K3Error, ModelWeights, QuantType, TensorView};
use jero_kernels_rvv::{
    BaselineKernel, VectorKernel, bf16_to_f32_ieee, dequantize_token_row, f16_to_f32_ieee,
    vec_dot_bf16, vec_dot_f16, vec_dot_i2_s, vec_dot_iq4_nl, vec_dot_q2_k, vec_dot_q3_k,
    vec_dot_q4_0, vec_dot_q4_1, vec_dot_q5_0, vec_dot_q5_1, vec_dot_q5_k, vec_dot_q6_k,
    vec_dot_q8_0,
};
#[cfg(not(miri))]
use rayon::prelude::*;

/// Returns `true` when `quant` is supported across SSM and hybrid recipe projections.
#[must_use]
pub(crate) const fn is_supported_quant(quant: QuantType) -> bool {
    matches!(
        quant,
        QuantType::F32
            | QuantType::F16
            | QuantType::BF16
            | QuantType::Q4_0
            | QuantType::Q4_1
            | QuantType::Q5_0
            | QuantType::Q5_1
            | QuantType::Q2_K
            | QuantType::Q3_K
            | QuantType::Q4_K_M
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::Q8_0
            | QuantType::IQ4_NL
    )
}

#[inline]
pub(crate) fn tensor_bounds_err(view: &TensorView<'_>, reason: impl Into<String>) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: view.name().to_owned(),
        offset: view.offset(),
        byte_len: view.byte_len(),
        buffer_len: view.as_bytes().len(),
        reason: reason.into(),
    }
}

/// Returns `true` if any tensor name in `candidates` is present in `weights`.
#[must_use]
pub(crate) fn has_any_tensor(weights: &dyn ModelWeights, candidates: &[&str]) -> bool {
    candidates.iter().any(|name| weights.contains_tensor(name))
}

/// Resolves a required tensor across GGUF and `SafeTensors` candidate names, failing closed with
/// [`K3Error::TensorBoundsError`] when absent or invalid.
pub(crate) fn resolve_required_tensor<'a>(
    weights: &'a dyn ModelWeights,
    candidates: &[&str],
) -> Result<TensorView<'a>, K3Error> {
    for &name in candidates {
        if weights.contains_tensor(name) {
            return weights.tensor(name);
        }
    }
    let primary = candidates.first().copied().unwrap_or("<unnamed>");
    weights.tensor(primary)
}

/// Resolves an optional tensor across GGUF and `SafeTensors` candidate names, returning `Ok(None)`
/// only when none of the candidate keys exist, while propagating bounds errors on corrupt entries.
pub(crate) fn resolve_optional_tensor<'a>(
    weights: &'a dyn ModelWeights,
    candidates: &[&str],
) -> Result<Option<TensorView<'a>>, K3Error> {
    for &name in candidates {
        if weights.contains_tensor(name) {
            return weights.tensor(name).map(Some);
        }
    }
    Ok(None)
}

fn validate_f32_matvec_shape(
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
) -> Result<(), K3Error> {
    let shape = weights.shape();
    if shape.len() == 2 {
        let matches =
            (shape[0] == rows && shape[1] == cols) || (shape[0] == cols && shape[1] == rows);
        if !matches {
            return Err(tensor_bounds_err(
                weights,
                format!(
                    "F32 matrix shape [{}, {}] does not match expected [{rows}, {cols}]",
                    shape[0], shape[1]
                ),
            ));
        }
    }
    let expected_elems = rows
        .checked_mul(cols)
        .ok_or_else(|| tensor_bounds_err(weights, "F32 matvec dimension overflowed usize"))?;
    if weights.num_elements() != expected_elems {
        return Err(tensor_bounds_err(
            weights,
            format!(
                "F32 tensor elements {} != out({rows}) * in({cols})",
                weights.num_elements()
            ),
        ));
    }
    Ok(())
}

fn matvec_f32(out: &mut [f32], weights: &TensorView<'_>, input: &[f32]) -> Result<(), K3Error> {
    let rows = out.len();
    let cols = input.len();
    validate_f32_matvec_shape(weights, rows, cols)?;
    let expected_elems = rows * cols;
    let raw = weights.as_bytes();
    let required_bytes = expected_elems
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| tensor_bounds_err(weights, "F32 required bytes overflowed usize"))?;
    if raw.len() < required_bytes {
        return Err(tensor_bounds_err(
            weights,
            format!(
                "F32 buffer length {} < required {required_bytes}",
                raw.len()
            ),
        ));
    }

    let kernel = BaselineKernel::new();
    let mut row_buf = vec![0.0_f32; cols];
    for (r, out_slot) in out.iter_mut().enumerate() {
        let base = r * cols * 4;
        for (c, slot) in row_buf.iter_mut().enumerate() {
            let off = base + c * 4;
            let val = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            if !val.is_finite() {
                return Err(tensor_bounds_err(
                    weights,
                    format!("non-finite F32 weight at row {r}, col {c}"),
                ));
            }
            *slot = val;
        }
        let dot = kernel.dot_product_f32(&row_buf, input);
        if !dot.is_finite() {
            return Err(tensor_bounds_err(
                weights,
                format!("non-finite F32 matvec output at row {r}"),
            ));
        }
        *out_slot = dot;
    }
    Ok(())
}

fn matvec_any_slice(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::F32 => matvec_f32(out, weights, input),
        QuantType::F16 => vec_dot_f16(out, weights, input),
        QuantType::BF16 => vec_dot_bf16(out, weights, input),
        QuantType::Q4_0 => vec_dot_q4_0(out, weights, input),
        QuantType::Q4_1 => vec_dot_q4_1(out, weights, input),
        QuantType::Q5_0 => vec_dot_q5_0(out, weights, input),
        QuantType::Q5_1 => vec_dot_q5_1(out, weights, input),
        QuantType::Q8_0 => vec_dot_q8_0(out, weights, input),
        QuantType::Q2_K => vec_dot_q2_k(out, weights, input),
        QuantType::Q3_K => vec_dot_q3_k(out, weights, input),
        QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(out, weights, input),
        QuantType::Q5_K => vec_dot_q5_k(out, weights, input),
        QuantType::Q6_K => vec_dot_q6_k(out, weights, input),
        QuantType::IQ4_NL => vec_dot_iq4_nl(out, weights, input),
        QuantType::I2_S => vec_dot_i2_s(out, weights, input),
        other => Err(K3Error::CapabilityMismatch {
            architecture: "ssm".to_owned(),
            reason: format!(
                "unsupported weight quantization {other:?} for tensor '{}'",
                weights.name()
            ),
        }),
    }
}

/// Dispatches matrix-vector multiplication across all quantization formats supported by SSM and
/// hybrid recipes, fanning out independent output row chunks across Rayon worker threads when
/// available outside Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if slices are empty, inputs contain non-finite scalars,
/// or tensor dimensions/bounds fail validation, and [`K3Error::CapabilityMismatch`] on unsupported
/// quantization formats.
pub fn matvec_any(out: &mut [f32], weights: &TensorView<'_>, input: &[f32]) -> Result<(), K3Error> {
    if out.is_empty() || input.is_empty() {
        return Err(tensor_bounds_err(
            weights,
            "matvec output and input slices cannot be empty",
        ));
    }
    if input.iter().any(|v| !v.is_finite()) {
        return Err(tensor_bounds_err(
            weights,
            "matvec input activation contains non-finite scalar",
        ));
    }
    #[cfg(not(miri))]
    if out.len() >= 64
        && weights.shape().len() == 2
        && weights.shape()[1] == out.len()
        && weights.shape()[0] == input.len()
        && weights.quant() != QuantType::I2_S
        && rayon::current_num_threads() > 1
    {
        let n_threads = rayon::current_num_threads();
        let chunk_rows = out.len().div_ceil(n_threads).clamp(16, 2048);
        out.par_chunks_mut(chunk_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                let chunk_view = weights.slice_rows(row_start, out_chunk.len())?;
                matvec_any_slice(out_chunk, &chunk_view, input)
            })?;
        return Ok(());
    }
    matvec_any_slice(out, weights, input)
}

fn decode_half_tensor(
    view: &TensorView<'_>,
    expected_len: usize,
    raw: &[u8],
    out: &mut [f32],
    decode_bits: fn(u16) -> f32,
) -> Result<(), K3Error> {
    let needed = expected_len * 2;
    if raw.len() < needed {
        return Err(tensor_bounds_err(
            view,
            format!(
                "{} payload length {} < required {needed}",
                view.quant().as_str(),
                raw.len()
            ),
        ));
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let off = i * 2;
        let val = decode_bits(u16::from_le_bytes([raw[off], raw[off + 1]]));
        if !val.is_finite() {
            return Err(tensor_bounds_err(
                view,
                format!("non-finite {} scalar at element {i}", view.quant().as_str()),
            ));
        }
        *slot = val;
    }
    Ok(())
}

/// Decodes a tensor (`F32`, `F16`, `BF16`, or block-quantized) into `expected_len` finite `f32`
/// scalars, failing closed on element-count mismatch, truncation, or non-finite values.
pub(crate) fn decode_tensor_f32(
    view: &TensorView<'_>,
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    if expected_len == 0 {
        return Err(tensor_bounds_err(
            view,
            "expected_len must be greater than zero",
        ));
    }
    if view.num_elements() != expected_len {
        return Err(tensor_bounds_err(
            view,
            format!(
                "tensor '{}' element count {} != expected {expected_len}",
                view.name(),
                view.num_elements()
            ),
        ));
    }

    let raw = view.as_bytes();
    let mut out = vec![0.0_f32; expected_len];
    match view.quant() {
        QuantType::F32 => {
            let needed = expected_len * size_of::<f32>();
            if raw.len() < needed {
                return Err(tensor_bounds_err(
                    view,
                    format!("F32 payload length {} < required {needed}", raw.len()),
                ));
            }
            for (i, slot) in out.iter_mut().enumerate() {
                let off = i * 4;
                let val = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                if !val.is_finite() {
                    return Err(tensor_bounds_err(
                        view,
                        format!("non-finite F32 scalar at element {i}"),
                    ));
                }
                *slot = val;
            }
        }
        QuantType::F16 => {
            decode_half_tensor(view, expected_len, raw, &mut out, f16_to_f32_ieee)?;
        }
        QuantType::BF16 => {
            decode_half_tensor(view, expected_len, raw, &mut out, bf16_to_f32_ieee)?;
        }
        _ => {
            let shape = view.shape();
            if shape.len() == 2 {
                let cols = shape[0];
                let rows = shape[1];
                if rows.checked_mul(cols) != Some(expected_len) {
                    return Err(tensor_bounds_err(
                        view,
                        format!("2D shape [{cols}, {rows}] != expected {expected_len}"),
                    ));
                }
                for r in 0..rows {
                    dequantize_token_row(view, r, &mut out[r * cols..(r + 1) * cols])?;
                }
            } else {
                dequantize_token_row(view, 0, &mut out)?;
            }
            if let Some((idx, _)) = out.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                return Err(tensor_bounds_err(
                    view,
                    format!("non-finite dequantized scalar at element {idx}"),
                ));
            }
        }
    }
    Ok(out)
}

/// Resolves and strictly decodes a mandatory `f32` vector/tensor across candidate names.
pub(crate) fn read_required_f32(
    weights: &dyn ModelWeights,
    candidates: &[&str],
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    let view = resolve_required_tensor(weights, candidates)?;
    decode_tensor_f32(&view, expected_len)
}

/// Resolves and strictly decodes an optional bias tensor across candidate names.
pub(crate) fn read_optional_f32(
    weights: &dyn ModelWeights,
    candidates: &[&str],
    expected_len: usize,
) -> Result<Option<Vec<f32>>, K3Error> {
    let Some(view) = resolve_optional_tensor(weights, candidates)? else {
        return Ok(None);
    };
    decode_tensor_f32(&view, expected_len).map(Some)
}

/// Strictly projects `input` through a mandatory weight matrix resolved from `candidates`,
/// failing closed on missing tensors, dimension mismatches, or non-finite outputs.
pub(crate) fn project_linear_strict(
    weights: &dyn ModelWeights,
    candidates: &[&str],
    input: &[f32],
    out_len: usize,
) -> Result<Vec<f32>, K3Error> {
    if out_len == 0 || input.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: candidates.first().copied().unwrap_or("linear").to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: format!(
                "linear projection requires non-zero dimensions (out_len={out_len}, in_len={})",
                input.len()
            ),
        });
    }
    let view = resolve_required_tensor(weights, candidates)?;
    let expected_elems = out_len
        .checked_mul(input.len())
        .ok_or_else(|| tensor_bounds_err(&view, "linear projection element count overflowed"))?;
    let shape = view.shape();
    if shape.len() == 2 {
        let matches = (shape[0] == out_len && shape[1] == input.len())
            || (shape[0] == input.len() && shape[1] == out_len);
        if !matches {
            return Err(tensor_bounds_err(
                &view,
                format!(
                    "projection shape [{}, {}] does not match expected [{out_len}, {}]",
                    shape[0],
                    shape[1],
                    input.len()
                ),
            ));
        }
    } else if view.num_elements() != expected_elems {
        return Err(tensor_bounds_err(
            &view,
            format!(
                "projection elements {} != expected {expected_elems}",
                view.num_elements()
            ),
        ));
    }

    let mut out = vec![0.0_f32; out_len];
    matvec_any(&mut out, &view, input)?;
    if let Some((idx, _)) = out.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(tensor_bounds_err(
            &view,
            format!("linear projection produced non-finite output at index {idx}"),
        ));
    }
    Ok(out)
}

/// Strictly adds an optional bias tensor to `target` when present in `weights`.
pub(crate) fn add_bias_if_present_strict(
    target: &mut [f32],
    weights: &dyn ModelWeights,
    candidates: &[&str],
) -> Result<(), K3Error> {
    if let Some(bias) = read_optional_f32(weights, candidates, target.len())? {
        for (dst, &b) in target.iter_mut().zip(bias.iter()) {
            *dst += b;
        }
    }
    Ok(())
}

/// Candidate tensor names for the token embedding table across GGUF and `SafeTensors`.
pub(crate) const EMBED_CANDIDATES: &[&str] = &[
    "token_embd.weight",
    "model.embed_tokens.weight",
    "backbone.embeddings.weight",
    "backbone.embedding.weight",
    "rwkv.embeddings.weight",
    "embeddings.weight",
    "embed_tokens.weight",
];

/// Candidate tensor names for the final output normalization weight across GGUF and `SafeTensors`.
pub(crate) const OUTPUT_NORM_CANDIDATES: &[&str] = &[
    "output_norm.weight",
    "model.norm.weight",
    "backbone.norm_f.weight",
    "rwkv.ln_out.weight",
    "norm.weight",
    "ln_f.weight",
];

/// Candidate tensor names for untied or tied LM head weights across GGUF and `SafeTensors`.
pub(crate) const LM_HEAD_CANDIDATES: &[&str] = &[
    "output.weight",
    "lm_head.weight",
    "head.weight",
    "token_embd.weight",
    "model.embed_tokens.weight",
    "backbone.embeddings.weight",
    "backbone.embedding.weight",
    "rwkv.embeddings.weight",
    "embeddings.weight",
    "embed_tokens.weight",
];

/// Looks up and dequantizes a token embedding row from `weights`, failing closed when the
/// embedding table is missing, mismatched, or `token_id` exceeds vocabulary row bounds.
pub(crate) fn embed_token_strict(
    token_id: u32,
    hidden_dim: usize,
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let embd = resolve_required_tensor(weights, EMBED_CANDIDATES)?;
    let token_idx = usize::try_from(token_id).map_err(|_| tensor_bounds_err(&embd, "bad token"))?;
    let row_count = embd
        .num_elements()
        .checked_div(hidden_dim)
        .filter(|&rows| rows > 0 && embd.num_elements() % hidden_dim == 0)
        .ok_or_else(|| {
            tensor_bounds_err(
                &embd,
                format!(
                    "embedding element count {} is not divisible by hidden_dim {hidden_dim}",
                    embd.num_elements()
                ),
            )
        })?;
    if token_idx >= row_count {
        return Err(tensor_bounds_err(
            &embd,
            format!("token_id {token_idx} out of bounds for embedding row count {row_count}"),
        ));
    }
    let mut hidden = vec![0.0_f32; hidden_dim];
    dequantize_token_row(&embd, token_idx, &mut hidden)?;
    if let Some((idx, _)) = hidden.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(tensor_bounds_err(
            &embd,
            format!("non-finite embedding element at index {idx} for token_id {token_idx}"),
        ));
    }
    Ok(hidden)
}

/// Projects `normed_final` through `output.weight` / `lm_head.weight` (or tied `token_embd.weight`),
/// supporting both GGUF `[hidden_dim, vocab]` and `SafeTensors` `[vocab, hidden_dim]` layouts and
/// failing closed on missing weights or out-of-bounds `logits` buffers.
pub(crate) fn project_lm_head_strict(
    weights: &dyn ModelWeights,
    normed_final: &[f32],
    logits: &mut [f32],
) -> Result<(), K3Error> {
    if logits.is_empty() {
        return Ok(());
    }
    let hidden_dim = normed_final.len();
    let view = resolve_required_tensor(weights, LM_HEAD_CANDIDATES)?;
    let shape = view.shape();
    if shape.len() == 2 {
        if shape[0] == hidden_dim && shape[1] > 0 {
            let weight_rows = shape[1];
            if logits.len() > weight_rows {
                return Err(tensor_bounds_err(
                    &view,
                    format!(
                        "logits buffer length {} exceeds LM head row count {weight_rows}",
                        logits.len()
                    ),
                ));
            }
            if logits.len() < weight_rows {
                let sub_view = view.slice_rows(0, logits.len())?;
                matvec_any(logits, &sub_view, normed_final)?;
            } else {
                matvec_any(logits, &view, normed_final)?;
            }
            return Ok(());
        }
        if shape[1] == hidden_dim && shape[0] > 0 {
            let weight_rows = shape[0];
            if logits.len() > weight_rows {
                return Err(tensor_bounds_err(
                    &view,
                    format!(
                        "logits buffer length {} exceeds LM head row count {weight_rows}",
                        logits.len()
                    ),
                ));
            }
            if logits.len() < weight_rows {
                let quant = view.quant();
                let block_elems = quant.block_elements();
                if hidden_dim % block_elems != 0 {
                    return Err(tensor_bounds_err(
                        &view,
                        format!(
                            "hidden_dim {hidden_dim} not divisible by {} block size {block_elems}",
                            quant.as_str()
                        ),
                    ));
                }
                let row_bytes = (hidden_dim / block_elems) * quant.block_bytes();
                let sub_bytes = logits.len() * row_bytes + quant.scale_trailer_bytes();
                if sub_bytes > view.as_bytes().len() {
                    return Err(tensor_bounds_err(
                        &view,
                        "row-major LM head slice exceeds tensor payload bounds",
                    ));
                }
                let sub_shape = [logits.len(), hidden_dim];
                let sub_view = TensorView::from_bytes(
                    view.name(),
                    quant,
                    &sub_shape,
                    0,
                    &view.as_bytes()[..sub_bytes],
                )?;
                matvec_any(logits, &sub_view, normed_final)?;
            } else {
                matvec_any(logits, &view, normed_final)?;
            }
            return Ok(());
        }
        return Err(tensor_bounds_err(
            &view,
            format!(
                "LM head shape [{}, {}] incompatible with hidden_dim={hidden_dim}, vocab={}",
                shape[0],
                shape[1],
                logits.len()
            ),
        ));
    }

    let expected_elems = logits
        .len()
        .checked_mul(hidden_dim)
        .ok_or_else(|| tensor_bounds_err(&view, "LM head element count overflowed"))?;
    if view.num_elements() == expected_elems {
        matvec_any(logits, &view, normed_final)?;
        return Ok(());
    }

    Err(tensor_bounds_err(
        &view,
        format!(
            "LM head elements {} != logits({}) * hidden_dim({hidden_dim})",
            view.num_elements(),
            logits.len()
        ),
    ))
}

/// Resolves a single-expert 2D matrix from either a 3D stacked `MoE` tensor or a per-expert tensor.
pub(crate) fn resolve_expert_matrix_strict<'a>(
    weights: &'a dyn ModelWeights,
    stacked_candidates: &[&str],
    indexed_candidates: &[&str],
    expert_idx: usize,
    num_experts: usize,
) -> Result<TensorView<'a>, K3Error> {
    for &stacked_name in stacked_candidates {
        if weights.contains_tensor(stacked_name) {
            let stacked = weights.tensor(stacked_name)?;
            if stacked.shape().len() == 3 {
                return stacked.slice_expert(expert_idx, num_experts);
            }
            return Err(tensor_bounds_err(
                &stacked,
                format!(
                    "expected 3D stacked MoE tensor, found shape {:?}",
                    stacked.shape()
                ),
            ));
        }
    }
    resolve_required_tensor(weights, indexed_candidates)
}
