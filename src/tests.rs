use std::collections::BTreeMap;

use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error, ModelWeights, QuantType,
    SafeTensorsFile, TensorView,
};

use crate::arena::RECURRENT_HEADER_FLOATS;
use crate::weights::matvec_any;
use crate::{
    CRATE_LICENSE, CRATE_NAME, GraniteRecipe, HybridLayerKind, HybridSsmTransformerRecipe,
    KV_BLOCK_SIZE, Lfm2MoeRecipe, Mamba2Recipe, Rwkv7Recipe, count_hybrid_schedule_layers,
    crate_description, estimate_hybrid_recurrent_bytes_per_layer, estimate_hybrid_ssm_arena_bytes,
    hybrid_recurrent_slot_floats, inspect_ping_pong_header, lfm2_shortconv_step,
    resolve_hybrid_layer_schedule, resolve_hybrid_layer_schedule_for_metadata,
    resolve_hybrid_layer_schedule_with_metadata, select_top_k_experts,
};

struct EmptyMockWeights;

impl ModelWeights for EmptyMockWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "empty mock weights".to_owned(),
        })
    }

    fn tensor_count(&self) -> usize {
        0
    }

    fn total_weight_bytes(&self) -> usize {
        0
    }
}

struct DeterministicWeights {
    tensors: BTreeMap<String, (QuantType, Vec<usize>, Vec<u8>)>,
    total_bytes: usize,
}

impl DeterministicWeights {
    fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
            total_bytes: 0,
        }
    }

    fn insert_f32(&mut self, name: &str, shape: &[usize], values: &[f32]) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.insert_raw(name, QuantType::F32, shape, bytes);
    }

    fn insert_raw(&mut self, name: &str, quant: QuantType, shape: &[usize], bytes: Vec<u8>) {
        if let Some((_, _, old)) = self.tensors.remove(name) {
            self.total_bytes = self.total_bytes.saturating_sub(old.len());
        }
        self.total_bytes += bytes.len();
        self.tensors
            .insert(name.to_owned(), (quant, shape.to_vec(), bytes));
    }

    fn remove(&mut self, name: &str) {
        if let Some((_, _, old)) = self.tensors.remove(name) {
            self.total_bytes = self.total_bytes.saturating_sub(old.len());
        }
    }

    fn fill_pattern(&mut self, name: &str, shape: &[usize], base: f32, step: f32) {
        let len: usize = shape.iter().product();
        let vals: Vec<f32> = (0..len)
            .map(|i| {
                let centered = f32::from(u16::try_from(i % 19).unwrap_or(0)) - 9.0;
                base + centered * step
            })
            .collect();
        self.insert_f32(name, shape, &vals);
    }

    fn fill_ones(&mut self, name: &str, len: usize) {
        self.insert_f32(name, &[len], &vec![1.0_f32; len]);
    }

    fn to_safetensors_bytes(&self, meta: &[(&str, &str)]) -> Vec<u8> {
        let refs: Vec<(&str, QuantType, &[usize], &[u8])> = self
            .tensors
            .iter()
            .map(|(name, (q, shape, bytes))| {
                (name.as_str(), *q, shape.as_slice(), bytes.as_slice())
            })
            .collect();
        create_synthetic_safetensors(meta, &refs)
    }

    fn to_populated_gguf_bytes(&self, arch: &str, extra_kvs: &[(&str, GgufValue<'_>)]) -> Vec<u8> {
        let specs: Vec<(&str, QuantType, &[usize])> = self
            .tensors
            .iter()
            .map(|(name, (q, shape, _))| (name.as_str(), *q, shape.as_slice()))
            .collect();
        let mut buf = create_synthetic_gguf_v3_with_kvs(arch, 32, extra_kvs, &specs);
        let offsets: Vec<(usize, usize)> = {
            let parsed = GgufFile::parse(&buf).expect("parse scaffold gguf");
            self.tensors
                .keys()
                .map(|name| {
                    let view = parsed.tensor(name).expect("tensor present in scaffold");
                    (view.offset(), view.byte_len())
                })
                .collect()
        };
        for ((_, _, raw), (off, len)) in self.tensors.values().zip(offsets) {
            assert_eq!(raw.len(), len);
            buf[off..off + len].copy_from_slice(raw);
        }
        buf
    }
}

impl ModelWeights for DeterministicWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let (resolved_name, (quant, shape, bytes)) =
            self.tensors
                .get_key_value(name)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: self.total_bytes,
                    reason: format!("missing tensor {name}"),
                })?;
        TensorView::from_bytes(resolved_name.as_str(), *quant, shape, 0, bytes)
    }

    fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.total_bytes
    }
}

#[derive(Clone, Copy)]
struct Mamba2TestDims {
    num_layers: usize,
    hidden: usize,
    d_inner: usize,
    d_conv: usize,
    d_state: usize,
    n_heads: usize,
    n_groups: usize,
    vocab: usize,
}

fn build_mamba2_weights(d: Mamba2TestDims) -> DeterministicWeights {
    let mut w = DeterministicWeights::new();
    w.fill_pattern("token_embd.weight", &[d.vocab, d.hidden], 0.2, 0.03);
    w.fill_ones("output_norm.weight", d.hidden);
    w.fill_pattern("output.weight", &[d.vocab, d.hidden], 0.1, 0.02);
    let proj_dim = 2 * d.d_inner + 2 * d.n_groups * d.d_state + d.n_heads;
    for layer in 0..d.num_layers {
        let scale = 0.02 * f32::from(u8::try_from(layer + 1).unwrap_or(1));
        w.fill_ones(&format!("blk.{layer}.attn_norm.weight"), d.hidden);
        w.fill_pattern(
            &format!("blk.{layer}.ssm_in.weight"),
            &[proj_dim, d.hidden],
            0.05,
            scale,
        );
        w.fill_pattern(
            &format!("blk.{layer}.ssm_conv1d.weight"),
            &[d.d_conv, d.d_inner],
            0.1,
            0.01,
        );
        w.insert_f32(
            &format!("blk.{layer}.ssm_conv1d.bias"),
            &[d.d_inner],
            &vec![0.02_f32; d.d_inner],
        );
        w.insert_f32(
            &format!("blk.{layer}.ssm_dt.bias"),
            &[d.n_heads],
            &vec![0.1_f32; d.n_heads],
        );
        w.insert_f32(
            &format!("blk.{layer}.ssm_a"),
            &[d.n_heads],
            &vec![-0.4_f32; d.n_heads],
        );
        w.insert_f32(
            &format!("blk.{layer}.ssm_d"),
            &[d.n_heads],
            &vec![0.75_f32; d.n_heads],
        );
        w.fill_pattern(
            &format!("blk.{layer}.ssm_out.weight"),
            &[d.hidden, d.d_inner],
            0.04,
            scale,
        );
    }
    w
}

fn build_rwkv7_weights(
    num_layers: usize,
    hidden: usize,
    ffn_dim: usize,
    n_heads: usize,
    head_size: usize,
    vocab: usize,
) -> DeterministicWeights {
    let mut w = DeterministicWeights::new();
    let vec_dim = n_heads * head_size;
    w.fill_pattern("token_embd.weight", &[vocab, hidden], 0.15, 0.02);
    w.fill_ones("output_norm.weight", hidden);
    w.fill_pattern("output.weight", &[vocab, hidden], 0.08, 0.02);
    for layer in 0..num_layers {
        let scale = 0.015 * f32::from(u8::try_from(layer + 1).unwrap_or(1));
        w.fill_ones(&format!("blk.{layer}.attn_norm.weight"), hidden);
        w.fill_ones(&format!("blk.{layer}.ffn_norm.weight"), hidden);
        w.insert_f32(
            &format!("blk.{layer}.time_mix_x_r"),
            &[hidden],
            &vec![0.5_f32; hidden],
        );
        w.insert_f32(
            &format!("blk.{layer}.channel_mix_lerp_k"),
            &[hidden],
            &vec![0.5_f32; hidden],
        );
        for suffix in [
            "time_mix_r.weight",
            "time_mix_w.weight",
            "time_mix_k.weight",
            "time_mix_v.weight",
            "time_mix_a.weight",
            "time_mix_g.weight",
        ] {
            w.fill_pattern(
                &format!("blk.{layer}.{suffix}"),
                &[vec_dim, hidden],
                0.03,
                scale,
            );
        }
        w.fill_pattern(
            &format!("blk.{layer}.time_mix_output.weight"),
            &[hidden, vec_dim],
            0.04,
            scale,
        );
        w.fill_pattern(
            &format!("blk.{layer}.channel_mix_key.weight"),
            &[ffn_dim, hidden],
            0.03,
            scale,
        );
        w.fill_pattern(
            &format!("blk.{layer}.channel_mix_value.weight"),
            &[hidden, ffn_dim],
            0.03,
            scale,
        );
    }
    w
}

#[derive(Clone, Copy)]
struct HybridTestDims {
    hidden: usize,
    ffn_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    d_inner: usize,
    d_conv: usize,
    d_state: usize,
    ssm_heads: usize,
    ssm_groups: usize,
    l_cache: usize,
    vocab: usize,
    moe_layers: Option<(usize, usize, usize)>,
}

impl HybridTestDims {
    const fn compact(moe_layers: Option<(usize, usize, usize)>) -> Self {
        Self {
            hidden: 4,
            ffn_dim: 8,
            q_dim: 4,
            kv_dim: 2,
            d_inner: 4,
            d_conv: 2,
            d_state: 2,
            ssm_heads: 2,
            ssm_groups: 1,
            l_cache: 3,
            vocab: 8,
            moe_layers,
        }
    }
}

fn populate_hybrid_ffn(w: &mut DeterministicWeights, layer: usize, d: HybridTestDims, scale: f32) {
    w.fill_ones(&format!("blk.{layer}.ffn_norm.weight"), d.hidden);
    if let Some((num_dense, num_experts, expert_ffn_dim)) = d.moe_layers
        && layer >= num_dense
    {
        w.fill_pattern(
            &format!("blk.{layer}.ffn_gate_inp.weight"),
            &[num_experts, d.hidden],
            0.05,
            scale,
        );
        for exp in 0..num_experts {
            let exp_bias = 0.03 + 0.005 * f32::from(u8::try_from(exp).unwrap_or(0));
            w.fill_pattern(
                &format!("blk.{layer}.ffn_gate_exps.{exp}.weight"),
                &[expert_ffn_dim, d.hidden],
                exp_bias,
                scale,
            );
            w.fill_pattern(
                &format!("blk.{layer}.ffn_up_exps.{exp}.weight"),
                &[expert_ffn_dim, d.hidden],
                exp_bias,
                scale,
            );
            w.fill_pattern(
                &format!("blk.{layer}.ffn_down_exps.{exp}.weight"),
                &[d.hidden, expert_ffn_dim],
                exp_bias,
                scale,
            );
        }
    } else {
        w.fill_pattern(
            &format!("blk.{layer}.ffn_gate.weight"),
            &[d.ffn_dim, d.hidden],
            0.04,
            scale,
        );
        w.fill_pattern(
            &format!("blk.{layer}.ffn_up.weight"),
            &[d.ffn_dim, d.hidden],
            0.05,
            scale,
        );
        w.fill_pattern(
            &format!("blk.{layer}.ffn_down.weight"),
            &[d.hidden, d.ffn_dim],
            0.03,
            scale,
        );
    }
}

fn build_hybrid_weights(schedule: &[HybridLayerKind], d: HybridTestDims) -> DeterministicWeights {
    let mut w = DeterministicWeights::new();
    w.fill_pattern("token_embd.weight", &[d.vocab, d.hidden], 0.15, 0.03);
    w.fill_ones("output_norm.weight", d.hidden);
    w.fill_pattern("output.weight", &[d.vocab, d.hidden], 0.1, 0.02);

    let proj_dim = 2 * d.d_inner + 2 * d.ssm_groups * d.d_state + d.ssm_heads;
    for (layer, &kind) in schedule.iter().enumerate() {
        let scale = 0.015 * f32::from(u8::try_from(layer + 1).unwrap_or(1));
        match kind {
            HybridLayerKind::GqaAttention { .. } => {
                w.fill_ones(&format!("blk.{layer}.attn_norm.weight"), d.hidden);
                w.fill_pattern(
                    &format!("blk.{layer}.attn_q.weight"),
                    &[d.q_dim, d.hidden],
                    0.05,
                    scale,
                );
                w.fill_pattern(
                    &format!("blk.{layer}.attn_k.weight"),
                    &[d.kv_dim, d.hidden],
                    0.04,
                    scale,
                );
                w.fill_pattern(
                    &format!("blk.{layer}.attn_v.weight"),
                    &[d.kv_dim, d.hidden],
                    0.06,
                    scale,
                );
                w.fill_pattern(
                    &format!("blk.{layer}.attn_output.weight"),
                    &[d.hidden, d.q_dim],
                    0.04,
                    scale,
                );
            }
            HybridLayerKind::Mamba2Ssd { .. } => {
                w.fill_ones(&format!("blk.{layer}.attn_norm.weight"), d.hidden);
                w.fill_pattern(
                    &format!("blk.{layer}.ssm_in.weight"),
                    &[proj_dim, d.hidden],
                    0.05,
                    scale,
                );
                w.fill_pattern(
                    &format!("blk.{layer}.ssm_conv1d.weight"),
                    &[d.d_conv, d.d_inner],
                    0.1,
                    0.01,
                );
                w.insert_f32(
                    &format!("blk.{layer}.ssm_a"),
                    &[d.ssm_heads],
                    &vec![-0.4_f32; d.ssm_heads],
                );
                w.insert_f32(
                    &format!("blk.{layer}.ssm_d"),
                    &[d.ssm_heads],
                    &vec![0.8_f32; d.ssm_heads],
                );
                w.fill_pattern(
                    &format!("blk.{layer}.ssm_out.weight"),
                    &[d.hidden, d.d_inner],
                    0.04,
                    scale,
                );
            }
            HybridLayerKind::Lfm2ShortConv { .. } => {
                w.fill_ones(&format!("blk.{layer}.operator_norm.weight"), d.hidden);
                w.fill_pattern(
                    &format!("blk.{layer}.shortconv.in_proj.weight"),
                    &[3 * d.hidden, d.hidden],
                    0.04,
                    scale,
                );
                w.fill_pattern(
                    &format!("blk.{layer}.shortconv.conv.weight"),
                    &[d.l_cache, d.hidden],
                    0.12,
                    0.01,
                );
                w.fill_pattern(
                    &format!("blk.{layer}.shortconv.out_proj.weight"),
                    &[d.hidden, d.hidden],
                    0.04,
                    scale,
                );
            }
        }
        populate_hybrid_ffn(&mut w, layer, d, scale);
    }
    w
}

/// Asserts exact bit-identity natively and tight ULP parity under Miri float-jitter testing.
fn assert_parity_slices(lhs: &[f32], rhs: &[f32]) {
    #[cfg(not(miri))]
    {
        assert_eq!(lhs, rhs);
    }
    #[cfg(miri)]
    {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5,
                "parity mismatch at index {idx}: lhs={a} rhs={b}"
            );
        }
    }
}

#[test]
fn crate_metadata_and_constants() {
    assert_eq!(CRATE_NAME, "jero-arch-ssm");
    assert_eq!(
        CRATE_LICENSE,
        "LGPL-3.0-only WITH LGPL-3.0-linking-exception"
    );
    assert_ne!(crate_description(), "");
    assert_eq!(KV_BLOCK_SIZE, 16);
    assert_eq!(RECURRENT_HEADER_FLOATS, 16);
}

#[test]
fn test_mamba2_recipe_forward_and_ping_pong_arena() {
    let mut recipe = Mamba2Recipe::new();
    recipe.num_layers = 2;
    recipe.hidden_dim = 4;
    recipe.d_inner = 8;
    recipe.d_conv = 3;
    recipe.d_state = 2;
    recipe.n_heads = 2;
    recipe.n_groups = 1;
    recipe.vocab_size = 8;
    assert_eq!(recipe.architecture_id(), "mamba2");
    assert_eq!(recipe.target_layouts(), &["native", "ime_tile_8x16"]);
    assert!(recipe.supports_quant(QuantType::Q4_K_M));
    assert!(recipe.supports_quant(QuantType::Q8_0));
    assert_eq!(Mamba2Recipe::new_mamba().architecture_id(), "mamba");

    let weights = build_mamba2_weights(Mamba2TestDims {
        num_layers: 2,
        hidden: 4,
        d_inner: 8,
        d_conv: 3,
        d_state: 2,
        n_heads: 2,
        n_groups: 1,
        vocab: 8,
    });
    let mut kv_arena = vec![0.0_f32; RECURRENT_HEADER_FLOATS + 2 * 2 * 32];
    let mut logits_prefill = vec![0.0_f32; 8];
    {
        let mut ctx =
            ExecutionContext::new(&[1_u32, 3], 64, &mut logits_prefill, &mut kv_arena).unwrap();
        recipe
            .prefill_step(&mut ctx, &weights)
            .expect("Mamba2Recipe chunked prefill succeeds");
        assert_eq!(ctx.position(), 2);
        assert!(ctx.logits().iter().all(|v| v.is_finite()));
    }
    assert!(logits_prefill.iter().any(|&v| v.abs() > 1e-6));
    assert_eq!(inspect_ping_pong_header(&kv_arena), Some((1, 2)));

    let mut logits_decode = vec![0.0_f32; 8];
    {
        let mut ctx_dec = ExecutionContext::new_prefill_chunk(
            &[5_u32],
            64,
            2,
            true,
            &mut logits_decode,
            &mut kv_arena,
        )
        .unwrap();
        ctx_dec.prepare_decode(5).unwrap();
        recipe
            .decode_step(&mut ctx_dec, &weights)
            .expect("Mamba2Recipe decode step succeeds");
        assert_eq!(ctx_dec.position(), 3);
        assert!(ctx_dec.logits().iter().all(|v| v.is_finite()));
    }
    assert_eq!(inspect_ping_pong_header(&kv_arena), Some((0, 3)));
    assert_ne!(logits_prefill, logits_decode);
}

#[test]
fn test_rwkv7_recipe_forward_and_hf_safetensors_aliases() {
    let mut recipe = Rwkv7Recipe::new();
    recipe.num_layers = 2;
    recipe.hidden_dim = 4;
    recipe.ffn_dim = 8;
    recipe.head_size = 2;
    recipe.n_heads = 2;
    recipe.vocab_size = 8;
    assert_eq!(recipe.architecture_id(), "rwkv7");
    assert!(recipe.supports_quant(QuantType::Q8_0));
    assert_eq!(Rwkv7Recipe::new_rwkv6().architecture_id(), "rwkv6");

    let weights = build_rwkv7_weights(2, 4, 8, 2, 2, 8);
    let slot_floats = 2 * 4 + 2 * 2 * 2;
    let mut kv_gguf = vec![0.0_f32; RECURRENT_HEADER_FLOATS + 2 * 2 * slot_floats];
    let mut logits_gguf = vec![0.0_f32; 8];
    {
        let mut ctx =
            ExecutionContext::new(&[2_u32, 4], 64, &mut logits_gguf, &mut kv_gguf).unwrap();
        recipe.prefill_step(&mut ctx, &weights).unwrap();
        assert_eq!(ctx.position(), 2);
    }
    assert_eq!(inspect_ping_pong_header(&kv_gguf), Some((0, 2)));
    {
        let mut ctx = ExecutionContext::new_prefill_chunk(
            &[6_u32],
            64,
            2,
            true,
            &mut logits_gguf,
            &mut kv_gguf,
        )
        .unwrap();
        ctx.prepare_decode(6).unwrap();
        recipe.decode_step(&mut ctx, &weights).unwrap();
        assert_eq!(ctx.position(), 3);
    }
    assert_eq!(inspect_ping_pong_header(&kv_gguf), Some((1, 3)));
    assert!(logits_gguf.iter().all(|v| v.is_finite()));

    // Verify SafeTensorsFile with HF `rwkv.blocks.*` names yields identical output.
    let mut hf_weights = DeterministicWeights::new();
    for (name, (q, shape, bytes)) in &weights.tensors {
        let mapped = match name.as_str() {
            "token_embd.weight" => "rwkv.embeddings.weight".to_owned(),
            "output_norm.weight" => "rwkv.ln_out.weight".to_owned(),
            "output.weight" => "head.weight".to_owned(),
            other => other
                .replace("blk.", "rwkv.blocks.")
                .replace(".attn_norm.weight", ".ln1.weight")
                .replace(".ffn_norm.weight", ".ln2.weight")
                .replace(".time_mix_x_r", ".att.x_r")
                .replace(".time_mix_r.weight", ".att.receptance.weight")
                .replace(".time_mix_w.weight", ".att.w.weight")
                .replace(".time_mix_k.weight", ".att.key.weight")
                .replace(".time_mix_v.weight", ".att.value.weight")
                .replace(".time_mix_a.weight", ".att.a.weight")
                .replace(".time_mix_g.weight", ".att.gate.weight")
                .replace(".time_mix_output.weight", ".att.output.weight")
                .replace(".channel_mix_lerp_k", ".ffn.x_k")
                .replace(".channel_mix_key.weight", ".ffn.key.weight")
                .replace(".channel_mix_value.weight", ".ffn.value.weight"),
        };
        hf_weights.insert_raw(&mapped, *q, shape, bytes.clone());
    }
    let st_bytes = hf_weights.to_safetensors_bytes(&[("general.architecture", "rwkv7")]);
    let st = SafeTensorsFile::from_bytes(&st_bytes).expect("parse rwkv7 safetensors");
    let mut kv_st = vec![0.0_f32; RECURRENT_HEADER_FLOATS + 2 * 2 * slot_floats];
    let mut logits_st = vec![0.0_f32; 8];
    {
        let mut ctx = ExecutionContext::new(&[2_u32, 4], 64, &mut logits_st, &mut kv_st).unwrap();
        recipe.prefill_step(&mut ctx, &st).unwrap();
        ctx.prepare_decode(6).unwrap();
        recipe.decode_step(&mut ctx, &st).unwrap();
    }
    assert_parity_slices(&logits_gguf, &logits_st);
    assert_parity_slices(&kv_gguf, &kv_st);
}

#[test]
fn test_hybrid_ssm_lfm2_granitehybrid_and_lfm2_shortconv_step() {
    for arch in ["jamba", "zamba2", "granitehybrid", "lfm2"] {
        let schedule = if arch == "lfm2" {
            vec![
                HybridLayerKind::Lfm2ShortConv {
                    recurrent_layer_slot: 0,
                },
                HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
            ]
        } else {
            vec![
                HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: 0,
                },
                HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
            ]
        };
        let mut recipe = HybridSsmTransformerRecipe::from_schedule(arch, schedule.clone());
        recipe.hidden_dim = 4;
        recipe.ffn_dim = 8;
        recipe.n_heads_q = 2;
        recipe.n_heads_kv = 1;
        recipe.head_dim = 2;
        recipe.d_inner = 4;
        recipe.d_conv = 2;
        recipe.d_state = 2;
        recipe.shortconv_l_cache = 3;
        recipe.vocab_size = 8;

        let weights = build_hybrid_weights(&schedule, HybridTestDims::compact(None));
        let mut kv_arena = vec![0.0_f32; 256];
        let mut logits = vec![0.0_f32; 8];
        let mut ctx = ExecutionContext::new(&[1_u32, 2], 64, &mut logits, &mut kv_arena).unwrap();
        recipe.prefill_step(&mut ctx, &weights).unwrap();
        assert_eq!(ctx.position(), 2);
        ctx.prepare_decode(3).unwrap();
        recipe.decode_step(&mut ctx, &weights).unwrap();
        assert_eq!(ctx.position(), 3);
        assert!(ctx.logits().iter().all(|v| v.is_finite()));
    }

    // Direct lfm2_shortconv_step helper check.
    let mut conv_state = [0.0_f32; 4]; // dim=2, l_cache=3 -> hist=2
    let b_gate = [1.0_f32, 0.5];
    let c_gate = [0.5_f32, 1.0];
    let x_proj = [2.0_f32, -1.0];
    let conv_weight = [0.1_f32, 0.2, 0.5, 0.1, 0.2, 0.5];
    let mut out = [0.0_f32; 2];
    lfm2_shortconv_step(
        &mut conv_state,
        &b_gate,
        &c_gate,
        &x_proj,
        &conv_weight,
        Some(&[0.1, -0.1]),
        &mut out,
        2,
        3,
    )
    .expect("lfm2_shortconv_step succeeds");
    assert!(out.iter().all(|v| v.is_finite()));
    assert_eq!(conv_state, [0.0, 2.0, 0.0, -0.5]);
}

#[test]
fn test_lfm2moe_recipe_per_expert_and_3d_stacked() {
    let schedule = vec![
        HybridLayerKind::Lfm2ShortConv {
            recurrent_layer_slot: 0,
        },
        HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
    ];
    let mut recipe = Lfm2MoeRecipe::new();
    recipe.base.layer_schedule = schedule.clone();
    recipe.base.hidden_dim = 4;
    recipe.base.ffn_dim = 8;
    recipe.base.n_heads_q = 2;
    recipe.base.n_heads_kv = 1;
    recipe.base.head_dim = 2;
    recipe.base.shortconv_l_cache = 3;
    recipe.base.vocab_size = 8;
    recipe.num_experts = 4;
    recipe.num_experts_per_tok = 2;
    recipe.expert_ffn_dim = 8;
    recipe.num_dense_layers = 1;

    assert_eq!(recipe.architecture_id(), "lfm2moe");
    let weights = build_hybrid_weights(&schedule, HybridTestDims::compact(Some((1, 4, 8))));
    let mut kv_arena = vec![0.0_f32; 256];
    let mut logits = vec![0.0_f32; 8];
    let mut ctx = ExecutionContext::new(&[1_u32, 3], 64, &mut logits, &mut kv_arena).unwrap();
    recipe.prefill_step(&mut ctx, &weights).unwrap();
    ctx.prepare_decode(2).unwrap();
    recipe.decode_step(&mut ctx, &weights).unwrap();
    assert_eq!(ctx.position(), 3);
    assert!(ctx.logits().iter().all(|v| v.is_finite()));

    let top = select_top_k_experts(&[0.2, 1.5, 1.5, -0.3], 2, true);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].0, 1);
    assert_eq!(top[1].0, 2);
}

#[test]
fn test_granite_recipe_multipliers_and_gguf_safetensors_parity() {
    let mut base = GraniteRecipe::new();
    base.num_layers = 2;
    base.hidden_dim = 4;
    base.ffn_dim = 8;
    base.n_heads_q = 2;
    base.n_heads_kv = 1;
    base.head_dim = 2;
    base.vocab_size = 8;

    let schedule = vec![
        HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
        HybridLayerKind::GqaAttention { kv_layer_slot: 1 },
    ];
    let mut weights = build_hybrid_weights(&schedule, HybridTestDims::compact(None));
    weights.insert_f32("blk.0.attn_q.bias", &[4], &[0.05, -0.05, 0.02, -0.02]);
    weights.insert_f32("blk.0.ffn_down.bias", &[4], &[0.01, -0.02, 0.03, -0.01]);

    let mut scaled = base.clone();
    scaled.embedding_scale = 2.0;
    scaled.residual_scale = 0.5;
    scaled.attention_scale = Some(0.25);
    scaled.logit_scale = 4.0;

    let mut kv_a = vec![0.0_f32; 256];
    let mut kv_b = vec![0.0_f32; 256];
    let mut logits_a = vec![0.0_f32; 8];
    let mut logits_b = vec![0.0_f32; 8];
    {
        let mut ctx_a = ExecutionContext::new(&[1_u32, 2], 64, &mut logits_a, &mut kv_a).unwrap();
        let mut ctx_b = ExecutionContext::new(&[1_u32, 2], 64, &mut logits_b, &mut kv_b).unwrap();
        base.prefill_step(&mut ctx_a, &weights).unwrap();
        scaled.prefill_step(&mut ctx_b, &weights).unwrap();
        ctx_a.prepare_decode(3).unwrap();
        ctx_b.prepare_decode(3).unwrap();
        base.decode_step(&mut ctx_a, &weights).unwrap();
        scaled.decode_step(&mut ctx_b, &weights).unwrap();
    }
    assert_ne!(logits_a, logits_b);
    assert!(logits_b.iter().all(|v| v.is_finite()));

    // Verify causal dependence on preserved KV prefix.
    let mut kv_alt = vec![0.0_f32; 256];
    let mut logits_alt = vec![0.0_f32; 8];
    {
        let mut ctx_alt =
            ExecutionContext::new(&[6_u32, 2], 64, &mut logits_alt, &mut kv_alt).unwrap();
        base.prefill_step(&mut ctx_alt, &weights).unwrap();
        ctx_alt.prepare_decode(3).unwrap();
        base.decode_step(&mut ctx_alt, &weights).unwrap();
    }
    assert_ne!(logits_a, logits_alt);

    // Verify GGUF vs SafeTensors bit-identical parity on GraniteRecipe.
    let gguf_bytes = weights.to_populated_gguf_bytes(
        "granite",
        &[
            ("granite.block_count", GgufValue::Uint32(2)),
            ("granite.embedding_length", GgufValue::Uint32(4)),
            ("granite.feed_forward_length", GgufValue::Uint32(8)),
            ("granite.attention.head_count", GgufValue::Uint32(2)),
            ("granite.attention.head_count_kv", GgufValue::Uint32(1)),
        ],
    );
    let st_bytes = weights.to_safetensors_bytes(&[("general.architecture", "granite")]);
    let gguf = GgufFile::parse(&gguf_bytes).unwrap();
    let st = SafeTensorsFile::from_bytes(&st_bytes).unwrap();

    let mut kv_g = vec![0.0_f32; 256];
    let mut kv_s = vec![0.0_f32; 256];
    let mut logits_g = vec![0.0_f32; 8];
    let mut logits_s = vec![0.0_f32; 8];
    {
        let mut ctx_g = ExecutionContext::new(&[1_u32, 2], 64, &mut logits_g, &mut kv_g).unwrap();
        let mut ctx_s = ExecutionContext::new(&[1_u32, 2], 64, &mut logits_s, &mut kv_s).unwrap();
        base.prefill_step(&mut ctx_g, &gguf).unwrap();
        base.prefill_step(&mut ctx_s, &st).unwrap();
        ctx_g.prepare_decode(3).unwrap();
        ctx_s.prepare_decode(3).unwrap();
        base.decode_step(&mut ctx_g, &gguf).unwrap();
        base.decode_step(&mut ctx_s, &st).unwrap();
    }
    assert_parity_slices(&logits_g, &logits_s);
    assert_parity_slices(&kv_g, &kv_s);
}

#[test]
fn test_resolve_hybrid_layer_schedule_gguf_and_safetensors() {
    let kvs = [
        ("lfm2.block_count", GgufValue::Uint32(4)),
        ("lfm2.embedding_length", GgufValue::Uint32(4)),
        ("lfm2.attention.head_count", GgufValue::Uint32(2)),
        ("lfm2.attention.head_count_kv", GgufValue::Uint32(1)),
        ("lfm2.context_length", GgufValue::Uint32(64)),
    ];
    let specs = [
        ("token_embd.weight", QuantType::F32, &[4_usize, 8][..]),
        ("blk.0.shortconv.conv.weight", QuantType::F32, &[4, 3][..]),
        ("blk.1.ssm_in.weight", QuantType::F32, &[4, 8][..]),
        ("blk.2.attn_q.weight", QuantType::F32, &[4, 4][..]),
        (
            "blk.3.shortconv.in_proj.weight",
            QuantType::F32,
            &[4, 12][..],
        ),
    ];
    let bytes = create_synthetic_gguf_v3_with_kvs("lfm2", 64, &kvs, &specs);
    let gguf = GgufFile::parse(&bytes).expect("parse hybrid schedule gguf");
    let expected = vec![
        HybridLayerKind::Lfm2ShortConv {
            recurrent_layer_slot: 0,
        },
        HybridLayerKind::Mamba2Ssd {
            recurrent_layer_slot: 1,
        },
        HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
        HybridLayerKind::Lfm2ShortConv {
            recurrent_layer_slot: 2,
        },
    ];
    assert_eq!(resolve_hybrid_layer_schedule(4, &gguf), expected);

    let st_bytes = create_synthetic_safetensors(
        &[("general.architecture", "lfm2")],
        &[
            (
                "model.embed_tokens.weight",
                QuantType::F32,
                &[8, 4],
                &[0; 128],
            ),
            (
                "model.layers.0.conv.conv.weight",
                QuantType::F32,
                &[4, 3],
                &[0; 48],
            ),
            (
                "model.layers.1.mamba.in_proj.weight",
                QuantType::F32,
                &[8, 4],
                &[0; 128],
            ),
            (
                "model.layers.2.self_attn.q_proj.weight",
                QuantType::F32,
                &[4, 4],
                &[0; 64],
            ),
            (
                "model.layers.3.conv.in_proj.weight",
                QuantType::F32,
                &[12, 4],
                &[0; 192],
            ),
        ],
    );
    let st = SafeTensorsFile::from_bytes(&st_bytes).expect("parse hybrid schedule safetensors");
    assert_eq!(resolve_hybrid_layer_schedule(4, &st), expected);
}

#[test]
fn test_resolve_hybrid_layer_schedule_metadata_and_budgeting() {
    let meta_kvs = [
        (
            "granitehybrid.attention.head_count_kv",
            GgufValue::Array(GgufArray::Uint32(vec![0, 2, 0, 0])),
        ),
        ("granitehybrid.block_count", GgufValue::Uint32(4)),
        ("granitehybrid.embedding_length", GgufValue::Uint32(16)),
        ("granitehybrid.attention.head_count", GgufValue::Uint32(4)),
        ("granitehybrid.ssm.inner_size", GgufValue::Uint32(16)),
        ("granitehybrid.ssm.conv_kernel", GgufValue::Uint32(4)),
        ("granitehybrid.ssm.state_size", GgufValue::Uint32(4)),
        ("granitehybrid.context_length", GgufValue::Uint32(1024)),
    ];
    let m_bytes = create_synthetic_gguf_v3_with_kvs("granitehybrid", 1024, &meta_kvs, &[]);
    let m_gguf = GgufFile::parse(&m_bytes).unwrap();
    let sched = resolve_hybrid_layer_schedule_with_metadata(
        "granitehybrid",
        4,
        Some(m_gguf.metadata()),
        &m_gguf,
    );
    assert_eq!(
        sched,
        vec![
            HybridLayerKind::Mamba2Ssd {
                recurrent_layer_slot: 0
            },
            HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
            HybridLayerKind::Mamba2Ssd {
                recurrent_layer_slot: 1
            },
            HybridLayerKind::Mamba2Ssd {
                recurrent_layer_slot: 2
            },
        ]
    );
    assert_eq!(count_hybrid_schedule_layers(m_gguf.metadata()), (1, 3));
    assert_eq!(
        hybrid_recurrent_slot_floats("granitehybrid", 16, 16, 4, 4, 3),
        Some(16 * (3 + 4))
    );
    assert_eq!(
        estimate_hybrid_recurrent_bytes_per_layer(m_gguf.metadata()).unwrap(),
        16 * 7 * 4
    );
    assert!(estimate_hybrid_ssm_arena_bytes(&sched, 64, 1024, 16 * 7 * 4).unwrap() > 1024);
    assert!(matches!(
        estimate_hybrid_ssm_arena_bytes(&sched, 0, 1024, 128),
        Err(K3Error::CapabilityMismatch { .. })
    ));
    assert_eq!(
        resolve_hybrid_layer_schedule_for_metadata("lfm2", 4, None).len(),
        4
    );
}

#[test]
fn negative_empty_weights_fail_closed_across_all_five_recipes() {
    let mamba = {
        let mut r = Mamba2Recipe::new();
        r.num_layers = 1;
        r.hidden_dim = 4;
        r.d_inner = 4;
        r.d_conv = 2;
        r.d_state = 2;
        r.n_heads = 2;
        r.vocab_size = 8;
        r
    };
    let rwkv = {
        let mut r = Rwkv7Recipe::new();
        r.num_layers = 1;
        r.hidden_dim = 4;
        r.ffn_dim = 8;
        r.head_size = 2;
        r.n_heads = 2;
        r.vocab_size = 8;
        r
    };
    let hybrid = {
        let mut r = HybridSsmTransformerRecipe::from_schedule(
            "lfm2",
            vec![HybridLayerKind::Lfm2ShortConv {
                recurrent_layer_slot: 0,
            }],
        );
        r.hidden_dim = 4;
        r.ffn_dim = 8;
        r.vocab_size = 8;
        r
    };
    let moe = {
        let mut r = Lfm2MoeRecipe::new();
        r.base.layer_schedule = vec![HybridLayerKind::GqaAttention { kv_layer_slot: 0 }];
        r.base.hidden_dim = 4;
        r.base.ffn_dim = 8;
        r.base.n_heads_q = 2;
        r.base.n_heads_kv = 1;
        r.base.head_dim = 2;
        r.base.vocab_size = 8;
        r.num_dense_layers = 0;
        r
    };
    let granite = {
        let mut r = GraniteRecipe::new();
        r.num_layers = 1;
        r.hidden_dim = 4;
        r.ffn_dim = 8;
        r.n_heads_q = 2;
        r.n_heads_kv = 1;
        r.head_dim = 2;
        r.vocab_size = 8;
        r
    };

    let recipes: [&dyn ArchRecipe; 5] = [&mamba, &rwkv, &hybrid, &moe, &granite];
    for recipe in recipes {
        let mut logits = vec![0.0_f32; 8];
        let mut kv = vec![0.0_f32; 256];
        {
            let mut ctx = ExecutionContext::new(&[0_u32, 1], 64, &mut logits, &mut kv).unwrap();
            let prefill_err = recipe
                .prefill_step(&mut ctx, &EmptyMockWeights)
                .expect_err("empty weights must fail prefill");
            assert!(
                matches!(prefill_err, K3Error::TensorBoundsError { .. }),
                "{}: expected TensorBoundsError, got {prefill_err:?}",
                recipe.architecture_id()
            );
            assert_eq!(ctx.position(), 0);
        }
        assert_eq!(inspect_ping_pong_header(&kv), None);

        let mut ctx = ExecutionContext::new(&[0_u32, 1], 64, &mut logits, &mut kv).unwrap();
        ctx.prepare_decode(2).unwrap();
        let decode_err = recipe
            .decode_step(&mut ctx, &EmptyMockWeights)
            .expect_err("empty weights must fail decode");
        assert!(
            matches!(decode_err, K3Error::TensorBoundsError { .. }),
            "{}: expected TensorBoundsError, got {decode_err:?}",
            recipe.architecture_id()
        );
    }
}

#[test]
fn negative_missing_mandatory_mamba2_and_rwkv7_tensors_fail_closed() {
    let mut mamba = Mamba2Recipe::new();
    mamba.num_layers = 1;
    mamba.hidden_dim = 4;
    mamba.d_inner = 4;
    mamba.d_conv = 2;
    mamba.d_state = 2;
    mamba.n_heads = 2;
    mamba.n_groups = 1;
    mamba.vocab_size = 8;

    let mamba_dims = Mamba2TestDims {
        num_layers: 1,
        hidden: 4,
        d_inner: 4,
        d_conv: 2,
        d_state: 2,
        n_heads: 2,
        n_groups: 1,
        vocab: 8,
    };
    for missing in [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ssm_in.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_a",
        "blk.0.ssm_d",
        "blk.0.ssm_out.weight",
        "output_norm.weight",
    ] {
        let mut w = build_mamba2_weights(mamba_dims);
        w.remove(missing);
        let mut kv = vec![0.0_f32; RECURRENT_HEADER_FLOATS + 2 * 16];
        let mut logits = vec![0.0_f32; 8];
        let mut ctx = ExecutionContext::new(&[0_u32, 1], 64, &mut logits, &mut kv).unwrap();
        let err = mamba
            .prefill_step(&mut ctx, &w)
            .expect_err("missing Mamba2 tensor must fail closed");
        assert!(
            matches!(err, K3Error::TensorBoundsError { .. }),
            "missing '{missing}' returned {err:?}"
        );
        assert_eq!(inspect_ping_pong_header(&kv), None);
    }

    let mut rwkv = Rwkv7Recipe::new();
    rwkv.num_layers = 1;
    rwkv.hidden_dim = 4;
    rwkv.ffn_dim = 8;
    rwkv.head_size = 2;
    rwkv.n_heads = 2;
    rwkv.vocab_size = 8;
    for missing in [
        "blk.0.attn_norm.weight",
        "blk.0.time_mix_x_r",
        "blk.0.time_mix_r.weight",
        "blk.0.time_mix_w.weight",
        "blk.0.time_mix_a.weight",
        "blk.0.time_mix_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.channel_mix_lerp_k",
        "blk.0.channel_mix_key.weight",
        "blk.0.channel_mix_value.weight",
        "output_norm.weight",
    ] {
        let mut w = build_rwkv7_weights(1, 4, 8, 2, 2, 8);
        w.remove(missing);
        let mut kv = vec![0.0_f32; 128];
        let mut logits = vec![0.0_f32; 8];
        let mut ctx = ExecutionContext::new(&[0_u32], 64, &mut logits, &mut kv).unwrap();
        let err = rwkv
            .prefill_step(&mut ctx, &w)
            .expect_err("missing RWKV7 tensor must fail closed");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
    }
}

#[test]
fn negative_missing_mandatory_hybrid_lfm2moe_and_granite_tensors_fail_closed() {
    let schedule = vec![
        HybridLayerKind::Lfm2ShortConv {
            recurrent_layer_slot: 0,
        },
        HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
    ];
    let mut moe = Lfm2MoeRecipe::new();
    moe.base.layer_schedule = schedule.clone();
    moe.base.hidden_dim = 4;
    moe.base.ffn_dim = 8;
    moe.base.n_heads_q = 2;
    moe.base.n_heads_kv = 1;
    moe.base.head_dim = 2;
    moe.base.shortconv_l_cache = 3;
    moe.base.vocab_size = 8;
    moe.num_experts = 2;
    moe.num_experts_per_tok = 2;
    moe.expert_ffn_dim = 8;
    moe.num_dense_layers = 1;

    for missing in [
        "blk.0.operator_norm.weight",
        "blk.0.shortconv.in_proj.weight",
        "blk.0.shortconv.conv.weight",
        "blk.0.shortconv.out_proj.weight",
        "blk.1.attn_norm.weight",
        "blk.1.attn_q.weight",
        "blk.1.ffn_gate_inp.weight",
        "blk.1.ffn_gate_exps.0.weight",
    ] {
        let mut w = build_hybrid_weights(&schedule, HybridTestDims::compact(Some((1, 2, 8))));
        w.remove(missing);
        let mut kv = vec![0.0_f32; 256];
        let mut logits = vec![0.0_f32; 8];
        let mut ctx = ExecutionContext::new(&[0_u32], 64, &mut logits, &mut kv).unwrap();
        let err = moe
            .prefill_step(&mut ctx, &w)
            .expect_err("missing Lfm2Moe tensor must fail closed");
        assert!(
            matches!(err, K3Error::TensorBoundsError { .. }),
            "missing '{missing}' returned {err:?}"
        );
    }

    let mut granite = GraniteRecipe::new();
    granite.num_layers = 1;
    granite.hidden_dim = 4;
    granite.ffn_dim = 8;
    granite.n_heads_q = 2;
    granite.n_heads_kv = 1;
    granite.head_dim = 2;
    granite.vocab_size = 8;
    let g_sched = vec![HybridLayerKind::GqaAttention { kv_layer_slot: 0 }];

    for missing in [
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "output_norm.weight",
    ] {
        let mut w = build_hybrid_weights(&g_sched, HybridTestDims::compact(None));
        w.remove(missing);
        let mut kv = vec![0.0_f32; 128];
        let mut logits = vec![0.0_f32; 8];
        let mut ctx = ExecutionContext::new(&[0_u32], 64, &mut logits, &mut kv).unwrap();
        assert!(matches!(
            granite.prefill_step(&mut ctx, &w),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }
}

#[test]
fn negative_corrupted_shapes_and_nan_weights_fail_closed() {
    let mut granite = GraniteRecipe::new();
    granite.num_layers = 1;
    granite.hidden_dim = 4;
    granite.ffn_dim = 8;
    granite.n_heads_q = 2;
    granite.n_heads_kv = 1;
    granite.head_dim = 2;
    granite.vocab_size = 8;
    let g_sched = vec![HybridLayerKind::GqaAttention { kv_layer_slot: 0 }];

    let mut bad_shape = build_hybrid_weights(&g_sched, HybridTestDims::compact(None));
    bad_shape.fill_pattern("blk.0.attn_q.weight", &[2, 2], 0.1, 0.01);
    let mut kv = vec![0.0_f32; 128];
    let mut logits = vec![0.0_f32; 8];
    let mut ctx = ExecutionContext::new(&[0_u32], 64, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        granite.prefill_step(&mut ctx, &bad_shape),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let mut nan_w = build_hybrid_weights(&g_sched, HybridTestDims::compact(None));
    nan_w.insert_f32("blk.0.attn_norm.weight", &[4], &[1.0, f32::NAN, 1.0, 1.0]);
    let mut ctx_nan = ExecutionContext::new(&[0_u32], 64, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        granite.prefill_step(&mut ctx_nan, &nan_w),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn negative_out_of_bounds_token_logits_and_matvec_validation() {
    let mut mamba = Mamba2Recipe::new();
    mamba.num_layers = 1;
    mamba.hidden_dim = 4;
    mamba.d_inner = 4;
    mamba.d_conv = 2;
    mamba.d_state = 2;
    mamba.n_heads = 2;
    mamba.n_groups = 1;
    mamba.vocab_size = 8;

    let weights = build_mamba2_weights(Mamba2TestDims {
        num_layers: 1,
        hidden: 4,
        d_inner: 4,
        d_conv: 2,
        d_state: 2,
        n_heads: 2,
        n_groups: 1,
        vocab: 8,
    });
    let mut kv = vec![0.0_f32; RECURRENT_HEADER_FLOATS + 2 * 16];

    // Token ID >= vocab rows must fail closed.
    let mut logits_ok = vec![0.0_f32; 8];
    let mut ctx_bad_tok = ExecutionContext::new(&[99_u32], 64, &mut logits_ok, &mut kv).unwrap();
    assert!(matches!(
        mamba.prefill_step(&mut ctx_bad_tok, &weights),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // Logits slice > vocab rows must fail closed.
    let mut logits_too_big = vec![0.0_f32; 16];
    let mut ctx_bad_logits =
        ExecutionContext::new(&[0_u32], 64, &mut logits_too_big, &mut kv).unwrap();
    assert!(matches!(
        mamba.prefill_step(&mut ctx_bad_logits, &weights),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // Direct matvec_any rejects empty and non-finite inputs.
    let mat_bytes = vec![0_u8; 4 * 4 * 4];
    let mat_view = TensorView::from_bytes("w", QuantType::F32, &[4, 4], 0, &mat_bytes).unwrap();
    assert!(matches!(
        matvec_any(&mut [], &mat_view, &[1.0; 4]),
        Err(K3Error::TensorBoundsError { .. })
    ));
    let mut out = [0.0_f32; 4];
    assert!(matches!(
        matvec_any(&mut out, &mat_view, &[1.0, f32::INFINITY, 0.0, 0.0]),
        Err(K3Error::TensorBoundsError { .. })
    ));
}
