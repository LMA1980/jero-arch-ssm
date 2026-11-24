use std::collections::BTreeMap;

use jero_arch_ssm::{
    GraniteRecipe, HybridLayerKind, HybridSsmTransformerRecipe, Lfm2MoeRecipe, Mamba2Recipe,
    RECURRENT_HEADER_FLOATS, Rwkv7Recipe, inspect_ping_pong_header,
};
use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, ModelWeights, QuantType,
    SafeTensorsFile, TensorView,
};

struct EmptyWeights;

impl ModelWeights for EmptyWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "missing".to_owned(),
        })
    }
    fn tensor_count(&self) -> usize {
        0
    }
    fn total_weight_bytes(&self) -> usize {
        0
    }
}

struct RawWeights {
    tensors: BTreeMap<String, (QuantType, Vec<usize>, Vec<u8>)>,
}

impl RawWeights {
    fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
        }
    }

    fn insert_f32(&mut self, name: &str, shape: &[usize], vals: &[f32]) {
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.tensors
            .insert(name.to_owned(), (QuantType::F32, shape.to_vec(), bytes));
    }

    fn fill_pattern(&mut self, name: &str, shape: &[usize], base: f32, step: f32) {
        let len: usize = shape.iter().product();
        let vals: Vec<f32> = (0..len)
            .map(|i| {
                let c = f32::from(u16::try_from(i % 13).unwrap_or(0)) - 6.0;
                base + c * step
            })
            .collect();
        self.insert_f32(name, shape, &vals);
    }

    fn to_safetensors(&self, arch: &str) -> Vec<u8> {
        let refs: Vec<(&str, QuantType, &[usize], &[u8])> = self
            .tensors
            .iter()
            .map(|(n, (q, s, b))| (n.as_str(), *q, s.as_slice(), b.as_slice()))
            .collect();
        create_synthetic_safetensors(&[("general.architecture", arch)], &refs)
    }

    fn to_gguf(&self, arch: &str, kvs: &[(&str, GgufValue<'_>)]) -> Vec<u8> {
        let specs: Vec<(&str, QuantType, &[usize])> = self
            .tensors
            .iter()
            .map(|(n, (q, s, _))| (n.as_str(), *q, s.as_slice()))
            .collect();
        let mut buf = create_synthetic_gguf_v3_with_kvs(arch, 32, kvs, &specs);
        let spans: Vec<(usize, usize)> = {
            let parsed = GgufFile::parse(&buf).unwrap();
            self.tensors
                .keys()
                .map(|n| {
                    let v = parsed.tensor(n).unwrap();
                    (v.offset(), v.byte_len())
                })
                .collect()
        };
        for ((_, _, raw), (off, len)) in self.tensors.values().zip(spans) {
            assert_eq!(raw.len(), len);
            buf[off..off + len].copy_from_slice(raw);
        }
        buf
    }
}

#[test]
fn integration_mamba2_gguf_vs_safetensors_trajectory_parity() {
    let mut recipe = Mamba2Recipe::new();
    recipe.num_layers = 2;
    recipe.hidden_dim = 4;
    recipe.d_inner = 8;
    recipe.d_conv = 3;
    recipe.d_state = 2;
    recipe.n_heads = 2;
    recipe.n_groups = 1;
    recipe.vocab_size = 8;

    let mut w = RawWeights::new();
    w.fill_pattern("token_embd.weight", &[8, 4], 0.2, 0.03);
    w.insert_f32("output_norm.weight", &[4], &[1.0; 4]);
    w.fill_pattern("output.weight", &[8, 4], 0.1, 0.02);
    let proj_dim = 2 * 8 + 2 * 2 + 2;
    for l in 0..2 {
        w.insert_f32(&format!("blk.{l}.attn_norm.weight"), &[4], &[1.0; 4]);
        w.fill_pattern(
            &format!("blk.{l}.ssm_in.weight"),
            &[proj_dim, 4],
            0.05,
            0.01,
        );
        w.fill_pattern(&format!("blk.{l}.ssm_conv1d.weight"), &[3, 8], 0.1, 0.02);
        w.insert_f32(&format!("blk.{l}.ssm_a"), &[2], &[-0.5, -0.3]);
        w.insert_f32(&format!("blk.{l}.ssm_d"), &[2], &[0.9, 1.1]);
        w.fill_pattern(&format!("blk.{l}.ssm_out.weight"), &[4, 8], 0.04, 0.02);
    }

    let gguf_bytes = w.to_gguf(
        "mamba2",
        &[
            ("mamba2.block_count", GgufValue::Uint32(2)),
            ("mamba2.embedding_length", GgufValue::Uint32(4)),
            ("mamba2.ssm.inner_size", GgufValue::Uint32(8)),
            ("mamba2.ssm.conv_kernel", GgufValue::Uint32(3)),
            ("mamba2.ssm.state_size", GgufValue::Uint32(2)),
            ("mamba2.ssm.time_step_rank", GgufValue::Uint32(2)),
            ("mamba2.context_length", GgufValue::Uint32(64)),
        ],
    );
    let st_bytes = w.to_safetensors("mamba2");
    let gguf = GgufFile::parse(&gguf_bytes).unwrap();
    let st = SafeTensorsFile::from_bytes(&st_bytes).unwrap();

    let mut kv_g = vec![0.0_f32; RECURRENT_HEADER_FLOATS + 2 * 2 * 32];
    let mut kv_s = vec![0.0_f32; RECURRENT_HEADER_FLOATS + 2 * 2 * 32];
    let mut logits_g = vec![0.0_f32; 8];
    let mut logits_s = vec![0.0_f32; 8];
    {
        let mut ctx_g = ExecutionContext::new(&[1_u32, 3], 64, &mut logits_g, &mut kv_g).unwrap();
        let mut ctx_s = ExecutionContext::new(&[1_u32, 3], 64, &mut logits_s, &mut kv_s).unwrap();
        recipe.prefill_step(&mut ctx_g, &gguf).unwrap();
        recipe.prefill_step(&mut ctx_s, &st).unwrap();
    }
    assert_eq!(inspect_ping_pong_header(&kv_g), Some((1, 2)));
    assert_eq!(inspect_ping_pong_header(&kv_s), Some((1, 2)));
    {
        let mut ctx_g =
            ExecutionContext::new_prefill_chunk(&[5_u32], 64, 2, true, &mut logits_g, &mut kv_g)
                .unwrap();
        let mut ctx_s =
            ExecutionContext::new_prefill_chunk(&[5_u32], 64, 2, true, &mut logits_s, &mut kv_s)
                .unwrap();
        ctx_g.prepare_decode(5).unwrap();
        ctx_s.prepare_decode(5).unwrap();
        recipe.decode_step(&mut ctx_g, &gguf).unwrap();
        recipe.decode_step(&mut ctx_s, &st).unwrap();
    }
    assert_eq!(logits_g, logits_s);
    assert_eq!(kv_g, kv_s);
}

#[test]
fn integration_lfm2moe_3d_stacked_experts_and_negative_missing() {
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
    recipe.num_experts = 2;
    recipe.num_experts_per_tok = 2;
    recipe.expert_ffn_dim = 8;
    recipe.num_dense_layers = 1;

    let mut w = RawWeights::new();
    w.fill_pattern("token_embd.weight", &[8, 4], 0.1, 0.02);
    w.insert_f32("output_norm.weight", &[4], &[1.0; 4]);
    w.fill_pattern("output.weight", &[8, 4], 0.1, 0.02);

    // Layer 0: ShortConv + dense SwiGLU
    w.insert_f32("blk.0.operator_norm.weight", &[4], &[1.0; 4]);
    w.fill_pattern("blk.0.shortconv.in_proj.weight", &[12, 4], 0.04, 0.01);
    w.fill_pattern("blk.0.shortconv.conv.weight", &[3, 4], 0.1, 0.01);
    w.fill_pattern("blk.0.shortconv.out_proj.weight", &[4, 4], 0.04, 0.01);
    w.insert_f32("blk.0.ffn_norm.weight", &[4], &[1.0; 4]);
    w.fill_pattern("blk.0.ffn_gate.weight", &[8, 4], 0.04, 0.01);
    w.fill_pattern("blk.0.ffn_up.weight", &[8, 4], 0.05, 0.01);
    w.fill_pattern("blk.0.ffn_down.weight", &[4, 8], 0.03, 0.01);

    // Layer 1: GQA + 3D stacked MoE (F32 -> block_elements=1)
    w.insert_f32("blk.1.attn_norm.weight", &[4], &[1.0; 4]);
    w.fill_pattern("blk.1.attn_q.weight", &[4, 4], 0.05, 0.01);
    w.fill_pattern("blk.1.attn_k.weight", &[2, 4], 0.04, 0.01);
    w.fill_pattern("blk.1.attn_v.weight", &[2, 4], 0.06, 0.01);
    w.fill_pattern("blk.1.attn_output.weight", &[4, 4], 0.04, 0.01);
    w.insert_f32("blk.1.ffn_norm.weight", &[4], &[1.0; 4]);
    w.fill_pattern("blk.1.ffn_gate_inp.weight", &[2, 4], 0.05, 0.02);
    w.fill_pattern("blk.1.ffn_gate_exps.weight", &[4, 8, 2], 0.03, 0.01);
    w.fill_pattern("blk.1.ffn_up_exps.weight", &[4, 8, 2], 0.04, 0.01);
    w.fill_pattern("blk.1.ffn_down_exps.weight", &[8, 4, 2], 0.03, 0.01);

    let st_bytes = w.to_safetensors("lfm2moe");
    let st = SafeTensorsFile::from_bytes(&st_bytes).unwrap();
    let mut kv = vec![0.0_f32; 256];
    let mut logits = vec![0.0_f32; 8];
    let mut ctx = ExecutionContext::new(&[0_u32, 1], 64, &mut logits, &mut kv).unwrap();
    recipe.prefill_step(&mut ctx, &st).unwrap();
    ctx.prepare_decode(2).unwrap();
    recipe.decode_step(&mut ctx, &st).unwrap();
    assert!(ctx.logits().iter().all(|v| v.is_finite()));

    // Negative: partial SafeTensors missing blk.1.ffn_down_exps.weight fails closed.
    w.tensors.remove("blk.1.ffn_down_exps.weight");
    let partial_st_bytes = w.to_safetensors("lfm2moe");
    let partial_st = SafeTensorsFile::from_bytes(&partial_st_bytes).unwrap();
    let mut ctx_partial = ExecutionContext::new(&[0_u32], 64, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        recipe.prefill_step(&mut ctx_partial, &partial_st),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let mut granite = GraniteRecipe::new();
    granite.num_layers = 1;
    granite.hidden_dim = 4;
    granite.ffn_dim = 8;
    granite.n_heads_q = 2;
    granite.n_heads_kv = 1;
    granite.head_dim = 2;
    granite.vocab_size = 8;
    let mut ctx_g = ExecutionContext::new(&[0_u32], 64, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        granite.prefill_step(&mut ctx_g, &EmptyWeights),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let hybrid = HybridSsmTransformerRecipe::from_schedule("jamba", schedule);
    let mut ctx_h = ExecutionContext::new(&[0_u32], 64, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        hybrid.prefill_step(&mut ctx_h, &EmptyWeights),
        Err(K3Error::TensorBoundsError { .. })
    ));
    let rwkv = Rwkv7Recipe::new();
    let mut ctx_r = ExecutionContext::new(&[0_u32], 64, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        rwkv.prefill_step(&mut ctx_r, &EmptyWeights),
        Err(K3Error::TensorBoundsError { .. })
    ));
}
