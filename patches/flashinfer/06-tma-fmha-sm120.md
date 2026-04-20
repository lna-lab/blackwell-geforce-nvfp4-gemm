# Patch 6: FlashInfer TMA FMHA for SM120

**Priority: Medium** — Optimized attention kernels for SM120.

## Fix

Pre-built TensorRT-LLM FMHA kernels targeting SM120 with tile shapes tuned for 99 KB SMEM:
- `fmha_v2_flash_attention_bf16_64_128_S_q_k_v_192x128_sm120.cu`
- `fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_192x128_sm120.cu`

Location: `flashinfer/jit/attention/modules.py` (lines ~1903-1916)
