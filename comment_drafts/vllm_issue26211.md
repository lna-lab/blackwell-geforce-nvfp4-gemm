# Draft comment for vLLM Issue #26211
**Target**: https://github.com/vllm-project/vllm/issues/26211
**Context**: "[Bug]: vLLM does not support DeepSeek series on RTX PRO 6000/SM120"

---

## Comment body

**DeepSeek-V4-Flash status on SM120 (2026-04-24)**

Tested the day-0 V4-Flash release on RTX PRO 6000 Blackwell (SM120) with
`vllm/vllm-openai:deepseekv4-cu130`. Wanted to share current state for other
SM120 users since many PRs have landed recently (e.g. #33416, #33417 for NVFP4
MoE family check).

### What now works on SM120

Most of the V4 stack initializes cleanly:
- ✅ `DeepseekV4ForCausalLM` model class loads (platform-agnostic)
- ✅ `deepseek_v4` tokenizer mode
- ✅ `DeepseekV4FP8Config` quantization
- ✅ MoE backend selection — `_supports_current_device()` now accepts
  `family(120)` thanks to #33417
- ✅ FP8 KV cache (`fp8_e4m3`)
- ✅ Weight loading — all 46 FP8 shards (159.6 GB) load at ~4.8 it/s
- ✅ TP=2 with `NCCL_P2P_DISABLE=1` (see "critical find" below)

### Critical finding: NCCL P2P hangs

Without `NCCL_P2P_DISABLE=1`, vLLM init freezes silently with workers at 100% CPU
but no logs. `py-spy dump` shows:

```
Thread 121 (active): "MainThread"
    synchronize (torch/cuda/streams.py:108)
    __init__ (vllm/distributed/device_communicators/pynccl.py:145)
```

`torch.cuda.synchronize()` never returns. Setting `NCCL_P2P_DISABLE=1` (and
falling back to SHM) unblocks init. Probably worth documenting this for
RTX PRO 6000 / RTX 5090 workstation class GPUs where NVLink is absent.

### What blocks

After weight loading, V4's mHC layer hits:

```
RuntimeError: Worker failed with error 'Assertion error
(/workspace/.deps/deepgemm-src/csrc/apis/hyperconnection.hpp:56):
Unsupported architecture'
```

Root cause: `vllm/third_party/deep_gemm/include/deep_gemm/impls/` has only
`sm90_tf32_hc_prenorm_gemm.cuh` and `sm100_tf32_hc_prenorm_gemm.cuh`. No SM120
variant exists. The SM100 impl uses TMEM/UMMA which SM120 doesn't have; SM90
uses WGMMA which SM120 also doesn't have.

Filed as a feature request on DeepGEMM: deepseek-ai/DeepGEMM#236

### Working startup command (partial — fails at mHC)

```bash
docker run -d --name ds-v4 \
  --gpus '"device=0,1"' \
  -v /path/to/DeepSeek-V4-Flash:/models/current:ro \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -e VLLM_USE_DEEP_GEMM_E8M0=0 \
  vllm/vllm-openai:deepseekv4-cu130 \
  --model /models/current \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
```

### Request

Two possible paths forward:

1. **Upstream DeepGEMM fix** — native SM120 `tf32_hc_prenorm_gemm` kernel
   (tracked at deepseek-ai/DeepGEMM#236)

2. **Interim TileLang fallback in vLLM** — `vllm/model_executor/layers/mhc.py`
   already has TileLang JIT kernels for the rest of the mHC path. Could the
   `tf32_hc_prenorm_gemm` GEMM+sqrsum primitive be re-implemented in TileLang
   for SM120 (and other architectures without DeepGEMM)?

Full technical report and reproducible Docker command:
https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm/blob/main/DEEPSEEK_V4_SM120_REPORT.md

— TonoKen3 / Lna-Lab
  (Maintainer of https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm —
   12 SM120 patches covering FP4 MoE, FlashInfer, Flash Attention, etc.)
