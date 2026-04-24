# DeepSeek-V4-Flash / V4-Pro on SM120 (RTX PRO 6000 Blackwell / RTX 5090)

**Status: Blocked — `hyperconnection.hpp:56` assertion, no SM120 kernel available**
**Date: 2026-04-24**
**Tested**: `vllm/vllm-openai:deepseekv4-cu130` (vllm 0.1.dev15830+g8d599d76a)

## TL;DR

DeepSeek-V4-Flash fails to initialize on SM120 (workstation Blackwell) hardware
because its **mHC (Manifold-Constrained Hyper-Connections)** layer depends on
DeepGEMM's `tf32_hc_prenorm_gemm` kernel, which only has SM90 and SM100
implementations. Neither works on SM120 due to missing hardware features
(TMEM on SM100, WGMMA on SM90).

## The Block

```
RuntimeError: Worker failed with error 'Assertion error
(/workspace/.deps/deepgemm-src/csrc/apis/hyperconnection.hpp:56):
Unsupported architecture'
```

Source location: `vllm/third_party/deep_gemm/include/deep_gemm/impls/`

| File | Target | Works on SM120? |
|------|--------|-----------------|
| `sm100_tf32_hc_prenorm_gemm.cuh` | SM100 (DC Blackwell) | ❌ Uses TMEM / UMMA |
| `sm90_tf32_hc_prenorm_gemm.cuh` | SM90 (Hopper) | ❌ Uses WGMMA |
| `sm120_tf32_hc_prenorm_gemm.cuh` | — | **Does not exist** |

## Path Before the Block

The full startup sequence works up to weight loading:

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

This successfully:
1. Loads the V4 model class (`DeepseekV4ForCausalLM`)
2. Parses `deepseek_v4_fp8` quantization config
3. Establishes NCCL TP=2 (only with `NCCL_P2P_DISABLE=1` — **critical finding**,
   default P2P hangs `torch.cuda.synchronize()` indefinitely on RTX PRO 6000)
4. Loads all 46 FP8 shards (159.6 GB, ~4.8 it/s) into VRAM
5. Fails at mHC init with the assertion above

## Attempted Workarounds (all failed)

### 1. Python-level `_impl` override
```python
vllm.utils.deep_gemm._tf32_hc_prenorm_gemm_impl = pytorch_fallback
```
Result: Installed successfully (logged), but assertion still fires → another code path
in `_C.so` directly invokes the kernel, bypassing the Python wrapper.

### 2. `torch.cuda.get_device_capability` spoof (12,0) → (10,0)
Result: Assertion bypassed, but execution fails:
```
CUDA driver error: no kernel image is available for execution on the device
```
The SM100-compiled cubin cannot actually execute on SM120 hardware.

### 3. `VLLM_USE_DEEP_GEMM=0` and related env vars
Result: Disables DeepGEMM for **FP8 GEMM** path only. The mHC `tf32_hc_prenorm_gemm`
is called via a separate code path that doesn't honor these flags.

## What's Needed

**A native SM120 implementation of `tf32_hc_prenorm_gemm`**, using:
- Warp-level `mma.sync` (SM80-style, which SM120 supports natively)
- SM120-compatible TMA (async bulk GMEM→SMEM loads)
- 99 KB shared memory limit (vs SM100's 228 KB)
- No TMEM, no UMMA/tcgen05, no cluster multicast

The math is simple (one GEMM + one squared-sum), so the kernel itself should be
straightforward — it just needs someone to write it with SM120 constraints in mind.

## Environment Variables Confirmed Working

| Variable | Value | Effect |
|----------|-------|--------|
| `NCCL_P2P_DISABLE` | `1` | **Required** — without, NCCL init hangs |
| `NCCL_CUMEM_ENABLE` | `0` | Helps with memory allocator stability |
| `VLLM_USE_DEEP_GEMM` | `0` | Partial bypass (FP8 GEMM only) |
| `VLLM_MOE_USE_DEEP_GEMM` | `0` | Partial bypass |
| `VLLM_USE_DEEP_GEMM_E8M0` | `0` | Disables UE8M0 scale path |
| `HF_HUB_OFFLINE` | `1` | Skips HF Hub lookups |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `1` | Required for multiproc executor |

## Hardware

- **7x NVIDIA RTX PRO 6000 Blackwell (96GB each)** @ SM120 compute capability (12.0)
- PCIe Gen5, no NVLink (NODE-only topology)
- CUDA 13.0, driver with SM120f support

## Author

**TonoKen3 / Lna-Lab** — private research lab, ~20k downloads on HF
([sakamakismile](https://huggingface.co/sakamakismile))

Maintainer of [lna-lab/blackwell-geforce-nvfp4-gemm](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm)
(12 SM120 patches for vLLM / FlashInfer / CUTLASS, published earlier this month).
