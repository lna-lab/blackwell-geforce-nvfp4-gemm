# Draft comment for DeepGEMM Issue #236
**Target**: https://github.com/deepseek-ai/DeepGEMM/issues/236
**Context**: "Feature Request: Support sm_120 (5090 and blackwell 6000 pro)"

---

## Comment body

Adding a concrete data point from today's DeepSeek-V4-Flash release (2026-04-24):

**SM120 is completely blocked on V4-family models** due to the mHC (Manifold-Constrained Hyper-Connections) layer calling `tf32_hc_prenorm_gemm`, which only has SM90 and SM100 implementations in the current DeepGEMM tree.

### Hardware
- 7x NVIDIA RTX PRO 6000 Blackwell (96GB), SM120, CUDA 13.0
- Image: `vllm/vllm-openai:deepseekv4-cu130` (vllm 0.1.dev15830)

### Exact failure
```
RuntimeError: Worker failed with error 'Assertion error
(/workspace/.deps/deepgemm-src/csrc/apis/hyperconnection.hpp:56):
Unsupported architecture'
```

The pre-mHC path works fine (NCCL with `NCCL_P2P_DISABLE=1`, FP8 weight loading,
V4 quantization config, MoE backend dispatch, attention backend). 149 GB of
FP8 weights load cleanly, then the assertion fires when mHC initializes.

### Kernel availability
Looking at `deep_gemm/include/deep_gemm/impls/`:
- `sm100_tf32_hc_prenorm_gemm.cuh` — uses TMEM, `tcgen05.mma`, cluster multicast → SM100 only
- `sm90_tf32_hc_prenorm_gemm.cuh` — uses WGMMA → SM90 only
- No `sm120_*` variant exists

### Why existing SM80-level MMA path should work

SM120 natively supports warp-level `mma.sync.aligned.m16n8k32` (SM80-era) plus
TMA (SM90-era). A naive port using:
- Warp-level MMA (no TMEM, no UMMA)
- TMA for GMEM→SMEM
- 99 KB SMEM limit (vs SM100's 228 KB)

...would be functionally correct for the `tf32_hc_prenorm_gemm` primitive
(one GEMM + one squared-sum). Not optimal for throughput, but unblocks the
entire V4 family on workstation Blackwell GPUs.

### Workarounds tried (none worked)

1. Python-level `_tf32_hc_prenorm_gemm_impl` override with a PyTorch fallback
   → Installed but assertion still fires (separate C++ call path)

2. `torch.cuda.get_device_capability` spoof to (10, 0)
   → Bypasses assertion, then fails with:
   `CUDA driver error: no kernel image is available for execution on the device`
   (SM100 cubin cannot execute on SM120)

3. All documented env vars (`VLLM_USE_DEEP_GEMM=0`, etc.)
   → Only affects the general FP8 GEMM path, not mHC.

### Ask

A minimal SM120 implementation of `tf32_hc_prenorm_gemm` (even a slow one)
would unblock the entire RTX 5090 / RTX PRO 6000 user base on V4-Flash / V4-Pro.
Happy to help test any proposed fix.

Full technical report:
https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm/blob/main/DEEPSEEK_V4_SM120_REPORT.md

— TonoKen3 / Lna-Lab
