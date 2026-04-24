# Patch 13: DeepGEMM HyperConnection SM120 Support (for DeepSeek-V4)

**Priority: Critical for DeepSeek-V4 family** — Without this, mHC (Manifold-Constrained Hyper-Connections) assertion fires.

## Problem

DeepSeek-V4-Flash / V4-Pro use a novel **mHC** (Manifold-Constrained Hyper-Connections) layer
that is implemented in DeepGEMM's C++ extension (`deepgemm-src/csrc/apis/hyperconnection.hpp:56`).

When launched on SM120 (RTX PRO 6000 Blackwell, RTX 5090), the kernel aborts with:

```
Assertion error (/workspace/.deps/deepgemm-src/csrc/apis/hyperconnection.hpp:56):
Unsupported architecture
```

The assertion check only whitelists SM100/SM101 (data-center Blackwell).
Unlike other DeepGEMM kernels that auto-fall-back via `_lazy_init()`,
hyperconnection is called directly from the V4 mHC forward pass and has no TileLang alternative
for the GEMM+sqrsum primitive (`tf32_hc_prenorm_gemm`).

## Symptoms

After NCCL init + weight loading (with `NCCL_P2P_DISABLE=1`):
```
RuntimeError: Worker failed with error 'Assertion error
(/workspace/.deps/deepgemm-src/csrc/apis/hyperconnection.hpp:56): Unsupported architecture'
```

## Attempted Workarounds (unsuccessful)

1. **Python `_impl` monkey-patch** — Patched `vllm.utils.deep_gemm._tf32_hc_prenorm_gemm_impl` to pure PyTorch fallback. Patch confirmed applied but same assertion still fires → the hyperconnection assertion is **in a different code path** (possibly `fp8_einsum` or mHC init).

2. **`torch.cuda.get_device_capability` spoof** — Returned (10,0) instead of (12,0). Assertion bypassed, weights loaded, but execution fails with:
   ```
   CUDA driver error: no kernel image is available for execution on the device
   ```
   → Expected: SM100-compiled cubins cannot run on SM120 hardware.

## Root Cause Analysis

`deep_gemm` is **not a separate Python package** — it is compiled directly into vLLM's
C++ extension (`_C.abi3.so`) as `/workspace/.deps/deepgemm-src`. The prebuilt vLLM docker
image `vllm/vllm-openai:deepseekv4-cu130` ships SM100-only cubins.

DeepGEMM uses JIT compilation for most kernels but `hyperconnection` appears to be
prebuilt or has a hardcoded SM list in its C++ source.

## Proposed Fix (requires source rebuild)

### Option A: Patch DeepGEMM source + vLLM rebuild

1. Clone `vllm-project/vllm` at the same commit as `deepseekv4-cu130` image
2. Apply patch to `deepgemm-src/csrc/apis/hyperconnection.hpp:56`:
   ```cpp
   // Original:
   static_assert(kSmArch == 100 || kSmArch == 101, "Unsupported architecture");
   
   // Fixed:
   static_assert(
       kSmArch == 100 || kSmArch == 101 ||
       kSmArch == 120 || kSmArch == 121,  // SM120 (RTX 5090/RTX PRO 6000), SM121 (DGX Spark)
       "Unsupported architecture"
   );
   ```
3. Build vLLM with:
   ```bash
   export TORCH_CUDA_ARCH_LIST='12.0+PTX;12.1'
   pip install -e . --no-build-isolation
   ```

### Option B: TileLang alternative for tf32_hc_prenorm_gemm

Re-implement the GEMM+sqrsum primitive in TileLang (same framework as `mhc_pre_big_fuse_tilelang`):

```python
@tilelang.jit
def tf32_hc_prenorm_gemm_tilelang(x, fn, out, sqrsum, num_split):
    # x: (M, N) bf16
    # fn: (K, N) fp32
    # out: (num_split, M, K) fp32
    # sqrsum: (num_split, M) fp32
    # Equivalent to: out[0] = x.float() @ fn.T; sqrsum[0] = x.float().square().sum(-1)
    ...
```

Then register via `direct_register_custom_op` to override the DeepGEMM call.

### Option C: PyTorch fallback (inefficient but universal)

Simple PyTorch replacement (written, tested):

```python
def _pytorch_tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_split):
    x_f = x.float()
    out[0].copy_(x_f @ fn.T)
    sqrsum[0].copy_(x_f.square().sum(-1))
    if num_split > 1:
        out[1:].zero_(); sqrsum[1:].zero_()
    return out
```

**Status**: Applied as sitecustomize.py monkey-patch — does not prevent the assertion because
another code path (likely `fp8_einsum`) hits the same DeepGEMM SM check.

## Verified Working Environment Variables

Even without Patch 13, the following env vars move past all pre-mHC checks:
```
NCCL_P2P_DISABLE=1         # critical — without this, NCCL init hangs
NCCL_CUMEM_ENABLE=0
VLLM_USE_DEEP_GEMM=0
VLLM_MOE_USE_DEEP_GEMM=0
VLLM_USE_DEEP_GEMM_E8M0=0
```

Weights load successfully. Only hyperconnection blocks final init.

## References

- DeepGEMM repo: https://github.com/deepseek-ai/DeepGEMM
- CUTLASS SM120 patches: https://github.com/NVIDIA/cutlass/issues/3096
- vLLM SM120 issue: https://github.com/vllm-project/vllm/issues/26211
- Mitko Vasilev (@iotcoi): Claims Ampere (SM86) works with 3 patches — likely same class of fix

## TODO

- [ ] Build vLLM from source with SM120 whitelisting
- [ ] Benchmark Option A vs Option C (PyTorch fallback perf)
- [ ] Contribute upstream PR to vLLM (`vllm-project/vllm`) and DeepGEMM
- [ ] Test on SGLang FP8 variant (`sgl-project/DeepSeek-V4-Flash-FP8`) to see if same path is hit
