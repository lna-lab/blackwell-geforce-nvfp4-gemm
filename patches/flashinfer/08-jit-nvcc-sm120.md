# Patch 8: FlashInfer JIT NVCC Flags for SM120

**Priority: Critical** — Required for any FlashInfer JIT compilation on SM120.

## Fix

Add SM120 gencode flags to FlashInfer's JIT compilation infrastructure:

```python
# flashinfer/jit/core.py
sm120a_nvcc_flags = ["-gencode=arch=compute_120a,code=sm_120a"] + common_nvcc_flags
sm120f_nvcc_flags = ["-gencode=arch=compute_120f,code=sm_120f"] + common_nvcc_flags
sm121a_nvcc_flags = ["-gencode=arch=compute_121a,code=sm_121a"] + common_nvcc_flags
```

`sm120a` = warp-level MMA variant (standard)
`sm120f` = "fast" / FP4 variant (used for FP4 quantization kernels)
