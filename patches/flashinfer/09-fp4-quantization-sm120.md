# Patch 9: FlashInfer FP4 Quantization JIT for SM120

**Priority: High** — Required for runtime activation quantization.

## Fix

JIT-compile TensorRT-LLM's `fp4Quantize.cpp` and `fp4Op.cpp` targeting SM120:

```python
# flashinfer/jit/fp4_quantization.py
def gen_fp4_quantization_sm120_module():
    return gen_fp4_quantization_module(sm120a_nvcc_flags, "120")

def gen_fp4_quantization_sm120f_module():
    return gen_fp4_quantization_module(sm120f_nvcc_flags, "120f")
```

Used for quantizing activations to NVFP4 before MoE GEMMs at inference time.
