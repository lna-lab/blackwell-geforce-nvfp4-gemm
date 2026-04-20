# Patch 4: FlashInferExperts SM120 Support

**Priority: Critical** — Required for FlashInfer CUTLASS MoE dispatch.

## Problem

`FlashInferExperts._supports_current_device()` does not recognize SM120.

## Location

`vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` (line ~131)

## Fix

```python
or p.is_device_capability(120)   # SM120 (not family — SM121 excluded)
```

Note: `is_device_capability(120)` (exact match) is used, not `is_device_capability_family(120)`. SM121 (DGX Spark) is intentionally excluded because FlashInfer 0.6.7's BF16 unquantized CUTLASS MoE GEMM lacks a Relu2 template instantiation for SM121. Fix is in upstream FlashInfer PR #2926.

## Impact

Without: `FLASHINFER_CUTLASS` MoE backend unavailable → falls to slower alternatives
With: FlashInfer CUTLASS MoE enabled
