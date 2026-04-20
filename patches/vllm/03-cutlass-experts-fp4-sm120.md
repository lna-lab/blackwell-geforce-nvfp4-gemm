# Patch 3: CutlassExpertsFp4 SM120 Support

**Priority: Critical** — Without this, MoE dispatch rejects SM120.

## Problem

`CutlassExpertsFp4._supports_current_device()` only checks for `family(100)` (SM100/SM101). SM120 is family 120, so the check returns False and the FP4 MoE expert path is disabled.

## Location

`vllm/model_executor/layers/fused_moe/cutlass_moe.py` (line ~684)

## Fix

```python
@staticmethod
def _supports_current_device() -> bool:
    p = current_platform
    return p.is_cuda() and (
        p.is_device_capability_family(100)
        or p.is_device_capability_family(110)
        or p.is_device_capability_family(120)  # <-- Added
    )
```

## Impact

Without: MoE falls back to Marlin or Triton → significantly slower
With: CUTLASS FP4 MoE enabled → full speed
