# Patch 12: Marlin W4A8-FP8 SM120

**Priority: Low** — Only needed for W4A8-FP8 quantized models (not NVFP4).

## Fix

Allow Marlin W4A8-FP8 kernels on SM120:

```python
# vllm/model_executor/layers/quantization/utils/marlin_utils.py
) and not current_platform.is_device_capability(120):
    "Marlin W4A8-FP8 only support SM89 or SM120 device"
```

SM120 is allowed alongside SM89 (Ada Lovelace) for Marlin FP8 kernels.
