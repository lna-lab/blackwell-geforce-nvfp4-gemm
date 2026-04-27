"""
Patch SGLang compressed_tensors.py to support NVFP4A16 (W4A16) MoE.

Problem: NVFP4A16 has input_quant=None + weight strategy=tensor_group.
This doesn't match any existing MoE scheme dispatcher, causing
AttributeError on input_quant.num_bits.

Fix: Add NVFP4A16 detection before _is_fp4a4_nvfp4 in get_moe_scheme,
routing to CompressedTensorsW4A4Nvfp4MoE with use_a16=True.
"""

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py"

with open(FILE) as f:
    content = f.read()

# Add _is_fp4a16_nvfp4 method after _is_fp4a4_nvfp4
OLD_METHOD = '''    def _is_fp4a4_nvfp4(
        self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ):
        if weight_quant is None or input_quant is None:
            return False'''

NEW_METHOD = '''    def _is_fp4a16_nvfp4(
        self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ):
        """NVFP4A16: FP4 weights, BF16 activations (no input quantization)."""
        if weight_quant is None or input_quant is not None:
            return False
        is_tensor_group = weight_quant.strategy == QuantizationStrategy.TENSOR_GROUP.value
        is_float = weight_quant.type == QuantizationType.FLOAT
        is_4_bits = weight_quant.num_bits == 4
        is_group_16 = weight_quant.group_size == 16
        is_symmetric = weight_quant.symmetric
        return is_tensor_group and is_float and is_4_bits and is_group_16 and is_symmetric

    def _is_fp4a4_nvfp4(
        self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ):
        if weight_quant is None or input_quant is None:
            return False'''

assert OLD_METHOD in content, "Cannot find _is_fp4a4_nvfp4 method"
content = content.replace(OLD_METHOD, NEW_METHOD)

# Add NVFP4A16 MoE dispatch in get_moe_scheme before _is_fp4a4_nvfp4
OLD_DISPATCH = '''        elif self._is_fp4a4_nvfp4(weight_quant, input_quant):
            logger.info_once("Using CompressedTensorsW4A4Nvfp4MoE")
            return CompressedTensorsW4A4Nvfp4MoE()'''

NEW_DISPATCH = '''        elif self._is_fp4a16_nvfp4(weight_quant, input_quant):
            logger.info_once("Using CompressedTensorsW4A4Nvfp4MoE (NVFP4A16 mode)")
            return CompressedTensorsW4A4Nvfp4MoE()
        elif self._is_fp4a4_nvfp4(weight_quant, input_quant):
            logger.info_once("Using CompressedTensorsW4A4Nvfp4MoE")
            return CompressedTensorsW4A4Nvfp4MoE()'''

assert OLD_DISPATCH in content, "Cannot find MoE dispatch block"
content = content.replace(OLD_DISPATCH, NEW_DISPATCH)

with open(FILE, "w") as f:
    f.write(content)

print("Patched: added _is_fp4a16_nvfp4 + MoE dispatch for NVFP4A16.")
