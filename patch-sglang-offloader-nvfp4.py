"""
Patch SGLang deepseek_v2.py offloader whitelist for NVFP4 packed weights.

Problem: OffloaderV2 whitelist expects 'w13_weight' and 'w2_weight',
but NVFP4 renames these to 'w13_weight_packed', 'w2_weight_packed' etc.

Fix: Dynamically detect which parameter names exist in the module.
"""

FILE = "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py"

with open(FILE) as f:
    content = f.read()

OLD = '''                whitelist_param_names_creator=lambda module: (
                    [
                        "w13_weight",
                        "w2_weight",
                        # only for nvfp4
                        *(
                            [
                                "w13_blockscale_swizzled",
                                "w2_blockscale_swizzled",
                            ]
                            if hasattr(module, "w13_blockscale_swizzled")
                            else []
                        ),
                    ]
                    if isinstance(module, FusedMoE)
                    else []
                ),'''

NEW = '''                whitelist_param_names_creator=lambda module: (
                    [
                        name for name in [
                            "w13_weight", "w2_weight",
                            # NVFP4 packed variants
                            "w13_weight_packed", "w2_weight_packed",
                            "w13_weight_scale", "w2_weight_scale",
                            "w13_weight_global_scale", "w2_weight_global_scale",
                            "w13_input_global_scale", "w2_input_global_scale",
                            # nvfp4 blockscale
                            "w13_blockscale_swizzled", "w2_blockscale_swizzled",
                        ]
                        if hasattr(module, name)
                    ]
                    if isinstance(module, FusedMoE)
                    else []
                ),'''

assert OLD in content, "Cannot find whitelist_param_names_creator block"
content = content.replace(OLD, NEW)

with open(FILE, "w") as f:
    f.write(content)

print("Patched: offloader whitelist now handles NVFP4 packed weight names.")
