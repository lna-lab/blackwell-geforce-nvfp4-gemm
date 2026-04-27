"""
Patch kt_ep_wrapper.py: set layer.num_experts to num_gpu_experts
before GPU method's process_weights_after_loading.
"""

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"

with open(FILE) as f:
    content = f.read()

OLD = '''        # 1. Process GPU weights
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)'''

NEW = '''        # Override num_experts so GPU MoE kernel only sees GPU experts
        layer._original_num_experts = getattr(layer, 'num_experts', None)
        layer.num_experts = self.num_gpu_experts

        # 1. Process GPU weights
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)'''

assert OLD in content, "Cannot find process_weights_after_loading body"
content = content.replace(OLD, NEW)

with open(FILE, "w") as f:
    f.write(content)

print("Patched: layer.num_experts = num_gpu_experts before GPU weight processing.")
