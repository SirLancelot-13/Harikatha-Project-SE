"""Sanity check for the local TTS environment: GPU visible, a real GPU computation, imports."""
import torch

print("python-side torch", torch.__version__, "| CUDA build", torch.version.cuda, "| available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| compute capability:", torch.cuda.get_device_capability(0))
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    b = a @ a
    torch.cuda.synchronize()
    print("bf16 matmul on the GPU ok:", bool(torch.isfinite(b).all()))
    free, total = (round(x / 1e9, 1) for x in torch.cuda.mem_get_info())
    print(f"VRAM free/total: {free}/{total} GB")
else:
    print("NO GPU visible to torch: the server would fall back to the (slow) CPU")

import transformers
from parler_tts import ParlerTTSForConditionalGeneration  # noqa: F401

print("transformers", transformers.__version__, "| parler_tts imports ok")
