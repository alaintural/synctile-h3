"""H3 VAE decode setting (ComfyUI custom node). Takes a VAE, applies the setting, returns the same VAE.
mode 0 = default path (fp16 accumulation off, tile overlap 64).
mode 1 = fp16 accumulation on + tile overlap 48.
mode 2 = fp16 accumulation on + tile overlap 32.
mode 3 = fp16 accumulation only, tile overlap 64 (used by synctile_4k.py).
The setting is applied on every call, so it does not stay for later jobs (mode 0 resets it).
Copy this file into ComfyUI/custom_nodes. The class name and the input names are used by synctile_4k.py: do not change them."""
import torch


class H3VAEAyar:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "mod": ("INT", {"default": 1, "min": 0, "max": 3})}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "ayarla"
    CATEGORY = "h3"

    def ayarla(self, vae, mod):
        torch.backends.cuda.matmul.allow_fp16_accumulation = mod > 0
        m = vae.first_stage_model
        if hasattr(m, "tile_overlap_min"):
            m.tile_overlap_min = {0: 64, 1: 48, 2: 32, 3: 64}[mod]
        return (vae,)


NODE_CLASS_MAPPINGS = {"H3VAEAyar": H3VAEAyar}
NODE_DISPLAY_NAME_MAPPINGS = {"H3VAEAyar": "H3 VAE decode settings"}
