"""ComfyUI entry point. Clone this repository into ComfyUI/custom_nodes and ComfyUI loads the three nodes:
SyncTileRefine, H3VAEAyar and H3BoruKaydet. The Studio scripts (studio.py and the others) do not use this file."""
from .comfy_node_synctile import NODE_CLASS_MAPPINGS as _a, NODE_DISPLAY_NAME_MAPPINGS as _an
from .comfy_node_vae_settings import NODE_CLASS_MAPPINGS as _b, NODE_DISPLAY_NAME_MAPPINGS as _bn
from .comfy_node_pipe_save import NODE_CLASS_MAPPINGS as _c, NODE_DISPLAY_NAME_MAPPINGS as _cn

NODE_CLASS_MAPPINGS = {**_a, **_b, **_c}
NODE_DISPLAY_NAME_MAPPINGS = {**_an, **_bn, **_cn}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
