"""SyncTileStreamSave (ComfyUI custom node): decodes a MiniMax H3 video latent and writes the mp4 at the same time.

VAEDecodeTiled keeps all decoded frames in RAM as float32 (a 30 s 4K clip is about 70 GB). This node gives the H3 VAE
an output that sends each finished chunk to ffmpeg as 8-bit frames, so RAM use does not grow with the length of the clip.
The frames are the same frames that the normal H3 decode makes: the VAE runs its own chunks and blending, only the place
where it writes them changes.

Inputs: vae (the H3 video VAE), samples (video latent), filename_prefix (relative to the ComfyUI output folder),
fps, crf, audio (optional, the AUDIO output of the source video).
Clone this repository into ComfyUI/custom_nodes, or copy this file there, then restart ComfyUI."""
import os, shutil, subprocess, tempfile
import torch
import comfy.model_management as mm
import folder_paths


def ffmpeg_exe():
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def checked_prefix(prefix):
    """A relative name under the ComfyUI output folder. A drive, a root, a network share or '..' stops here,
    before the first filesystem call."""
    p = str(prefix).strip().replace("\\", "/")
    if not p or p.startswith("/") or ":" in p or ".." in p.split("/"):
        raise ValueError("filename_prefix must be a relative path inside the ComfyUI output folder")
    return p


class _Sink:
    """Looks like the output tensor [B, 3, T, H, W] to the H3 VAE. The VAE writes frames in order with
    out[:, :, a:b].copy_(part); each write goes to ffmpeg as rgb24."""
    def __init__(self, shape, pipe):
        self.shape, self.pipe, self.pos = shape, pipe, 0

    def __getitem__(self, idx):
        return _Slot(self, idx[2])

    def write(self, part, start):
        if start != self.pos:
            raise RuntimeError(f"H3 VAE wrote frame {start}, expected {self.pos}")
        x = part[0].clamp(0, 1).mul(255).add(0.5).byte().permute(1, 2, 3, 0).contiguous().cpu()   # [T, H, W, 3]
        self.pipe.write(x.numpy().tobytes())
        self.pos += x.shape[0]


class _Slot:
    def __init__(self, sink, sl):
        self.sink, self.sl = sink, sl

    def copy_(self, part):
        self.sink.write(part, self.sl.start)
        return self


class SyncTileStreamSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "samples": ("LATENT",),
                             "filename_prefix": ("STRING", {"default": "synctile/upscale"}),
                             "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                             "crf": ("INT", {"default": 14, "min": 0, "max": 51})},
                "optional": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "h3"

    def save(self, vae, samples, filename_prefix, fps, crf, audio=None):
        z = samples["samples"]
        if getattr(z, "is_nested", False):
            z = z.unbind()[0]
        m = vae.first_stage_model
        if z.ndim != 5 or not hasattr(m, "decode_temporal"):
            raise RuntimeError("SyncTileStreamSave needs a MiniMax H3 video latent and the H3 video VAE")
        shape = m.decode_output_shape(z.shape)
        _, _, n, h, w = shape

        folder, base, counter, sub, _ = folder_paths.get_save_image_path(checked_prefix(filename_prefix), folder_paths.get_output_directory(), w, h)
        name = f"{base}_{counter:05}_.mp4"
        out = os.path.join(folder, name)

        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
        wav = None
        if audio is not None:
            wf = audio["waveform"][0].float().cpu()                      # [C, N]
            fd, wav = tempfile.mkstemp(suffix=".f32")
            with os.fdopen(fd, "wb") as f:
                f.write(wf.t().contiguous().numpy().tobytes())
            cmd += ["-f", "f32le", "-ar", str(audio["sample_rate"]), "-ac", str(wf.shape[0]), "-i", wav, "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]

        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        sink = _Sink(shape, p.stdin)
        try:
            mm.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(z.shape, vae.vae_dtype), force_full_load=vae.disable_offload)
            with torch.inference_mode(), mm.cuda_device_context(vae.device):
                m.decode(z.to(vae.vae_dtype).to(vae.device), output_buffer=sink)
        finally:
            p.stdin.close()
            rc = p.wait()
            if wav:
                os.remove(wav)
        if rc != 0:
            raise RuntimeError(f"ffmpeg stopped with code {rc}")
        if sink.pos != n:
            raise RuntimeError(f"wrote {sink.pos} frames, expected {n}")
        return {"ui": {"images": [{"filename": name, "subfolder": sub, "type": "output"}], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {"SyncTileStreamSave": SyncTileStreamSave}
NODE_DISPLAY_NAME_MAPPINGS = {"SyncTileStreamSave": "SyncTile stream decode + save (low RAM)"}
