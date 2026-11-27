"""H3 pipe save (ComfyUI custom node): sends decoded frames to ffmpeg through a pipe (no PNG files) and writes an mp4.
Inputs: images, filename_prefix (relative to the ComfyUI output folder), crf, fps, audio (optional AUDIO input).
The file goes only into the ComfyUI output folder. The node takes no file path from the workflow.
Copy this file into ComfyUI/custom_nodes. The class name and the input names are used by synctile_4k.py: do not change them."""
import os, shutil, subprocess, tempfile
import folder_paths


def checked_prefix(prefix):
    """A relative name under the ComfyUI output folder. A drive, a root, a network share or '..' stops here,
    before the first filesystem call."""
    p = str(prefix).strip().replace("\\", "/")
    if not p or p.startswith("/") or ":" in p or ".." in p.split("/"):
        raise ValueError("filename_prefix must be a relative path inside the ComfyUI output folder")
    return p


class H3BoruKaydet:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "synctile/video"}),
                             "crf": ("INT", {"default": 14, "min": 0, "max": 51}),
                             "fps": ("INT", {"default": 24, "min": 1, "max": 120})},
                "optional": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "h3"

    def save(self, images, filename_prefix, crf, fps, audio=None):
        n, h, w, _ = images.shape
        folder, base, counter, sub, _ = folder_paths.get_save_image_path(checked_prefix(filename_prefix), folder_paths.get_output_directory(), w, h)
        name = f"{base}_{counter:05}_.mp4"
        ff = shutil.which("ffmpeg")
        if not ff:
            import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        raw = None
        if audio is not None:
            wf = audio["waveform"][0].float().cpu()                      # [C, N]
            fd, raw = tempfile.mkstemp(suffix=".f32")
            with os.fdopen(fd, "wb") as f:
                f.write(wf.t().contiguous().numpy().tobytes())
            cmd += ["-f", "f32le", "-ar", str(audio["sample_rate"]), "-ac", str(wf.shape[0]), "-i", raw,
                    "-c:a", "aac", "-b:a", "192k", "-shortest_buf_duration", "120", "-shortest"]
        cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", os.path.join(folder, name)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(0, n, 8):
                p.stdin.write((images[i:i + 8].clamp(0, 1) * 255 + 0.5).byte().cpu().numpy().tobytes())
        finally:
            p.stdin.close()
            rc = p.wait()
            if raw:
                os.remove(raw)
        if rc != 0:
            raise RuntimeError("ffmpeg error")
        return {"ui": {"images": [{"filename": name, "subfolder": sub, "type": "output"}], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {"H3BoruKaydet": H3BoruKaydet}
NODE_DISPLAY_NAME_MAPPINGS = {"H3BoruKaydet": "H3 pipe save (ffmpeg mp4)"}
