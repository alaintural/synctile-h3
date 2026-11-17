"""H3 pipe save (ComfyUI custom node): sends decoded frames to ffmpeg through a pipe (no PNG files) and writes an mp4.
Inputs: images, cikti = output mp4 path, ses = audio file path (optional), crf, fps.
Copy this file into ComfyUI/custom_nodes. The class name and the input names are used by synctile_4k.py: do not change them."""
import subprocess, shutil, os


class H3BoruKaydet:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "cikti": ("STRING", {"default": ""}),
                             "ses": ("STRING", {"default": ""}), "crf": ("INT", {"default": 14, "min": 0, "max": 51}),
                             "fps": ("INT", {"default": 24, "min": 1, "max": 120})}}

    RETURN_TYPES = ()
    FUNCTION = "kaydet"
    OUTPUT_NODE = True
    CATEGORY = "h3"

    def kaydet(self, images, cikti, ses, crf, fps):
        n, h, w, _ = images.shape
        ff = shutil.which("ffmpeg")
        if not ff:
            import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
        kmt = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if ses and os.path.exists(ses):
            kmt += ["-i", ses, "-c:a", "aac", "-b:a", "192k", "-shortest_buf_duration", "120", "-shortest"]
        kmt += ["-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", cikti]
        p = subprocess.Popen(kmt, stdin=subprocess.PIPE)
        for i in range(0, n, 8):
            p.stdin.write((images[i:i + 8].clamp(0, 1) * 255 + 0.5).byte().cpu().numpy().tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError("ffmpeg error")
        return {}


NODE_CLASS_MAPPINGS = {"H3BoruKaydet": H3BoruKaydet}
NODE_DISPLAY_NAME_MAPPINGS = {"H3BoruKaydet": "H3 pipe save (ffmpeg mp4)"}
