"""Video upscale with RTX Video Super Resolution (nvidia-vfx, local GPU).
Usage: python rtx_upscale.py <input.mp4> <output.mp4> <width> <height> [quality]
quality: ULTRA (default), HIGH, MEDIUM, LOW, HIGHBITRATE_ULTRA ... (names from nvvfx.QualityLevel)
Needs a python with torch (CUDA), numpy and nvvfx; ffmpeg and ffprobe on PATH."""
import sys, subprocess, json, time
import numpy as np, torch, nvvfx

gir, cik, W, H = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
kalite = sys.argv[5] if len(sys.argv) > 5 else "ULTRA"
bilgi = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,r_frame_rate", "-of", "json", gir],
                                  capture_output=True, text=True).stdout)["streams"][0]
w, h, fps = bilgi["width"], bilgi["height"], bilgi["r_frame_rate"]
oku = subprocess.Popen(["ffmpeg", "-v", "error", "-i", gir, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
yaz = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                        "-r", fps, "-i", "-", "-i", gir, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264",
                        "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", cik],
                       stdin=subprocess.PIPE)
t0, n = time.time(), 0
with nvvfx.VideoSuperRes(quality=nvvfx.QualityLevel[kalite], strength=1.0, device=0,
                         image_encoding=nvvfx.ImageEncoding.RGB8) as ef:
    ef.output_width, ef.output_height = W, H
    ef.load()
    while True:
        b = b""
        while len(b) < w * h * 3:  # a pipe read can return a partial frame
            p = oku.stdout.read(w * h * 3 - len(b))
            if not p:
                break
            b += p
        if len(b) < w * h * 3:
            break
        kare = torch.from_numpy(np.frombuffer(b, np.uint8).reshape(h, w, 3).copy()).cuda().float().div(255)
        o = torch.from_dlpack(ef.run(kare.permute(2, 0, 1).contiguous()).image).clone()
        yaz.stdin.write(o.permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy().tobytes())
        n += 1
yaz.stdin.close(); yaz.wait()
if n == 0 or yaz.returncode:
    sys.exit(f"ERROR rtx_upscale: {n} frames read from {gir}, ffmpeg writer exit code {yaz.returncode}")
print(f"DONE {cik} {n} frames {time.time() - t0:.1f} s", flush=True)
