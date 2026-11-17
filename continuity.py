"""Continuity check: python continuity.py video.mp4 [output.png]
At every cut it puts the LAST 4 frames of the previous shot next to the FIRST 4 frames of the next shot (one row = one cut).
Read the image and compare item by item: object in hand, accessories, clothes, door/location state,
light/effects, body pose, background. Default output: <video>_cuts.png. Needs ffmpeg and Pillow."""
import io, re, subprocess, sys
from pathlib import Path
from PIL import Image

if len(sys.argv) < 2:
    sys.exit(__doc__)
v = sys.argv[1]; c = sys.argv[2] if len(sys.argv) > 2 else str(Path(v).with_name(Path(v).stem + "_cuts.png"))
e = subprocess.run(["ffmpeg", "-i", v, "-vf", "select='gt(scene,0.25)',showinfo", "-f", "null", "-"], capture_output=True, text=True).stderr
kes = [float(x) for x in re.findall(r"pts_time:([\d.]+)", e)]
fps = 24.0


def kare(t):
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", v, "-frames:v", "1", "-vf", "scale=300:-1",
                        "-f", "image2pipe", "-c:v", "png", "-"], capture_output=True).stdout
    return Image.open(io.BytesIO(p)).convert("RGB")


sat = []
for t in kes:
    sat.append([kare(t - k / fps) for k in (12, 6, 3, 1)] + [kare(t + k / fps) for k in (0, 2, 6, 12)])
if not sat:
    print("no cuts"); sys.exit()
w, h = sat[0][0].size; im = Image.new("RGB", (w * 8 + 20, h * len(sat)), (255, 0, 255))
for y, s in enumerate(sat):
    for x, k in enumerate(s):
        im.paste(k, (x * w + (20 if x > 3 else 0), y * h))
im.save(c); print("cuts", [round(t, 2) for t in kes], c)
