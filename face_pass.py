"""Face and motion pass for 2K and 4K: a box is cropped, sent through eye_pass.py at 1280x1280 and laid back over the frame.
Faces: every face gets its own box, so it has more pixels than in a full-frame pass and the eyes come out clean.
Motion: the box with the most motion (hands, objects) gets the same pass; it removes the blur and the blocks of fast motion.
The rest of the frame is not touched.

    python face_pass.py <input.mp4> <output.mp4>

environment: EYE_PROMPT and EYE_REFS (see eye_pass.py); the face prompt and the motion prompt are made from EYE_PROMPT
Face detector: OpenCV YuNet (face_detection_yunet_2023mar.onnx, MIT license, from opencv_zoo).
"""
import os, re, shutil, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import h3_engine as h

HERE = Path(__file__).parent
YUZ = 1280        # the face box is processed at this size
PAY = 1.15        # margin around the face
DENOISE = "0.2"   # 0.35 moves the iris
BOLGE = 1152      # side of the motion box
ESIK = 0.5        # mean motion in the box per frame (0-255); below this the shot is still and the motion pass is skipped
YUZ_OZET = "[reference generation] Close-up of the face of the person, the head fills the frame, soft light on the face."
BOLGE_OZET = "[reference generation] Close-up detail of the scene, hands and objects in motion."
BOLGE_DETAY = ("crisp detail on hands and objects, fast shutter, no motion blur.\n[Shot 1] Hands, fingers, objects and clothes stay exactly as in the video, "
               "with the same motion. Colors and background stay exactly as in the video. Camera stays the same.")


def istemler(cikti):
    """-> (face prompt file, motion prompt file): the eye prompt (EYE_PROMPT) with a summary that fits the crop."""
    gp = Path(os.environ["EYE_PROMPT"]).read_text(encoding="utf-8"); oz = r"\[reference generation\].*"
    y, b = cikti + ".face.txt", cikti + ".motion.txt"
    Path(y).write_text(re.sub(oz, YUZ_OZET, gp, count=1), encoding="utf-8")
    Path(b).write_text(re.sub(r"natural detailed eyes.*?Camera stays the same\.", lambda m: BOLGE_DETAY,
                              re.sub(oz, BOLGE_OZET, gp, count=1), flags=re.S), encoding="utf-8")
    return y, b


def kutular(video):
    """-> (frame count, [(x, y, side)]): one square box per face that holds the face in every frame."""
    v = cv2.VideoCapture(str(video)); W, H = int(v.get(3)), int(v.get(4))
    k = 1280 / max(W, H); kw, kh = round(W * k), round(H * k)
    d = cv2.FaceDetectorYN.create(str(HERE / "face_detection_yunet_2023mar.onnx"), "", (kw, kh), 0.6, 0.3, 50)
    izler, kare, ornek = [], 0, 0   # track: [x0, y0, x1, y1, hits]
    while v.grab():
        kare += 1
        if (kare - 1) % 6: continue
        ornek += 1
        _, r = d.detect(cv2.resize(v.retrieve()[1], (kw, kh)))
        for a in ([] if r is None else r):
            x0, y0, x1, y1 = a[0] / k, a[1] / k, (a[0] + a[2]) / k, (a[1] + a[3]) / k
            for i in izler:     # a face that overlaps a track belongs to it; the track box grows
                if x0 < i[2] and x1 > i[0] and y0 < i[3] and y1 > i[1]:
                    i[:] = [min(i[0], x0), min(i[1], y0), max(i[2], x1), max(i[3], y1), i[4] + 1]; break
            else:
                izler.append([x0, y0, x1, y1, 1])
    out = []
    for x0, y0, x1, y1, n in izler:
        s = min(int(max(x1 - x0, y1 - y0) * PAY), W, H) // 2 * 2
        if n < ornek / 4: continue      # a short false hit
        if not YUZ / 5 <= s < YUZ:      # too small to rebuild, or large already
            print(f"face box {s} px skipped", flush=True); continue
        x, y = (min(max(int((a + b - s) / 2), 0), L - s) // 2 * 2 for a, b, L in ((x0, x1, W), (y0, y1, H)))
        out.append((x, y, s))
    return kare, out


def hareket(video, yuzler):
    """-> [(x, y, side)]: the box with the most motion, faces left out. [] when the shot is still."""
    v = cv2.VideoCapture(str(video)); W, H = int(v.get(3)), int(v.get(4))
    k = 480 / max(W, H); s = min(BOLGE, W, H) // 2 * 2; ks = int(s * k)
    e, once, n = 0, None, 0
    while v.grab():
        g = cv2.cvtColor(cv2.resize(v.retrieve()[1], None, fx=k, fy=k, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if once is not None:
            d = cv2.absdiff(g, once); e = e + d * (d > 8); n += 1     # small differences are grain, not motion
        once = g
    if not n: return []
    for x, y, f in yuzler: e[int(y * k):int((y + f) * k), int(x * k):int((x + f) * k)] = 0
    ii = cv2.integral(e); t = ii[ks:, ks:] - ii[:-ks, ks:] - ii[ks:, :-ks] + ii[:-ks, :-ks]   # motion sum for every box position
    y, x = np.unravel_index(t.argmax(), t.shape); guc = t[y, x] / ks / ks / n
    print(f"motion {guc:.2f}", flush=True)
    if guc < ESIK: return []
    return [(min(int(x / k), W - s) // 2 * 2, min(int(y / k), H - s) // 2 * 2, s)]


def bindir(taban, yuz, x, y, s, cikti, kare):
    """Lay the new face over the frame: soft edge, low-frequency color taken from the original."""
    m = cikti + ".mask.png"; e = np.minimum(np.arange(s), s - 1 - np.arange(s))
    cv2.imwrite(m, (np.clip(np.minimum.outer(e, e) / (s / 16), 0, 1) * 255).astype(np.uint8))
    yuv = "setparams=range=tv:colorspace=bt709"   # the same tags on both inputs: without them ffmpeg converts colors in between
    b = f"gblur=sigma={s / 32:.1f}"
    subprocess.run([h.FFMPEG, "-y", "-v", "error", "-i", taban, "-i", yuz, "-loop", "1", "-i", m, "-filter_complex",
                    f"[0:v]{yuv},split[h][hk];[hk]crop={s}:{s}:{x}:{y},format=yuv444p,{b}[ob];"
                    f"[1:v]{yuv},scale={s}:{s}:flags=lanczos,format=yuv444p,split[f][fk];[fk]{b}[nb];"
                    "[ob][nb]blend=all_expr='clip(A-B+128,0,255)'[d];[f][d]blend=all_expr='clip(A+B-128,0,255)'[fc];"
                    f"[2:v]format=gray[m];[fc][m]alphamerge[fa];[h][fa]overlay={x}:{y}:format=yuv420:shortest=1[v]",
                    "-map", "[v]", "-map", "0:a?", "-c:a", "copy", "-frames:v", str(kare),
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", cikti], check=True)
    os.unlink(m)


if __name__ == "__main__":
    girdi, cikti = sys.argv[1], sys.argv[2]
    kare, ks = kutular(girdi)
    bs = hareket(girdi, ks); yp, bp = istemler(cikti)
    print(f"{len(ks)} face box(es): {ks}, motion box: {bs}", flush=True)
    simdi = girdi
    # the motion box goes first: a face that overlaps it is laid over it and keeps its own pass
    for i, (x, y, s, prompt) in enumerate([b + (bp,) for b in bs] + [b + (yp,) for b in ks], 1):
        kirp, yuz, yeni = (f"{cikti}.{t}{i}.mp4" for t in "feb")
        subprocess.run([h.FFMPEG, "-y", "-v", "error", "-i", girdi, "-vf", f"crop={s}:{s}:{x}:{y}", "-an",
                        "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", kirp], check=True)
        subprocess.run([sys.executable, str(HERE / "eye_pass.py"), kirp, "0", str(min(kare, 124) - 1), str(YUZ), str(YUZ), DENOISE],
                       cwd=HERE, check=True, env={**os.environ, "EYE_PROMPT": prompt, "EYE_OUT": yuz})
        bindir(simdi, yuz, x, y, s, yeni, kare)
        for f in (kirp, yuz) + ((simdi,) if simdi != girdi else ()): os.unlink(f)
        simdi = yeni
    shutil.move(simdi, cikti) if simdi != girdi else shutil.copy2(girdi, cikti)
    print(f"DONE {cikti}", flush=True)
