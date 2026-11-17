"""MiniMax H3 engine helpers for SyncTile Studio: ComfyUI API, base graph, latent files, chain and fast modes.

    python h3_engine.py fast  "prompt" [--seconds 5] [--name NAME] [--seed N]
    python h3_engine.py chain "block 1\\n---\\nblock 2" [--seconds 10] [--name NAME] [--seed N] [--ref img1.jpg ...]
                              [--width 1344 --height 768] [--out out.mp4] [--scale 1280:720] [--transition latent_handoff]

fast  : one shot from a prompt, 8-step turbo LoRA + int8 attention, 1344x768 (no references, no SyncTile, no eye pass).
chain : one continuous take (H3MultishotSampler from ComfyUI-H3-Multishot), 25 steps, no LoRA. Blocks are split by '---'.
Studio calls 'chain' for scenes with "single_take". Other scripts import this file as a library.
"""
import argparse, json, math, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

import settings as S

COMFY = S.COMFY_URL
GIRDI = S.COMFY_INPUT                 # ComfyUI input folder
COMFY_CIKTI = S.MEDIA                 # ComfyUI --output-directory
ARA = "synctile/_tmp"                 # prefix for intermediate files (latents, raw videos)
FFMPEG = shutil.which("ffmpeg") or str(S.COMFY_DIR / "python_embeded" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg.exe")

# --- models and settings (measured in tests) ---
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_V = "minimax_h3_video_vae_int8_convrot.safetensors"
VAE_A = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
W, H, ADIM, SEED = 1344, 768, 8, 20260927
LAT = 16                               # H3 spatial downscale
KW, KH = 1472, 896                     # tile size in pixels (latent 92 x 56)

# Sentences added to every chain and fast prompt (in the source engine these were on by default).
HANDHELD = ("Handheld smartphone camera with a subtle constant natural shake and slight amateur micro-jitter, gentle, never jerky; no drone, no gimbal, no perfect glide. "
            "People far from the camera keep their back to the camera and never turn around. No readable text, logos or labels in focus; the camera never zooms in on text.")
MICRO = ("Natural micro-expressions on the face: an occasional soft blink, a tiny lift of one eyebrow, "
         "the corner of the lips twitching slightly, and a light, quiet breath between phrases; small, realistic and not exaggerated.")


def add_options(prompt, handheld=True, micro=True):
    """Adds the sentences before the first soundscape line, or at the end."""
    ek = " ".join(x for x, on in ((HANDHELD, handheld), (MICRO, micro)) if on)
    if not ek:
        return prompt
    i = prompt.find("\noverall_soundscape:")
    if i >= 0:
        return prompt[:i] + " " + ek + prompt[i:]
    return prompt.rstrip() + " " + ek


# ------------------------------------------------------------------ ComfyUI
def istek(yol, veri=None):
    r = urllib.request.Request(COMFY + yol, data=json.dumps(veri).encode() if veri is not None else None,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=60) as c:
        return json.loads(c.read().decode() or "{}")


def comfy_hazir():
    """Starts ComfyUI (output folder = media_dir) if it does not answer."""
    try:
        istek("/system_stats")
        return
    except Exception:
        pass
    print("Starting ComfyUI...", flush=True)
    subprocess.Popen(S.comfy_command(["--output-directory", str(COMFY_CIKTI)]),
                     cwd=S.COMFY_DIR, env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                     creationflags=S.DETACHED, close_fds=True)
    for _ in range(120):
        time.sleep(5)
        try:
            istek("/system_stats")
            return
        except Exception:
            pass
    raise RuntimeError("ComfyUI did not start in 10 minutes")


def calis(g):
    """Runs a graph and waits. Returns (seconds, mp4 path or None)."""
    pid = istek("/prompt", {"prompt": g})["prompt_id"]
    t0 = time.time()
    while True:
        time.sleep(2)
        h = istek(f"/history/{pid}").get(pid)
        if h and h.get("status", {}).get("completed") is not None:
            if h["status"].get("status_str") != "success":
                raise RuntimeError(json.dumps(h["status"].get("messages"))[:1500])
            for cikti in h["outputs"].values():
                for liste in cikti.values():
                    for o in (liste if isinstance(liste, list) else []):
                        if isinstance(o, dict) and str(o.get("filename", "")).endswith(".mp4") and o.get("type", "output") == "output":  # LoadVideo lists its input too
                            return time.time() - t0, COMFY_CIKTI / o.get("subfolder", "") / o["filename"]
            return time.time() - t0, None


def ara_glob(pattern):
    """Intermediate files: normal ComfyUI layout (output/synctile/_tmp/x) and a flat layout (output/synctile___tmp__x)."""
    flat = ARA.replace("/", "__") + "__"
    return list(COMFY_CIKTI.glob(flat + pattern)) + list((COMFY_CIKTI / ARA).glob(pattern))


def son_latent(onek):
    return max(ara_glob(f"{onek}_*.latent"), key=lambda p: p.stat().st_mtime)


def latent_yaz(t, ad):
    import torch
    from safetensors.torch import save_file
    save_file({"latent_tensor": t.contiguous(), "latent_format_version_0": torch.tensor([])}, str(GIRDI / ad))
    return ad


# ------------------------------------------------------------------ graphs
def temel(prompt, kare, w, h, seed):
    """H3 T2V + 8-step LoRA + int8 attention. Node ids: 5 condition, 8 schedule, 9 guider, 10 sampler."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_V}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_A}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                                                               "width": w, "height": h, "length": kare}},
        "L": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": LORA, "strength_model": 1.0, "model": ["1", 0]}},
        "CK": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["L", 0], "attention": "comfy kitchen attention"}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "8": {"class_type": "BasicScheduler", "inputs": {"model": ["CK", 0], "scheduler": "simple", "steps": ADIM, "denoise": 1.0}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["CK", 0], "conditioning": ["5", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0],
                                                                 "sigmas": ["8", 0], "latent_image": ["5", 1]}},
    }


def hizli_grafik(prompt, kare, seed, onek):
    g = temel(prompt, kare, W, H, seed)
    g.update({
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["12", 0], "fps": 24}},
        "14": {"class_type": "SaveVideo", "inputs": {"video": ["13", 0], "filename_prefix": f"{ARA}/{onek}",
                                                     "format": "mp4", "format.codec": "h264"}},
    })
    return g


# ------------------------------------------------------------------ SyncTile tile helpers
def konumlar(L, t, en_az=16):
    n = max(1, math.ceil((L - en_az) / (t - en_az)))
    return [round(i * (L - t) / (n - 1)) for i in range(n)] if n > 1 else [0]


def agirlik(p, t, kom):
    """1D tile weight: 0->1 ramp on the side that overlaps a neighbour, flat at the canvas edge."""
    import torch
    w = torch.ones(t)
    once, sonra = [q for q in kom if q < p], [q for q in kom if q > p]
    if once:
        o = once[-1] + t - p
        w[:o] = torch.linspace(0, 1, o + 2)[1:-1]
    if sonra:
        o = p + t - sonra[0]
        w[-o:] = torch.minimum(w[-o:], torch.linspace(1, 0, o + 2)[1:-1])
    return w


# ------------------------------------------------------------------ chain (one continuous take)
def zincir_grafik(script, kare, seed, onek, zw=W, zh=H, refler=(), gecis="latent_handoff", steps=25):
    """Seamless chain (ComfyUI-H3-Multishot): fl2va int8, no LoRA, 25 steps simple, res_multistep,
    comfy kitchen attention. Shots are split by '---' in the script; kare = frames per shot."""
    g = temel("", kare, zw, zh, seed)
    g["CK"]["inputs"]["model"] = ["1", 0]
    del g["L"]
    for n in ("5", "6", "7", "8", "9", "10"):
        g.pop(n)
    for i, y in enumerate(refler):
        y = Path(y)
        if not (GIRDI / y.name).exists():
            shutil.copy2(y, GIRDI / y.name)
        g[f"RF{i}"] = {"class_type": "LoadImage", "inputs": {"image": y.name}}
        if i:  # ImageBatch scales the second image to the size of the first
            g[f"RB{i}"] = {"class_type": "ImageBatch", "inputs": {"image1": [f"RB{i-1}" if i > 1 else "RF0", 0], "image2": [f"RF{i}", 0]}}
    g.update({
        "S": {"class_type": "H3MultishotSampler", "inputs": {
            "model": ["CK", 0], "clip": ["2", 0], "video_vae": ["3", 0], "audio_vae": ["4", 0], "script": script,
            "shot_count": 0, "width": zw, "height": zh, "frames_per_shot": kare, "seed": seed, "steps": steps,
            "seed_per_shot": True, "sampler_name": "res_multistep", "scheduler": "simple", "continuity": gecis}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["S", 0], "audio": ["S", 1], "fps": 24}},
        "14": {"class_type": "SaveVideo", "inputs": {"video": ["13", 0], "filename_prefix": f"{ARA}/{onek}",
                                                     "format": "mp4", "format.codec": "h264"}},
    })
    if refler:
        g["S"]["inputs"]["reference_images"] = [f"RB{len(refler)-1}" if len(refler) > 1 else "RF0", 0]
    return g


def kare_say(sn):
    """H3 length rule: 17n+5 frames at 24 fps."""
    return 17 * max(1, round((sn * 24 - 5) / 17)) + 5


def main():
    p = argparse.ArgumentParser(description="MiniMax H3 fast and chain modes")
    p.add_argument("mode", choices=["fast", "chain"])
    p.add_argument("prompt", nargs="?")
    p.add_argument("--prompt-file")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--name", default="video")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--ref", nargs="+", default=[], metavar="IMAGE", help="chain only: reference images (1-9), <Picture N> in the prompt")
    p.add_argument("--width", type=int, default=W)
    p.add_argument("--height", type=int, default=H)
    p.add_argument("--out", help="chain only: final mp4 path")
    p.add_argument("--scale", default="1280:720", help="chain only: final ffmpeg scale")
    p.add_argument("--transition", default="latent_handoff", help="chain only: H3MultishotSampler 'continuity' value")
    a = p.parse_args()
    prompt = Path(a.prompt_file).read_text(encoding="utf-8").strip() if a.prompt_file else a.prompt
    if not prompt:
        sys.exit("ERROR: no prompt")
    prompt = add_options(prompt)
    kare = kare_say(a.seconds)
    anahtar = time.strftime("%Y%m%d_%H%M%S") + "_" + a.name
    if len(a.ref) > 9:
        sys.exit("ERROR: at most 9 reference images")
    for r in a.ref:
        if not Path(r).is_file():
            sys.exit(f"ERROR: reference image not found: {r}")
    comfy_hazir()
    if a.mode == "chain":
        n = prompt.count("---") + 1
        print(f"chain: {n} shots x {kare} frames, 25 steps, {len(a.ref)} refs", flush=True)
        sn, yol = calis(zincir_grafik(prompt, kare, a.seed, anahtar, a.width, a.height, [str(Path(r).resolve()) for r in a.ref], a.transition))
        hedef = Path(a.out or S.MEDIA / f"{anahtar}_chain_{a.scale.replace(':', 'x')}.mp4")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(yol), "-vf", f"scale={a.scale}:flags=lanczos", "-c:v", "libx264",
                        "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", str(hedef)], check=True)
        print(f"DONE {hedef} ({sn/60:.1f} min)", flush=True)
        return
    if a.ref:
        sys.exit("ERROR: fast mode takes no reference images")
    print(f"fast: {kare} frames, 8 steps + int8 attention, {W}x{H}", flush=True)
    s1, yol = calis(hizli_grafik(prompt, kare, a.seed, anahtar))
    hedef = S.MEDIA / f"{anahtar}_fast_{W}x{H}.mp4"
    shutil.copy2(yol, hedef)
    (S.MEDIA / f"{anahtar}_fast.json").write_text(json.dumps(
        {"mode": "fast", "prompt": prompt, "frames": kare, "seed": a.seed, "video": str(hedef), "seconds": round(s1, 1)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"DONE {hedef} ({s1:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
