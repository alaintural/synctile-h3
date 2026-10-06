#!/usr/bin/env python3
"""SyncTile for MiniMax H3: fast local generation and synchronized-tile 4K refinement.

Two modes, both driven through a running ComfyUI instance (HTTP API on localhost):

  fast : MiniMax H3 + 8-step distillation LoRA + int8 attention, 1344x768.
  4k   : fast base clip -> MiniMax H3 latent upscaler straight to a 3840x2160 latent
         -> SyncTile: 9 tiles re-sampled for the last 3 of 8 steps, blended in latent space
         after EVERY step (MultiDiffusion-style) -> decoded once -> optional unsharp mask.

Examples:
  python synctile_h3.py fast "A drone shot over a red-rock canyon at golden hour" --seconds 5
  python synctile_h3.py 4k   "A drone shot over a red-rock canyon at golden hour" --seconds 5
  python synctile_h3.py 4k   --prompt-file prompt.txt --name canyon --sharpen 0

See README.md for setup, benchmarks and limits.
Copyright (c) 2026 Alain Tural, Pixedi. MIT License.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# --------------------------------------------------------------------------- model files
# File names as distributed for ComfyUI. Change them here if yours differ.
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
LATENT_UPSCALER = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"

# --------------------------------------------------------------------------- settings (measured, see README)
BASE_W, BASE_H = 1344, 768
STEPS = 8
SEED = 20260927
# 8-step 'simple' schedule for MiniMax H3 (shift 12), identical to ComfyUI's simple_scheduler
SIGMAS = [1.0, 0.9882, 0.973, 0.9524, 0.9231, 0.878, 0.8, 0.6316, 0.0]
OUT_W, OUT_H = 3840, 2160
LATENT_STRIDE = 16                 # MiniMax H3 spatial downscale
TILE_W, TILE_H = 1472, 896         # pixels; 92 x 56 in latent space
START_STEP = 5                     # refine steps 5,6,7 (last 3 of 8). 4 was tested: slower, small shifts.
AUDIO_SCALE = 12.0 / 3.0           # MiniMax H3 audio_scale = shift / audio_shift
TMP = "synctile/_tmp"              # sub-folder of ComfyUI's output directory for intermediate files


class Comfy:
    def __init__(self, url, input_dir, output_dir):
        self.url = url.rstrip("/")
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def req(self, path, data=None):
        r = urllib.request.Request(self.url + path, data=json.dumps(data).encode() if data is not None else None,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=60) as c:
            return json.loads(c.read().decode() or "{}")

    def check(self):
        try:
            self.req("/system_stats")
        except Exception as e:
            sys.exit(f"ComfyUI is not reachable at {self.url} ({e}). Start ComfyUI first.")

    def run(self, graph):
        """Queue a graph, wait for it. Returns (seconds, path of the saved mp4 or None)."""
        pid = self.req("/prompt", {"prompt": graph})["prompt_id"]
        t0 = time.time()
        while True:
            time.sleep(2)
            h = self.req(f"/history/{pid}").get(pid)
            if h and h.get("status", {}).get("completed") is not None:
                if h["status"].get("status_str") != "success":
                    raise RuntimeError(json.dumps(h["status"].get("messages"))[-1500:])
                for out in h["outputs"].values():
                    for items in out.values():
                        for o in (items if isinstance(items, list) else []):
                            if isinstance(o, dict) and str(o.get("filename", "")).endswith(".mp4"):
                                return time.time() - t0, self.output_dir / o.get("subfolder", "") / o["filename"]
                return time.time() - t0, None

    def latest_latent(self, prefix):
        return max((self.output_dir / TMP).glob(f"{prefix}_*.latent"), key=lambda p: p.stat().st_mtime)

    def put_latent(self, tensor, name):
        save_file({"latent_tensor": tensor.contiguous(), "latent_format_version_0": torch.tensor([])},
                  str(self.input_dir / name))
        return name


# --------------------------------------------------------------------------- graphs
def base_graph(prompt, frames, w, h, seed):
    """MiniMax H3 text-to-video + 8-step LoRA + int8 attention. Node ids used below:
    5 conditioning/empty latent, 8 scheduler, 9 guider, 10 sampler."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                                                               "width": w, "height": h, "length": frames}},
        "L": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": LORA, "strength_model": 1.0, "model": ["1", 0]}},
        "A": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["L", 0], "attention": "comfy kitchen attention"}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "8": {"class_type": "BasicScheduler", "inputs": {"model": ["A", 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["A", 0], "conditioning": ["5", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0],
                                                                 "sigmas": ["8", 0], "latent_image": ["5", 1]}},
    }


def fast_graph(prompt, frames, seed, prefix):
    g = base_graph(prompt, frames, BASE_W, BASE_H, seed)
    g.update({
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["12", 0], "fps": 24}},
        "14": {"class_type": "SaveVideo", "inputs": {"video": ["13", 0], "filename_prefix": f"{TMP}/{prefix}",
                                                     "format": "mp4", "format.codec": "h264"}},
    })
    return g


def tile_step_graph(prompt, frames, video_in, audio_in, step, prefix):
    """One tile, one step (SIGMAS[step] -> SIGMAS[step+1]). No noise is added: the input already is that step's state."""
    g = base_graph(prompt, frames, TILE_W, TILE_H, SEED)
    g.update({
        "LV": {"class_type": "LoadLatent", "inputs": {"latent": video_in}},
        "LA": {"class_type": "LoadLatent", "inputs": {"latent": audio_in}},
        "AV": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["LV", 0], "audio_latent": ["LA", 0]}},
        "S1": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["8", 0], "step": step}},
        "S2": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["S1", 1], "step": 1}},
        "DN": {"class_type": "DisableNoise", "inputs": {}},
        "SP": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["10", 0]}},
        "SV": {"class_type": "SaveLatent", "inputs": {"samples": ["SP", 0], "filename_prefix": f"{TMP}/{prefix}_v"}},
        "SA": {"class_type": "SaveLatent", "inputs": {"samples": ["SP", 1], "filename_prefix": f"{TMP}/{prefix}_a"}},
    })
    g["7"]["inputs"]["sampler_name"] = "euler"
    g["10"]["inputs"].update({"noise": ["DN", 0], "sigmas": ["S2", 0], "latent_image": ["AV", 0]})
    g.pop("6")
    return g


# --------------------------------------------------------------------------- SyncTile
def positions(length, tile, min_overlap=16):
    n = max(1, math.ceil((length - min_overlap) / (tile - min_overlap)))
    return [round(i * (length - tile) / (n - 1)) for i in range(n)] if n > 1 else [0]


def ramp(p, tile, all_pos):
    """1D tile weight: 0->1 ramp where it overlaps a neighbour, flat at the canvas edge."""
    w = torch.ones(tile)
    before, after = [q for q in all_pos if q < p], [q for q in all_pos if q > p]
    if before:
        o = before[-1] + tile - p
        w[:o] = torch.linspace(0, 1, o + 2)[1:-1]
    if after:
        o = p + tile - after[0]
        w[-o:] = torch.minimum(w[-o:], torch.linspace(1, 0, o + 2)[1:-1])
    return w


def synctile_4k(comfy, prompt, frames, base_clip, key):
    clip_name = f"{key}_base.mp4"
    shutil.copy2(base_clip, comfy.input_dir / clip_name)
    # 1) base clip -> latent -> MiniMax H3 latent upscaler straight to 4K (no pixel upscale)
    comfy.run({
        "1": {"class_type": "LoadVideo", "inputs": {"file": clip_name}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["3", 0]}},
        "5": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
            "latent": ["4", 0], "model_name": LATENT_UPSCALER, "mode": "target dimensions",
            "mode.width": OUT_W, "mode.height": OUT_H, "align": 16, "enable_temporal_chunking": True,
            "force_unload": True, "device": "cuda", "precision": "fp16"}},
        "6": {"class_type": "SaveLatent", "inputs": {"samples": ["5", 0], "filename_prefix": f"{TMP}/{key}_up"}},
    })
    clean = load_file(str(comfy.latest_latent(f"{key}_up")))["latent_tensor"].float()      # [1,24,T,135,240]
    # 2) shape of the empty audio latent at tile size
    g = base_graph(prompt, frames, TILE_W, TILE_H, SEED)
    comfy.run({"2": g["2"], "3": g["3"], "5": g["5"],
               "SP": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["5", 1]}},
               "S": {"class_type": "SaveLatent", "inputs": {"samples": ["SP", 1], "filename_prefix": f"{TMP}/{key}_audio"}}})
    audio_shape = load_file(str(comfy.latest_latent(f"{key}_audio")))["latent_tensor"].shape
    # 3) one shared noise image for the whole canvas. ComfyUI chaining rule: an intermediate output is x/(1-s)
    #    and the next run multiplies by (1-s), so the input state is clean + s/(1-s) * noise.
    s0 = SIGMAS[START_STEP]
    x = clean + s0 / (1 - s0) * torch.randn(clean.shape, generator=torch.Generator().manual_seed(SEED))
    hl, wl = clean.shape[-2:]
    tw, th = TILE_W // LATENT_STRIDE, TILE_H // LATENT_STRIDE
    xs, ys = positions(wl, tw), positions(hl, th)
    print(f"  4K latent {tuple(clean.shape)}, {len(xs) * len(ys)} tiles", flush=True)
    audio, total = {}, 0.0
    for step in range(START_STEP, STEPS):
        acc, wsum = torch.zeros_like(x), torch.zeros(hl, wl)
        for py in ys:
            for px in xs:
                t = f"{px}_{py}"
                if t not in audio:
                    audio[t] = s0 * torch.randn(audio_shape, generator=torch.Generator().manual_seed(SEED + 7)) / (AUDIO_SCALE * (1 - s0))
                prefix = f"{key}_{step}_{t}"
                sec, _ = comfy.run(tile_step_graph(prompt, frames,
                                                   comfy.put_latent(x[..., py:py + th, px:px + tw], f"{prefix}_vin.latent"),
                                                   comfy.put_latent(audio[t], f"{prefix}_ain.latent"), step, prefix))
                total += sec
                new = load_file(str(comfy.latest_latent(f"{prefix}_v")))["latent_tensor"].float()
                audio[t] = load_file(str(comfy.latest_latent(f"{prefix}_a")))["latent_tensor"].float()
                w = ramp(py, th, ys)[:, None] * ramp(px, tw, xs)[None, :]
                acc[..., py:py + th, px:px + tw] += new * w
                wsum[py:py + th, px:px + tw] += w
        x = acc / wsum.clamp(min=1e-6)            # all tiles agree before the next step starts
        print(f"  step {step + 1}/{STEPS} done ({total:.0f} s)", flush=True)
    # 4) decode the whole canvas once (no pixel seams)
    sec, out = comfy.run({
        "1": {"class_type": "LoadLatent", "inputs": {"latent": comfy.put_latent(x, f"{key}_final.latent")}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["1", 0], "vae": ["3", 0], "tile_size": 1024,
                                                        "overlap": 128, "temporal_size": 32, "temporal_overlap": 8}},
        "5": {"class_type": "CreateVideo", "inputs": {"images": ["4", 0], "fps": 24}},
        "6": {"class_type": "SaveVideo", "inputs": {"video": ["5", 0], "filename_prefix": f"{TMP}/{key}_4k",
                                                    "format": "mp4", "format.codec": "h264"}},
    })
    return out, total + sec


# --------------------------------------------------------------------------- main
def frames_for(seconds):
    """MiniMax H3 length rule: 17n + 5 frames at 24 fps."""
    return 17 * max(1, round((seconds * 24 - 5) / 17)) + 5


def ffmpeg_bin():
    return os.environ.get("FFMPEG") or shutil.which("ffmpeg") or sys.exit("ffmpeg not found (set FFMPEG=path)")


def main():
    ap = argparse.ArgumentParser(description="SyncTile for MiniMax H3")
    ap.add_argument("mode", choices=["fast", "4k"])
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--prompt-file")
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--name", default="clip")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sharpen", type=float, default=1.0, help="unsharp mask amount on the 4K result (5x5), 0 = off")
    ap.add_argument("--out", default="synctile_output", help="folder for the finished clips")
    ap.add_argument("--comfy-url", default=os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"))
    ap.add_argument("--comfy-input", default=os.environ.get("COMFYUI_INPUT"), help="ComfyUI input folder")
    ap.add_argument("--comfy-output", default=os.environ.get("COMFYUI_OUTPUT"), help="ComfyUI output folder")
    a = ap.parse_args()
    if not a.comfy_input or not a.comfy_output:
        sys.exit("Set --comfy-input and --comfy-output (or COMFYUI_INPUT / COMFYUI_OUTPUT).")
    prompt = Path(a.prompt_file).read_text(encoding="utf-8").strip() if a.prompt_file else a.prompt
    if not prompt:
        sys.exit("No prompt.")
    frames = frames_for(a.seconds)
    if a.mode == "4k" and frames > 124:
        print(f"Note: 4K was tested up to 124 frames (5 s). {frames} frames may exceed 24 GB VRAM per tile.", flush=True)
    comfy = Comfy(a.comfy_url, a.comfy_input, a.comfy_output)
    comfy.check()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_bin()
    key = time.strftime("%Y%m%d_%H%M%S") + "_" + a.name
    print(f"[1] fast: {frames} frames, 8 steps + int8 attention", flush=True)
    sec, raw = comfy.run(fast_graph(prompt, frames, a.seed, key))
    base = out / f"{key}_fast_{BASE_W}x{BASE_H}.mp4"
    shutil.copy2(raw, base)
    print(f"    {sec:.0f} s -> {base}", flush=True)
    log = {"mode": a.mode, "prompt": prompt, "frames": frames, "seed": a.seed, "fast": str(base), "fast_s": round(sec, 1)}
    if a.mode == "4k":
        print("[2] SyncTile 4K: latent upscale + 9 synchronized tiles (last 3 of 8 steps)", flush=True)
        raw4k, sec4k = synctile_4k(comfy, prompt, frames, base, key)
        final = out / f"{key}_4k_{OUT_W}x{OUT_H}.mp4"
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw4k), "-i", str(base), "-map", "0:v", "-map", "1:a?",
                        "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", str(final)], check=True)
        log.update({"4k": str(final), "4k_s": round(sec4k, 1)})
        if a.sharpen > 0:
            sharp = out / f"{key}_4k_{OUT_W}x{OUT_H}_sharpen{a.sharpen:g}.mp4"
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(final), "-vf", f"unsharp=5:5:{a.sharpen}:5:5:0.0",
                            "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", str(sharp)],
                           check=True)
            log["4k_sharpened"] = str(sharp)
    (out / f"{key}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("DONE", log.get("4k_sharpened") or log.get("4k") or log["fast"], flush=True)


if __name__ == "__main__":
    main()
