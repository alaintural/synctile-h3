"""SyncTile pass: base latent -> MiniMax H3 latent upscaler -> synchronized tiles -> one decode.
Tail model bf16 without LoRA, one step from sigma 0.45 to 0.

    python synctile_4k.py [girdi|yuks|karo|coz ...]     (empty = girdi yuks karo coz when SYNCTILE_INPUT is set)

stages:
  girdi : SYNCTILE_INPUT=<mp4> is scaled to the base size and VAE-encoded (used for the 4K pass after the eye pass)
  yuks  : latent upscale to SYNCTILE_W x SYNCTILE_H
  karo  : synchronized tile refinement
  coz   : tiled VAE decode + mp4 (needs the custom nodes H3VAEAyar and H3BoruKaydet)
environment:
  SYNCTILE_KEY     latent name key (base latents <key>_k2v / <key>_k2a / <key>_ses)
  SYNCTILE_W/H     target size (default 3840x2160; 1080p: 1920x1088, cropped later)
  SYNCTILE_PROMPT  prompt file for the tiles (required)
  SYNCTILE_OUT     output mp4
  SYNCTILE_FRAMES  frame count for 'girdi' (default 124)
  SYNCTILE_BASE_W/H base size for 'girdi' (default 1344x768)
  SYNCTILE_SIGMA   tile sigmas, comma separated (default "0.45,0": one step)
"""
import os, subprocess, sys, time
from pathlib import Path
import torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).parent))
import h3_engine as h
import settings as S

KEY = os.environ.get("SYNCTILE_KEY", "synctile")
BF16 = "minimax_h3_fl2va_pruned_bf16.safetensors"
SIGMA_KUYRUK = [float(x) for x in os.environ.get("SYNCTILE_SIGMA", "0.45,0").split(",")]
S0 = SIGMA_KUYRUK[0]
SES_OLCEK = 4.0
VAE_MOD = 3                    # H3VAEAyar mode 3: fp16 accumulation only, tile overlap 64
OW, OH = int(os.environ.get("SYNCTILE_W", "3840")), int(os.environ.get("SYNCTILE_H", "2160"))
if OH > OW:
    h.KW, h.KH = h.KH, h.KW    # vertical 9:16: tiles are vertical too (896x1472)
FW, FH = int(os.environ.get("SYNCTILE_BASE_W", "1344")), int(os.environ.get("SYNCTILE_BASE_H", "768"))
KARE = int(os.environ.get("SYNCTILE_FRAMES", "124"))
_pf = os.environ.get("SYNCTILE_PROMPT")
if not _pf:
    raise SystemExit("ERROR: SYNCTILE_PROMPT (prompt file) is not set")
PROMPT = Path(_pf).read_text(encoding="utf-8").strip()


def oku(onek):
    return load_file(str(h.son_latent(onek)))["latent_tensor"].float()


def girdi():
    """SYNCTILE_INPUT=<mp4> -> ready video instead of a base; brought to FW x FH and KARE frames, then encoded."""
    ad = f"{KEY}_girdi.mp4"
    gv = os.environ["SYNCTILE_INPUT"]; gecici = S.WORK / ad
    probe = str(Path(h.FFMPEG).with_name("ffprobe" + Path(h.FFMPEG).suffix)) if Path(h.FFMPEG).is_absolute() else "ffprobe"
    sesli = subprocess.run([probe, "-v", "error", "-select_streams", "a", "-show_entries",
                            "stream=index", "-of", "csv=p=0", gv], capture_output=True, text=True).stdout.strip()
    subprocess.run([h.FFMPEG, "-y", "-loglevel", "error", "-i", gv, "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                    "-vf", f"scale={FW}:{FH}:flags=lanczos:force_original_aspect_ratio=increase,crop={FW}:{FH},fps=24,tpad=stop=-1:stop_mode=clone",
                    "-af", "apad", "-t", f"{KARE / 24:.4f}", "-map", "0:v", "-map", "0:a" if sesli else "1:a",
                    "-c:v", "libx264", "-crf", "10", "-c:a", "aac", str(gecici)], check=True)
    # upload through the API (the ComfyUI input folder can differ)
    subprocess.run(["curl", "-s", "-F", f"image=@{gecici}", "-F", "overwrite=true", h.COMFY + "/upload/image"], check=True, capture_output=True)
    gecici.unlink()
    t0 = time.time()
    h.calis({"1": {"class_type": "LoadVideo", "inputs": {"file": ad}},
             "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
             "3": {"class_type": "VAELoader", "inputs": {"vae_name": h.VAE_V}},
             "4": {"class_type": "VAELoader", "inputs": {"vae_name": h.VAE_A}},
             "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["3", 0]}},
             "6": {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["2", 1], "vae": ["4", 0]}},
             "SV": {"class_type": "SaveLatent", "inputs": {"samples": ["5", 0], "filename_prefix": f"{h.ARA}/{KEY}_k2v"}},
             "SA": {"class_type": "SaveLatent", "inputs": {"samples": ["6", 0], "filename_prefix": f"{h.ARA}/{KEY}_k2a"}},
             "16": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 1], "filename_prefix": f"{h.ARA}/{KEY}_ses"}}})
    print(f"input encoded {time.time() - t0:.0f} s", flush=True)


def yukselt():
    t = oku(f"{KEY}_k2v"); h.latent_yaz(t, f"{KEY}_k2v.latent")   # LoadLatent reads from the input folder
    t0 = time.time()
    h.calis({"1": {"class_type": "LoadLatent", "inputs": {"latent": f"{KEY}_k2v.latent"}},
             "5": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
                 "latent": ["1", 0], "model_name": h.UPSCALER, "mode": "target dimensions", "mode.width": OW, "mode.height": OH,
                 "align": 16, "enable_temporal_chunking": True, "force_unload": True, "device": "cuda", "precision": "fp16"}},
             "6": {"class_type": "SaveLatent", "inputs": {"samples": ["5", 0], "filename_prefix": f"{h.ARA}/{KEY}_up"}}})
    print(f"latent upscale {time.time() - t0:.0f} s", flush=True)


def karo_grafigi(vin, ain, sig_a, sig_b, onek):
    g = h.temel(PROMPT, KARE, h.KW, h.KH, h.SEED)
    for n in ("L", "CK", "6"):
        g.pop(n)
    g["1"]["inputs"]["unet_name"] = BF16
    g["9"]["inputs"]["model"] = ["1", 0]
    g["8"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": f"{sig_a}, {sig_b}"}}
    g["7"]["inputs"]["sampler_name"] = "euler"
    g.update({
        "LV": {"class_type": "LoadLatent", "inputs": {"latent": vin}},
        "LA": {"class_type": "LoadLatent", "inputs": {"latent": ain}},
        "BR": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["LV", 0], "audio_latent": ["LA", 0]}},
        "DN": {"class_type": "DisableNoise", "inputs": {}},
        "AY": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["10", 0]}},
        "SV": {"class_type": "SaveLatent", "inputs": {"samples": ["AY", 0], "filename_prefix": f"{h.ARA}/{onek}_v"}},
        "SA": {"class_type": "SaveLatent", "inputs": {"samples": ["AY", 1], "filename_prefix": f"{h.ARA}/{onek}_a"}},
    })
    g["10"]["inputs"].update({"noise": ["DN", 0], "latent_image": ["BR", 0]})
    return g


def karolar():
    c = oku(f"{KEY}_up")                      # clean upscaled latent
    ak = oku(f"{KEY}_k2a")                    # base audio latent (clean)
    gen = torch.Generator().manual_seed(h.SEED)
    x = c + S0 / (1 - S0) * torch.randn(c.shape, generator=gen)   # one shared noise for the whole canvas
    Hl, Wl = c.shape[-2:]; tw, th = h.KW // h.LAT, h.KH // h.LAT
    xs, ys = h.konumlar(Wl, tw), h.konumlar(Hl, th)
    print(f"latent {tuple(c.shape)}, {len(xs) * len(ys)} tiles, audio {tuple(ak.shape)}", flush=True)
    ses = {}
    for a in range(len(SIGMA_KUYRUK) - 1):
        acc, top = torch.zeros_like(x), torch.zeros(Hl, Wl); ts = 0.0
        for py in ys:
            for px in xs:
                k = f"{px}_{py}"
                if k not in ses:
                    ses[k] = ak + S0 * torch.randn(ak.shape, generator=torch.Generator().manual_seed(h.SEED + 7)) / (SES_OLCEK * (1 - S0))
                onek = f"{KEY}_{a}_{k}"
                s, _ = h.calis(karo_grafigi(h.latent_yaz(x[..., py:py + th, px:px + tw], f"{onek}_vin.latent"),
                                            h.latent_yaz(ses[k], f"{onek}_ain.latent"), SIGMA_KUYRUK[a], SIGMA_KUYRUK[a + 1], onek))
                ts += s
                yeni = oku(f"{onek}_v"); ses[k] = oku(f"{onek}_a")
                w = h.agirlik(py, th, ys)[:, None] * h.agirlik(px, tw, xs)[None, :]
                acc[..., py:py + th, px:px + tw] += yeni * w
                top[py:py + th, px:px + tw] += w
                print(f"  step {a + 1} tile {k}: {s:.0f} s", flush=True)
        x = acc / top.clamp(min=1e-6)        # tiles are blended in latent space after every step
        print(f"step {a + 1} done {ts:.0f} s", flush=True)
    h.latent_yaz(x, f"{KEY}_son.latent")


def ses_yolu():
    return max(h.ara_glob(f"{KEY}_ses_*.flac"), key=lambda p: p.stat().st_mtime)


def cikti():
    return Path(os.environ["SYNCTILE_OUT"]) if os.environ.get("SYNCTILE_OUT") else S.MEDIA / f"{KEY}_synctile_{OW}x{OH}.mp4"


def coz():
    t0 = time.time()
    h.calis({"1": {"class_type": "LoadLatent", "inputs": {"latent": f"{KEY}_son.latent"}},
             "3": {"class_type": "VAELoader", "inputs": {"vae_name": h.VAE_V}},
             "6": {"class_type": "H3VAEAyar", "inputs": {"vae": ["3", 0], "mod": VAE_MOD}},
             "4": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["1", 0], "vae": ["6", 0], "tile_size": 1024,
                                                             "overlap": 128, "temporal_size": 32, "temporal_overlap": 8}},
             "5": {"class_type": "H3BoruKaydet", "inputs": {"images": ["4", 0], "cikti": str(cikti()), "ses": str(ses_yolu()), "crf": 14, "fps": 24}}})
    print(f"DONE {cikti()} (decode {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    h.comfy_hazir()
    ad = sys.argv[1:] or ["girdi", "yuks", "karo", "coz"]
    if "girdi" in ad and not os.environ.get("SYNCTILE_INPUT"):
        raise SystemExit("ERROR: stage 'girdi' needs SYNCTILE_INPUT=<mp4>")
    F = {"girdi": girdi, "yuks": yukselt, "karo": karolar, "coz": coz}
    for a in ad:
        print("==", a, flush=True); F[a]()
