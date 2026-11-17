"""Eye pass: full-frame second pass with identity references. Ref2VA + video latent (VAEEncode) + low denoise,
8 steps with the ref2v turbo LoRA.

    python eye_pass.py <input.mp4> <first> <last> <W> <H> <denoise...>   (last=123; padded to 124 frames with ping-pong)

environment:
  EYE_PROMPT  prompt file (required)
  EYE_REFS    reference images separated by '|' (required, absolute paths or names in media_dir)
  EYE_OUT     copy the result to this mp4 (optional; one denoise value)
Output: <media_dir>/synctile/_tmp/eye_<W>_<denoise>_00001_.mp4 (audio is taken from the original later)
"""
import os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import h3_engine as h
import settings as S

REF_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
REF_TURBO = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"

if not os.environ.get("EYE_PROMPT") or not os.environ.get("EYE_REFS"):
    raise SystemExit("ERROR: set EYE_PROMPT (prompt file) and EYE_REFS (images separated by '|')")
PROMPT = Path(os.environ["EYE_PROMPT"]).read_text(encoding="utf-8")
REFS = [Path(r) if Path(r).is_absolute() else S.MEDIA / r for r in os.environ["EYE_REFS"].split("|")]


def grafik(ad, kare, w, hh, denoise, onek):
    g = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": REF_UNET, "weight_dtype": "default"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": h.CLIP, "type": "minimax", "device": "default"}},
         "3": {"class_type": "VAELoader", "inputs": {"vae_name": h.VAE_V}},
         "4": {"class_type": "VAELoader", "inputs": {"vae_name": h.VAE_A}}}
    g["V"] = {"class_type": "LoadVideo", "inputs": {"file": ad}}
    g["VC"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["V", 0]}}
    g["VS"] = {"class_type": "ImageScale", "inputs": {"image": ["VC", 0], "upscale_method": "lanczos",
                                                      "width": w, "height": hh, "crop": "center"}}
    ref = {}
    for i, r in enumerate(REFS, 1):
        g[f"RI{i}"] = {"class_type": "LoadImage", "inputs": {"image": r.name}}
        ref[f"ref_images.ref_image_{i}"] = [f"RI{i}", 0]
    g["5"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0], "prompt": PROMPT, "width": w, "height": hh,
        "length": kare, "ref_image_size": "match", **ref}}
    g["KE"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["VS", 0], "vae": ["3", 0]}}
    g["AY"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["5", 1]}}
    g["BR"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["KE", 0], "audio_latent": ["AY", 1]}}
    g["LR0"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": REF_TURBO, "strength_model": 1.0, "model": ["1", 0]}}
    g.update({
        "CK": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["LR0", 0], "attention": "comfy kitchen attention"}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": h.SEED}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "8": {"class_type": "BasicScheduler", "inputs": {"model": ["CK", 0], "scheduler": "simple", "steps": 8, "denoise": denoise}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["CK", 0], "conditioning": ["5", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0],
                                                                 "sigmas": ["8", 0], "latent_image": ["BR", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["12", 0], "fps": 24}},
        "14": {"class_type": "SaveVideo", "inputs": {"video": ["13", 0], "filename_prefix": f"{h.ARA}/{onek}",
                                                     "format": "mp4", "format.codec": "h264"}},
    })
    return g


if __name__ == "__main__":
    v, ilk, son, w, hh = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    kare = 124  # H3 length is 17k+5 (107/124). Input: first..last(123) + 9 frames ping-pong padding
    ad = f"eye_{ilk}_{son}.mp4"
    subprocess.run([h.FFMPEG, "-y", "-v", "error", "-i", str(v), "-filter_complex",
                    (f"[0:v]select='between(n,{ilk},{son})',setpts=N/24/TB,split[a][b];"
                     f"[b]select='between(n,{son-ilk-10},{son-ilk-2})',setpts=N/24/TB,reverse[c];[a][c]concat=n=2:v=1:a=0[o]" if son-ilk+1 < kare else
                     f"[0:v]select='between(n,{ilk},{son})',setpts=N/24/TB[o]"), "-map", "[o]",
                    "-an", "-r", "24", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(h.GIRDI / ad)], check=True)
    for r in REFS:
        if not (h.GIRDI / r.name).exists():
            shutil.copy2(r, h.GIRDI / r.name)
    h.comfy_hazir()
    for d in map(float, sys.argv[6:]):
        sn, yol = h.calis(grafik(ad, kare, w, hh, d, f"eye_{w}_{int(d*100)}"))
        print(f"DONE denoise {d} {w}x{hh} {sn:.0f} s -> {yol}", flush=True)
        if os.environ.get("EYE_OUT") and yol:
            shutil.copy2(str(yol), os.environ["EYE_OUT"])
            try:
                Path(yol).unlink()
            except OSError:
                pass
