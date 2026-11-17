"""Continuity segment -> input for the SyncTile 'yuks' stage.

    python continuity_latent.py <continuity_job_folder> <segment_no> <KEY>

Writes <KEY>_k2v / <KEY>_k2a latents and <KEY>_ses audio to <media_dir>/synctile/_tmp/.
"""
import json, subprocess, sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch
sys.path.insert(0, str(Path(__file__).parent))
import h3_engine as h

d, n, key = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(open(Path(d) / "manifest.json", encoding="utf-8"))
e = [x for x in m["segments"] if x["index"] == n][0]
f = safe_open(str(Path(d) / f"seg_{n:04d}.safetensors"), "pt")
v, a, w = f.get_tensor("video").float(), f.get_tensor("audio").float(), f.get_tensor("waveform").float()
M = h.COMFY_CIKTI / h.ARA
M.mkdir(parents=True, exist_ok=True)
fmt = {"latent_format_version_0": torch.tensor([])}
save_file({"latent_tensor": v.contiguous(), **fmt}, str(M / f"{key}_k2v_00001_.latent"))
save_file({"latent_tensor": a.contiguous(), **fmt}, str(M / f"{key}_k2a_00001_.latent"))
sr = int(e["audio_sample_rate"])
subprocess.run([h.FFMPEG, "-y", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-", str(M / f"{key}_ses_00001_.flac")],
               input=w.numpy().astype("<f4").T.tobytes(), check=True)
print(key, "video", tuple(v.shape), "audio", tuple(a.shape), sr, "Hz", w.shape[1] / sr, "s", d)
