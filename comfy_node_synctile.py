"""SyncTile refine (ComfyUI custom node): the synchronized tile step of synctile_4k.py as one node.
The video latent is cut into overlapping tiles. Each tile is sampled with the guider. After every step the tiles are
blended in latent space, so they stay one picture. The result is the same as the 'karo' stage of synctile_4k.py.

Inputs:
  guider        for example BasicGuider (H3 model + the prompt conditioning)
  sampler       for example KSamplerSelect "euler"
  sigmas        for example ManualSigmas "0.45, 0" (one step; the first sigma sets the noise level)
  video_latent  the upscaled latent (MinimaxH3LatentUpscaler3D)
  audio_latent  the audio latent of the same clip (the model is an audio-video model; the audio is not changed)
  tile_width, tile_height   tile size in pixels (1472x896 for horizontal video; swap them for vertical video)
  seed          noise seed
  window_frames   optional, for long clips: the length of one time window in frames (17k+5: 124, 141, 243). 0 = off.
                  The clip is then also cut on the time axis. Each window is refined alone, so memory use does not grow
                  with the length of the clip. Windows are blended the same way as tiles, after every step.
  window_overlap  frames that two neighbour windows share (22, 39, 56)
Output: the refined video latent. Decode it with VAEDecodeTiled.

Clone this repository into ComfyUI/custom_nodes, or copy this file there, then restart ComfyUI."""
import math
import torch
import comfy.utils
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, Noise_EmptyNoise
from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent

LAT = 16          # H3 spatial downscale
AUDIO_SCALE = 4.0
BLOCK = 5         # H3 time grid: 17 frames are 5 latent frames, and a clip of 17k+5 frames is 5k+2 latent frames

def latent_frames(frames):
    return 2 + BLOCK * ((frames - 5) // 17)

def frames_of(T):
    return (T - 2) // BLOCK * 17 + 5

def windows(T, L, O):
    """Window start positions on the time axis. Each start is a multiple of 5 latent frames, so each window is cut
    where a clip can start. The last window ends at the last latent frame."""
    if L >= T:
        return [0]
    n = max(2, math.ceil((T - O) / (L - O)))
    return sorted({min(T - L, BLOCK * round(i * (T - L) / (n - 1) / BLOCK)) for i in range(n)})


def positions(L, t, least=16):
    """Tile start positions on one axis: as few tiles as possible, with an overlap of 'least' or more."""
    n = max(1, math.ceil((L - least) / (t - least)))
    return [round(i * (L - t) / (n - 1)) for i in range(n)] if n > 1 else [0]


def weight(p, t, pos):
    """1D tile weight: 0->1 ramp on the side that overlaps a neighbour, flat at the canvas edge."""
    w = torch.ones(t)
    before, after = [q for q in pos if q < p], [q for q in pos if q > p]
    if before:
        o = before[-1] + t - p
        w[:o] = torch.linspace(0, 1, o + 2)[1:-1]
    if after:
        o = p + t - after[0]
        w[-o:] = torch.minimum(w[-o:], torch.linspace(1, 0, o + 2)[1:-1])
    return w


class SyncTileRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "guider": ("GUIDER",), "sampler": ("SAMPLER",), "sigmas": ("SIGMAS",),
            "video_latent": ("LATENT",), "audio_latent": ("LATENT",),
            "tile_width": ("INT", {"default": 1472, "min": 256, "max": 8192, "step": LAT}),
            "tile_height": ("INT", {"default": 896, "min": 256, "max": 8192, "step": LAT}),
            "seed": ("INT", {"default": 4242, "min": 0, "max": 0xffffffffffffffff})},
            "optional": {
            "window_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
            "window_overlap": ("INT", {"default": 39, "min": 5, "max": 100000})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "refine"
    CATEGORY = "h3"

    def refine(self, guider, sampler, sigmas, video_latent, audio_latent, tile_width, tile_height, seed, window_frames=0, window_overlap=39):
        c = video_latent["samples"].detach().float().cpu()      # clean upscaled latent
        ak = audio_latent["samples"].detach().float().cpu()     # clean audio latent
        sig = [round(float(s), 6) for s in sigmas]             # 0.45 as float32 is 0.44999999: round it, so the noise is the same as in synctile_4k.py
        if len(sig) < 2 or not 0 < sig[0] < 1:
            raise ValueError("SyncTile: sigmas need 2 or more values and the first must be between 0 and 1, for example '0.45, 0'")
        s0 = sig[0]
        x = c + s0 / (1 - s0) * torch.randn(c.shape, generator=torch.Generator().manual_seed(seed))  # one shared noise for the whole canvas
        Hl, Wl = c.shape[-2:]
        tw, th = min(tile_width // LAT, Wl), min(tile_height // LAT, Hl)
        xs, ys = positions(Wl, tw), positions(Hl, th)
        T, A = c.shape[2], ak.shape[-1]        # audio latent: [B, C, channels, time]
        L = T
        if window_frames and latent_frames(window_frames) < T:
            if T % BLOCK != 2:
                raise ValueError(f"SyncTile: time windows need a clip of 17k+5 frames. This clip has {T} latent frames; cut it to {frames_of(T)} frames")
            L = max(2 + BLOCK, latent_frames(window_frames))
        O = min(max(2, 2 + BLOCK * round((window_overlap - 5) / 17)), L - BLOCK)
        ts = windows(T, L, O)
        AL = A if L == T else min(A, round(frames_of(L) * A / frames_of(T)))   # audio steps of one window
        noisy = ak + s0 * torch.randn(ak.shape, generator=torch.Generator().manual_seed(seed + 7)) / (AUDIO_SCALE * (1 - s0))
        bar = comfy.utils.ProgressBar((len(sig) - 1) * len(ts) * len(xs) * len(ys))
        audio = {}
        for a in range(len(sig) - 1):
            acc, total = torch.zeros_like(x), torch.zeros(T, Hl, Wl)
            for t0 in ts:
                a0 = A - AL if t0 + L >= T else min(A - AL, round(17 * (t0 // BLOCK) * A / frames_of(T)))   # the audio of the same seconds
                wt = weight(t0, L, ts)
                for py in ys:
                    for px in xs:
                        k = (t0, px, py)
                        if k not in audio:
                            audio[k] = noisy[..., a0:a0 + AL].clone()
                        av = LTXVConcatAVLatent.execute({"samples": x[:, :, t0:t0 + L, py:py + th, px:px + tw].contiguous()}, {"samples": audio[k]}).result[0]
                        out = SamplerCustomAdvanced.execute(Noise_EmptyNoise(), guider, sampler, torch.tensor(sig[a:a + 2]), av).result[0]
                        v, au = LTXVSeparateAVLatent.execute(out).result
                        audio[k] = au["samples"].float().cpu()
                        w = wt[:, None, None] * weight(py, th, ys)[None, :, None] * weight(px, tw, xs)[None, None, :]
                        acc[:, :, t0:t0 + L, py:py + th, px:px + tw] += v["samples"].float().cpu() * w
                        total[t0:t0 + L, py:py + th, px:px + tw] += w
                        bar.update(1)
            x = acc / total.clamp(min=1e-6)      # tiles and windows are blended in latent space after every step
        out = video_latent.copy()
        out["samples"] = x
        return (out,)


NODE_CLASS_MAPPINGS = {"SyncTileRefine": SyncTileRefine}
NODE_DISPLAY_NAME_MAPPINGS = {"SyncTileRefine": "SyncTile refine (synchronized tiles)"}
