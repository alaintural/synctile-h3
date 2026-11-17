# SyncTile Studio

SyncTile Studio turns a JSON scene file into a MiniMax H3 video. It runs on a local ComfyUI.

The pipeline has these steps:

1. **Base.** The H3 Continuity extension renders all shots at 1344x768 (25 steps, with reference images).
2. **SyncTile.** For 1080, 2K and 4K, the base latent goes through the MiniMax H3 latent upscaler. Then synchronized tiles refine it in one step. The decode is one tiled VAE decode.
3. **Eye pass.** An optional second pass with the identity references at low denoise (0.2, 8 steps). At 720 and 1080 it runs on the full frame. At 2K and 4K it is a face and motion pass: each face and the box with the most motion are cropped, processed at 1280x1280 and laid back over the frame.
4. **Assembly.** Crop, cut, audio tail (L-cut) and concat with ffmpeg. Horizontal 4K also gets an RTX Video Super Resolution quad pass.

SyncTile is described in the SyncTile repository: https://github.com/alaintural/synctile-h3 (DOI 10.5281/zenodo.23196934).

## Status

This is a research release. The code is the code we use on one Windows machine with one RTX GPU. We did not test it on Linux or macOS. Read "Limits" before you use it.

## Requirements

**Programs**
- ComfyUI (the Windows portable build is the default; other installs: set `comfyui_python`)
- ffmpeg and ffprobe on PATH, curl
- bash (only for the horizontal 4K RTX step; on Windows, for example Git for Windows)
- Python packages in the Python that runs the scripts: `torch`, `safetensors`, `Pillow` (continuity.py), `opencv-python` 4.8 or later and `numpy` (face_pass.py, 2K and 4K only). Optional: `faster-whisper` (audio check in single-take mode)
- SageAttention (ComfyUI starts with `--use-sage-attention` by default; change `comfyui_extra_args` if you do not have it)
- `nvvfx` (NVIDIA RTX Video Super Resolution) in the ComfyUI Python, only for the 4K RTX step

**ComfyUI nodes**
- MiniMax H3 core nodes: `MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`
- `MinimaxH3LatentUpscaler3D` (Comfyui_Minimax_h3_latent_Upscaler)
- `H3MultishotSampler` (ComfyUI-H3-Multishot), only for single-take scenes
- ComfyUI-H3-Continuity-Studio with its API at `/h3c/api/v1`, for the base pass
- Core nodes: `LTXVConcatAVLatent`, `LTXVSeparateAVLatent`, `ModelAttentionBackend` ("comfy kitchen attention"), `ManualSigmas`, `DisableNoise`, `LoadLatent`, `SaveLatent`, `LoadVideo`, `GetVideoComponents`, `VAEEncode`, `VAEEncodeAudio`, `VAEDecodeTiled`, `CreateVideo`, `SaveVideo`
- The two nodes in this package. Copy `comfy_node_vae_settings.py` and `comfy_node_pipe_save.py` into `ComfyUI/custom_nodes` and restart ComfyUI. They give `H3VAEAyar` and `H3BoruKaydet`.

**Models** (file names as the scripts use them)
- `minimax_h3_fl2va_pruned_int8_convrot.safetensors` and `minimax_h3_fl2va_pruned_bf16.safetensors` (SyncTile tiles)
- `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (eye pass)
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (text encoder)
- `minimax_h3_video_vae_int8_convrot.safetensors`, `minimax_h3_audio_vae_fp32.safetensors`
- `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`
- LoRA `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` (eye pass)
- LoRA `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` (fast mode only)
- The base pass uses the models of your Continuity profile.

## Setup

1. Copy `config.example.json` to `config.json`.
2. Set the paths. Relative paths start at the folder of `config.json`.
3. Run `python settings.py`. It prints the paths it uses.

| key | meaning |
| --- | --- |
| `comfyui_dir` | ComfyUI folder (portable: the folder with `python_embeded`) |
| `comfyui_python` | Python that runs ComfyUI. Empty = `python_embeded/python.exe` |
| `comfyui_input_dir` | ComfyUI input folder. Empty = `<comfyui_dir>/ComfyUI/input` |
| `comfyui_url` | ComfyUI address |
| `comfyui_extra_args` | extra ComfyUI start arguments |
| `media_dir` | output folder (ComfyUI output for SyncTile and eye pass, final videos) |
| `continuity_input_dir`, `continuity_output_dir` | ComfyUI input and output folders for the Continuity pass. The base result is read from `<continuity_output_dir>/h3_sequences/api_*` |
| `continuity_profile` | name of the Continuity profile for the base pass (the example `h3-25-ref` is a name: make your own 25-step profile with references) |
| `work_dir` | logs, prompt files and parts of the current job |
| `lessons_file` | lesson file (see "Lessons") |
| `bash` | bash program for the RTX step |
| `open_command` | command that opens the final video, for example `["vlc"]`. Empty = nothing opens |

The scripts stop and start ComfyUI themselves (one start for the Continuity pass, one for the rest), so every pass starts with clean VRAM. The process id is in `<work_dir>/comfy_pid.txt`.

## Use

```
python studio.py examples/single_shot.json --dry     # print the prompts only
python studio.py examples/single_shot.json           # render
python studio.py scene.json --resume                 # continue after a stop
python studio.py scene.json --try                    # allow trial styles
python studio.py --fast "A cat sleeps on a sofa." --seconds 5
```

`--dry` does not start ComfyUI and does not render. It writes the log, the scene and the Continuity boxes to `work_dir`.
Set `SYNCTILE_NO_OPEN=1` if the final video must not open.

The example scenes name reference images that are not in this repository. Put your own images in `media_dir` and change the names in `ref`.

## Scene file

Top level:

| key | values |
| --- | --- |
| `name` | job name (file names) |
| `format` | `horizontal` (1344x768 base), `vertical` (768x1344), `square` (1024x1024). Required |
| `resolution` | `720`, `1080`, `2k`, `4k`. Required |
| `style` | `iphone` (default), `sitcom`, `a3d`; trial styles with `--try`: `cinema`, `film` |
| `camera` | `fixed` (default), `handheld`, `action`, `news` |
| `eyes` | `true` (default) / `false`: eye pass for shots where the first person is seen |
| `people` | list of persons (below). The first person is the main person |
| `location` | location text for all shots |
| `shots` | list of shots (below) |
| `seed` | first seed (default 9000; shot n uses seed + n - 1) |
| `ref_video` | path of a motion and camera reference video (24 fps, 2 to 15 s), `@video1` in the prompt |
| `skip_lessons` | list of lesson numbers that do not apply to this scene |
| `single_take` | `true`: one continuous take without cuts (chain mode, below) |
| `transition` | single take only: `continuity` value of `H3MultishotSampler` (default `latent_handoff`) |

Person:

| key | meaning |
| --- | --- |
| `id` | name used in `who` |
| `ref` | list of reference images (absolute path or file name in `media_dir`). With refs the person is `<Subject N>` |
| `description` | text; `{img}` becomes the image tags. Without refs this text replaces the name in the prompt |
| `short` | short text for the SyncTile tile prompts |
| `outfit` | only used with a named entry in `CHARACTERS` in studio.py (empty in this release) |

Shot:

| key | meaning |
| --- | --- |
| `framing` | `medium` (default), `two`, `close`, `reverse`, `detail`, `wide` |
| `who` | person id (default: first person) |
| `action` | what happens (required) |
| `start` | opening state; repeat the end of the previous shot with the same words |
| `end` | end state ("The shot ends with ...") |
| `audio` | soundscape text. If it contains "nobody speaks", a no-voice sentence is added |
| `eyes` | eye pass for this shot (default: true when `who` is the first person) |
| `alone` | `true`: only this person is in the framing sentence |
| `location` | location for this shot only |
| `summary` | summary line (default: first sentence of `action`) |
| `seconds` | shot length (default 5) |
| `audio_tail` | extra seconds of audio that continue under the next shots (L-cut) |
| `memory` | shot number to use as memory (default: the last shot with the same view) |

Notes:

- `detail`: `who` is the thing that fills the frame. Add it to `people` as an entry with a `description` (see `examples/demo_jar_story.json`, id `jar`).
- A later shot with `start` gets the sentence "Nobody moves from their spot." If a person walks in that shot, leave `start` out and write the opening state as the first sentence of `action`.
- `wide` and `two` do not put the person descriptions in the framing sentence. Without reference images, write the descriptions in `action`.

The keys of the source engine (Turkish) also work, for example `cekimler`, `kadraj`, `eylem`.

## Modes

- **720:** the Continuity segment of each shot, scaled to 1280x720 (or 720x1280, 720x720).
- **1080:** SyncTile at 1920x1088 (portrait 1088x1920, square 1088x1088), then a crop to 1080.
- **2k:** SyncTile at 2560x1440 (portrait 1440x2560, square 1440x1440), no crop. With `eyes` on, it gets the face and motion pass, like 4k.
- **4k:** SyncTile at 3840x2160 (portrait 2160x3840, square 2176x2176 cropped to 2160). Horizontal 4K then gets the RTX quad pass (`rtx_quad.sh`).
- **Face and motion pass** (2k and 4k, `face_pass.py`): it runs when `eyes` is on and the first person has reference images. OpenCV YuNet finds the faces. Each face gets its own square box. One more box (1152 px) goes on the area with the most motion, for example hands and objects. Each box is cropped, goes through the eye pass at 1280x1280 (denoise 0.2) and is laid back with a soft edge. The low-frequency color comes from the original, so the box edge does not show. The rest of the frame is not touched. Each box adds about 3.5 minutes per shot on our GPU. A shot with almost no motion gets no motion box. Very fast motion can keep some blur, because the blur is already in the 768p base.
- **Single take** (`"single_take": true`): one call of `H3MultishotSampler`, 25 steps, no LoRA. Each shot is one block, all blocks use the `seconds` of the first shot (10 s is a good value). Output is scaled to 720. There is no SyncTile and no eye pass in this mode. After the render, blocks whose audio says "nobody speaks" get the voice band removed, and faster-whisper (if installed) reports non-English speech.
- **Fast** (`--fast "prompt"`): one shot from a prompt with the 8-step turbo LoRA, 1344x768, saved to `media_dir`. It is a separate entry point: no scene file, no multi-shot, no references, no SyncTile, no eye pass. The scene path always uses the 25-step base from the Continuity profile.

Not tested by us: square in all resolutions, vertical 2K and vertical 4K. Studio prints a warning for them.

## Prompts

The prompt format follows the H3 reference format (`subject_definitions`, `summary`, `detailed_description`, `[Shot 1]`, `overall_soundscape`, `non_diegetic_music`). Reference images are `@image17`, `@image18` ... in the Continuity pass.

Chain and fast mode add two fixed sentences (handheld camera and micro-expressions) before the first `overall_soundscape:` line. You can see them with `--dry` on a single-take scene. They are `HANDHELD` and `MICRO` in `h3_engine.py`.

Studio stops when the scene text contains a peace or V sign (`YASAK` in studio.py).

## Lessons

`lessons.py` keeps a list of problems and fixes. Studio adds the text of every active `prompt` lesson to `detailed_description` and prints `not` lessons in the log.

```
python lessons.py add --video out.mp4 --frame "c2 f40" --problem "text on a sign" --cause "readable sign" --fix prompt --text "No readable text anywhere."
python lessons.py list
python lessons.py approve 1
python lessons.py close 1 "did not help"
```

The release ships 6 prompt lessons in `lessons.json` (no readable text, nobody looks into the camera, realistic sound only, lines are said once, clear skin).

**The scene wins.** A lesson can have an `unless` list of phrases. If the text of a shot asks for one of them, the lesson is not added to that shot, and Studio prints which lesson it skipped. Example: the sound lesson has `music` and `comic sound` in `unless`. When your `audio` text says "a comic sound effect", you get it. A phrase after no, not, without or never does not count ("no music" keeps the lesson). Add your own with `--unless "phrase one,phrase two"`. To switch a lesson off for a full scene, use `skip_lessons`.

## Tools

- `continuity.py video.mp4 [out.png]`: one row per cut, the last 4 frames of the shot before and the first 4 frames of the shot after. Use it to check objects, clothes and poses across cuts.
- `frame_player.html?v=file:///path/video.mp4`: frame-by-frame player with shot and frame numbers. An optional `frame_list.js` gives a list: `VIDS=[{path:"file:///...", name:"...", info:"...", cuts:[0,124]}]`.
- `synctile_4k.py`, `eye_pass.py`, `face_pass.py`, `continuity_latent.py`, `rtx_quad.sh`, `rtx_upscale.py`: the steps that studio.py calls. Each file explains its arguments at the top.

## Limits

- The eye-pass prompt uses the description of the first person only if it is a named entry in `CHARACTERS`. Otherwise it says "the person in <Picture 1>, <Picture 2> and <Picture 3>", also when there are not 3 images. The eye pass uses the references of the first person that has references.
- The eye pass and the face and motion pass need reference images. A shot without them gets no second pass, so its fast motion stays as SyncTile made it.
- The face and motion pass makes one motion box per shot. Motion outside this box is not changed.
- `outfit` has no effect without a `CHARACTERS` entry.
- The base pass needs the H3 Continuity extension and a matching profile. Studio cannot run without it, except in single-take and fast mode.
- Studio starts and stops ComfyUI. Do not use it on a ComfyUI that other people use at the same time.
- Paths and the process handling were only used on Windows.

## Credits

SyncTile uses the idea of MultiDiffusion: Omer Bar-Tal, Lior Yariv, Yaron Lipman, Tali Dekel. "MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation". ICML 2023.

The base pass uses ComfyUI-H3-Continuity-Studio. The chain mode uses ComfyUI-H3-Multishot. The 4K upscale uses Comfyui_Minimax_h3_latent_Upscaler. Thanks to their authors.

The face and motion pass finds faces with OpenCV YuNet. The model file `face_detection_yunet_2023mar.onnx` comes from the OpenCV Zoo and has the MIT license.

## License

MIT. Copyright (c) 2026 Alain Tural, Pixedi. See `LICENSE` and `CITATION.cff`.
