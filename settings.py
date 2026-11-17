"""Shared settings for SyncTile Studio.

All paths come from config.json (next to this file, or the file in the SYNCTILE_CONFIG
environment variable). Relative paths resolve against the folder of the config file.
Copy config.example.json to config.json and edit it.
"""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WINDOWS = os.name == "nt"
NO_WINDOW = 0x08000000 if WINDOWS else 0                              # CREATE_NO_WINDOW
DETACHED = (0x08000000 | 0x00000008 | 0x00000200) if WINDOWS else 0  # no window, detached, new process group

DEFAULTS = {
    "comfyui_dir": "../ComfyUI",
    "comfyui_python": "",
    "comfyui_input_dir": "",
    "comfyui_url": "http://127.0.0.1:8188",
    "comfyui_extra_args": ["--use-sage-attention"],
    "media_dir": "./media",
    "continuity_input_dir": "./inputs",
    "continuity_output_dir": "./outputs",
    "continuity_profile": "h3-25-ref",
    "work_dir": "./work",
    "lessons_file": "./lessons.json",
    "bash": "",
    "open_command": [],
}


def _load():
    f = Path(os.environ.get("SYNCTILE_CONFIG") or HERE / "config.json")
    if not f.exists():
        raise SystemExit(f"ERROR: config file not found: {f}\nCopy config.example.json to config.json and set your paths.")
    c = {**DEFAULTS, **json.loads(f.read_text(encoding="utf-8-sig"))}
    c["_base"] = f.resolve().parent
    return c


CFG = _load()


def path(key):
    """Config value as an absolute Path (relative values resolve against the config folder)."""
    p = Path(os.path.expandvars(os.path.expanduser(CFG[key])))
    return p if p.is_absolute() else (CFG["_base"] / p).resolve()


COMFY_DIR = path("comfyui_dir")
COMFY_URL = CFG["comfyui_url"].rstrip("/")
MEDIA = path("media_dir")
CONT_IN = path("continuity_input_dir")
CONT_OUT = path("continuity_output_dir")
WORK = path("work_dir")
LESSONS = path("lessons_file")
COMFY_INPUT = path("comfyui_input_dir") if CFG["comfyui_input_dir"] else COMFY_DIR / "ComfyUI" / "input"
for _d in (MEDIA, WORK):
    _d.mkdir(parents=True, exist_ok=True)


def comfy_python():
    """Python that runs ComfyUI. Default: the Windows portable build (python_embeded/python.exe)."""
    if CFG["comfyui_python"]:
        return str(path("comfyui_python"))
    p = COMFY_DIR / "python_embeded" / "python.exe"
    if not p.exists():
        raise SystemExit(f"ERROR: ComfyUI Python not found at {p}. On Linux/macOS or a non-portable install, "
                         "set \"comfyui_python\" in config.json (for example the python of your ComfyUI venv).")
    return str(p)


def comfy_command(extra):
    """Command line that starts ComfyUI. Run it with cwd=COMFY_DIR."""
    main = COMFY_DIR / "ComfyUI" / "main.py"
    if not main.exists():
        main = COMFY_DIR / "main.py"
    if not main.exists():
        raise SystemExit(f"ERROR: ComfyUI main.py not found under {COMFY_DIR}. Check \"comfyui_dir\" in config.json.")
    cmd = [comfy_python(), "-s", str(main), "--disable-auto-launch"]
    if WINDOWS and (COMFY_DIR / "python_embeded").exists():
        cmd.append("--windows-standalone-build")
    return cmd + list(CFG["comfyui_extra_args"]) + list(extra)


def stop_pid(pid):
    if WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
    else:
        try:
            os.kill(int(pid), 15)
        except (OSError, ValueError):
            pass


def bash():
    b = CFG["bash"]
    if b:
        return str(path("bash")) if ("/" in b or "\\" in b) else b
    if WINDOWS:
        raise SystemExit("ERROR: the 4K RTX step runs a bash script. Set \"bash\" in config.json "
                         "(for example the bash.exe of Git for Windows).")
    return "bash"


def open_video(f):
    """Runs open_command + [file] if it is set. Empty open_command = nothing opens."""
    cmd = CFG.get("open_command") or []
    if isinstance(cmd, str):
        cmd = [cmd]
    if cmd:
        subprocess.Popen([*cmd, str(f)], creationflags=NO_WINDOW)


if __name__ == "__main__":
    for k in ("comfyui_dir", "media_dir", "continuity_input_dir", "continuity_output_dir", "work_dir", "lessons_file"):
        print(f"{k:24} {path(k)}")
    print(f"{'comfyui_input_dir':24} {COMFY_INPUT}")
    print(f"{'comfyui_url':24} {COMFY_URL}")
    print(f"{'python':24} {sys.version.split()[0]}")
