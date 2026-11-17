"""SyncTile Studio: scene file -> MiniMax H3 video (base -> SyncTile -> eye pass -> assembly).

Usage:
    python studio.py scene.json             # render the scene
    python studio.py scene.json --dry       # only print the prompts, no ComfyUI, no render
    python studio.py scene.json --try       # allow trial styles (cinema, film)
    python studio.py scene.json --resume    # continue a stopped job from the last finished step
    python studio.py --fast "prompt" [--seconds 5]   # one fast shot from a prompt (see README)

Scene keys are in English (see README and examples/). The Turkish keys of the source engine also work.
Method comes from the shot count: 1 shot = one scene (no cut); 2+ shots = cuts with memory
(the opening pose of a shot repeats the end of the previous shot).
"""
import json, re, os, subprocess, sys, time, shutil, uuid, urllib.request as U
from pathlib import Path

import settings as S
import lessons
import h3_engine as h

VERSION = "0.1.0 (engine v11)"
HERE = Path(__file__).parent
MEDIA = S.MEDIA
API = S.COMFY_URL + "/h3c/api/v1"
SEQUENCES = S.CONT_OUT / "h3_sequences"
if S.WINDOWS:
    _PI = subprocess.Popen.__init__
    def _pi(self, *a, **k):  # every child process is hidden: no console window opens
        k.setdefault("creationflags", S.NO_WINDOW)
        k.setdefault("stdin", subprocess.DEVNULL)  # without a redirect a windowless child gets no stdout: its output is lost and bash stops at the first echo
        _PI(self, *a, **k)
    subprocess.Popen.__init__ = _pi

# ---------------- DEFAULTS ----------------
PROFILE = S.CFG["continuity_profile"]   # H3 Continuity profile: 25-step full H3 + references (server side)
TAIL = "5"                              # memory: last frames of the previous shot
KARE = 121                              # 5 s per shot
GOZ_DENOISE = "0.2"                     # eye pass denoise

TABAN = {"yatay": (1344, 768), "dikey": (768, 1344), "kare": (1024, 1024)}
HAT = {  # (format, resolution) -> (SyncTile W, H, final crop W:H:x:y)
    ("yatay", "1080"): (1920, 1088, "1920:1080:0:4"), ("dikey", "1080"): (1088, 1920, "1080:1920:4:0"),
    ("kare", "1080"): (1088, 1088, "1080:1080:4:4"),
    ("yatay", "2k"): (2560, 1440, None), ("dikey", "2k"): (1440, 2560, None), ("kare", "2k"): (1440, 1440, None),
    ("yatay", "4k"): (3840, 2160, None), ("dikey", "4k"): (2160, 3840, None), ("kare", "4k"): (2176, 2176, "2160:2160:8:8"),
}
SON720 = {"yatay": "1280:720", "dikey": "720:1280", "kare": "720:720"}
DENENMEDI = {("kare", "1080"), ("kare", "2k"), ("kare", "4k"), ("kare", "720"), ("dikey", "2k"), ("dikey", "4k")}
BUYUK = ("2k", "4k")                    # these get the face and motion pass (face_pass.py) instead of the full-frame eye pass
FORMAT_EN = {"yatay": "horizontal", "dikey": "vertical", "kare": "square"}

# Named characters: {"id": {"ref": [images], "tanim": "the man in {img}, ...", "kisa": "a man ..."}}.
# Empty in the release. A person can always give "ref", "description" and "short" in the scene file.
CHARACTERS = {}
STIL = {
    "iphone": "The target video is live-action, natural iPhone footage, slightly amateur, sharp clean natural image, real skin texture, faces clearly visible.",
    "sitcom": "The target video is live-action, a 90s-style multi-camera sitcom look, bright warm even studio lighting, sharp clean natural image.",
    "a3d": "The target video is an angular stylized 3D animated film look, sharp geometric faceted shapes, hand-painted textures on 3D models, painterly stylized materials, visible volume and depth, cinematic lighting, smooth CGI render.",
}
DENEME_STIL = {  # only with --try
    "sinema": "The target video is live-action, a dark moody science-fiction thriller film look, cold blue-gray color grade, low-key lighting with deep shadows, anamorphic lens, shallow depth of field, soft film grain, real skin texture, faces clearly visible.",
    "film": "The target video is live-action, a cinematic 35mm film still look, warm natural colors, soft film grain, shallow depth of field, natural daylight, real skin texture, faces clearly visible.",
}
KAMERA = {"sabit": "locked-off camera", "el": "handheld camera with a slight natural shake",
          "aksiyon": "shaky handheld action camera that never stops moving",
          "haber": "shoulder-mounted TV news camera held by a reporter, it follows the action, slightly shaky, the operator zooms in slowly and zooms back out a little"}
SIRA = ["", "second ", "third ", "fourth ", "fifth ", "sixth ", "seventh ", "eighth ", "ninth ", "tenth "]
SESSIZ = " No human voices at all, no talking, no murmur in any language."
SUREKLI = "This shot starts exactly where the previous moment ended:"
YASAK = ["peace sign", "v sign", "v-sign", "victory sign", "two fingers"]  # blocked gesture in every video

# ---------------- English scene keys -> internal keys ----------------
TOP = {"name": "ad", "resolution": "cozunurluk", "style": "stil", "camera": "kamera", "eyes": "goz", "people": "kisiler",
       "location": "mekan", "shots": "cekimler", "skip_lessons": "ders_haric", "single_take": "tek_plan", "transition": "gecis"}
PERSON = {"description": "tanim", "short": "kisa", "outfit": "kiyafet"}
SHOT = {"framing": "kadraj", "who": "kim", "action": "eylem", "start": "baslangic", "end": "bitis", "audio": "ses", "eyes": "goz",
        "alone": "yalniz", "location": "mekan", "summary": "ozet", "seconds": "sn", "audio_tail": "ses_kuyruk", "memory": "hafiza"}
VALUES = {"format": {"horizontal": "yatay", "vertical": "dikey", "square": "kare"},
          "stil": {"cinema": "sinema"},
          "kamera": {"fixed": "sabit", "handheld": "el", "action": "aksiyon", "news": "haber"},
          "kadraj": {"two": "iki", "close": "yakin", "reverse": "ters", "detail": "detay", "wide": "genis", "medium": "orta"}}


def normalize(s):
    """English keys and values -> internal keys. Unknown keys stay as they are."""
    def keys(d, m):
        return {m.get(k, k): v for k, v in d.items()}
    s = keys(s, TOP)
    if "cozunurluk" in s:
        s["cozunurluk"] = str(s["cozunurluk"]).lower().rstrip("p")
    for k in ("format", "stil", "kamera"):
        if k in s:
            s[k] = VALUES[k].get(s[k], s[k])
    s["kisiler"] = [keys(p, PERSON) for p in s.get("kisiler", [])]
    s["cekimler"] = [keys(c, SHOT) for c in s.get("cekimler", [])]
    for c in s["cekimler"]:
        if "kadraj" in c:
            c["kadraj"] = VALUES["kadraj"].get(c["kadraj"], c["kadraj"])
    return s


def kadraj(c, i, cam, ad1, ad2):
    k = c.get("kadraj", "orta"); cm = f"a {SIRA[min(i, 9)]}{cam}"
    if k == "iki": return f"A medium close two-shot from {cm}, both faces large in the frame."
    if k == "yakin": return (f"A close-up from {cm} on {ad1}, the face fills half the frame"
                             + (f", {ad2} at the edge of the frame, out of focus." if ad2 else "."))
    if k == "ters": return (f"A medium close two-shot from {cm}, the reverse angle: {ad2} in the foreground and {ad1} beside, both faces large in the frame."
                            if ad2 else f"A medium close shot from {cm}, the reverse angle on {ad1}.")
    if k == "detay": return f"An extreme close-up detail shot from {cm} on {ad1}: it fills the whole frame."
    if k == "genis": return f"A wide shot from {cm}, the whole room visible."
    return f"A medium shot from {cm} on {ad1}."


def gorunum(c, ilk):
    return c.get("kim", ilk) if c.get("yalniz") else ("genis" if c.get("kadraj") == "genis" else "ikili")


def promptlar(s):
    """scene -> (Continuity boxes, ref files, tile prompts, eye prompts, eye ref names)"""
    metin = json.dumps(s, ensure_ascii=False).lower()
    for y in YASAK:
        if y in metin: raise SystemExit(f"ERROR blocked gesture: '{y}'")
    refs, tanimlar, adlar, kisalar, goz_refs, tanim_kim = [], [], {}, {}, [], {}
    for n, k in enumerate(s["kisiler"], 1):
        kk = CHARACTERS.get(k["id"], {}); r = k.get("ref", kk.get("ref", []))
        if isinstance(r, str): r = [r]
        img = ", ".join(f"@image{17 + len(refs) + j}" for j in range(len(r)))
        img = img.rsplit(", ", 1)[0] + " and " + img.rsplit(", ", 1)[1] if len(r) > 1 else img
        refs += r
        if r and not goz_refs: goz_refs = r
        tanim = (k.get("tanim") or kk.get("tanim", "a person")).replace("{img}", img)
        if k.get("kiyafet"): tanim = tanim.split(", with")[0] + ", " + kk.get("kisa", "").replace("a man ", "") + ", wearing " + k["kiyafet"] if kk else tanim
        if r:
            tanimlar.append(f"<Subject {n}> is {tanim}."); adlar[k["id"]] = f"<Subject {n}>"; tanim_kim[k["id"]] = tanimlar[-1]
        else:
            adlar[k["id"]] = tanim  # a second person without references is described in text
        kisalar[k["id"]] = k.get("kisa") or kk.get("kisa", tanim)
    cam = KAMERA[s.get("kamera", "sabit")]; st = s.get("stil", "iphone")
    if st not in STIL and not s.get("_dene"): raise SystemExit(f"ERROR style '{st}' is a trial style; run with --try")
    stil = STIL.get(st) or DENEME_STIL[st]
    ilk = s["kisiler"][0]["id"]; diger = s["kisiler"][1]["id"] if len(s["kisiler"]) > 1 else None
    kutular, karo, goz = [], [], []
    for i, c in enumerate(s["cekimler"]):
        kim = c.get("kim", ilk); obur = diger if kim == ilk else ilk
        ad1, ad2 = adlar[kim], (adlar.get(obur) if obur and not c.get("yalniz") else None)  # alone: no second person in the framing sentence
        coklu = len(s["cekimler"]) > 1
        tek = "One single continuous shot from start to end, no cuts, no scene change. " if not coklu or s.get("tek_plan") else ""
        cumle = (f"[Shot 1] {tek}{kadraj(c, i, cam, ad1, ad2)} " + (f"{SUREKLI} {c['baslangic']} " if i and c.get("baslangic") else "")
                 + c["eylem"] + (f" The shot ends with {c['bitis']}" if c.get("bitis") else "")
                 + (" Nobody moves from their spot." if coklu and i and c.get("baslangic") and s.get("kamera") != "aksiyon" else ""))
        metin = " ".join(str(c.get(k, "")) for k in ("eylem", "baslangic", "bitis", "ses", "ozet")) + " " + c.get("mekan", s["mekan"])
        dersler = [x for x in lessons.aktif(st, c.get("kadraj"), "prompt") if x.get("ek") and x["no"] not in s.get("ders_haric", [])]
        for x in dersler:  # the scene wins: a lesson is skipped when the shot text asks for the opposite
            if lessons.wanted(x, metin): print(f"LESSON #{x['no']} skipped in shot {i + 1}: the scene asks for \"{lessons.wanted(x, metin)}\"")
        dersek = " ".join(x["ek"] for x in dersler if not lessons.wanted(x, metin))
        detay = f"{stil} {c.get('mekan', s['mekan'])}" + (f" {dersek}" if dersek else "")
        # a shot whose person has no reference: the referenced person stays out of it, so the face does not move to him
        refsiz = bool(c.get("yalniz")) and not adlar[kim].startswith("<Subject")
        p = (("" if refsiz else "subject_definitions:\n" + (tanim_kim[kim] if c.get("yalniz") else "\n".join(tanimlar)) + "\n\n") + f"summary:\n{'' if refsiz else '[reference generation] '}{c.get('ozet', c['eylem'].split('.')[0] + '.')}\n\n"
             f"detailed_description:\n{detay}\n{cumle}\n\noverall_soundscape:\n{c.get('ses', 'Quiet room tone.')}{SESSIZ if 'nobody speaks' in c.get('ses', '').lower() else ''}\n\nnon_diegetic_music:\nN/A")
        b = {"id": f"c{i + 1}", "frames": int(round((c.get("sn", 5) + c.get("ses_kuyruk", 0)) * 24)) | 1, "seed": s.get("seed", 9000) + i, "prompt": p}
        if i:
            # memory chain: the reference is the last shot of the SAME view, not simply the previous shot
            # (person A -> the previous shot of A; wide -> the previous wide). "memory": N picks it by hand.
            b["cut"] = True
            kay = c.get("hafiza") or next((k + 1 for k in range(i - 1, -1, -1) if gorunum(s["cekimler"][k], ilk) == gorunum(c, ilk)), None)
            if kay and kay != i:
                oyn = 17 * max(1, round((kutular[kay - 1]["frames"] - 5) / 17)) + 5
                b["memory"] = "selected"; b["memories"] = [{"box": f"c{kay}", "frame": oyn - 5, "label": "the same people and place a moment earlier"}]
            else:
                b["memory"] = "off" if c.get("yalniz") else "scene"
        kutular.append(b)
        duz = cumle
        for kid, a in adlar.items(): duz = duz.replace(a, kisalar[kid])
        karo.append(f"{detay}\n{duz}")
        if s.get("goz", True) and c.get("goz", kim == ilk) and goz_refs:
            gp = ("subject_definitions:\n<Subject 1> is " + CHARACTERS.get(ilk, {}).get("tanim", "the person in {img}").format(
                img="<Picture 1>, <Picture 2> and <Picture 3>") + ".\n\n"
                  f"summary:\n[reference generation] {c['eylem'].split('.')[0]}.\n\ndetailed_description:\n{stil} natural detailed eyes, real skin texture, sharp focus on the faces.\n"
                  f"[Shot 1] <Subject 1> keeps his own face, eyes and expression exactly as in the video. Other people keep their own faces. Clothes, colors, hair and background stay exactly as in the video. Camera stays the same.\n\n"
                  "overall_soundscape:\nLight room tone.\n\nnon_diegetic_music:\nN/A")
            goz.append(gp)
        else:
            goz.append(None)
    return kutular, refs, karo, goz, goz_refs


def chain_prompt(kutular):
    """The script the chain mode sends to the model: @imageN -> <Picture N-16>, blocks joined by '---', options added."""
    blok = [re.sub(r"@image(\d+)", lambda m: f"<Picture {int(m.group(1)) - 16}>", b["prompt"]) for b in kutular]
    return blok, h.add_options("\n---\n".join(blok))


def ref_path(r):
    return Path(r) if Path(r).is_absolute() else MEDIA / r


# ---------------- helpers ----------------
class Is:
    def __init__(s, sahne):
        s.ad = sahne["ad"]; s.on = str(S.WORK / f"studio_{s.ad}"); s.log = open(f"{s.on}.log", "a", encoding="utf-8")
    def yaz(s, m): s.log.write(f"{time.strftime('%H:%M:%S')} {m}\n"); s.log.flush(); print(m, flush=True)
    def var(s, ad): return Path(f"{s.on}_{ad}").exists()
    def isaret(s, ad, icerik="ok"): Path(f"{s.on}_{ad}").write_text(icerik, encoding="utf-8")
    def oku(s, ad): return Path(f"{s.on}_{ad}").read_text(encoding="utf-8")


def ff(*a): subprocess.run([h.FFMPEG, "-y", "-v", "error", *map(str, a)], check=True)


def probe(*a): return subprocess.run(["ffprobe", "-v", "error", *map(str, a)], capture_output=True, text=True).stdout.strip()


def comfy(mod):
    """mod cont = Continuity API (base), std = SyncTile/eye pass. Every pass starts with clean VRAM."""
    pf = S.WORK / "comfy_pid.txt"
    if pf.exists():
        S.stop_pid(pf.read_text().strip())
        time.sleep(5)
    ek = (["--input-directory", str(S.CONT_IN), "--output-directory", str(S.CONT_OUT)]
          if mod == "cont" else ["--output-directory", str(MEDIA)])
    lg = open(S.WORK / f"comfy_{mod}.log", "ab")
    pr = subprocess.Popen(S.comfy_command(ek), cwd=S.COMFY_DIR, stdout=lg, stderr=lg, creationflags=S.DETACHED, close_fds=True,
                          env={**os.environ, "H3C_MEMORY_TAIL": TAIL, "H3C_SELECTED_ONLY": "1", "H3C_NO_LAST_PNG": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    pf.write_text(str(pr.pid))
    for _ in range(120):
        time.sleep(5)
        try: U.urlopen(S.COMFY_URL + "/system_stats", timeout=5); return
        except Exception: pass
    raise SystemExit("ERROR ComfyUI did not start: " + mod)


def api(m, p, b=None, hd={}):
    q = U.Request(API + p, data=json.dumps(b).encode() if b is not None else None, method=m, headers={"Content-Type": "application/json", **hd})
    return json.load(U.urlopen(q, timeout=120))


def taban(j, s, kutular, refs):
    W, H = TABAN[s["format"]]
    assets = []
    for i, r in enumerate(refs):
        o = subprocess.run(["curl", "-s", "-F", "kind=picture", "-F", f"file=@{ref_path(r)}", API + "/assets"], capture_output=True, text=True).stdout
        assets.append({"asset_id": json.loads(o)["id"], "bindings": [{"kind": "picture", "slot": f"image{17 + i}", "title": f"Ref {i + 1}"}]})
    tl = {"version": 1, "boxes": kutular}
    if s.get("ref_video"):  # motion and camera reference video (24 fps, 2-15 s), @video1 in the prompt
        d = json.loads(subprocess.run(["curl", "-s", "-F", "kind=video", "-F", f"file=@{s['ref_video']}", API + "/assets"], capture_output=True, text=True).stdout)
        assets.append({"asset_id": d["id"], "bindings": [{"kind": "reference_replacement", "slot": "video1"}]})
        tl["reference_replacements"] = {"video1": {"file": d["files"]["video"]}}
    body = {"name": f"h3-{s['ad']}", "profile": PROFILE, "assets": assets, "timeline": tl,
            "settings": {"width": W, "height": H, "context_frames": 39, "carry_audio": False, "ref_image_size": "match"}}
    p = api("POST", "/projects", body, {"Idempotency-Key": str(uuid.uuid4())}); sel = {"kind": "all"}
    pl = api("POST", f"/projects/{p['id']}/plan", {"expected_revision": p["revision"], "selection": sel})
    job = api("POST", f"/projects/{p['id']}/jobs", {"expected_revision": p["revision"], "selection": sel, "plan_sha256": pl["plan_sha256"]},
              {"Idempotency-Key": str(uuid.uuid4())})["id"]
    j.yaz(f"base job {job}")
    while True:
        time.sleep(60)
        d = api("GET", f"/jobs/{job}"); j.yaz(f"base {d['state']} {d.get('progress')}")
        if d["state"] == "succeeded": break
        if d["state"] in ("failed", "cancelled"): raise SystemExit("ERROR base " + json.dumps(d)[:800])
    k = sorted(SEQUENCES.glob("api_*"), key=lambda x: x.stat().st_mtime)[-1]
    j.isaret("klasor", str(k)); j.yaz(f"base done {k}")


def hat(j, s, n, karo_p):
    """shot n (1..) Continuity latent -> SyncTile pass (1080/2k/4k)"""
    W, H, _ = HAT[(s["format"], s["cozunurluk"])]; key = f"st_{s['ad']}_c{n}"
    subprocess.run([sys.executable, str(HERE / "continuity_latent.py"), j.oku("klasor"), str(n), key], check=True)
    pd = f"{j.on}_karo{n}.txt"; Path(pd).write_text(karo_p, encoding="utf-8")
    subprocess.run([sys.executable, str(HERE / "synctile_4k.py"), "yuks", "karo", "coz"], cwd=HERE, check=True,
                   env={**os.environ, "SYNCTILE_W": str(W), "SYNCTILE_H": str(H), "SYNCTILE_KEY": key, "SYNCTILE_PROMPT": pd,
                        "SYNCTILE_OUT": f"{j.on}_c{n}_hat.mp4"})


def kesme_kirp(j, ham, n):
    """If the model puts the cut late, the start of the shot still shows the previous view. A cut in the first 2 s: the part before it is removed."""
    o = subprocess.run([h.FFMPEG, "-v", "error", "-i", ham, "-vf", "select='lt(n,48)*gt(scene,0.3)',metadata=print:file=-", "-f", "null", "-"],
                       capture_output=True, text=True).stdout
    m = re.findall(r"pts_time:([\d.]+)", o)
    if not m or float(m[0]) <= 0: return
    tmp = ham + ".kirp.mp4"
    ff("-i", ham, "-ss", m[0], "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", tmp)
    shutil.move(tmp, ham); j.yaz(f"c{n}: {round(float(m[0]) * 24)} frames of the previous view at the start, removed")


def goz(j, girdi, cikti, W, H, gp, refs, yuz=False):
    """Eye pass on the full frame. yuz=True (2K/4K): face and motion pass, every face and the box with the most motion
    are cropped and processed on their own (face_pass.py)."""
    pd = f"{cikti}.prompt.txt"; Path(pd).write_text(gp, encoding="utf-8")
    env = {**os.environ, "EYE_PROMPT": pd, "EYE_REFS": "|".join(str(ref_path(r)) for r in refs)}
    if yuz:
        subprocess.run([sys.executable, str(HERE / "face_pass.py"), girdi, cikti], cwd=HERE, check=True, env=env); return
    son = int(probe("-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", girdi)) - 1
    subprocess.run([sys.executable, str(HERE / "eye_pass.py"), girdi, "0", str(son), str(W), str(H), GOZ_DENOISE], cwd=HERE, check=True,
                   env={**env, "EYE_OUT": cikti})


def sure(f):
    return float(probe("-show_entries", "format=duration", "-of", "csv=p=0", f))


def kuyruk_bindir(j, s, parcalar, cikti):
    """L-cut: the rest of the audio of a shot with audio_tail plays under the next shots, fades out, and ducks under dialog."""
    ks = [(n, f"{j.on}_c{n}_kuyruk.wav") for n in range(1, len(parcalar) + 1)
          if s["cekimler"][n - 1].get("ses_kuyruk") and Path(f"{j.on}_c{n}_kuyruk.wav").exists()]
    if not ks: return
    gir, fl = [], [f"[0:a]asplit={len(ks) + 1}[ana]" + "".join(f"[k{i}]" for i in range(len(ks)))]
    for i, (n, w) in enumerate(ks):
        bas, d = sum(sure(p) for p in parcalar[:n]), sure(w); gir += ["-i", w]
        fl.append(f"[{i + 1}:a]afade=t=in:d=0.03,afade=t=out:st={d * 0.35:.2f}:d={d * 0.65:.2f},adelay={int(bas * 1000)}:all=1[t{i}];"
                  f"[t{i}][k{i}]sidechaincompress=threshold=0.04:ratio=5:attack=30:release=400[s{i}]")
        j.yaz(f"audio tail c{n}: {d:.1f} s, from {bas:.2f} s")
    fl.append("[ana]" + "".join(f"[s{i}]" for i in range(len(ks))) + f"amix=inputs={len(ks) + 1}:normalize=0:duration=first,alimiter=limit=0.95[a]")
    tmp = str(cikti) + ".kuyruk.mp4"
    ff("-i", cikti, *gir, "-filter_complex", ";".join(fl), "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", tmp)
    shutil.move(tmp, cikti)


def cikti_adi(s):
    return MEDIA / f"{time.strftime('%Y%m%d')}_{s['ad']}_{FORMAT_EN[s['format']]}_{s['cozunurluk']}.mp4"


def tek_plan(j, s, kutular, refs):
    """One long take: seamless chain (H3MultishotSampler, latent_handoff), no cut.
    Every shot is one block (same seconds, 10 recommended); blocks join without a seam. Calls h3_engine.py chain."""
    W, H = TABAN[s["format"]]; cikti = cikti_adi(s)
    blok, _ = chain_prompt(kutular)
    gecis = s.get("gecis", "latent_handoff")
    sn = s["cekimler"][0].get("sn", 10)
    j.yaz(f"SINGLE TAKE {len(blok)} blocks x {sn} s, {W}x{H}, transition {gecis}")
    r = subprocess.run([sys.executable, str(HERE / "h3_engine.py"), "chain", "\n---\n".join(blok), "--seconds", str(sn), "--seed", str(s.get("seed", 9000)),
                        "--name", s["ad"], "--width", str(W), "--height", str(H), "--out", str(cikti), "--scale", SON720[s["format"]],
                        "--transition", gecis, *(["--ref", *[str(ref_path(x)) for x in refs]] if refs else [])],
                       cwd=HERE, capture_output=True, text=True, errors="ignore")
    j.yaz(r.stdout[-600:] + r.stderr[-1500:])
    if not cikti.exists(): raise SystemExit("ERROR single take: no output")
    ses_temizle(j, s, cikti)
    j.yaz(f"DONE {cikti}")
    return cikti


def ses_temizle(j, s, cikti):
    """H3 can make meaningless speech in a block without dialog. In a block whose audio says 'nobody speaks'
    the voice band is removed; then faster-whisper (optional) looks for non-English speech in every block."""
    sn = s["cekimler"][0].get("sn", 10); ar = []
    for i, c in enumerate(s["cekimler"]):
        if "nobody speaks" in c.get("ses", "").lower():
            a, b = i * sn, (i + 1) * sn
            ar.append(f"between(t,{a},{b})")
    if ar:
        e = "+".join(ar); gec = str(cikti) + ".tmp.mp4"
        # the voice band is cut fully: low brown noise (<400 Hz) + high band (>7000 Hz) stay
        ff("-i", cikti, "-c:v", "copy", "-filter_complex",
           f"[0:a]asplit=2[k][u];anoisesrc=c=brown:a=0.02:d=600:r=48000,lowpass=f=400,aformat=channel_layouts=stereo[d2];[u]highpass=f=7000,highpass=f=7000,volume=1.8[u2];"
           f"[d2][u2]amix=2:normalize=0:duration=shortest[t];[k]volume='if({e},0,1)':eval=frame[k2];[t]volume='if({e},1,0)':eval=frame[t2];[k2][t2]amix=2:normalize=0[a]",
           "-map", "0:v", "-map", "[a]", "-c:a", "aac", "-b:a", "192k", gec)
        os.replace(gec, cikti); j.yaz(f"AUDIO CLEAN: blocks without speech {e}")
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel("small", device="cpu", compute_type="int8"); w = str(cikti) + ".wav"
        for i in range(len(s["cekimler"])):
            ff("-ss", i * sn, "-t", sn, "-i", cikti, "-vn", "-ar", "16000", "-ac", "1", w)
            seg, inf = m.transcribe(w, condition_on_previous_text=False, vad_filter=True)
            metin = " ".join(x.text for x in seg).strip()
            j.yaz(f"AUDIO CHECK block {i + 1}: {inf.language} {inf.language_probability:.2f} '{metin[:120]}'"
                  + (" WARNING non-English speech" if metin and inf.language != "en" else ""))
        os.remove(w)
    except Exception as x: j.yaz(f"AUDIO CHECK skipped: {x}")


def calistir(s, kuru=False, devam=False):
    for a in ("format", "cozunurluk"):
        if s.get(a) not in ({"yatay", "dikey", "kare"} if a == "format" else {"720", "1080", "2k", "4k"}):
            raise SystemExit(f"ERROR: '{'resolution' if a == 'cozunurluk' else a}' is missing or wrong. Set it in the scene file.")
    fm, cz = s["format"], s["cozunurluk"]
    kutular, refs, karo_p, goz_p, goz_refs = promptlar(s)
    j = Is(s)
    Path(f"{j.on}_sahne.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(f"{j.on}_kutular.json").write_text(json.dumps(kutular, ensure_ascii=False, indent=1), encoding="utf-8")
    if (fm, cz) in DENENMEDI: j.yaz(f"WARNING {FORMAT_EN[fm]}/{cz} is not tested on this path")
    for x in lessons.aktif(s.get("stil"), tur="not"): j.yaz(f"LESSON #{x['no']} (note): {x['sorun']} -> {x.get('neden', '')}")
    if kuru:
        if s.get("tek_plan"):
            print("\n===== single take: script sent to the model\n" + chain_prompt(kutular)[1])
        else:
            for b in kutular: print("\n=====", b["id"], "\n" + b["prompt"])
        return
    if s.get("tek_plan"):
        cikti = tek_plan(j, s, kutular, refs)
        if not os.environ.get("SYNCTILE_NO_OPEN"): S.open_video(cikti)
        return
    if not devam:
        for f in S.WORK.glob(f"studio_{s['ad']}_*"):
            if f.suffix in (".mp4", ".ok") or f.name.endswith("_klasor"): f.unlink()
    j.yaz(f"START {VERSION} {s['ad']} {FORMAT_EN[fm]} {cz} style={s.get('stil')} camera={s.get('kamera')} shots={len(kutular)}")
    t0 = time.time()
    if not j.var("klasor"):
        comfy("cont"); taban(j, s, kutular, refs)
    k = Path(j.oku("klasor")); full = next(k.glob("*_full.mp4"))
    shutil.copy2(full, MEDIA / f"{time.strftime('%Y%m%d')}_{s['ad']}_base.mp4")
    W0, H0 = TABAN[fm]; parcalar = []
    comfy("std")
    for n in range(1, len(kutular) + 1):
        son = f"{j.on}_c{n}_son.mp4"
        if Path(son).exists(): parcalar.append(son); continue
        if cz == "720":
            ham = f"{j.on}_c{n}_hat.mp4"
            # shot borders come from the real segment in the manifest (H3 makes 124 frames when 121 are asked)
            shutil.copy2(k / f"seg_{n:04d}.mp4", ham)
            gw, gh = W0, H0
        else:
            ham = f"{j.on}_c{n}_hat.mp4"
            if not (devam and Path(ham).exists()): j.yaz(f"synctile c{n}"); hat(j, s, n, karo_p[n - 1])
            gw, gh = HAT[(fm, cz)][:2]
        if n > 1 and not Path(f"{j.on}_c{n}_goz.mp4").exists(): kesme_kirp(j, ham, n)
        if goz_p[n - 1]:
            temiz = f"{j.on}_c{n}_goz.mp4"
            if not (devam and Path(temiz).exists()):
                j.yaz(f"{'face and motion' if cz in BUYUK else 'eye'} pass c{n}"); goz(j, ham, temiz, gw, gh, goz_p[n - 1], goz_refs, yuz=cz in BUYUK)
            gor = temiz
        else:
            gor = ham
        kirp = SON720[fm] if cz == "720" else HAT[(fm, cz)][2]
        vf = (f"scale={kirp}:flags=lanczos" if cz == "720" else (f"crop={kirp}" if kirp else "null")) + ",fps=24"
        kuy = s["cekimler"][n - 1].get("ses_kuyruk")  # the picture is cut at 'seconds', the rest of the audio becomes the tail
        if kuy: ff("-ss", s["cekimler"][n - 1]["sn"], "-i", ham, "-vn", "-ar", "48000", "-ac", "2", f"{j.on}_c{n}_kuyruk.wav")
        ff("-i", gor, "-i", ham, "-map", "0:v", "-map", "1:a?", "-vf", vf, *(["-t", s["cekimler"][n - 1]["sn"]] if kuy else []),
           "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", son)
        parcalar.append(son); j.yaz(f"shot {n} ready")
    liste = f"{j.on}_liste.txt"; Path(liste).write_text("".join(f"file '{p}'\n" for p in parcalar))
    cikti = cikti_adi(s)
    ff("-f", "concat", "-safe", "0", "-i", liste, "-c", "copy", cikti)
    kuyruk_bindir(j, s, parcalar, cikti)
    if cz == "4k" and fm == "yatay":  # RTX quad; only for horizontal 3840x2160
        j.yaz("rtx quad"); rtx = cikti.with_name(cikti.stem + "_rtx.mp4")
        subprocess.run([S.bash(), str(HERE / "rtx_quad.sh"), str(cikti), str(rtx)], check=True,
                       env={**os.environ, "RTX_PYTHON": S.comfy_python(), "RTX_TMP": str(S.WORK)}); cikti = rtx
    j.yaz(f"DONE {cikti} {(time.time() - t0) / 60:.1f} min")
    if not os.environ.get("SYNCTILE_NO_OPEN"):
        S.open_video(cikti)


def fast(a):
    """--fast "prompt" [--seconds N]: one shot through h3_engine.py fast."""
    i = a.index("--fast")
    if i + 1 >= len(a): sys.exit("ERROR: --fast needs a prompt")
    cmd = [sys.executable, str(HERE / "h3_engine.py"), "fast", a[i + 1]]
    if "--seconds" in a: cmd += ["--seconds", a[a.index("--seconds") + 1]]
    sys.exit(subprocess.run(cmd, cwd=HERE).returncode)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit()
    if "--fast" in a: fast(a)
    s = normalize(json.load(open(a[0], encoding="utf-8-sig")))
    calistir({**s, "_dene": "--try" in a or "--dene" in a}, kuru="--dry" in a or "--kuru" in a, devam="--resume" in a or "--devam" in a)
