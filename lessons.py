"""Lesson log: every rejected video becomes a lesson; studio.py applies active lessons on the next render.

  python lessons.py add --video X.mp4 --frame "c2 f40" --problem "bad hand" --cause "hand detail in a crowd" --fix prompt --text "hands stay in pockets or out of frame." [--style iphone] [--framing wide] [--unless "hands,gloves"]
  python lessons.py list             # active lessons
  python lessons.py approve 3        # the fix worked in the next video -> approved
  python lessons.py close 3 "reason" # the lesson did not help or was wrong

fix types:
  prompt = 'ek' text is added to the detailed_description of every matching shot (studio.py does it)
  ayar   = a setting in the code changes (the lesson stays as a record)
  not    = a note for the scene writer (studio.py prints it in the log)
The file is lessons_file in config.json (default lessons.json). Stored keys:
no, tarih, durum (acik|onayli|kapali), cozum (prompt|ayar|not), sorun, neden, ek, video, kare, stil, kadraj, unless.

unless = list of phrases. If the text of a shot asks for one of them (for example "comic sound"), the scene wins
and the lesson is not added to that shot. A phrase after no / not / without / never does not count.
"""
import json, re, sys, time
from pathlib import Path

import settings as S

DOSYA = S.LESSONS
FIX = {"setting": "ayar", "note": "not", "prompt": "prompt", "ayar": "ayar", "not": "not"}


def yukle(): return json.loads(DOSYA.read_text(encoding="utf-8")) if DOSYA.exists() else []
def kaydet(d): DOSYA.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def aktif(stil=None, kadraj=None, tur=None):
    """studio.py calls this. Lessons with durum acik or onayli that match the style/framing filter."""
    return [d for d in yukle() if d["durum"] in ("acik", "onayli") and (tur is None or d["cozum"] == tur)
            and (not d.get("stil") or d["stil"] == stil) and (not d.get("kadraj") or d["kadraj"] == kadraj)]


def wanted(x, text):
    """The 'unless' phrase of lesson x that the shot text asks for, or None. studio.py skips the lesson then."""
    text = text.lower()
    for ph in x.get("unless") or []:
        for m in re.finditer(re.escape(ph.lower()), text):
            if not re.search(r"(\bno|\bnot|\bwithout|\bnever|n't)\W+(\w+\W+){0,2}$", text[:m.start()]): return ph
    return None


if __name__ == "__main__":
    a = sys.argv[1:]; d = yukle()
    if not a or a[0] == "list":
        for x in d:
            if x["durum"] != "kapali": print(f"#{x['no']} [{x['durum']}/{x['cozum']}] {x['sorun']} -> {x.get('ek') or x.get('neden')}")
    elif a[0] == "add":
        keys = {"video": "video", "frame": "kare", "problem": "sorun", "cause": "neden", "fix": "cozum",
                "text": "ek", "style": "stil", "framing": "kadraj"}
        o = {v: a[a.index("--" + k) + 1] for k, v in keys.items() if "--" + k in a}
        o["cozum"] = FIX.get(o.get("cozum", "not"), "not")
        if "--unless" in a: o["unless"] = [w.strip() for w in a[a.index("--unless") + 1].split(",") if w.strip()]
        o.update(no=max([x["no"] for x in d], default=0) + 1, tarih=time.strftime("%Y-%m-%d %H:%M"), durum="acik")
        d.append(o); kaydet(d); print("lesson", o["no"], "saved")
    elif a[0] in ("approve", "close"):
        for x in d:
            if x["no"] == int(a[1]):
                x["durum"] = "onayli" if a[0] == "approve" else "kapali"
                if len(a) > 2: x["kapanis"] = a[2]
        kaydet(d); print("ok")
    else:
        print(__doc__)
