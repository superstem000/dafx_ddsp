"""Build a self-contained MUSHRA page from what ds_compare_audio.py wrote.

    python scripts/ds_compare_audio.py --arms synth_hybrid synth_mag synth_log \
        --domain id --n 8 --out compare_id --no-png --no-tar
    python scripts/make_mushra.py --dir compare_id --out mushra_id.html

Then scp the one HTML file and open it. No server, no assets, no network -- the
audio is embedded, which is also what keeps the test honest offline.

WHY MUSHRA RATHER THAN "LISTEN TO THEM SIDE BY SIDE". ds_compare_audio already
does the latter, and it is how we discovered that a masking gate we had called
inaudible was in fact audible on the attack. What it cannot do is tell you
whether a difference you hear is a difference at all: with no hidden reference
you have no estimate of your own false-positive rate, and with no anchor you
have no scale. ITU-R BS.1534 exists to supply both, and the parts that make it
work are cheap to implement:

  hidden reference  the unprocessed target, unlabelled, among the conditions.
                    Rating it below ~90 means that trial's ratings are noise --
                    reported here per trial rather than silently dropped.
  anchor            a 3.5 kHz lowpass of the target, which fixes what "bad"
                    means so ratings are comparable across trials and sessions.
  blinding          condition order reshuffled per trial, with the key written
                    into the results rather than shown.
  same instant      switching between stimuli keeps the playback position, so
                    a passage can be compared rather than remembered.

ONE DEVIATION FROM THE STANDARD, STATED. BS.1534 specifies a 3.5 kHz low anchor
and a 7 kHz mid anchor for full-band material. diffsynth runs at 16 kHz, so
Nyquist is 8 kHz and a 7 kHz anchor is very nearly the signal itself; the mid
anchor is therefore omitted rather than included as something that would sit
indistinguishably next to the hidden reference and inflate every score.

AND THIS IS A SELF-TEST, NOT AN EXPERIMENT. One listener who knows the
hypothesis is not a MUSHRA panel, and the output says so. It is worth doing
because the ID numbers are metric differences of a few percent and nobody has
yet checked whether they correspond to anything audible -- which is the same
question the masking work asked on the real branch, and got a surprising answer
to.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter


def lowpass(x: np.ndarray, sr: int, cutoff: float, taps: int = 511) -> np.ndarray:
    """Linear-phase FIR lowpass, the BS.1534 anchor.

    firwin rather than a brickwall in the FFT domain: a rectangular cut leaves
    ringing that is itself an artefact, and the anchor is supposed to sound like
    band-limiting rather than like a bug.
    """
    if cutoff >= sr / 2:
        return x
    b = firwin(taps, cutoff, fs=sr)
    y = lfilter(b, [1.0], x)
    # Compensate the FIR's constant group delay so the anchor stays aligned with
    # every other stimulus -- MUSHRA switching compares the same instant, and a
    # 255-sample offset would be audible as a click on switch.
    return np.roll(y, -(taps // 2))


def wav_b64(x: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, x, sr, subtype="PCM_16", format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def discover(d: Path):
    """clip{N}_target.wav and clip{N}_{arm}.wav, as ds_compare_audio writes them."""
    clips, arms = {}, set()
    for p in sorted(d.glob("clip*_*.wav")):
        m = re.match(r"clip(\d+)_(.+)\.wav$", p.name)
        if not m:
            continue
        idx, tag = int(m.group(1)), m.group(2)
        clips.setdefault(idx, {})[tag] = p
        if tag != "target":
            arms.add(tag)
    usable = {i: v for i, v in clips.items() if "target" in v}
    return usable, sorted(arms)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", type=Path, required=True,
                   help="Directory ds_compare_audio.py wrote")
    p.add_argument("--out", type=Path, default=Path("mushra.html"))
    p.add_argument("--anchor-hz", type=float, default=3500.0)
    p.add_argument("--trials", type=int, default=0,
                   help="How many clips to use, 0 = all found")
    p.add_argument("--seed", type=int, default=0,
                   help="Blinding order. Recorded in the results.")
    p.add_argument("--title", default="MUSHRA -- diffsynth in-domain")
    args = p.parse_args()

    clips, arms = discover(args.dir)
    if not clips:
        raise SystemExit(f"no clip*_target.wav under {args.dir}")
    idxs = sorted(clips)[: args.trials or None]
    print(f"{len(idxs)} trials, arms: {', '.join(arms)}")

    rng = random.Random(args.seed)
    trials, key, total = [], [], 0
    for n, i in enumerate(idxs):
        ref, sr = sf.read(clips[i]["target"], dtype="float32", always_2d=False)
        if ref.ndim > 1:
            ref = ref.mean(axis=1)
        stimuli = [("reference", ref)]
        for a in arms:
            if a not in clips[i]:
                print(f"  clip{i}: missing {a}, trial skipped")
                stimuli = None
                break
            y, sr2 = sf.read(clips[i][a], dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr2 != sr:
                raise SystemExit(f"clip{i} {a}: {sr2} Hz against target {sr}")
            stimuli.append((a, y))
        if stimuli is None:
            continue
        stimuli.append((f"anchor{int(args.anchor_hz)}",
                        lowpass(ref, sr, args.anchor_hz)))

        order = list(range(len(stimuli)))
        rng.shuffle(order)
        conds = []
        for slot, s in enumerate(order):
            name, audio = stimuli[s]
            b = wav_b64(audio, sr)
            total += len(b)
            conds.append({"label": chr(65 + slot), "audio": b})
            key.append({"trial": n, "label": chr(65 + slot), "condition": name})
        trials.append({"n": n, "clip": i, "ref": wav_b64(ref, sr),
                       "conds": conds, "sr": sr})
        total += len(trials[-1]["ref"])

    mb = total / 1e6
    print(f"embedded audio: {mb:.1f} MB")
    if mb > 60:
        print("  large for a single page -- consider fewer --trials")

    payload = json.dumps({"trials": trials, "key": key, "seed": args.seed,
                          "arms": arms, "anchor_hz": args.anchor_hz})
    args.out.write_text(PAGE.replace("__TITLE__", args.title)
                            .replace("__DATA__", payload))
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    print("\nscp it over and open it. Ratings stay in the page until you export;")
    print("the condition key is embedded, so scoring needs nothing else.")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;background:#12141a;color:#e8e8ea}
 h1{font-size:18px;margin:0 0 4px} .sub{color:#9a9aa4;margin-bottom:20px}
 .card{background:#1b1e26;border:1px solid #2a2e3a;border-radius:10px;padding:18px;margin-bottom:16px}
 .row{display:flex;align-items:center;gap:14px;margin:10px 0}
 .lab{width:34px;font-weight:600;font-size:16px}
 button{background:#2a2e3a;color:#e8e8ea;border:1px solid #3a3f4d;border-radius:6px;
        padding:7px 13px;cursor:pointer;font:inherit}
 button:hover{background:#343948} button.on{background:#4a6ee0;border-color:#4a6ee0}
 input[type=range]{flex:1;accent-color:#4a6ee0}
 .val{width:34px;text-align:right;font-variant-numeric:tabular-nums}
 .scale{display:flex;justify-content:space-between;color:#7a7a84;font-size:11px;
        margin:0 0 6px 48px;padding-right:44px}
 .refbtn{background:#3a4d2a;border-color:#4d6636}
 textarea{width:100%;height:160px;background:#0e1015;color:#b8c0d0;border:1px solid #2a2e3a;
          border-radius:6px;padding:10px;font:12px/1.4 ui-monospace,monospace}
 .warn{color:#e0a44a} .note{color:#9a9aa4;font-size:12px;margin-top:6px}
</style></head><body>
<h1>__TITLE__</h1>
<div class="sub">Rate each condition against the reference. One is the hidden reference
 and one is a lowpass anchor &mdash; both unlabelled.</div>
<div id="app"></div>
<div class="card">
 <button onclick="exportResults()">Export results</button>
 <span class="note" id="progress"></span>
 <textarea id="out" placeholder="Results appear here."></textarea>
</div>
<script>
const D = __DATA__;
let cur = null;           // {audio, startedAt, offset}
const ratings = {};

function stopAll(){
  document.querySelectorAll('audio').forEach(a=>{a.pause();});
  document.querySelectorAll('button.on').forEach(b=>b.classList.remove('on'));
}
// Switching keeps the playback position: MUSHRA compares the same instant, and
// restarting each stimulus from zero would make that impossible.
function play(id, btn){
  const el = document.getElementById(id);
  const pos = cur && cur.el && !cur.el.paused ? cur.el.currentTime : (cur ? cur.pos : 0);
  const wasSame = cur && cur.el === el && !el.paused;
  stopAll();
  if (wasSame){ cur = {el, pos}; return; }
  el.currentTime = Math.min(pos, Math.max(0, el.duration || 0) - 0.01) || 0;
  el.play(); btn.classList.add('on');
  cur = {el, pos};
  el.ontimeupdate = ()=>{ cur.pos = el.currentTime; };
  el.onended = ()=>{ el.currentTime = 0; el.play(); };   // loop
}

const app = document.getElementById('app');
D.trials.forEach(t=>{
  const c = document.createElement('div'); c.className='card';
  let h = `<b>Trial ${t.n+1}</b> <span class="note">(clip ${t.clip})</span>
   <div class="row"><span class="lab"></span>
     <button class="refbtn" onclick="play('r${t.n}',this)">Reference</button>
     <audio id="r${t.n}" src="data:audio/wav;base64,${t.ref}"></audio>
     <button onclick="stopAll()">Stop</button></div>
   <div class="scale"><span>Bad</span><span>Poor</span><span>Fair</span>
     <span>Good</span><span>Excellent</span></div>`;
  t.conds.forEach(k=>{
    const id = `t${t.n}${k.label}`;
    h += `<div class="row"><span class="lab">${k.label}</span>
      <button onclick="play('${id}',this)">Play</button>
      <audio id="${id}" src="data:audio/wav;base64,${k.audio}"></audio>
      <input type="range" min="0" max="100" value="0"
             oninput="rate(${t.n},'${k.label}',this.value)">
      <span class="val" id="v${id}">0</span></div>`;
  });
  c.innerHTML = h; app.appendChild(c);
});

function rate(n,label,v){
  ratings[n+'|'+label] = +v;
  document.getElementById('vt'+n+label).textContent = v;
  const done = Object.keys(ratings).length;
  const need = D.trials.length * D.trials[0].conds.length;
  document.getElementById('progress').textContent = ` ${done}/${need} rated`;
}

function exportResults(){
  const rows = D.key.map(k=>({...k, score: ratings[k.trial+'|'+k.label] ?? null}));
  // Post-screening, reported rather than applied. BS.1534 excludes a listener
  // who scores the hidden reference below 90 on more than 15% of trials; with
  // one self-testing listener the right move is to show the number and let it
  // be judged, not to silently drop data.
  const hr = rows.filter(r=>r.condition==='reference' && r.score!==null);
  const bad = hr.filter(r=>r.score < 90).length;
  const byArm = {};
  rows.forEach(r=>{ if(r.score!==null){ (byArm[r.condition] ||= []).push(r.score); } });
  const mean = a => a.reduce((x,y)=>x+y,0)/a.length;
  const summary = Object.entries(byArm).map(([k,v])=>
      `${k.padEnd(20)} n=${String(v.length).padStart(2)}  mean ${mean(v).toFixed(1)}`)
    .join('\n');
  document.getElementById('out').value =
    `seed ${D.seed}   anchor ${D.anchor_hz} Hz\n` +
    `hidden reference scored <90 on ${bad}/${hr.length} trials` +
    (hr.length && bad/hr.length > 0.15
      ? '   <-- above the 15% BS.1534 post-screen threshold; treat this session as unreliable\n'
      : '\n') +
    `\n${summary}\n\n` + JSON.stringify(rows, null, 1);
}
</script></body></html>
"""


if __name__ == "__main__":
    main()
