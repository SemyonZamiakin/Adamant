"""Terminal GIF of a real generation, rendered from an `adamant-view --record` log.

Everything shown comes from the recording: token text and the time each token
arrived, the forward-pass times (tokens/s), and the CPU-per-core / RAM snapshots
published by Euclase Monitor during the run. Playback is real time.

usage: python scripts/visuals/terminal_gif.py events.jsonl "command line shown" [out.gif]
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]

W, H = 1000, 620
BG = (10, 31, 39)
WIN = (15, 41, 51)
BAR = (11, 34, 43)
EDGE = (27, 69, 86)
TEAL = (45, 212, 191)
MINT = (153, 246, 228)
TEXT = (248, 250, 252)
MUTED = (143, 179, 188)
DIM = (78, 122, 134)
FPS = 12


def font(names, size):
    for n in names:
        for base in ("C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/"):
            try:
                return ImageFont.truetype(base + n, size)
            except OSError:
                pass
    return ImageFont.load_default()


MONO = font(["consola.ttf", "DejaVuSansMono.ttf"], 15)
MONO_S = font(["consola.ttf", "DejaVuSansMono.ttf"], 13)
MONO_B = font(["consolab.ttf", "DejaVuSansMono-Bold.ttf"], 15)
CW = MONO.getlength("M")


def load(path):
    ev = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    sess = next(e for e in ev if e["ev"] == "session")
    toks = [e for e in ev if e["ev"] == "token"]
    mets = [e for e in ev if e["ev"] == "metrics"]
    end = next((e for e in ev if e["ev"] == "end"), None)
    prompt = next((e for e in ev if e["ev"] == "prompt"), None)
    return sess, prompt, toks, mets, end


def wrap(text, cols):
    lines = [""]
    for ch in text:
        if ch == "\n":
            lines.append("")
            continue
        if len(lines[-1]) >= cols:
            cur = lines[-1]
            sp = cur.rfind(" ")
            if sp > 0:
                lines[-1], carry = cur[:sp], cur[sp + 1:]
            else:
                carry = ""
            lines.append(carry)
        lines[-1] += ch
    return lines


def bar(dr, x, y, w, h, frac, col):
    dr.rounded_rectangle([x, y, x + w, y + h], radius=3, fill=BAR)
    if frac > 0:
        dr.rounded_rectangle([x, y, x + max(4, w * min(frac, 1)), y + h], radius=3, fill=col)


def render(sess, prompt, toks, mets, end, cmd, t, t_start, t_type_end):
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle([12, 12, W - 12, H - 12], radius=12, fill=WIN, outline=EDGE)
    for i, c in enumerate([(251, 113, 133), (251, 191, 36), (52, 211, 153)]):
        dr.ellipse([30 + i * 20, 26, 42 + i * 20, 38], fill=c)
    dr.text((W / 2 - 110, 22), "adamant-view · live", font=MONO_S, fill=DIM)

    # typed command
    typed = cmd[: int(len(cmd) * min(1, max(0, (t - 0.0) / max(t_type_end, 0.01))))]
    y = 56
    for line in wrap("$ " + typed + ("▌" if t < t_type_end + 0.6 and int(t * 3) % 2 == 0 else ""), 104):
        dr.text((30, y), line, font=MONO, fill=MINT)
        y += 20
    if t < t_type_end:
        return im
    y += 8
    rt = t - t_type_end + t_start  # time on the recording clock
    shown = [e for e in toks if e["t"] <= rt]
    m = None
    for e in mets:
        if e["t"] <= rt:
            m = e
    fwd = [e["fwd_ns"] for e in shown[-16:] if e["seq"] > 0 and e["fwd_ns"] > 0]
    tps = 1e9 * len(fwd) / sum(fwd) if fwd else 0.0
    gbs = sess["weight_bytes"] * tps / 1e9
    done = end is not None and end["t"] <= rt

    dr.polygon([(36, y + 2), (43, y + 9), (36, y + 16), (29, y + 9)], fill=TEAL)  # the gem
    dr.text((50, y), "ADAMANT SIMD-VIC", font=MONO_B, fill=TEAL)
    dr.text((210, y), f"{sess['model']} · {sess['isa']} · {sess['threads']} threads", font=MONO, fill=TEXT)
    y += 26
    dr.line([30, y, W - 30, y], fill=EDGE)
    y += 10
    dr.text((30, y), f"tokens {len(shown):<4}", font=MONO, fill=MUTED)
    dr.text((150, y), f"{tps:5.1f} tok/s", font=MONO_B, fill=TEAL)
    # sparkline of instantaneous speed
    hist = [1e9 / e["fwd_ns"] for e in shown[-40:] if e["seq"] > 0 and e["fwd_ns"] > 0]
    if hist:
        mx = max(hist)
        for i, v in enumerate(hist):
            hh = 3 + 15 * v / mx
            dr.rectangle([290 + i * 5, y + 17 - hh, 293 + i * 5, y + 17], fill=TEAL)
    dr.text((510, y), f"weights {gbs:5.1f} GB/s", font=MONO, fill=MUTED)
    if prompt:
        dr.text((740, y), f"prefill {prompt['n'] * 1e9 / prompt['prefill_ns']:.1f} tok/s", font=MONO, fill=MUTED)
    y += 30

    if m:
        cores = m["cores"]
        dr.text((30, y), f"CPU {m['cpu']:5.1f}%", font=MONO, fill=MUTED)
        dr.text((170, y), "per core, from Euclase Monitor (EMS1 over Corundum)" if "Euclase" in m["src"] else "per core, local sampler", font=MONO_S, fill=DIM)
        y += 22
        cols = 4
        cw = (W - 60) / cols
        for i, v in enumerate(cores):
            cx = 30 + (i % cols) * cw
            cy = y + (i // cols) * 20
            dr.text((cx, cy), f"{i:2d}", font=MONO_S, fill=DIM)
            bar(dr, cx + 24, cy + 3, cw - 80, 11, v / 100, TEAL if v < 85 else MINT)
            dr.text((cx + cw - 50, cy), f"{v:3.0f}%", font=MONO_S, fill=TEXT)
        y += 20 * ((len(cores) + cols - 1) // cols) + 6
        if m["ram_total_gb"] > 0:
            frac = m["ram_used_gb"] / m["ram_total_gb"]
            dr.text((30, y), f"RAM {m['ram_used_gb']:5.1f} / {m['ram_total_gb']:.1f} GB", font=MONO, fill=MUTED)
            bar(dr, 270, y + 4, W - 380, 12, frac, TEAL)
            dr.text((W - 96, y), f"{100 * frac:3.0f}%", font=MONO, fill=TEXT)
            y += 26
    dr.line([30, y, W - 30, y], fill=EDGE)
    y += 10
    text = "".join(e["text"] for e in shown).lstrip()
    lines = wrap(text, int((W - 70) / CW))
    max_lines = int((H - 60 - y) / 20)
    for line in lines[-max_lines:]:
        dr.text((34, y), line, font=MONO, fill=TEXT)
        y += 20
    if not done and int(t * 4) % 2 == 0:
        dr.text((34 + CW * len(lines[-1] if lines else ""), y - 20), "▌", font=MONO, fill=TEAL)
    status = (f"done · {end['n']} tokens · {(end['n'] - 1) * 1e9 / end['gen_ns']:.1f} tok/s average · {end['dropped']} records dropped"
              if done else "streaming over Corundum HFT ring /adamant-tokens")
    dr.text((30, H - 42), status, font=MONO_S, fill=TEAL if done else DIM)
    return im


def main():
    path, cmd = sys.argv[1], sys.argv[2]
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "docs" / "img" / "terminal.gif"
    sess, prompt, toks, mets, end = load(path)
    t_start = sess["t"]
    t_type_end = 1.6
    t_total = t_type_end + ((end["t"] if end else toks[-1]["t"]) - t_start) + 3.0
    frames = []
    n = int(t_total * FPS)
    for k in range(n):
        frames.append(render(sess, prompt, toks, mets, end, cmd, k / FPS, t_start, t_type_end))
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    durations = [int(1000 / FPS)] * len(pal)
    durations[-1] = 2500
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB, {len(frames)} frames, {t_total:.1f} s)")


if __name__ == "__main__":
    main()
