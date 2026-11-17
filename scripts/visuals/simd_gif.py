"""Animated GIF: one Q4_K x Q8_K sub-block dot product, the way the AVX2 kernel does it.

32 lanes of 4-bit weights and int8 activations -> vpmaddubsw (16 x int16) ->
vpmaddwd with the sub-block scale (8 x int32) -> horizontal sum -> x d.
The numbers are real: every stage is computed and the final integer is checked
against a plain scalar loop. Writes docs/img/simd-dot.gif.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "img" / "simd-dot.gif"

W, H = 1000, 470
PETROL = (15, 41, 51)
PANEL = (11, 34, 43)
EDGE = (27, 69, 86)
TEAL = (45, 212, 191)
MINT = (153, 246, 228)
TEXT = (248, 250, 252)
MUTED = (143, 179, 188)
DIM = (78, 122, 134)
AMBER = (251, 191, 36)
CORAL = (251, 113, 133)


def font(names, size):
    for n in names:
        for base in ("C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/"):
            try:
                return ImageFont.truetype(base + n, size)
            except OSError:
                continue
    return ImageFont.load_default()


MONO = lambda s: font(["consola.ttf", "DejaVuSansMono.ttf"], s)  # noqa: E731
MONOB = lambda s: font(["consolab.ttf", "DejaVuSansMono-Bold.ttf"], s)  # noqa: E731
SANS = lambda s: font(["segoeui.ttf", "DejaVuSans.ttf"], s)  # noqa: E731
SANSB = lambda s: font(["seguisb.ttf", "DejaVuSans-Bold.ttf"], s)  # noqa: E731

random.seed(11)
q4 = [random.randint(0, 15) for _ in range(32)]
q8 = [random.randint(-127, 127) for _ in range(32)]
scale = 37
d = 0.000183
p16 = [q4[2 * i] * q8[2 * i] + q4[2 * i + 1] * q8[2 * i + 1] for i in range(16)]
p32 = [scale * (p16[2 * i] + p16[2 * i + 1]) for i in range(8)]
total = sum(p32)
assert total == scale * sum(a * b for a, b in zip(q4, q8))  # same integer as the scalar loop

LANE_W, GAP, X0 = 27, 2, 36


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lane_x(i, n, width_total=32 * (LANE_W + GAP)):
    w = width_total / n
    return X0 + i * w, w - GAP


def box(dr, x, y, w, h, fill, outline=None, r=5):
    dr.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=fill, outline=outline, width=1)


def ctext(dr, cx, y, s, f, fill):
    tw = dr.textlength(s, font=f)
    dr.text((cx - tw / 2, y), s, font=f, fill=fill)


def frame(stage, t):
    im = Image.new("RGB", (W, H), PETROL)
    dr = ImageDraw.Draw(im)
    dr.text((36, 22), "AVX2 dot product · one Q4_K sub-block × Q8_K activations", font=SANSB(22), fill=TEXT)
    steps = ["1  load + unpack", "2  vpmaddubsw", "3  vpmaddwd × scale", "4  horizontal sum", "5  × d  (float)"]
    for i, s in enumerate(steps):
        active = i == stage
        x = 36 + i * 190
        box(dr, x, 62, 180, 28, TEAL if active else PANEL, EDGE, 14)
        ctext(dr, x + 90, 67, s, MONOB(13) if active else MONO(13), PETROL if active else MUTED)

    f9, f10 = MONO(10), MONO(11)
    # row 1: weights (unsigned nibbles), row 2: activations
    yw, yx = 128, 178
    dr.text((36, yw - 18), "w  · 32 × uint4  (low nibbles of one 32-byte load)", font=SANS(13), fill=MUTED)
    dr.text((36, yx + 30), "x  · 32 × int8   (Q8_K activation block)", font=SANS(13), fill=MUTED)
    appear = ease(t * 2) if stage == 0 else 1.0
    fade_rows = 1.0 if stage <= 1 else 0.45
    for i in range(32):
        x, w = lane_x(i, 32)
        k = ease((appear * 40 - i) / 8) if stage == 0 else 1.0
        cw = mix(PANEL, mix(EDGE, TEAL, q4[i] / 15), k * fade_rows)
        box(dr, x, yw, w, 26, cw)
        if k > 0.5:
            ctext(dr, x + w / 2, yw + 6, str(q4[i]), f10, PETROL if q4[i] > 7 and fade_rows == 1 else TEXT)
        cx = mix(PANEL, mix((40, 60, 70), MINT if q8[i] >= 0 else CORAL, abs(q8[i]) / 127), k * fade_rows)
        box(dr, x, yx, w, 26, cx)
        if k > 0.5:
            ctext(dr, x + w / 2, yx + 7, str(q8[i]), f9, TEXT)

    # stage 2: 16 int16
    y16 = 262
    if stage >= 1:
        tt = ease(t * 1.6) if stage == 1 else 1.0
        dr.text((36, y16 - 18), "vpmaddubsw → 16 × int16:  w[2i]·x[2i] + w[2i+1]·x[2i+1]", font=SANS(13), fill=MUTED)
        for i in range(16):
            xa, wa = lane_x(2 * i, 32)
            xb, _ = lane_x(2 * i + 1, 32)
            x, w = lane_x(i, 16)
            if stage == 1:
                for src in (xa + wa / 2, xb + wa / 2):
                    sx, sy = src, yx + 26
                    ex, ey = x + w / 2, y16
                    mx, my = lerp(sx, ex, tt), lerp(sy, ey, tt)
                    dr.line([sx, sy, mx, my], fill=mix(EDGE, TEAL, 0.6), width=1)
            k = ease((tt * 22 - i) / 6) if stage == 1 else 1.0
            box(dr, x, y16, w, 28, mix(PANEL, mix(EDGE, AMBER, min(1, abs(p16[i]) / 2500)), k * (1.0 if stage <= 2 else 0.5)))
            if k > 0.5:
                ctext(dr, x + w / 2, y16 + 8, str(p16[i]), f10, TEXT)

    # stage 3: 8 int32 x scale
    y32 = 330
    if stage >= 2:
        tt = ease(t * 1.6) if stage == 2 else 1.0
        dr.text((36, y32 - 18), f"vpmaddwd with scale[j] = {scale} → 8 × int32", font=SANS(13), fill=MUTED)
        for i in range(8):
            x, w = lane_x(i, 8)
            if stage == 2:
                for s in (2 * i, 2 * i + 1):
                    xs, ws = lane_x(s, 16)
                    dr.line([xs + ws / 2, y16 + 28, lerp(xs + ws / 2, x + w / 2, tt), lerp(y16 + 28, y32, tt)], fill=mix(EDGE, AMBER, 0.6), width=1)
            k = ease((tt * 12 - i) / 4) if stage == 2 else 1.0
            box(dr, x, y32, w, 30, mix(PANEL, mix(EDGE, TEAL, min(1, abs(p32[i]) / 150000)), k * (1.0 if stage <= 3 else 0.55)))
            if k > 0.5:
                ctext(dr, x + w / 2, y32 + 8, str(p32[i]), MONO(12), TEXT if stage != 2 else PETROL)

    # stage 4: tree
    yt = 392
    if stage >= 3:
        tt = ease(t * 1.4) if stage == 3 else 1.0
        levels = [p32, [p32[i] + p32[i + 4] for i in range(4)]]
        levels.append([levels[1][i] + levels[1][i + 2] for i in range(2)])
        levels.append([levels[2][0] + levels[2][1]])
        assert levels[3][0] == total
        cx_total = X0 + 16 * (LANE_W + GAP)
        shown = min(3, int(tt * 3.2))
        dr.text((36, yt + 6), "hadd tree: 8 → 4 → 2 → 1   (integer: exact in any order)", font=SANS(13), fill=MUTED)
        for lvl in range(1, shown + 1):
            vals = levels[lvl]
            for i, v in enumerate(vals):
                x = cx_total + 120 + (lvl - 1) * 0 + (i - (len(vals) - 1) / 2) * 110
                y = yt - 6 + (lvl - 1) * 0
                if lvl == shown:
                    box(dr, x - 50, y, 100, 28, PANEL if lvl < 3 else TEAL, TEAL)
                    ctext(dr, x, y + 7, str(v), MONOB(13), TEXT if lvl < 3 else PETROL)

    if stage >= 4:
        tt = ease(t * 1.5)
        res = total * d
        box(dr, 600, 418, 364, 36, mix(PANEL, TEAL, tt), TEAL, 10)
        ctext(dr, 782, 425, f"{total} × d = {res:+.6f}", MONOB(15), PETROL if tt > 0.5 else TEXT)
        dr.text((36, 430), "bit-identical to the scalar loop (checked on 3.8 M results in test_kernels)", font=SANS(13), fill=MINT)
    return im


def main():
    frames, durations = [], []
    plan = [(0, 14, 70), (1, 16, 70), (2, 14, 70), (3, 12, 80), (4, 12, 80)]
    for stage, n, ms in plan:
        for k in range(n):
            frames.append(frame(stage, k / (n - 1)))
            durations.append(ms)
        frames.append(frame(stage, 1.0))
        durations.append(900 if stage < 4 else 2600)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB, {len(frames)} frames), total = {total}")


if __name__ == "__main__":
    main()
