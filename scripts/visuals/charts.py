"""Benchmark charts (SVG) generated from the raw CSV logs in docs/bench/.

  docs/img/chart-kernels.svg   matrix-vector throughput per format, scalar vs AVX2, 1 and N threads
  docs/img/chart-tokens.svg    generation tokens/s vs threads, per model, scalar vs AVX2

Palette validated with the dataviz validator on the petrol surface (#0F2933, dark mode):
scalar #D06A4C, AVX2 #0E9F90 (lightness band, chroma, CVD and contrast all pass).
Run: python scripts/visuals/charts.py
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "docs" / "bench"
IMG = ROOT / "docs" / "img"

SURFACE = "#0F2933"
GRID = "#1B4556"
INK = "#F8FAFC"
INK2 = "#8FB3BC"
INK3 = "#5E828C"
SCALAR = "#D06A4C"
AVX2 = "#0E9F90"
CEIL = "#B08A14"
SANS = "Segoe UI, Inter, Helvetica, Arial, sans-serif"
MONO = "Consolas, 'JetBrains Mono', monospace"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_max(v):
    for step in (1, 2, 2.5, 5):
        for mag in (1, 10, 100, 1000):
            if step * mag * 5 >= v:
                return step * mag * 5, step * mag
    return v, v / 5


def bar_end(x, y, w, h, color):
    """Horizontal bar anchored at x with a 4px rounded data end."""
    r = min(4, h / 2, max(w, 0) / 2)
    if w <= 0:
        return ""
    return (f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" fill="{color}"/>')


def legend(x, y, items):
    out, cx = [], x
    for label, color, dashed in items:
        if dashed:
            out.append(f'<line x1="{cx}" y1="{y-4}" x2="{cx+18}" y2="{y-4}" stroke="{color}" stroke-width="2" stroke-dasharray="4 3"/>')
        else:
            out.append(f'<rect x="{cx}" y="{y-10}" width="12" height="12" rx="3" fill="{color}"/>')
        out.append(f'<text x="{cx+24}" y="{y}" fill="{INK2}" font-family="{SANS}" font-size="13">{esc(label)}</text>')
        cx += 34 + 7.2 * len(label)
    return "".join(out)


def chart_kernels(rows, cpu, out):
    data = defaultdict(dict)  # (threads) -> {(type, isa): gbs}
    ceil = {}
    shape = None
    for r in rows:
        t = int(r["threads"])
        if r["type"] == "stream":
            ceil[t] = float(r["gb_s"])
            continue
        if shape is None:
            shape = (int(r["rows"]), int(r["cols"]))
        if (int(r["rows"]), int(r["cols"])) != shape:
            continue  # first (large) shape only
        data[t][(r["type"], r["isa"])] = float(r["gb_s"])
    threads = sorted(data)
    types = ["Q4_K", "Q6_K", "Q8_0", "F16"]
    W, H = 1200, 410
    pw = (W - 120) / len(threads)
    vmax = max([v for d in data.values() for v in d.values()] + list(ceil.values()))
    top, step = nice_max(vmax * 1.08)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Matrix-vector throughput, scalar vs AVX2">',
         f'<rect width="{W}" height="{H}" rx="16" fill="{SURFACE}"/>',
         f'<text x="40" y="46" fill="{INK}" font-family="{SANS}" font-size="22" font-weight="700">Matrix × vector throughput, GB/s of weights read</text>',
         f'<text x="40" y="72" fill="{INK2}" font-family="{SANS}" font-size="14">{esc(cpu)} · {shape[0]}×{shape[1]} weights per format (larger than L3) · measured with adamant kbench</text>',
         legend(40, 104, [("scalar reference", SCALAR, False), ("AVX2", AVX2, False), ("memory read bandwidth, AVX2 loads (same machine)", CEIL, True)])]
    for pi, t in enumerate(threads):
        x0 = 100 + pi * pw
        plot_w = pw - 70
        y0, gh = 140, 52
        o.append(f'<text x="{x0}" y="{y0 - 8}" fill="{INK}" font-family="{SANS}" font-size="15" font-weight="600">{t} thread{"s" if t > 1 else ""}</text>')
        for g in range(0, int(top / step) + 1):
            gx = x0 + plot_w * g * step / top
            o.append(f'<line x1="{gx:.1f}" y1="{y0+4}" x2="{gx:.1f}" y2="{y0 + gh * len(types) + 4}" stroke="{GRID}" stroke-width="1"/>')
            o.append(f'<text x="{gx:.1f}" y="{y0 + gh * len(types) + 22}" text-anchor="middle" fill="{INK3}" font-family="{MONO}" font-size="12">{g*step:g}</text>')
        if t in ceil:
            cx = x0 + plot_w * ceil[t] / top
            o.append(f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0 + gh * len(types) + 6}" stroke="{CEIL}" stroke-width="2" stroke-dasharray="5 4"/>')
            o.append(f'<text x="{cx - 4:.1f}" y="{y0 - 8}" text-anchor="end" fill="{INK2}" font-family="{MONO}" font-size="12">read {ceil[t]:.1f}</text>')
        for ti, ty in enumerate(types):
            gy = y0 + 8 + ti * gh
            if pi == 0:
                o.append(f'<text x="{x0 - 12}" y="{gy + 25}" text-anchor="end" fill="{INK2}" font-family="{MONO}" font-size="13">{ty}</text>')
            for k, (isa, col) in enumerate((("scalar", SCALAR), ("avx2", AVX2))):
                v = data[t].get((ty, isa))
                if v is None:
                    continue
                by = gy + k * 20
                bw = plot_w * v / top
                o.append(bar_end(x0, by, bw, 18, col))
                o.append(f'<text x="{x0 + bw + 6:.1f}" y="{by + 14}" fill="{INK}" stroke="{SURFACE}" stroke-width="4" paint-order="stroke" font-family="{MONO}" font-size="12">{v:.1f}</text>')
    o.append("</svg>")
    out.write_text("\n".join(o), encoding="utf-8")


def chart_tokens(series, cpu, out):
    """series: {model_label: {isa: {threads: tok_s}}}"""
    models = list(series)
    W, H = 1200, 440
    pw = (W - 80) / len(models)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Generation speed vs threads">',
         f'<rect width="{W}" height="{H}" rx="16" fill="{SURFACE}"/>',
         f'<text x="40" y="46" fill="{INK}" font-family="{SANS}" font-size="22" font-weight="700">Generation speed, tokens per second</text>',
         f'<text x="40" y="72" fill="{INK2}" font-family="{SANS}" font-size="14">TinyLlama-1.1B-Chat · {esc(cpu)} · greedy, median of runs · measured with adamant bench</text>',
         legend(40, 104, [("scalar reference", SCALAR, False), ("AVX2", AVX2, False)])]
    allv = [v for m in series.values() for d in m.values() for v in d.values()]
    top, step = nice_max(max(allv) * 1.12)
    for mi, m in enumerate(models):
        x0, y0 = 90 + mi * pw, 140
        plot_w, plot_h = pw - 110, 230
        threads = sorted({t for d in series[m].values() for t in d})
        tmax = max(threads)
        o.append(f'<text x="{x0}" y="{y0 - 10}" fill="{INK}" font-family="{SANS}" font-size="15" font-weight="600">{esc(m)}</text>')
        for g in range(0, int(top / step) + 1):
            gy = y0 + plot_h - plot_h * g * step / top
            o.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0 + plot_w}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
            o.append(f'<text x="{x0 - 10}" y="{gy + 4:.1f}" text-anchor="end" fill="{INK3}" font-family="{MONO}" font-size="12">{g*step:g}</text>')
        for t in threads:
            tx = x0 + plot_w * (t - 1) / max(tmax - 1, 1)
            o.append(f'<text x="{tx:.1f}" y="{y0 + plot_h + 20}" text-anchor="middle" fill="{INK3}" font-family="{MONO}" font-size="12">{t}</text>')
        o.append(f'<text x="{x0 + plot_w/2:.1f}" y="{y0 + plot_h + 42}" text-anchor="middle" fill="{INK2}" font-family="{SANS}" font-size="13">threads</text>')
        for isa, col in (("scalar", SCALAR), ("avx2", AVX2)):
            d = series[m].get(isa, {})
            if not d:
                continue
            pts = [(x0 + plot_w * (t - 1) / max(tmax - 1, 1), y0 + plot_h - plot_h * d[t] / top, d[t]) for t in sorted(d)]
            o.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
            for x, y, _ in pts:
                o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{col}" stroke="{SURFACE}" stroke-width="2"/>')
            best = max(pts, key=lambda p: p[2])
            o.append(f'<text x="{best[0]:.1f}" y="{best[1] - 12:.1f}" text-anchor="middle" fill="{INK}" font-family="{MONO}" font-size="13">{best[2]:.1f}</text>')
    o.append("</svg>")
    out.write_text("\n".join(o), encoding="utf-8")


def main():
    kcsv = BENCH / "kbench-windows-i9-9900k.csv"
    cpu = "Intel Core i9-9900K, Windows 11"
    if kcsv.exists():
        with kcsv.open() as f:
            chart_kernels(list(csv.DictReader(f)), cpu, IMG / "chart-kernels.svg")
        print("wrote chart-kernels.svg")
    series = {}
    for path in sorted(BENCH.glob("bench-windows-*.csv")):
        acc = defaultdict(lambda: defaultdict(list))
        label = None
        with path.open() as f:
            for r in csv.DictReader(f):
                acc[r["isa"]][int(r["threads"])].append(float(r["gen_tok_s"]))
        label = "Q4_K_M (669 MB)" if "q4_k_m" in path.name else "Q8_0 (1.17 GB)" if "q8_0" in path.name else path.stem
        series[label] = {isa: {t: statistics.median(v) for t, v in d.items()} for isa, d in acc.items()}
    if series:
        chart_tokens(series, cpu, IMG / "chart-tokens.svg")
        print("wrote chart-tokens.svg")


if __name__ == "__main__":
    main()
