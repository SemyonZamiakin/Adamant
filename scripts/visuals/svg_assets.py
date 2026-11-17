"""Static and animated SVG assets for the Adamant README (Euclase palette).

Writes docs/img/banner.svg, docs/img/q4k-layout.svg, docs/img/tiling.svg.
Run: python scripts/visuals/svg_assets.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "docs" / "img"

PETROL = "#0F2933"
PETROL_DEEP = "#0A1F27"
PANEL = "#0B222B"
EDGE = "#1B4556"
TEAL = "#2DD4BF"
MINT = "#99F6E4"
TEXT = "#F8FAFC"
MUTED = "#8FB3BC"
DIM = "#4E7A86"
AMBER = "#FBBF24"
CORAL = "#FB7185"
SANS = "Segoe UI, Inter, Helvetica, Arial, sans-serif"
MONO = "Consolas, 'JetBrains Mono', 'DejaVu Sans Mono', monospace"


def banner() -> str:
    w, h = 1280, 340
    lanes = []
    # ymm register: 8 lanes, staggered pulse = a dot product being accumulated
    for i in range(8):
        x = 962 + i * 34
        delay = i * 0.18
        lanes.append(
            f'<rect class="lane" x="{x}" y="96" width="25" height="150" rx="5" '
            f'style="animation-delay:{delay:.2f}s" fill="url(#gem)"/>'
            f'<rect x="{x}" y="96" width="25" height="150" rx="5" fill="none" stroke="{EDGE}" stroke-width="1.5"/>'
        )
    tokens = ["The", "▁quick", "▁brown", "▁fox", "▁jumps", "▁over", "▁the", "▁lazy", "▁dog", "."]
    chips, x = [], 0
    for t in tokens:
        wpx = 18 + 10 * len(t)
        chips.append(
            f'<g transform="translate({x} 0)"><rect width="{wpx}" height="26" rx="13" fill="{PANEL}" stroke="{EDGE}"/>'
            f'<text x="{wpx/2}" y="18" text-anchor="middle" fill="{MINT}" font-size="14" font-family="{MONO}">{t}</text></g>'
        )
        x += wpx + 10
    strip_w = x
    chips_svg = "".join(chips)
    grid = "".join(f'<line x1="{gx}" y1="0" x2="{gx}" y2="{h}" stroke="#123642" stroke-width="1"/>' for gx in range(0, w, 40))
    grid += "".join(f'<line x1="0" y1="{gy}" x2="{w}" y2="{gy}" stroke="#123642" stroke-width="1"/>' for gy in range(0, h, 40))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Adamant SIMD-VIC by Euclase">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{PETROL}"/><stop offset="1" stop-color="{PETROL_DEEP}"/></linearGradient>
    <linearGradient id="gem" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{MINT}"/><stop offset="1" stop-color="{TEAL}"/></linearGradient>
    <radialGradient id="glow" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="{TEAL}" stop-opacity=".45"/><stop offset="1" stop-color="{TEAL}" stop-opacity="0"/></radialGradient>
    <linearGradient id="fade" x1="0" x2="1"><stop offset="0" stop-color="{PETROL}" stop-opacity="1"/><stop offset=".12" stop-color="{PETROL}" stop-opacity="0"/><stop offset=".88" stop-color="{PETROL_DEEP}" stop-opacity="0"/><stop offset="1" stop-color="{PETROL_DEEP}" stop-opacity="1"/></linearGradient>
    <clipPath id="stripclip"><rect x="0" y="282" width="{w}" height="40"/></clipPath>
    <style>
      .lane {{ transform-box: fill-box; transform-origin: 50% 100%; animation: pulse 1.8s ease-in-out infinite; }}
      @keyframes pulse {{ 0%,100% {{ transform: scaleY(.18); opacity:.55 }} 50% {{ transform: scaleY(1); opacity:1 }} }}
      .halo {{ transform-box: fill-box; transform-origin: center; animation: breathe 3.2s ease-in-out infinite; }}
      @keyframes breathe {{ 0%,100% {{ transform: scale(.85); opacity:.55 }} 50% {{ transform: scale(1.1); opacity:1 }} }}
      .strip {{ animation: flow 14s linear infinite; }}
      @keyframes flow {{ from {{ transform: translateX(0) }} to {{ transform: translateX(-{strip_w}px) }} }}
      .caret {{ animation: blink 1s steps(1) infinite; }}
      @keyframes blink {{ 50% {{ opacity: 0 }} }}
      .sum {{ animation: sumglow 1.8s ease-in-out infinite; }}
      @keyframes sumglow {{ 0%,100% {{ opacity:.35 }} 60% {{ opacity:1 }} }}
    </style>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#bg)"/>
  <g opacity=".55">{grid}</g>
  <!-- the gem: adamant -->
  <g transform="translate(150 160)">
    <circle class="halo" r="115" fill="url(#glow)"/>
    <polygon points="0,-88 70,-30 0,92 -70,-30" fill="url(#gem)"/>
    <polygon points="0,-88 70,-30 0,-12 -70,-30" fill="{TEXT}" fill-opacity=".38"/>
    <polygon points="-70,-30 0,-12 0,92" fill="{PETROL}" fill-opacity=".22"/>
    <polygon points="70,-30 0,-12 0,92" fill="{PETROL}" fill-opacity=".40"/>
    <line x1="-35" y1="-59" x2="0" y2="-12" stroke="{PETROL}" stroke-opacity=".35" stroke-width="2"/>
    <line x1="35" y1="-59" x2="0" y2="-12" stroke="{PETROL}" stroke-opacity=".35" stroke-width="2"/>
  </g>
  <g font-family="{SANS}">
    <text x="280" y="138" fill="{TEXT}" font-size="66" font-weight="700" letter-spacing="-1.5">Adamant <tspan fill="{TEAL}">SIMD-VIC</tspan></text>
    <text x="284" y="184" fill="{MUTED}" font-size="22">SIMD-Vectorized Inference Core for quantized LLMs</text>
    <text x="284" y="218" fill="{DIM}" font-size="17" font-family="{MONO}">AVX2 · Q4_K · Q6_K · Q8_0 · zero-copy mmap · alignas(64) · C++17<tspan class="caret" fill="{TEAL}"> ▌</tspan></text>
    <text x="284" y="258" fill="{TEAL}" font-size="17" font-weight="600" letter-spacing="5">BY EUCLASE</text>
  </g>
  <!-- ymm0: eight 32-bit lanes accumulating -->
  <g font-family="{MONO}">
    <text x="962" y="84" fill="{DIM}" font-size="14">ymm0 · 8 × int32 lanes</text>
    {''.join(lanes)}
    <path d="M 962 256 L 1230 256" stroke="{EDGE}" stroke-width="2"/>
    <path d="M 1093 256 L 1093 270" stroke="{TEAL}" stroke-width="2"/>
    <text class="sum" x="1093" y="68" text-anchor="middle" fill="{TEAL}" font-size="14" opacity=".8">Σ lanes → logit</text>
  </g>
  <!-- tokens streaming out (Corundum HFT ring) -->
  <g clip-path="url(#stripclip)">
    <g transform="translate(0 290)"><g class="strip">{chips_svg}<g transform="translate({strip_w} 0)">{chips_svg}</g><g transform="translate({2*strip_w} 0)">{chips_svg}</g></g></g>
  </g>
  <rect x="0" y="282" width="{w}" height="40" fill="url(#fade)"/>
</svg>
"""


def q4k_layout() -> str:
    w, h = 1200, 570
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Q4_K block layout">',
         f'<rect width="{w}" height="{h}" rx="16" fill="{PETROL}"/>',
         f'<text x="40" y="52" fill="{TEXT}" font-family="{SANS}" font-size="26" font-weight="700">Q4_K super-block <tspan fill="{TEAL}">· 256 weights in 144 bytes · 4.5 bits/weight</tspan></text>',
         f'<text x="40" y="80" fill="{MUTED}" font-family="{SANS}" font-size="15">w = d · scale[j] · q − dmin · min[j]   with q ∈ [0,15] (4 bits), scale/min 6-bit per 32-weight sub-block j, d/dmin fp16 per super-block</text>']
    # byte bar: header segments get readable widths, qs the rest (not to scale)
    x0, y0 = 40, 110
    segs = [(0, 2, 70, "d", AMBER), (2, 2, 70, "dmin", CORAL), (4, 12, 170, "scales[12]", MINT),
            (16, 128, 810, "qs[128]: 256 × 4-bit quants, two per byte", TEAL)]
    x = x0
    for start, n, wpx, label, col in segs:
        o.append(f'<rect x="{x}" y="{y0}" width="{wpx - 4}" height="54" rx="6" fill="{col}" fill-opacity=".9"/>')
        o.append(f'<text x="{x + wpx/2 - 2}" y="{y0+26}" text-anchor="middle" fill="{PETROL}" font-family="{MONO}" font-size="15" font-weight="700">{label}</text>')
        sub = {"d": "fp16", "dmin": "fp16", "scales[12]": "16 × 6 bit"}.get(label, "128 bytes")
        o.append(f'<text x="{x + wpx/2 - 2}" y="{y0+45}" text-anchor="middle" fill="{PETROL}" fill-opacity=".75" font-family="{MONO}" font-size="12">{sub}</text>')
        o.append(f'<text x="{x}" y="{y0+72}" fill="{DIM}" font-family="{MONO}" font-size="12">{start}</text>')
        x += wpx
    # 6-bit packing
    yp = 220
    o.append(f'<text x="40" y="{yp}" fill="{TEXT}" font-family="{SANS}" font-size="18" font-weight="600">6-bit scale / min packing in scales[12]</text>')
    for i in range(12):
        x = 40 + i * 62
        col = MINT if i < 4 else (CORAL if i < 8 else AMBER)
        o.append(f'<rect x="{x}" y="{yp+14}" width="56" height="34" rx="5" fill="{PANEL}" stroke="{col}" stroke-width="2"/>')
        o.append(f'<text x="{x+28}" y="{yp+36}" text-anchor="middle" fill="{TEXT}" font-family="{MONO}" font-size="13">[{i}]</text>')
    notes = [(MINT, "[0-3]: scale j in bits 0-5, top 2 bits of scale j+4 in 6-7"),
             (CORAL, "[4-7]: min j in bits 0-5, top 2 bits of min j+4 in 6-7"),
             (AMBER, "[8-11]: low 4 bits of scale j+4 | low 4 bits of min j+4 &lt;&lt; 4")]
    for k, (col, t) in enumerate(notes):
        o.append(f'<circle cx="{46 + k*380}" cy="{yp+68}" r="5" fill="{col}"/><text x="{56 + k*380}" y="{yp+73}" fill="{MUTED}" font-family="{SANS}" font-size="12.5">{t}</text>')
    # nibble mapping
    yn = 345
    o.append(f'<text x="40" y="{yn}" fill="{TEXT}" font-family="{SANS}" font-size="18" font-weight="600">Nibble order: each 32-byte chunk of qs holds 64 weights</text>')
    for c in range(4):
        x = 40 + c * 285
        o.append(f'<rect x="{x}" y="{yn+16}" width="270" height="40" rx="6" fill="{PANEL}" stroke="{EDGE}"/>')
        o.append(f'<text x="{x+135}" y="{yn+41}" text-anchor="middle" fill="{TEAL}" font-family="{MONO}" font-size="14">qs[{32*c}..{32*c+31}]</text>')
        o.append(f'<rect x="{x}" y="{yn+80}" width="130" height="34" rx="6" fill="{TEAL}" fill-opacity=".85"/>')
        o.append(f'<rect x="{x+140}" y="{yn+80}" width="130" height="34" rx="6" fill="{MINT}" fill-opacity=".85"/>')
        o.append(f'<text x="{x+65}" y="{yn+102}" text-anchor="middle" fill="{PETROL}" font-family="{MONO}" font-size="12" font-weight="700">low  → w{64*c}..{64*c+31}</text>')
        o.append(f'<text x="{x+205}" y="{yn+102}" text-anchor="middle" fill="{PETROL}" font-family="{MONO}" font-size="12" font-weight="700">high → w{64*c+32}..{64*c+63}</text>')
        o.append(f'<path d="M {x+70} {yn+57} L {x+65} {yn+79}" stroke="{TEAL}" stroke-width="2"/><path d="M {x+200} {yn+57} L {x+205} {yn+79}" stroke="{MINT}" stroke-width="2"/>')
        o.append(f'<text x="{x+65}" y="{yn+132}" text-anchor="middle" fill="{DIM}" font-family="{MONO}" font-size="11">sub-block {2*c} · scale[{2*c}]</text>')
        o.append(f'<text x="{x+205}" y="{yn+132}" text-anchor="middle" fill="{DIM}" font-family="{MONO}" font-size="11">sub-block {2*c+1} · scale[{2*c+1}]</text>')
    ya = 508
    o.append(f'<text x="40" y="{ya}" fill="{TEXT}" font-family="{SANS}" font-size="15" font-weight="600">AVX2 unpack, one 32-byte load:</text>')
    o.append(f'<text x="280" y="{ya}" fill="{MINT}" font-family="{MONO}" font-size="13.5">lo = bits &amp; 0x0F · hi = (bits &gt;&gt; 4) &amp; 0x0F → vpmaddubsw(lo|hi, q8) → vpmaddwd(·, scale[j]) → int32 lanes</text>')
    o.append(f'<text x="40" y="{ya+30}" fill="{MUTED}" font-family="{SANS}" font-size="14">The mins never touch the quants again: Σ min[j] · Σ(q8 of sub-block j) comes from the 16 pre-summed bsums of the Q8_K activation block.</text>')
    o.append("</svg>")
    return "\n".join(o)


def tiling() -> str:
    w, h = 1200, 520
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Matrix-vector tiling">',
         f'<rect width="{w}" height="{h}" rx="16" fill="{PETROL}"/>',
         f'<text x="40" y="52" fill="{TEXT}" font-family="{SANS}" font-size="26" font-weight="700">Matrix × vector, one token <tspan fill="{TEAL}">· how the work is cut</tspan></text>',
         f'<text x="40" y="80" fill="{MUTED}" font-family="{SANS}" font-size="15">Weights are read once per token, straight from the mmap. The activation vector is quantized once (Q8_K / Q8_0) and stays hot in L1/L2 for every row.</text>']
    # matrix
    mx, my, mw, mh = 60, 120, 520, 340
    threads = [TEAL, MINT, AMBER, CORAL, "#60A5FA", "#A78BFA", "#34D399", "#F472B6"]
    rows = 17
    for r in range(rows):
        y = my + r * (mh / rows)
        col = threads[(r * 3 + r * r) % 8]
        o.append(f'<rect x="{mx}" y="{y:.1f}" width="{mw}" height="{mh/rows - 3:.1f}" rx="3" fill="{col}" fill-opacity=".22" stroke="{col}" stroke-opacity=".7"/>')
        if r == 4:
            for b in range(10):
                bx = mx + 6 + b * 51
                o.append(f'<rect x="{bx}" y="{y+3:.1f}" width="45" height="{mh/rows - 9:.1f}" rx="2" fill="{col}" fill-opacity=".8"/>')
    o.append(f'<text x="{mx}" y="{my-12}" fill="{TEXT}" font-family="{SANS}" font-size="16" font-weight="600">W  (rows × cols, Q4_K / Q6_K / Q8_0 blocks)</text>')
    o.append(f'<text x="{mx+mw+12}" y="{my + 4.6*(mh/rows):.1f}" fill="{MUTED}" font-family="{SANS}" font-size="13">← one task = 32 rows,</text>')
    o.append(f'<text x="{mx+mw+12}" y="{my + 5.6*(mh/rows):.1f}" fill="{MUTED}" font-family="{SANS}" font-size="13">   streamed block by block</text>')
    o.append(f'<text x="{mx}" y="{my+mh+28}" fill="{DIM}" font-family="{MONO}" font-size="13">colour = thread that took the task (atomic counter, fastest thread takes more)</text>')
    # vector
    vx = 780
    o.append(f'<text x="{vx-30}" y="{my-12}" fill="{TEXT}" font-family="{SANS}" font-size="16" font-weight="600">x  (activations)</text>')
    for i in range(12):
        o.append(f'<rect x="{vx}" y="{my + i*28}" width="44" height="24" rx="4" fill="{TEAL}" fill-opacity="{0.35 + 0.05*(i%3)}" stroke="{TEAL}"/>')
    o.append(f'<text x="{vx+60}" y="{my+40}" fill="{MUTED}" font-family="{SANS}" font-size="13">float32, alignas(64)</text>')
    o.append(f'<path d="M {vx+22} {my+340} L {vx+22} {my+360}" stroke="{TEAL}" stroke-width="2"/>')
    o.append(f'<text x="{vx-30}" y="{my+382}" fill="{MINT}" font-family="{MONO}" font-size="13">quantize once → Q8_K (256 / block)</text>')
    # threads panel
    tx, ty = 960, 150
    o.append(f'<text x="{tx}" y="{ty-30}" fill="{TEXT}" font-family="{SANS}" font-size="16" font-weight="600">ThreadPool</text>')
    for t in range(8):
        y = ty + t * 36
        o.append(f'<rect x="{tx}" y="{y}" width="200" height="28" rx="6" fill="{PANEL}" stroke="{threads[t]}"/>')
        o.append(f'<circle cx="{tx+16}" cy="{y+14}" r="6" fill="{threads[t]}"/>')
        o.append(f'<text x="{tx+32}" y="{y+19}" fill="{TEXT}" font-family="{MONO}" font-size="13">core {t} · dot rows</text>')
    o.append(f'<text x="{tx}" y="{ty+8*36+22}" fill="{DIM}" font-family="{SANS}" font-size="12">spin between jobs, sleep when idle;</text>')
    o.append(f'<text x="{tx}" y="{ty+8*36+40}" fill="{DIM}" font-family="{SANS}" font-size="12">Q, K, V (and gate, up) go in one job</text>')
    o.append("</svg>")
    return "\n".join(o)


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    (IMG / "banner.svg").write_text(banner(), encoding="utf-8")
    (IMG / "q4k-layout.svg").write_text(q4k_layout(), encoding="utf-8")
    (IMG / "tiling.svg").write_text(tiling(), encoding="utf-8")
    print("wrote banner.svg, q4k-layout.svg, tiling.svg")


if __name__ == "__main__":
    main()
