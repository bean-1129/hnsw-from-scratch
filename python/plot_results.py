#!/usr/bin/env python3
"""Render the benchmark CSVs as a recall-vs-throughput chart (SVG, no dependencies).

Produces a light and a dark variant so the README can serve the right one per
viewer theme via <picture>. SVG rather than PNG: it stays sharp at any zoom, it
diffs as text, and it needs no plotting library.

Usage:
    python3 python/plot_results.py                     # uses results/*.csv
    python3 python/plot_results.py --out-dir results
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, List, Sequence, Tuple

# --- theme -----------------------------------------------------------------
# Categorical slots 1-3 of the reference palette, validated for all-pairs CVD
# separation in both modes. Three series is the documented cap for scatter-like
# forms, which is exactly what this chart needs.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, "
        "Arial, sans-serif")

# Geometry. The right margin holds the direct end-labels; it is generous because
# rendered text is wider than a naive characters-times-width estimate suggests,
# and a clipped label is worse than no label at all.
WIDTH, HEIGHT = 960, 530
LEFT, RIGHT, TOP, BOTTOM = 74, 190, 104, 64
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - TOP - BOTTOM

X_MIN, X_MAX = 0.55, 1.02
Y_MIN, Y_MAX = 30.0, 30000.0

# Exact scan reference, measured by `hnsw benchmark --brute-force`.
BRUTE_FORCE_QPS = 41.8
BRUTE_FORCE_RECALL = 1.0


def read_csv(path: str) -> List[Dict[str, float]]:
    with open(path, newline="") as handle:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def x_of(recall: float) -> float:
    return LEFT + (recall - X_MIN) / (X_MAX - X_MIN) * PLOT_W


def y_of(qps: float) -> float:
    """Log scale: throughput spans nearly three decades."""
    lo, hi = math.log10(Y_MIN), math.log10(Y_MAX)
    return TOP + (hi - math.log10(qps)) / (hi - lo) * PLOT_H


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def spread_labels(anchors: Sequence[float], min_gap: float = 19.0) -> List[float]:
    """Push colliding end-labels apart around their mean, preserving order.

    The M=16 and FAISS curves sit ~13px apart at the right edge, which is the
    whole point of the chart -- so the labels have to be nudged and given leader
    lines rather than stacked or dropped.
    """
    order = sorted(range(len(anchors)), key=lambda i: anchors[i])
    placed = [anchors[i] for i in order]
    for i in range(1, len(placed)):
        if placed[i] - placed[i - 1] < min_gap:
            placed[i] = placed[i - 1] + min_gap
    # Re-centre the block on the original mean so it does not drift downward.
    shift = (sum(anchors) / len(anchors)) - (sum(placed) / len(placed))
    placed = [p + shift for p in placed]
    result = [0.0] * len(anchors)
    for slot, i in enumerate(order):
        result[i] = placed[slot]
    return result


def render(series: List[Tuple[str, str, List[Dict[str, float]]]], mode: str) -> str:
    t = THEMES[mode]
    out: List[str] = []
    add = out.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{FONT}" '
        f'role="img" aria-label="Recall at 10 versus queries per second on '
        f'glove-100-angular">')
    add(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{t["surface"]}"/>')

    # -- titles --------------------------------------------------------------
    add(f'<text x="{LEFT}" y="34" fill="{t["ink"]}" font-size="17" '
        f'font-weight="600">Recall@10 vs. throughput on glove-100-angular '
        f'(1.18M &#215; 100)</text>')
    add(f'<text x="{LEFT}" y="55" fill="{t["secondary"]}" font-size="12.5">'
        f'Single thread, batch size 1, Apple M4. Each point is one efSearch value '
        f'(16 &#8594; 512, left to right). Up and to the right is better.</text>')

    # -- legend (identity never rests on color alone) ------------------------
    lx = LEFT
    for i, (name, _short, _rows) in enumerate(series):
        color = t["series"][i]
        add(f'<line x1="{lx}" y1="76" x2="{lx + 16}" y2="76" stroke="{color}" '
            f'stroke-width="2" stroke-linecap="round"/>')
        add(f'<circle cx="{lx + 8}" cy="76" r="3.5" fill="{color}"/>')
        add(f'<text x="{lx + 23}" y="80" fill="{t["secondary"]}" font-size="12">'
            f'{escape(name)}</text>')
        lx += 27 + len(name) * 6.6
    add(f'<line x1="{lx}" y1="76" x2="{lx + 16}" y2="76" stroke="{t["muted"]}" '
        f'stroke-width="2" stroke-dasharray="1 3" stroke-linecap="round"/>')
    add(f'<text x="{lx + 23}" y="80" fill="{t["secondary"]}" font-size="12">'
        f'exact scan</text>')

    # -- gridlines and ticks -------------------------------------------------
    for decade in (100, 1000, 10000):
        y = y_of(decade)
        add(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + PLOT_W}" y2="{y:.1f}" '
            f'stroke="{t["grid"]}" stroke-width="1"/>')
        add(f'<text x="{LEFT - 10}" y="{y + 4:.1f}" fill="{t["muted"]}" '
            f'font-size="11.5" text-anchor="end">{decade:,}</text>')
    for recall in (0.6, 0.7, 0.8, 0.9, 1.0):
        x = x_of(recall)
        add(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{TOP + PLOT_H}" '
            f'stroke="{t["grid"]}" stroke-width="1"/>')
        add(f'<text x="{x:.1f}" y="{TOP + PLOT_H + 22}" fill="{t["muted"]}" '
            f'font-size="11.5" text-anchor="middle">{recall:.1f}</text>')

    add(f'<line x1="{LEFT}" y1="{TOP + PLOT_H}" x2="{LEFT + PLOT_W}" '
        f'y2="{TOP + PLOT_H}" stroke="{t["axis"]}" stroke-width="1"/>')
    add(f'<text x="{LEFT + PLOT_W / 2}" y="{HEIGHT - 18}" fill="{t["secondary"]}" '
        f'font-size="12.5" text-anchor="middle">Recall@10</text>')
    add(f'<text transform="translate(20,{TOP + PLOT_H / 2}) rotate(-90)" '
        f'fill="{t["secondary"]}" font-size="12.5" text-anchor="middle">'
        f'Queries per second (log scale)</text>')

    # -- exact-scan reference ------------------------------------------------
    bx, by = x_of(BRUTE_FORCE_RECALL), y_of(BRUTE_FORCE_QPS)
    add(f'<line x1="{LEFT}" y1="{by:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
        f'stroke="{t["muted"]}" stroke-width="1" stroke-dasharray="1 3"/>')
    add(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4.5" fill="{t["muted"]}" '
        f'stroke="{t["surface"]}" stroke-width="2"/>')
    add(f'<text x="{bx - 12:.1f}" y="{by - 10:.1f}" fill="{t["secondary"]}" '
        f'font-size="11.5" text-anchor="end">brute force (exact): '
        f'{BRUTE_FORCE_QPS:.1f} QPS</text>')

    # -- series --------------------------------------------------------------
    anchors = [y_of(rows[-1]["qps"]) for _, _short, rows in series]
    label_y = spread_labels(anchors)

    for i, (_name, short, rows) in enumerate(series):
        color = t["series"][i]
        points = " ".join(f'{x_of(r["recall"]):.1f},{y_of(r["qps"]):.1f}' for r in rows)
        add(f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        # 2px surface ring keeps markers legible where the curves overlap.
        for r in rows:
            add(f'<circle cx="{x_of(r["recall"]):.1f}" cy="{y_of(r["qps"]):.1f}" '
                f'r="4.5" fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')

        end = rows[-1]
        ex, ey = x_of(end["recall"]), y_of(end["qps"])
        ly = label_y[i]
        add(f'<path d="M {ex + 7:.1f} {ey:.1f} L {LEFT + PLOT_W + 12:.1f} {ly:.1f}" '
            f'fill="none" stroke="{t["muted"]}" stroke-width="1"/>')
        add(f'<text x="{LEFT + PLOT_W + 17:.1f}" y="{ly + 4:.1f}" fill="{t["ink"]}" '
            f'font-size="12" font-weight="500">{escape(short)}</text>')

    return "\n".join(out) + "\n</svg>\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the benchmark CSVs as SVG")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    # (legend name, short end-label, csv). The legend carries the full name so
    # the direct labels can stay short enough to never collide or overflow.
    wanted = [
        ("this impl, M=16", "this M=16", "hnsw_cpp.csv"),
        ("this impl, M=32", "this M=32", "hnsw_cpp_m32.csv"),
        ("FAISS, M=16", "FAISS M=16", "faiss.csv"),
    ]
    series = []
    for name, short, filename in wanted:
        path = os.path.join(args.results_dir, filename)
        if not os.path.exists(path):
            print(f"skipping {name}: {path} not found")
            continue
        series.append((name, short, read_csv(path)))

    if not series:
        print("error: no benchmark CSVs found; run `hnsw benchmark --csv ...` first")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    for mode in ("light", "dark"):
        path = os.path.join(args.out_dir, f"benchmark_{mode}.svg")
        with open(path, "w") as handle:
            handle.write(render(series, mode))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
