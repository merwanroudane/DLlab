"""Lightweight SVG building blocks for animation frames (multicolour palette).

Heat maps (decision surfaces, feature maps, attention-like grids) are encoded
as tiny PNG data URIs with a hand-written encoder, so a 60-frame animation
stays a few hundred kilobytes and needs no plotting library.
"""

from __future__ import annotations

import base64
import math
import struct
import zlib
from typing import Sequence

import numpy as np

from core.rtl import esc

# multicolour palette (mirrors assets/styles/colors.css)
VIOLET, BLUE, CYAN, EMERALD, AMBER, ORANGE, PINK, RED, INK, MUTED = (
    "#7C3AED", "#2563EB", "#0891B2", "#059669", "#D97706", "#EA580C", "#DB2777", "#DC2626", "#1E1B4B", "#6B7280")
PALETTE = [VIOLET, BLUE, CYAN, EMERALD, AMBER, ORANGE, PINK]
FONT = "Inter, Segoe UI, sans-serif"
MONO = "JetBrains Mono, Consolas, monospace"


# ------------------------------------------------------------------ PNG heat maps
def _png(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 array as PNG bytes."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _lerp_colors(t: np.ndarray, stops: Sequence[str]) -> np.ndarray:
    cols = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in stops], float)
    t = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return (cols[i] * (1 - f) + cols[i + 1] * f).astype(np.uint8)


DIVERGING = ["#93C5FD", "#DBEAFE", "#FFFFFF", "#FBCFE8", "#F9A8D4"]     # class 0 blue  ↔  class 1 pink (light, so points stay visible)
SEQUENTIAL = ["#F8FAFC", "#C4B5FD", "#7C3AED", "#DB2777", "#F59E0B"]     # low → high activation


def heat_uri(values: np.ndarray, *, stops: Sequence[str] = DIVERGING, vmin: float | None = None, vmax: float | None = None,
             flip_y: bool = True) -> str:
    """values: (H, W) grid -> PNG data URI. Row 0 is the *bottom* of the plot when flip_y."""
    v = np.asarray(values, float)
    lo = np.nanmin(v) if vmin is None else vmin
    hi = np.nanmax(v) if vmax is None else vmax
    t = (v - lo) / (hi - lo if hi > lo else 1.0)
    rgb = _lerp_colors(np.nan_to_num(t, nan=0.5), stops)
    if flip_y:
        rgb = rgb[::-1]
    return "data:image/png;base64," + base64.b64encode(_png(np.ascontiguousarray(rgb))).decode()


def grid(extent: tuple[float, float, float, float], n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xx, yy, points) for a surface over extent = (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = extent
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


# ------------------------------------------------------------------ panels
def svg_open(w: int, h: int) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" xmlns="http://www.w3.org/2000/svg">'
            '<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
            f'<path d="M0,0 L8,4 L0,8 z" fill="{MUTED}"/></marker></defs>')


def text(x: float, y: float, s: str, *, size: int = 12, color: str = INK, anchor: str = "middle", bold: bool = False,
         mono: bool = False) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MONO if mono else FONT}" font-size="{size}" '
            f'font-weight="{600 if bold else 400}" fill="{color}">{esc(s)}</text>')


def box(x: float, y: float, w: float, h: float, label: str, color: str, *, filled: bool = False, size: int = 12,
        rx: int = 10, sub: str | None = None) -> str:
    fill = color if filled else _tint(color)
    tc = "#fff" if filled else color
    s = f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{color}" stroke-width="1.6"/>'
    cy = y + h / 2 + size * 0.35 - (6 if sub else 0)
    s += text(x + w / 2, cy, label, size=size, color=tc, bold=True)
    if sub:
        s += text(x + w / 2, cy + size + 2, sub, size=size - 2, color=tc, mono=True)
    return s


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = MUTED, width: float = 1.8, dash: str = "") -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" marker-end="url(#ah)"{d}/>'


def _tint(hex_color: str, k: float = 0.88) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(c + (255 - c) * k) for c in (r, g, b))


def tint(hex_color: str, k: float = 0.88) -> str:
    return _tint(hex_color, k)


class Panel:
    """A data-coordinate panel inside an SVG: maps (x, y) data to pixels."""

    def __init__(self, x: float, y: float, w: float, h: float, extent: tuple[float, float, float, float]):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.x0, self.x1, self.y0, self.y1 = extent

    def px(self, v: float) -> float:
        return self.x + (v - self.x0) / (self.x1 - self.x0) * self.w

    def py(self, v: float) -> float:
        return self.y + self.h - (v - self.y0) / (self.y1 - self.y0) * self.h

    def frame(self, title: str = "", xlabel: str = "", ylabel: str = "", bg: str = "#FFFFFF") -> str:
        s = f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}" fill="{bg}" stroke="#DDD6FE" rx="4"/>'
        if title:
            s += text(self.x + self.w / 2, self.y - 8, title, size=12, bold=True, color=VIOLET)
        if xlabel:
            s += text(self.x + self.w / 2, self.y + self.h + 16, xlabel, size=10, color=MUTED)
        if ylabel:
            s += (f'<text x="{self.x - 10}" y="{self.y + self.h / 2}" text-anchor="middle" font-family="{FONT}" font-size="10" '
                  f'fill="{MUTED}" transform="rotate(-90 {self.x - 10} {self.y + self.h / 2})">{esc(ylabel)}</text>')
        return s

    def image(self, uri: str, opacity: float = 0.9) -> str:
        return (f'<image href="{uri}" x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}" preserveAspectRatio="none" '
                f'opacity="{opacity}" style="image-rendering:auto"/>')

    def points(self, X: np.ndarray, y: np.ndarray, colors: Sequence[str] = ("#1D4ED8", "#BE185D"), r: float = 2.8, opacity: float = 0.95) -> str:
        out = []
        for (a, b), c in zip(X, y):
            if self.x0 <= a <= self.x1 and self.y0 <= b <= self.y1:
                out.append(f'<circle cx="{self.px(a):.1f}" cy="{self.py(b):.1f}" r="{r}" fill="{colors[int(c)]}" '
                           f'fill-opacity="{opacity}" stroke="#fff" stroke-width=".6"/>')
        return "".join(out)

    def line_wb(self, w1: float, w2: float, b: float, color: str = INK, width: float = 2.4, dash: str = "") -> str:
        """Draw the line w1*x + w2*y + b = 0 clipped to the panel."""
        pts = []
        if abs(w2) > 1e-9:
            for xv in (self.x0, self.x1):
                pts.append((xv, -(w1 * xv + b) / w2))
        if abs(w1) > 1e-9:
            for yv in (self.y0, self.y1):
                pts.append((-(w2 * yv + b) / w1, yv))
        pts = [(a, c) for a, c in pts if self.x0 - 1e-9 <= a <= self.x1 + 1e-9 and self.y0 - 1e-9 <= c <= self.y1 + 1e-9]
        if len(pts) < 2:
            return ""
        (a1, c1), (a2, c2) = pts[0], pts[-1]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<line x1="{self.px(a1):.1f}" y1="{self.py(c1):.1f}" x2="{self.px(a2):.1f}" y2="{self.py(c2):.1f}" stroke="{color}" stroke-width="{width}"{d}/>'

    def polyline(self, xs: Sequence[float], ys: Sequence[float], color: str, width: float = 2.2, dash: str = "") -> str:
        pts = " ".join(f"{self.px(a):.1f},{self.py(max(self.y0, min(self.y1, b))):.1f}" for a, b in zip(xs, ys) if math.isfinite(b))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" points="{pts}"{d}/>'

    def dot(self, a: float, b: float, color: str, r: float = 5) -> str:
        return f'<circle cx="{self.px(a):.1f}" cy="{self.py(max(self.y0, min(self.y1, b))):.1f}" r="{r}" fill="{color}" stroke="#fff" stroke-width="1.5"/>'

    def yticks(self, values: Sequence[float], fmt: str = "{:g}") -> str:
        return "".join(text(self.x - 5, self.py(v) + 3, fmt.format(v), size=9, color=MUTED, anchor="end") for v in values)

    def xticks(self, values: Sequence[float], fmt: str = "{:g}") -> str:
        return "".join(text(self.px(v), self.y + self.h + 11, fmt.format(v), size=9, color=MUTED) for v in values)


def legend(x: float, y: float, items: Sequence[tuple[str, str]], size: int = 11) -> str:
    s, cx = "", x
    for color, label in items:
        s += f'<circle cx="{cx + 5}" cy="{y - 4}" r="5" fill="{color}"/>' + text(cx + 14, y, label, size=size, anchor="start", color=INK)
        cx += 22 + len(label) * size * 0.55
    return s


def confusion_svg(tn: int, fp: int, fn: int, tp: int, x: float = 0, y: float = 0, cell: int = 70) -> str:
    """2×2 confusion matrix with coloured cells (rows = truth, cols = prediction)."""
    cells = [(tn, "TN", EMERALD, 0, 0), (fp, "FP", ORANGE, 0, 1), (fn, "FN", RED, 1, 0), (tp, "TP", BLUE, 1, 1)]
    s = text(x + cell + 20, y + 10, "predicted 0      predicted 1", size=10, color=MUTED)
    for v, name, c, r, k in cells:
        cx, cy = x + 40 + k * (cell + 6), y + 18 + r * (cell + 6)
        s += f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" rx="10" fill="{_tint(c, 0.8)}" stroke="{c}" stroke-width="2"/>'
        s += text(cx + cell / 2, cy + cell / 2 - 2, str(v), size=20, bold=True, color=c)
        s += text(cx + cell / 2, cy + cell / 2 + 16, name, size=11, color=c, bold=True)
    s += text(x + 18, y + 18 + cell / 2 + 4, "true 0", size=10, color=MUTED)
    s += text(x + 18, y + 18 + cell * 1.5 + 10, "true 1", size=10, color=MUTED)
    return s
