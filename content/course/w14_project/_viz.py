"""Weeks 14–15 frames: the 17 project stages as a phased roadmap and the
10-slide presentation storyboard (schematic thumbnails)."""

from __future__ import annotations

from components import svgkit as K

PHASES = [("Plan", range(0, 6), K.VIOLET), ("Build", range(6, 10), K.BLUE), ("Diagnose", range(10, 13), K.AMBER), ("Report", range(13, 17), K.EMERALD)]


def phase_of(i: int) -> tuple[str, str]:
    for name, rng, col in PHASES:
        if i in rng:
            return name, col
    return "", K.MUTED


def roadmap_svg(stages: list[tuple], k: int) -> str:
    """17 numbered stops in four coloured phases; stop k highlighted."""
    s = K.svg_open(700, 250)
    for pi, (name, rng, col) in enumerate(PHASES):
        xs = [30 + i * 38.5 for i in rng]
        s += f'<rect x="{xs[0] - 16}" y="30" width="{xs[-1] - xs[0] + 32}" height="110" rx="14" fill="{K.tint(col, 0.9)}" stroke="{col}" stroke-dasharray="5 4"/>'
        s += K.text((xs[0] + xs[-1]) / 2, 24, name, size=13, bold=True, color=col)
    s += f'<line x1="30" y1="85" x2="{30 + 16 * 38.5}" y2="85" stroke="#CBD5E1" stroke-width="3"/>'
    for i in range(len(stages)):
        _, col = phase_of(i)
        x = 30 + i * 38.5
        on, done = i == k, i < k
        s += f'<circle cx="{x}" cy="85" r="{15 if on else 11}" fill="{col if (on or done) else "#FFFFFF"}" fill-opacity="{1 if on else 0.55 if done else 1}" stroke="{col}" stroke-width="2"/>'
        s += K.text(x, 90, str(i + 1), size=11 if on else 10, bold=True, color="#fff" if (on or done) else col)
    title = stages[k][0].split(". ", 1)[1]
    _, col = phase_of(k)
    s += K.text(350, 175, f"stage {k + 1}: {stages[k][2]}", size=12, color=K.MUTED)
    s += K.box(150, 190, 400, 44, title, col, filled=True, size=14)
    return s + "</svg>"


def slide_svg(slides: list[tuple], k: int) -> str:
    """Storyboard: ten thumbnails, the current one enlarged with its content type."""
    s = K.svg_open(700, 270)
    for i in range(len(slides)):
        x = 12 + i * 68
        col = K.PALETTE[i % 7]
        on = i == k
        s += f'<rect x="{x}" y="14" width="60" height="40" rx="5" fill="{col if on else K.tint(col, 0.85)}" stroke="{col}" stroke-width="{2.5 if on else 1}"/>'
        s += K.text(x + 30, 39, str(i + 1), size=13, bold=True, color="#fff" if on else col)
    name, content, form = slides[k]
    col = K.PALETTE[k % 7]
    s += f'<rect x="120" y="72" width="460" height="186" rx="12" fill="#FFFFFF" stroke="{col}" stroke-width="2.5"/>'
    s += f'<rect x="120" y="72" width="460" height="34" rx="12" fill="{col}"/>'
    s += K.text(350, 95, f"slide {k + 1}", size=13, bold=True, color="#fff")
    # schematic content by form
    if "جدول" in form:
        for r in range(4):
            for c in range(3):
                s += f'<rect x="{180 + c * 115}" y="{122 + r * 26}" width="108" height="20" rx="3" fill="{K.tint(col, 0.8 if r == 0 else 0.93)}"/>'
    if "رسم" in form or "مصفوفة" in form:
        pts = " ".join(f"{410 + j * 18},{220 - (j * 7) % 40 - j * 3}" for j in range(8))
        s += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3"/>'
    if "نص" in form or "قائمت" in form:
        for r in range(4):
            s += f'<rect x="170" y="{130 + r * 24}" width="{300 - r * 40}" height="10" rx="5" fill="{K.tint(col, 0.75)}"/>'
    s += K.text(350, 250, "schematic thumbnail", size=10, color=K.MUTED)
    return s + "</svg>"
