"""Week 13 frames: CPU vs GPU filling the cells of a matrix product, the
training-memory stack piece by piece, and the Colab workflow."""

from __future__ import annotations

from components import svgkit as K

# the week's example MLP (same as the code lab): 784 -> 512 -> 256 -> 10
P_MLP = 784 * 512 + 512 + 512 * 256 + 256 + 256 * 10 + 10
A_MLP = 784 + 512 + 256 + 10


def parallel_svg(tick: int, n: int = 8, cpu_cores: int = 4) -> str:
    """Each output cell of an n×n product is an independent dot product.
    CPU: `cpu_cores` cells per tick. GPU: all n² cells in one tick."""
    s = K.svg_open(700, 270)
    c = 22
    total = n * n
    for side, (x0, name, per_tick, col) in enumerate([(40, f"CPU — {cpu_cores} cores", cpu_cores, K.BLUE), (390, "GPU — thousands of cores", total, K.EMERALD)]):
        done = min(total, tick * per_tick)
        s += K.text(x0 + n * c / 2, 24, name, size=13, bold=True, color=col)
        for q in range(total):
            r, cc = divmod(q, n)
            fill = col if q < done else "#F1F5F9"
            new = done - per_tick <= q < done and tick > 0
            s += f'<rect x="{x0 + cc * c}" y="{36 + r * c}" width="{c - 2}" height="{c - 2}" rx="3" fill="{fill}" fill-opacity="{1 if new else 0.55 if q < done else 1}" stroke="{col if new else "#E2E8F0"}"/>'
        ticks_needed = -(-total // per_tick)
        s += K.text(x0 + n * c / 2, 36 + n * c + 22, f"{done}/{total} cells · tick {min(tick, ticks_needed)}/{ticks_needed}", size=11, mono=True, color=col, bold=True)
    s += K.text(350, 262, "each cell = one row · one column (independent) → all can be computed at the same time", size=11, color=K.MUTED)
    return s + "</svg>"


MEM_PARTS = [
    ("parameters", 4, K.VIOLET, "the weights themselves (float32 = 4 bytes each)"),
    ("gradients", 4, K.PINK, "one gradient per weight, same size"),
    ("Adam m", 4, K.AMBER, "first moment (momentum) per weight"),
    ("Adam v", 4, K.ORANGE, "second moment per weight"),
    ("activations", None, K.CYAN, "values kept for the backward pass: grows with batch size"),
]


def mem_parts(batch: int) -> list[tuple[str, float, str]]:
    out = []
    for name, b, col, _ in MEM_PARTS:
        mb = (P_MLP * b if b else A_MLP * 4 * batch * 2) / 1e6
        out.append((name, mb, col))
    return out


def memory_svg(batch: int, k: int, cap_mb: float = 40.0) -> str:
    parts = mem_parts(batch)
    s = K.svg_open(700, 270)
    x0, w0, y0, h0 = 60, 560, 50, 160
    total = sum(mb for _, mb, _ in parts)
    scale = w0 / (total * 1.02)
    x = x0
    for i, (name, mb, col) in enumerate(parts[: k + 1]):
        wdt = mb * scale
        s += f'<rect x="{x:.1f}" y="{y0}" width="{max(2, wdt):.1f}" height="{h0}" fill="{col}" fill-opacity="{1 if i == k else 0.7}" stroke="#fff"/>'
        if wdt > 50:
            s += K.text(x + wdt / 2, y0 + h0 / 2, name, size=11, bold=True, color="#fff")
            s += K.text(x + wdt / 2, y0 + h0 / 2 + 16, f"{mb:.1f} MB", size=10, color="#fff", mono=True)
        x += wdt
    used = sum(mb for _, mb, _ in parts[: k + 1])
    s += K.text(350, 30, f"MLP 784→512→256→10 · {P_MLP:,} parameters · batch {batch} · Adam", size=12, bold=True, color=K.INK)
    s += K.text(350, 240, f"so far: {used:.1f} MB", size=14, bold=True, color=K.PINK, mono=True)
    return s + "</svg>"


COLAB = [
    ("open", "New notebook (or upload .ipynb)"),
    ("GPU", "Runtime → Change runtime type → GPU"),
    ("verify", "!nvidia-smi · list_physical_devices('GPU')"),
    ("data", "drive.mount('/content/drive')"),
    ("train", "model.fit(..., callbacks=[ModelCheckpoint(→ Drive)])"),
    ("save", "model.save('/content/drive/MyDrive/.../m.keras')"),
]


def colab_svg(k: int) -> str:
    s = K.svg_open(700, 200)
    for i, (name, detail) in enumerate(COLAB):
        x = 8 + i * 115
        col = K.PALETTE[i % 7]
        s += K.box(x, 30, 104, 46, name, col, filled=i == k, size=13)
        if i < len(COLAB) - 1:
            s += K.arrow(x + 105, 53, x + 114, 53, color=col)
    name, detail = COLAB[k]
    s += f'<rect x="60" y="105" width="580" height="44" rx="10" fill="#0F172A"/>'
    s += K.text(350, 132, detail, size=13, mono=True, color="#E2E8F0")
    s += K.text(350, 180, "schematic of the workflow — menu names as in Colab at the time of writing", size=10, color=K.MUTED)
    return s + "</svg>"
