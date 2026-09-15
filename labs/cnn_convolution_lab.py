"""CNN Convolution Lab — a kernel slides over a small image; every step shows
the window, the element-wise products and the feature-map value (spec §41)."""

import html as _html

import numpy as np
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.cnn import KERNELS, conv2d, conv_steps, out_size

LAB = Lab(
    id="labs.cnn_convolution_lab",
    title_ar="معمل الالتفاف: حركة النواة المتحركة",
    title_en="CNN Convolution Lab (Animated Kernel)",
    category="cnn",
    description_ar="اختر صورة صغيرة ونواة (مرشّحًا)، وشاهد النواة تنزلق موضعًا موضعًا: نافذة الصورة، الضرب عنصريًا، الجمع، والقيمة الناتجة في خريطة الخصائص — مع الحشو والخطوة.",
    related_lessons=["course.w08.convolution", "course.w08.padding_stride_pooling"],
)

IMAGES = {
    "حافة عمودية": lambda: np.concatenate([np.zeros((6, 3)), np.ones((6, 3))], 1),
    "حافة أفقية": lambda: np.concatenate([np.zeros((3, 6)), np.ones((3, 6))], 0),
    "مربع في الوسط": lambda: np.pad(np.ones((2, 2)), 2),
    "خط قطري": lambda: np.eye(6),
    "أرقام 0..35": lambda: np.arange(36.0).reshape(6, 6) / 35,
}


def _grid_svg(img: np.ndarray, kernel: np.ndarray, fmap: np.ndarray, step: dict, k: int, pad: int) -> str:
    cell = 34
    n = img.shape[0]
    s = [f'<svg viewBox="0 0 {cell * (n + kernel.shape[0] + fmap.shape[0]) + 120} {cell * max(n, 4) + 60}" width="100%" style="max-width:760px;font-family:JetBrains Mono,Consolas,monospace">']
    # image (padded shown)
    for i in range(n):
        for j in range(n):
            v = img[i, j]; inwin = step["r0"] <= i < step["r0"] + k and step["c0"] <= j < step["c0"] + k
            is_pad = i < pad or j < pad or i >= n - pad or j >= n - pad
            fill = f"rgb({int(255 - 160 * v)},{int(255 - 120 * v)},{int(255 - 60 * v)})" if not is_pad else "#EEE9DF"
            stroke = "#C8473A" if inwin else ("#D9D3C7" if not is_pad else "#B9B2A6")
            s.append(f'<rect x="{10 + j * cell}" y="{30 + i * cell}" width="{cell - 2}" height="{cell - 2}" fill="{fill}" stroke="{stroke}" stroke-width="{3 if inwin else 1}"/>')
            s.append(f'<text x="{10 + j * cell + cell / 2 - 1}" y="{30 + i * cell + cell / 2 + 4}" text-anchor="middle" font-size="10" fill="#2B2A28">{v:.1f}</text>')
    s.append(f'<text x="{10 + n * cell / 2}" y="18" text-anchor="middle" font-size="12" font-weight="600" fill="#2B2A28">image {img.shape[0] - 2 * pad}×{img.shape[1] - 2 * pad}{" (+pad " + str(pad) + ")" if pad else ""}</text>')
    x0 = 10 + n * cell + 30
    s.append(f'<text x="{x0 + k * cell / 2}" y="18" text-anchor="middle" font-size="12" font-weight="600" fill="#C8473A">kernel {k}×{k}</text>')
    for i in range(k):
        for j in range(k):
            s.append(f'<rect x="{x0 + j * cell}" y="{30 + i * cell}" width="{cell - 2}" height="{cell - 2}" fill="#FBE6E2" stroke="#C8473A"/>')
            s.append(f'<text x="{x0 + j * cell + cell / 2 - 1}" y="{30 + i * cell + cell / 2 + 4}" text-anchor="middle" font-size="10" fill="#2B2A28">{kernel[i, j]:.2g}</text>')
    s.append(f'<text x="{x0 + k * cell / 2}" y="{30 + k * cell + 18}" text-anchor="middle" font-size="11" fill="#6B675F">Σ(window ⊙ kernel)</text>')
    s.append(f'<text x="{x0 + k * cell / 2}" y="{30 + k * cell + 36}" text-anchor="middle" font-size="13" font-weight="600" fill="#C8473A">= {step["value"]:.2f}</text>')
    x1 = x0 + k * cell + 40; m = fmap.shape[0]
    s.append(f'<text x="{x1 + m * cell / 2}" y="18" text-anchor="middle" font-size="12" font-weight="600" fill="#1F7A78">feature map {m}×{fmap.shape[1]}</text>')
    vmax = max(1e-9, float(np.abs(fmap).max()))
    for i in range(m):
        for j in range(fmap.shape[1]):
            done = (i, j) <= (step["i"], step["j"]) and (i < step["i"] or j <= step["j"])
            v = fmap[i, j] if done else 0.0
            shade = int(200 - 160 * abs(v) / vmax) if done else 245
            fill = f"rgb({shade},{shade + 20 if v >= 0 else shade},{shade + 30 if v >= 0 else shade})" if done else "#F7F5F0"
            cur = (i, j) == (step["i"], step["j"])
            s.append(f'<rect x="{x1 + j * cell}" y="{30 + i * cell}" width="{cell - 2}" height="{cell - 2}" fill="{fill}" stroke="{"#C8473A" if cur else "#1F7A78"}" stroke-width="{3 if cur else 1}"/>')
            if done:
                s.append(f'<text x="{x1 + j * cell + cell / 2 - 1}" y="{30 + i * cell + cell / 2 + 4}" text-anchor="middle" font-size="10" fill="#2B2A28">{v:.1f}</text>')
    s.append("</svg>")
    return "".join(s)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        img_name = st.selectbox("الصورة (6×6)", list(IMAGES), key="cv_img")
    with c2:
        k_name = st.selectbox("النواة", list(KERNELS), index=1, key="cv_k")
    with c3:
        pad = st.select_slider("الحشو padding", options=[0, 1], value=0, key="cv_pad")
    with c4:
        stride = st.select_slider("الخطوة stride", options=[1, 2], value=1, key="cv_stride")
    img = IMAGES[img_name](); kernel = KERNELS[k_name]
    fmap = conv2d(img, kernel, padding=pad, stride=stride); steps = conv_steps(img, kernel, padding=pad, stride=stride)
    padded = np.pad(img, pad) if pad else img
    n = img.shape[0]; k = kernel.shape[0]
    st.code(f"output size = floor((n + 2p − k) / s) + 1 = floor(({n} + 2·{pad} − {k}) / {stride}) + 1 = {out_size(n, k, pad, stride)}   → feature map {fmap.shape[0]}×{fmap.shape[1]}", language="text")
    frames = []
    for idx, stp in enumerate(steps):
        prods = np.array(stp["products"])
        detail = " + ".join(f"{w:.1f}×{kk:.2g}" for w, kk in zip(np.array(stp["window"]).ravel(), kernel.ravel()) if kk != 0) or "0"
        frames.append(Frame(_grid_svg(padded, kernel, fmap, stp, k, pad), caption(f"**الموضع {idx + 1}/{len(steps)}** — النافذة عند الصف {stp['r0']} والعمود {stp['c0']} (في الصورة المحشوّة). الضرب عنصريًا ثم الجمع: {detail} = **{stp['value']:.2f}** → يُكتب في خريطة الخصائص عند ({stp['i']}, {stp['j']})."),
                            action=f"({stp['i']}, {stp['j']})", equation=f"Σ(window ⊙ kernel) = {stp['value']:.2f}", values=[("feature_map[i,j]", "", f"{stp['value']:.2f}")]))
    animation_player("cv_anim", frames, title_ar="النواة تنزلق: نافذة × نواة → قيمة", interval_ms=900)
    st.markdown("### خريطة الخصائص كاملة")
    table([""] + [str(j) for j in range(fmap.shape[1])], [(str(i),) + tuple(f"{v:.1f}" for v in row) for i, row in enumerate(fmap)], ["num"] * (fmap.shape[1] + 1))
    intuition("النواة «حافة عمودية» تعطي قيمًا كبيرة حيث ينتقل السطوع من يسار إلى يمين، وصفرًا في المناطق الموحدة. جرّبها على «حافة أفقية»: لا تكتشف شيئًا. كل نواة كاشف نمط محلي — وفي CNN **تُتعلَّم** قيم النواة بالتدرج بدل كتابتها يدويًا.")
    practical_note("ما يسمّيه التعلم العميق «التفافًا» هو رياضيًا ارتباط متقاطع (بلا قلب النواة). لا فرق عمليًا لأن النواة تُتعلَّم. عدد معلمات نواة 3×3 على قناة واحدة = 9 + انحياز واحد — مهما كان حجم الصورة: هذه **مشاركة الأوزان**.")
