"""Chain Rule Lab — build a chain of operations and watch gradients flow back."""

import numpy as np
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table

LAB = Lab(
    id="labs.chain_rule_lab",
    title_ar="معمل قاعدة السلسلة",
    title_en="Chain Rule Lab",
    category="math",
    description_ar="ركّب سلسلة من العمليات (خطي، sigmoid، tanh، ReLU، مربع)، أدخل قيمة، وشاهد التمرير الأمامي ثم عودة التدرج حلقة بحلقة مع حاصل الضرب المتراكم.",
    related_lessons=["foundations.calculus.chain_rule"],
)

OPS = {
    "خطي: 2x + 1": (lambda x: 2 * x + 1, lambda x: 2.0, "2"),
    "خطي: 0.5x − 1": (lambda x: 0.5 * x - 1, lambda x: 0.5, "0.5"),
    "sigmoid": (lambda x: 1 / (1 + np.exp(-x)), lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))), "σ(1−σ)"),
    "tanh": (np.tanh, lambda x: 1 - np.tanh(x) ** 2, "1−tanh²"),
    "ReLU": (lambda x: max(0.0, x), lambda x: 1.0 if x > 0 else 0.0, "1 or 0"),
    "مربع: x²": (lambda x: x ** 2, lambda x: 2 * x, "2x"),
}


def _chain_svg(names: list[str], vals: list[float], active: int | None, phase: str) -> str:
    n = len(names)
    w = 120; gap = 40
    total = n * w + (n + 1) * gap
    s = f'<svg viewBox="0 0 {total} 130" width="100%" style="max-width:{total}px">'
    s += '<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6B675F"/></marker></defs>'
    x = gap
    s += f'<text x="{gap / 2}" y="60" text-anchor="middle" font-size="12" font-family="JetBrains Mono, monospace">x={vals[0]:.3g}</text>'
    for i, name in enumerate(names):
        fill = "#1F7A78" if i == active else "#FBF4E8"
        color = "#fff" if i == active else "#2B2A28"
        s += f'<rect x="{x}" y="30" width="{w}" height="50" rx="10" fill="{fill}" stroke="#EADFCD"/>'
        s += f'<text x="{x + w / 2}" y="52" text-anchor="middle" font-size="12" fill="{color}" font-family="Inter, sans-serif">{esc(name)}</text>'
        s += f'<text x="{x + w / 2}" y="70" text-anchor="middle" font-size="11" fill="{color}" font-family="JetBrains Mono, monospace">→ {vals[i + 1]:.4g}</text>'
        if i < n - 1:
            s += f'<line x1="{x + w}" y1="55" x2="{x + w + gap - 4}" y2="55" stroke="#6B675F" stroke-width="2" marker-end="url(#ah)"/>'
        x += w + gap
    s += f'<text x="{total / 2}" y="115" text-anchor="middle" font-size="12" fill="#6B675F">{esc(phase)}</text></svg>'
    return s


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    chain = st.multiselect("السلسلة (بالترتيب من المدخل إلى المخرج)", list(OPS), default=["خطي: 2x + 1", "sigmoid", "مربع: x²"], key="crl_chain")
    x0 = st.slider("المدخل x", -3.0, 3.0, 0.5, 0.1, key="crl_x")
    if not chain:
        st.info("اختر عملية واحدة على الأقل.")
        return
    intuition("أمامًا: كل صندوق يطبق دالته ويحفظ مخرجه. خلفًا: نبدأ بـ 1 عند المخرج ونضرب في المشتقة المحلية لكل صندوق بالترتيب المعكوس.")

    vals = [x0]
    for name in chain:
        vals.append(float(OPS[name][0](vals[-1])))
    locals_ = [float(OPS[name][1](vals[i])) for i, name in enumerate(chain)]
    frames = []
    for i, name in enumerate(chain):
        frames.append(Frame(_chain_svg(chain, vals, i, "forward"), caption(f"**أمامي** — `{name}`: المدخل `{vals[i]:.4g}` ← المخرج `{vals[i + 1]:.4g}`."), action="Forward", highlight=0))
    acc = 1.0
    for i in reversed(range(len(chain))):
        acc_new = acc * locals_[i]
        frames.append(Frame(_chain_svg(chain, vals, i, "backward"),
                            caption(f"**خلفي** — `{chain[i]}`: المشتقة المحلية `{OPS[chain[i]][2]}` = `{locals_[i]:.4g}`؛ التراكمي `{acc:.4g} × {locals_[i]:.4g} = {acc_new:.4g}`."),
                            action="Backward", equation=f"d(out)/d(in_{i}) = {acc:.4g} × {locals_[i]:.4g} = {acc_new:.4g}",
                            values=[("accumulated", f"{acc:.4g}", f"{acc_new:.4g}")], highlight=1))
        acc = acc_new
    animation_player("chain_lab_anim", frames, title_ar="أمامي ثم خلفي", stages=["Forward", "Backward"], interval_ms=1300)

    h = 1e-6
    def full(x):
        for name in chain:
            x = OPS[name][0](x)
        return x
    numeric = (full(x0 + h) - full(x0 - h)) / (2 * h)
    st.markdown("### النتيجة")
    table(["الطريقة", "d(out)/dx"], [("قاعدة السلسلة (حاصل ضرب المحليات)", f"{acc:.6g}"), ("فروق منتهية (تحقق)", f"{numeric:.6g}")], ["rtl", "num"])
    st.code(" × ".join(f"{v:.4g}" for v in reversed(locals_)) + f" = {acc:.6g}", language="text")
    if abs(acc) < 1e-3:
        warning_note("التدرج شبه معدوم: إحدى الحلقات (sigmoid/tanh مشبعة، أو ReLU بمدخل سالب) قتلت الإشارة. هذا بالضبط تلاشي التدرج — جرّب تغيير x أو استبدال العملية.")
    if any(n in ("sigmoid", "tanh") for n in chain) and len([n for n in chain if n in ("sigmoid", "tanh")]) >= 2:
        warning_note("تنشيطان مشبعان متتاليان: حاصل ضرب مشتقتين ≤ 0.25 كل منهما. أضف المزيد وراقب التدرج يتقلص هندسيًا.")
