"""Epoch / Batch simulator (spec §20) — a *real* mini-batch gradient-descent
run on the study-hours dataset, replayed step by step."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, intuition, practical_note, warning_note
from components.math_explainer import equation
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, pipeline, table
from labs.datasets import study_hours

LAB = Lab(
    id="labs.epoch_batch_simulator",
    title_ar="محاكي الحقبة والدفعة",
    title_en="Epoch / Batch Simulator",
    category="training",
    description_ar="غيّر حجم البيانات وحجم الدفعة وعدد الحقب والخلط ومعدل التعلم، ثم شاهد كل خطوة تدريب حقيقية: دفعة ← تمرير أمامي ← تنبؤ ← خسارة ← تدرج ← تحديث.",
    related_lessons=["foundations.start.model_training_prediction"],
)

STAGES = ["Batch", "Forward", "Prediction", "Loss", "Backward", "Update"]
STAGES_AR = ["اختيار الدفعة", "التمرير الأمامي", "التنبؤ", "الخسارة", "الانتشار الخلفي (التدرج)", "تحديث المعلمات"]


@dataclass
class Step:
    epoch: int
    batch_index: int
    indices: np.ndarray
    w_before: float
    b_before: float
    pred: np.ndarray
    loss: float
    dw: float
    db: float
    w_after: float
    b_after: float
    is_partial: bool
    epoch_end: bool


def batches_for_epoch(n: int, batch_size: int, drop_remainder: bool) -> int:
    return n // batch_size if drop_remainder else math.ceil(n / batch_size)


@st.cache_data(max_entries=32)
def simulate(n: int, batch_size: int, epochs: int, shuffle: bool, lr: float, drop_remainder: bool,
             seed: int = 0) -> list[Step]:
    df = study_hours(n=n, seed=5)
    x = df["hours"].to_numpy() / 10.0          # scaled to [0, 1]
    y = df["score"].to_numpy() / 100.0         # scaled to [0, 1]
    rng = np.random.default_rng(seed)
    w, b = 0.0, 0.0
    steps: list[Step] = []
    for ep in range(1, epochs + 1):
        order = rng.permutation(n) if shuffle else np.arange(n)
        nb = batches_for_epoch(n, batch_size, drop_remainder)
        for bi in range(nb):
            idx = order[bi * batch_size:(bi + 1) * batch_size]
            xb, yb = x[idx], y[idx]
            pred = w * xb + b
            err = pred - yb
            loss = float(np.mean(err ** 2))
            dw = float(np.mean(2 * err * xb))
            db = float(np.mean(2 * err))
            w_new, b_new = w - lr * dw, b - lr * db
            steps.append(Step(ep, bi + 1, idx, w, b, pred, loss, dw, db, w_new, b_new,
                              is_partial=len(idx) < batch_size, epoch_end=(bi == nb - 1)))
            w, b = w_new, b_new
    return steps


def _batch_strip_html(n: int, batch_size: int, current: np.ndarray, order_shuffled: bool) -> str:
    cur = set(current.tolist())
    cells = []
    for i in range(n):
        in_batch = i in cur
        style = ("background:#1F7A78;color:#fff;border-color:#1F7A78" if in_batch
                 else "background:#FBF4E8;color:#6B675F")
        cells.append(f'<span style="display:inline-block;min-width:1.7rem;padding:.1rem .25rem;margin:.1rem;'
                     f'border:1px solid #EADFCD;border-radius:6px;font-size:.75rem;text-align:center;'
                     f'font-family:JetBrains Mono,monospace;{style}">{i + 1}</span>')
    note = "الترتيب مخلوط: الدفعة تجمع ملاحظات متفرقة" if order_shuffled else "بدون خلط: الدفعة تجمع ملاحظات متتالية"
    return (f'<div style="direction:ltr;line-height:1.2">{"".join(cells)}</div>'
            f'<div class="dlia-rtl" style="color:#6B675F;font-size:.85rem;margin-top:.3rem">{esc(note)}</div>')


MAX_ANIM_STEPS = 40


def _loss_sparkline_svg(losses: list[float], upto: int, w: int = 420, h: int = 90) -> str:
    """Tiny LTR SVG of batch loss up to `upto` (inclusive), with the current point marked."""
    if not losses:
        return ""
    finite = [v for v in losses if math.isfinite(v)]
    top = max(finite) if finite else 1.0
    top = top if top > 0 else 1.0
    n = len(losses)
    pts = []
    for i, v in enumerate(losses[: upto + 1]):
        x = 20 + (w - 40) * (i / max(1, n - 1))
        y = h - 12 - (h - 30) * (min(v, top) / top)
        pts.append((x, y))
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    cx, cy = pts[-1]
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">'
            f'<line x1="20" y1="{h - 12}" x2="{w - 20}" y2="{h - 12}" stroke="#EADFCD"/>'
            f'<path d="{path}" fill="none" stroke="#C8473A" stroke-width="2"/>'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#1F7A78"/>'
            f'<text x="{w - 20}" y="12" text-anchor="end" font-size="11" fill="#6B675F" font-family="JetBrains Mono, monospace">loss (batch)</text>'
            f'</svg>')


def _animation_frames(steps: list[Step], n: int, batch_size: int, steps_per_epoch: int, lr: float,
                      shuffle: bool) -> list[Frame]:
    losses = [t.loss for t in steps]
    frames: list[Frame] = []
    for k, s in enumerate(steps[:MAX_ANIM_STEPS]):
        where = f"الحقبة {s.epoch} · الدفعة {s.batch_index}/{steps_per_epoch} · الخطوة {k + 1}"
        strip = _batch_strip_html(n, batch_size, s.indices, shuffle)
        spark = _loss_sparkline_svg(losses, k)
        frames.append(Frame(strip, caption(f"**{where}** — نختار {len(s.indices)} ملاحظة" + (" (دفعة جزئية)" if s.is_partial else "") + "."),
                            action="Batch", highlight=0))
        frames.append(Frame(spark, caption(f"{where} — نمرر ملاحظات الدفعة عبر النموذج الحالي: `w = {s.w_before:.4f}`, `b = {s.b_before:.4f}`."),
                            action="Forward", equation="ŷ = w·x + b", highlight=1))
        frames.append(Frame(spark, caption(f"{where} — أول تنبؤات الدفعة: `{[round(float(v), 3) for v in s.pred[:4]]}`."),
                            action="Prediction", highlight=2))
        frames.append(Frame(spark, caption(f"{where} — خسارة هذه الدفعة (متوسط مربع الخطأ) = **{s.loss:.5f}**."),
                            action="Loss", equation="L = mean((ŷ − y)²)", values=[("loss", "", f"{s.loss:.5f}")], highlight=3))
        frames.append(Frame(spark, caption(f"{where} — التدرج يخبرنا اتجاه زيادة الخسارة: `dL/dw = {s.dw:+.4f}`, `dL/db = {s.db:+.4f}`."),
                            action="Backward", equation="dL/dw = mean(2(ŷ−y)·x),  dL/db = mean(2(ŷ−y))", highlight=4))
        frames.append(Frame(spark, caption(f"{where} — نتحرك عكس التدرج بخطوة `η = {lr}`." + (" **انتهت الحقبة.**" if s.epoch_end else "")),
                            action="Update" + (" · Epoch done" if s.epoch_end else ""),
                            equation=f"w ← w − η·dL/dw = {s.w_before:.4f} − {lr}×({s.dw:+.4f}) = {s.w_after:.4f}",
                            values=[("w", f"{s.w_before:.5f}", f"{s.w_after:.5f}"), ("b", f"{s.b_before:.5f}", f"{s.b_after:.5f}")],
                            highlight=5))
    return frames


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    intuition(
        "**الحقبة** `Epoch` = مرور كامل على كل الملاحظات. **الدفعة** `Batch` = مجموعة صغيرة منها تُعالج معًا. "
        "كل دفعة تعطي **خطوة** `Step` واحدة = تحديثًا واحدًا للمعلمات. عدد الخطوات في الحقبة = عدد الدفعات."
    )

    with st.container(border=True):
        st.markdown("**⚙️ الإعدادات (معلمات فائقة يحددها الباحث)**")
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.slider("حجم البيانات n", 10, 200, 100, 5, key="ebs_n")
            batch_size = st.slider("حجم الدفعة batch_size", 1, 64, 20, 1, key="ebs_bs")
        with c2:
            epochs = st.slider("عدد الحقب epochs", 1, 5, 2, key="ebs_epochs")
            lr = st.select_slider("معدل التعلم learning_rate", options=[0.001, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0, 1.5],
                                  value=0.1, key="ebs_lr")
        with c3:
            shuffle = st.toggle("خلط الملاحظات كل حقبة shuffle", value=True, key="ebs_shuffle")
            drop_remainder = st.toggle("إسقاط الدفعة الناقصة drop_remainder", value=False, key="ebs_drop")

    steps_per_epoch = batches_for_epoch(n, batch_size, drop_remainder)
    remainder = n % batch_size
    total_steps = steps_per_epoch * epochs

    st.markdown("### 1) الحساب قبل التشغيل")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("steps_per_epoch", steps_per_epoch)
    c2.metric("إجمالي الخطوات", total_steps)
    c3.metric("آخر دفعة", f"{remainder if remainder else batch_size} ملاحظة")
    c4.metric("التحديثات لكل ملاحظة", epochs)
    st.code(
        f"n = {n}, batch_size = {batch_size}\n"
        f"n / batch_size = {n / batch_size:.2f}\n"
        f"steps_per_epoch = {'floor' if drop_remainder else 'ceil'}({n} / {batch_size}) = {steps_per_epoch}\n"
        f"total_steps = steps_per_epoch × epochs = {steps_per_epoch} × {epochs} = {total_steps}",
        language="text",
    )
    if remainder and not drop_remainder:
        warning_note(
            f"`n = {n}` لا يقبل القسمة على `batch_size = {batch_size}`: آخر دفعة في كل حقبة تحتوي {remainder} ملاحظة فقط "
            f"(**دفعة جزئية**). الأطر تعالجها افتراضيًا؛ `drop_remainder=True` يسقطها فتخسر {remainder} ملاحظة كل حقبة."
        )
    elif remainder and drop_remainder:
        warning_note(f"أُسقطت {remainder} ملاحظة من كل حقبة. مفيد عندما تتطلب البنية دفعات بحجم ثابت، ومضرّ مع البيانات القليلة.")

    steps = simulate(n, batch_size, epochs, shuffle, float(lr), drop_remainder)
    if not steps:
        st.error("لا توجد خطوات: حجم الدفعة أكبر من البيانات مع إسقاط الدفعة الناقصة.")
        return

    st.markdown("### 2) شاهد كل خطوة")
    # The slider widget state is the single source of truth (1-based). Buttons
    # change it in callbacks, which run before the slider renders.
    total = len(steps)
    st.session_state.setdefault("ebs_slider", 1)
    st.session_state["ebs_slider"] = max(1, min(total, st.session_state["ebs_slider"]))

    def _jump(delta: int | None = None, to: int | None = None) -> None:
        cur = st.session_state["ebs_slider"]
        new = to if to is not None else cur + (delta or 0)
        st.session_state["ebs_slider"] = max(1, min(total, new))

    with st.container(horizontal=True, vertical_alignment="center"):
        st.button("من البداية", icon=":material/restart_alt:", key="ebs_restart", on_click=_jump, kwargs={"to": 1})
        st.button("السابق", icon=":material/arrow_forward:", key="ebs_prev", on_click=_jump, kwargs={"delta": -1})
        st.button("التالي", icon=":material/arrow_back:", type="primary", key="ebs_next", on_click=_jump, kwargs={"delta": 1})
        st.button("آخر خطوة", icon=":material/last_page:", key="ebs_last", on_click=_jump, kwargs={"to": total})
    if total > 1:
        st.slider("الخطوة", 1, total, key="ebs_slider")
    step_idx = st.session_state["ebs_slider"] - 1
    s = steps[step_idx]
    stage = st.segmented_control("المرحلة داخل الخطوة", STAGES, default="Update", key="ebs_stage") or "Update"
    si = STAGES.index(stage)
    pipeline(STAGES, active=si)

    st.markdown(
        f"**الحقبة {s.epoch} / {epochs} · الدفعة {s.batch_index} / {steps_per_epoch} · الخطوة الكلية "
        f"{step_idx + 1} / {len(steps)}** — {STAGES_AR[si]}"
    )
    if s.is_partial:
        st.info(f"هذه دفعة جزئية: {len(s.indices)} ملاحظة بدل {batch_size}.", icon="ℹ️")

    if si == 0:
        st.markdown(f"نأخذ الملاحظات ذات الأرقام (1-based): `{(s.indices + 1).tolist()}`")
        st.html(_batch_strip_html(n, batch_size, s.indices, shuffle))
    elif si == 1:
        st.markdown("نمرر كل ملاحظة في الدفعة عبر النموذج الحالي (قبل التحديث):")
        equation(r"\hat{y}_i = w \cdot x_i + b", [("w", f"الوزن الحالي = {s.w_before:.4f}"), ("b", f"الانحياز الحالي = {s.b_before:.4f}"),
                                                    ("x_i", "ساعات المذاكرة (مقسومة على 10 لتبقى في [0,1])")],
                 meaning_ar="نفس المعادلة تُطبَّق على كل ملاحظات الدفعة دفعة واحدة (عملية متجهية).")
    elif si == 2:
        df = study_hours(n=n, seed=5)
        xb = df["hours"].to_numpy()[s.indices] / 10.0
        yb = df["score"].to_numpy()[s.indices] / 100.0
        rows = [(str(i + 1), f"{xv:.2f}", f"{p:.3f}", f"{yv:.3f}", f"{p - yv:+.3f}")
                for i, xv, p, yv in zip(s.indices[:8], xb[:8], s.pred[:8], yb[:8])]
        table(["#", "x (مقيس)", "ŷ التنبؤ", "y الحقيقة", "الخطأ ŷ−y"], rows, ["num", "num", "num", "num", "num"],
              caption=f"أول {min(8, len(rows))} ملاحظات من الدفعة")
    elif si == 3:
        equation(r"L = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2",
                 [("m", f"عدد ملاحظات الدفعة = {len(s.indices)}"), ("L", f"خسارة هذه الدفعة = {s.loss:.5f}")],
                 meaning_ar="متوسط مربع الخطأ **على هذه الدفعة فقط**، لا على كل البيانات. لذلك تتذبذب الخسارة بين الدفعات.")
    elif si == 4:
        equation(r"\frac{\partial L}{\partial w} = \frac{2}{m}\sum_i (\hat{y}_i - y_i)\,x_i \qquad \frac{\partial L}{\partial b} = \frac{2}{m}\sum_i (\hat{y}_i - y_i)",
                 [(r"\partial L/\partial w", f"= {s.dw:+.5f}"), (r"\partial L/\partial b", f"= {s.db:+.5f}")],
                 meaning_ar="التدرج يخبرنا: إذا زدنا w قليلًا، هل تزيد الخسارة (إشارة موجبة) أم تنقص (سالبة)؟ نتحرك عكس الإشارة.")
    else:
        equation(r"w \leftarrow w - \eta\,\frac{\partial L}{\partial w}, \qquad b \leftarrow b - \eta\,\frac{\partial L}{\partial b}",
                 [(r"\eta", f"معدل التعلم = {lr}")],
                 meaning_ar="خطوة واحدة في الاتجاه المعاكس للتدرج. هذا هو «التعلم» فعليًا.")
        table(["المعلمة", "قبل", "التدرج", "الخطوة −η·∇", "بعد"],
              [("w", f"{s.w_before:.5f}", f"{s.dw:+.5f}", f"{-lr * s.dw:+.5f}", f"{s.w_after:.5f}"),
               ("b", f"{s.b_before:.5f}", f"{s.db:+.5f}", f"{-lr * s.db:+.5f}", f"{s.b_after:.5f}")],
              ["code", "num", "num", "num", "num"])
        if s.epoch_end:
            st.success(f"✅ Epoch {s.epoch} completed — مرّت كل الملاحظات مرة واحدة ({steps_per_epoch} تحديثًا).", icon="🏁")

    st.markdown("### 3) شغّل التدريب كاملًا (تحريك)")
    st.markdown(
        "اضغط ▶ لمشاهدة الخطوات تتوالى تلقائيًا: كل خطوة تمر بالمراحل الست. يمكنك الإيقاف والرجوع وتغيير السرعة. "
        f"يعرض التحريك أول {min(len(steps), MAX_ANIM_STEPS)} خطوة."
    )
    animation_player(
        "ebs_anim",
        _animation_frames(steps, n, batch_size, steps_per_epoch, float(lr), shuffle),
        title_ar="دفعة ← تمرير أمامي ← تنبؤ ← خسارة ← انتشار خلفي ← تحديث",
        stages=STAGES,
        interval_ms=1000,
    )

    st.markdown("### 4) الخسارة عبر الخطوات")
    losses = [t.loss for t in steps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, x=list(range(1, len(losses) + 1)), mode="lines+markers", name="loss per batch",
                             line=dict(color="#C8473A")))
    fig.add_vline(x=step_idx + 1, line=dict(color="#1F7A78", dash="dot"))
    for e in range(1, epochs):
        fig.add_vline(x=e * steps_per_epoch + 0.5, line=dict(color="#B9B2A6", dash="dash"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="step", yaxis_title="batch loss (MSE)",
                      plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="ebs_loss_chart")
    if max(losses) > 10 or any(math.isnan(v) or math.isinf(v) for v in losses):
        st.error("🔥 الخسارة تنفجر: معدل التعلم كبير جدًا فالخطوات تقفز فوق الحد الأدنى وتتباعد. جرّب 0.1 أو أقل.", icon="❌")
    elif lr <= 0.001:
        st.warning("🐢 الخسارة تنخفض ببطء شديد: معدل التعلم صغير جدًا. ستحتاج حقبًا كثيرة.", icon="⚠️")

    st.markdown("### 5) الخلاصة الاصطلاحية")
    table(
        ["المفهوم", "English", "التفسير", "في هذه التجربة"],
        [
            ("الحقبة", "Epoch", "مرور كامل على بيانات التدريب", f"epochs={epochs}"),
            ("الدفعة", "Batch", "مجموعة ملاحظات تُعالج معًا", f"{steps_per_epoch} دفعة/حقبة"),
            ("حجم الدفعة", "Batch Size", "عدد الملاحظات في الدفعة", f"batch_size={batch_size}"),
            ("الخطوة / التكرار", "Step / Iteration", "تحديث واحد للمعلمات", f"{len(steps)} خطوة كلية"),
            ("التحديث", "Update", "w ← w − η∇L على دفعة واحدة", f"η={lr}"),
            ("الدفعة الجزئية", "Last partial batch", "عندما لا يقبل n القسمة على batch_size", f"{remainder or 'لا توجد'}"),
        ],
        ["rtl", "ltr", "rtl", "code"],
    )
    common_mistake("«حقبة واحدة = تحديث واحد». خطأ: الحقبة الواحدة تحتوي `steps_per_epoch` تحديثًا، واحد لكل دفعة.")
    practical_note("جرّب: `batch_size = n` (دفعة واحدة = الانحدار التدريجي الكامل) ثم `batch_size = 1` (SGD خالص) وقارن نعومة منحنى الخسارة.")
