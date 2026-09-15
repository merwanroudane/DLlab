import streamlit as st

from core.registry import get_registry
from core.routing import go
from core.rtl import esc

CATEGORY_AR = {
    "general": "عام", "data": "بيانات", "math": "رياضيات", "model": "نموذج", "training": "تدريب",
    "evaluation": "تقييم", "frameworks": "أطر العمل",
}


def render() -> None:
    reg = get_registry()
    st.markdown("# المصطلحات والبحث")
    st.markdown("<div class='en' style='color:#6B675F'>Glossary & Concept Search</div>", unsafe_allow_html=True)
    st.markdown("قاموس مركزي: ترجمة عربية واحدة معتمدة لكل مصطلح، تعريف قصير، الرمز الرياضي، المفاهيم المرتبطة، ورابط الدرس الكامل.")

    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_input("ابحث بالعربية أو الإنجليزية", key="glossary_q", placeholder="مثال: gradient، الحقبة، shape")
    with c2:
        cats = ["الكل"] + sorted({t.category for t in reg.terms.values()})
        cat = st.selectbox("الفئة", cats, format_func=lambda c: CATEGORY_AR.get(c, c), key="glossary_cat")

    ql = q.strip().lower()
    terms = sorted(reg.terms.values(), key=lambda t: t.en.lower())
    if cat != "الكل":
        terms = [t for t in terms if t.category == cat]
    if ql:
        terms = [t for t in terms if ql in " ".join([t.ar, t.en, t.definition_ar, *t.aliases_ar]).lower()]

    if ql:
        other = [h for h in reg.search(q, limit=10) if h[0] != "term"]
        if other:
            st.markdown("**دروس ومعامل مطابقة:**")
            with st.container(horizontal=True, gap="small"):
                for kind, hid, label in other:
                    st.button(label, key=f"gs_{hid}", type="tertiary",
                              icon=":material/menu_book:" if kind == "lesson" else ":material/science:",
                              on_click=go, args=(hid,))

    st.caption(f"{len(terms)} مصطلحًا")
    for t in terms:
        with st.container(border=True):
            head = f"**{t.ar}** — <span class='en'>{esc(t.en)}</span>"
            if t.notation:
                head += f" &nbsp; ${t.notation}$"
            st.markdown(head, unsafe_allow_html=True)
            st.markdown(t.definition_ar)
            meta = [f"الفئة: {CATEGORY_AR.get(t.category, t.category)}"]
            if t.aliases_ar:
                meta.append("مرادفات: " + "، ".join(t.aliases_ar))
            st.caption(" · ".join(meta))
            with st.container(horizontal=True, gap="small"):
                if t.lesson and t.lesson in reg.lessons:
                    st.button(f"الدرس الكامل: {reg.lessons[t.lesson].title_ar}", key=f"gl_{t.id}", type="secondary",
                              icon=":material/menu_book:", on_click=go, args=(t.lesson,))
                for rid in t.related:
                    r = reg.terms[rid]
                    st.button(f"{r.ar} · {r.en}", key=f"gr_{t.id}_{rid}", type="tertiary",
                              on_click=lambda rid=rid: st.session_state.__setitem__("glossary_q", reg.terms[rid].en))
