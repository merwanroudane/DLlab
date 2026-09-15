import streamlit as st

from components.callouts import intuition, takeaway, why
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.progress import module_progress
from core.registry import get_registry
from core.routing import go

LESSON = Lesson(
    id="foundations.ready.concept_map",
    title_ar="خريطة المفاهيم وخريطة الأطر",
    title_en="Concept Map & Framework Map",
    module="foundations.ready",
    order=1,
    prerequisites=["foundations.gallery.before_after"],
    objectives_ar=["رؤية الوحدات 0–22 كرسم اعتماديات واحد مع تقدمك عليه.", "خريطة الأطر: من الرياضيات إلى Keras/TensorFlow/PyTorch بالمفاهيم لا بالأسماء.", "الانتقال بنقرة إلى أي وحدة غير مكتملة."],
    terms=[],
    difficulty="beginner",
    summary_ar="كل شيء في الأسس يخدم حلقة التدريب: بيانات ← موترات ← نموذج (خلية/طبقات/تنشيط) ← أمامي ← خسارة ← خلفي ← محسّن ← تقييم/تعميم/تنظيم ← إطار عمل. الخريطة تلوّن ما زرته.",
)

# concept nodes: (module id, short label, cluster)
CONCEPTS = [
    ("foundations.data", "Data · tensors", "data"), ("foundations.python", "Python · NumPy", "data"), ("foundations.prep", "Preparation · splits", "data"),
    ("foundations.math", "Math", "math"), ("foundations.linalg", "Linear algebra", "math"), ("foundations.calculus", "Calculus · chain rule", "math"), ("foundations.prob", "Probability · CE", "math"),
    ("foundations.ml", "ML · regression", "model"), ("foundations.neuron", "Neuron", "model"), ("foundations.architecture", "Architecture", "model"), ("foundations.activations", "Activations", "model"), ("foundations.forward", "Forward pass", "model"),
    ("foundations.loss", "Loss", "train"), ("foundations.backprop", "Backprop", "train"), ("foundations.optim", "Optimization", "train"), ("foundations.training_loop", "Training loop", "train"), ("foundations.batch_epoch", "Batch · epoch", "train"),
    ("foundations.eval", "Evaluation", "quality"), ("foundations.generalization", "Generalization", "quality"), ("foundations.regularization", "Regularization", "quality"),
    ("foundations.frameworks", "Frameworks", "fw"), ("foundations.gallery", "Gallery", "fw"),
]
EDGES = [("foundations.python", "foundations.data"), ("foundations.data", "foundations.prep"), ("foundations.math", "foundations.linalg"), ("foundations.math", "foundations.calculus"), ("foundations.math", "foundations.prob"),
         ("foundations.linalg", "foundations.forward"), ("foundations.ml", "foundations.neuron"), ("foundations.neuron", "foundations.architecture"), ("foundations.architecture", "foundations.activations"), ("foundations.activations", "foundations.forward"),
         ("foundations.prep", "foundations.training_loop"), ("foundations.prob", "foundations.loss"), ("foundations.forward", "foundations.loss"), ("foundations.calculus", "foundations.backprop"), ("foundations.loss", "foundations.backprop"),
         ("foundations.backprop", "foundations.optim"), ("foundations.optim", "foundations.training_loop"), ("foundations.batch_epoch", "foundations.training_loop"), ("foundations.training_loop", "foundations.eval"), ("foundations.eval", "foundations.generalization"),
         ("foundations.generalization", "foundations.regularization"), ("foundations.regularization", "foundations.frameworks"), ("foundations.training_loop", "foundations.frameworks"), ("foundations.frameworks", "foundations.gallery")]
CLUSTER_COLORS = {"data": "#E6F1FB", "math": "#EFE9F8", "model": "#E3F3F0", "train": "#FBE6E2", "quality": "#FFF3D6", "fw": "#F1EFEA"}
CLUSTER_AR = {"data": "البيانات", "math": "الرياضيات", "model": "النموذج", "train": "التدريب", "quality": "الجودة", "fw": "الأطر"}


def _dot() -> str:
    lines = ['digraph G { rankdir="LR"; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11, color="#B9B2A6"]; edge [color="#B9B2A6"];']
    for cl in CLUSTER_COLORS:
        lines.append(f'subgraph cluster_{cl} {{ label="{CLUSTER_AR[cl]}"; fontname="IBM Plex Sans Arabic"; fontsize=11; color="#E5E0D6"; style="rounded";')
        for mid, lbl, c in CONCEPTS:
            if c == cl:
                p = module_progress(mid)
                fill = "#BDE8D2" if p.percent == 100 else ("#DDF5EA" if p.done else CLUSTER_COLORS[cl])
                lines.append(f'"{mid}" [label="{lbl}\\n{p.done}/{p.total}", fillcolor="{fill}"];')
        lines.append("}")
    for a, b in EDGES:
        lines.append(f'"{a}" -> "{b}";')
    lines.append("}")
    return "\n".join(lines)


FW_DOT = '''digraph F { rankdir="TB"; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11, color="#B9B2A6"]; edge [color="#B9B2A6", fontname="Inter", fontsize=9];
"math" [label="Mathematics\\nz = xW+b · L · ∇L · θ ← θ−η∇L", fillcolor="#EFE9F8"];
"numpy" [label="NumPy (modules 2–20)\\nforward · backward · update by hand", fillcolor="#E6F1FB"];
"tf" [label="TensorFlow\\ntf.Tensor · tf.Variable · GradientTape · tf.data", fillcolor="#FBE6E2"];
"keras" [label="Keras\\nlayers · compile · fit · callbacks", fillcolor="#E3F3F0"];
"torch" [label="PyTorch\\ntorch.Tensor · Autograd · nn.Module · explicit loop", fillcolor="#EFE9F8"];
"tools" [label="Beside the stack\\nscikit-learn · Colab/Jupyter · TensorBoard", fillcolor="#FFF3D6", style="rounded,filled,dashed"];
"math" -> "numpy" [label="you wrote it"];
"numpy" -> "tf" [label="same math, autodiff + GPU"];
"numpy" -> "torch" [label="same math, autodiff + GPU"];
"tf" -> "keras" [label="high-level API on top"];
"keras" -> "tools" [style=dashed]; "torch" -> "tools" [style=dashed];
}'''


def render() -> None:
    lesson_header(LESSON)
    why("22 وحدة و150 درسًا تبدو كثيرة حتى تُرسم: ست عناقيد وسلسلة اعتماديات واحدة تنتهي عند حلقة التدريب ثم الأطر. الخريطة أدناه تُلوَّن بتقدمك في هذه الجلسة.")
    h2("خريطة المفاهيم", "Concept map")
    st.graphviz_chart(_dot(), width="stretch")
    st.caption("أخضر داكن = وحدة مكتملة · أخضر فاتح = بدأت · لون العنقود = لم تُفتح بعد. الأسهم: «يبنى على».")
    reg = get_registry()
    incomplete = [(mid, lbl) for mid, lbl, _ in CONCEPTS if module_progress(mid).percent < 100]
    if incomplete:
        st.markdown("**وحدات غير مكتملة — انتقل إليها:**")
        cols = st.columns(4)
        for i, (mid, lbl) in enumerate(incomplete):
            with cols[i % 4]:
                st.button(reg.modules[mid].title_ar, key=f"cm_go_{mid}", type="tertiary", on_click=go, args=(mid,))
    else:
        st.success("كل وحدات الأسس مكتملة في هذه الجلسة.", icon="✅")
    intuition("اقرأ الخريطة من اليسار: البيانات والرياضيات تغذيان النموذج؛ النموذج والخسارة يغذيان الانتشار الخلفي والتحسين؛ كل ذلك يلتقي في حلقة التدريب؛ التقييم والتعميم والتنظيم يحكمون عليها؛ والأطر تغلّف الكل.")
    h2("خريطة الأطر", "Framework map")
    st.graphviz_chart(FW_DOT, width="stretch")
    st.markdown("""
- **الرياضيات** واحدة في كل مستوى: `z = xW + b`، الخسارة، التدرج، التحديث.
- **NumPy**: كتبتها بيدك (الوحدات 2–20) — هذا ما يجعل الطبقات العليا شفافة.
- **TensorFlow / PyTorch**: نفس الرياضيات + اشتقاق تلقائي + أجهزة. **Keras** واجهة فوق TensorFlow.
- **بجانب المكدس**: أدوات وبيئات، لا أطر.
""")
    quiz("ready.map", [
        Q("العقدة التي تلتقي فيها كل الأسس قبل الأطر:", ["الخسارة", "حلقة التدريب", "التقييم"], 1, "المركز."),
        Q("Keras في خريطة الأطر تقع…", ["تحت TensorFlow", "فوق TensorFlow", "بجانب المكدس"], 1, "واجهة عليا."),
        Q("scikit-learn وColab وTensorBoard…", ["أطر عمل", "بجانب المكدس: أدوات وبيئات", "خلفيات"], 1, "ليست أطرًا."),
    ])
    takeaway("خريطة واحدة: بيانات + رياضيات → نموذج → خسارة → خلفي → محسّن → حلقة → جودة → أطر. الأطر تغلّف رياضيات كتبتها بنفسك.")
    lesson_footer(LESSON, ["خريطة المفاهيم بتقدمك.", "الوحدات غير المكتملة بنقرة.", "خريطة الأطر بالمفاهيم."])
