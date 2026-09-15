import streamlit as st

from core.rtl import table


def render() -> None:
    st.markdown("# حول المنصة")
    st.markdown("<div class='en' style='color:#6B675F'>About — Deep Learning Interactive Academy</div>", unsafe_allow_html=True)
    st.html(
        '<div class="dlia-hero">'
        "<h1>الدكتور مروان رودان</h1>"
        '<p class="subtitle"><span class="en">Dr. Marwan Roudane</span> — إعداد وتصميم أكاديمي</p>'
        '<p class="subtitle"><span class="en">Deep Learning Interactive Academy · From Foundations to Deep Neural Networks · Designed for Teaching and Research</span></p>'
        "</div>"
    )
    st.markdown("## معلومات المقرر")
    table(
        ["البند", "التفصيل"],
        [
            ("المادة", "التعلم العميق — Deep Learning"),
            ("المدة", "15 أسبوعًا (المحاور الرسمية السبعة + مشروع تطبيقي وعرض)"),
            ("الأدوات", "Python، Keras/TensorFlow، توجيه مقابل في PyTorch، NumPy، pandas، Google Colab/GPU"),
            ("الجمهور", "باحثون وأساتذة وطلبة في التخصصات الاقتصادية والإدارية وغيرها"),
            ("اللغة", "العربية بالدرجة الأولى، مع المصطلحات والأكواد بالإنجليزية"),
            ("نظام التعلم", "أهداف بصياغة بلوم، اختبار قبلي وبعدي، تمارين، معامل، تشخيص مدمج"),
        ],
        ["rtl", "rtl"],
    )
    st.markdown("## الهدف العام")
    st.markdown(
        "تمكين الطالب/الباحث من فهم وتطبيق خوارزميات التعلم العميق باستخدام `Keras`، وبناء نماذج ذكية قادرة على دعم "
        "اتخاذ القرار في المجالات الاقتصادية والإدارية أو مجال تخصصه من خلال بيانات تخصصية."
    )
    st.markdown("## بنهاية المقرر يستطيع الطالب")
    st.markdown(
        """
- التمييز بين التعلم الآلي والتعلم العميق.
- التعامل مع `Keras`/`TensorFlow` وفهم ما يقابلها في `PyTorch`.
- تصميم وتدريب الشبكات العصبية، وبناء نماذج انحدار وتصنيف.
- تحليل أداء النموذج وتفسير نتائجه واستخدام المقاييس الملائمة.
- تطبيق CNN وLSTM وGRU حسب متطلبات الحالة.
- استخدام GPU/Google Colab لتسريع التنفيذ.
- تشخيص مشاكل التدريب واتخاذ قرارات مبنية على البيانات والمنحنيات بدل التخمين.
"""
    )
    st.markdown("## المنهجية")
    st.markdown(
        "> Learn → See → Manipulate → Calculate → Code → Run → Break → Diagnose → Fix → Compare → Understand → Practice\n\n"
        "قاعدة غير قابلة للتفاوض: لا يُستخدم مصطلح تقني قبل شرحه أو ربطه بدرس يشرحه."
    )
    st.markdown("## التقنية")
    st.markdown("مبنية بـ `Python` و`Streamlit`، بواجهة عربية ذات تنقل يميني ونظام طباعة ثنائي الاتجاه، وبأكواد قابلة للتشغيل داخل الصفحة.")
