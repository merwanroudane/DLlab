from core.models import Module

MODULE = Module(
    id="foundations.frameworks",
    section="foundations",
    title_ar="الوحدة 21 — أكاديمية أطر التعلم العميق: Keras وTensorFlow وPyTorch",
    title_en="Module 21 — Deep Learning Frameworks Academy: Keras, TensorFlow & PyTorch",
    order=21,
    icon=":material/deployed_code:",
    prerequisites=["foundations.training_loop", "foundations.eval", "foundations.regularization"],
    purpose_ar="فهم ما هو إطار العمل، ما الذي يفعله بدلًا عنا (الموترات، الاشتقاق التلقائي، الطبقات، المحسّنات، حلقة التدريب، CPU/GPU) وما الذي يبقى مسؤولية الباحث. مسار كامل لكل من Keras وTensorFlow وPyTorch مع مستكشفات مخرجات، تفكيك `model.summary()`، محرك `fit()`، ومحرك حلقة التدريب في PyTorch.",
    why_ar="بعد عشرين وحدة تفهم كل خطوة في حلقة التدريب. الآن تتعلم كيف تطلبها من إطار عمل — دون أن تصبح `model.fit()` أو `loss.backward()` تعويذة غامضة.",
    objectives_ar=["تمييز اللغة والمكتبة وإطار العمل وواجهة API العليا والخلفية وبيئة التشغيل.", "قراءة مخرجات Keras (summary، سجل fit، History، evaluate، predict) سطرًا سطرًا.", "كتابة حلقة تدريب صريحة في PyTorch وتفسير كل سطر فيها بمقابله الرياضي.", "ربط كل مفهوم بين Keras/TensorFlow وPyTorch بجدول واحد.", "اختيار إطار العمل بحسب السياق لا بحسب الموضة."],
    challenges_ar=["حفظ الصيغة دون فهم ما يحدث داخلها.", "الخلط بين إطار العمل والمنصة (Colab/Jupyter) والأداة (TensorBoard).", "الاعتقاد أن `fit()` سحر، أو أن `zero_grad()` تعليمة عشوائية."],
)

LESSON_MODULES = [
    "from_python_to_framework", "ecosystem_map",
    "keras", "keras_layers_models", "keras_compile", "keras_fit", "keras_evaluate_predict", "keras_callbacks_saving",
    "keras_summary_deconstruction", "keras_output_explorer", "keras_errors", "keras_code_lab_standard",
    "tensorflow", "tf_tensors", "tf_gradient_tape", "tf_data", "tf_keras_custom_loop", "tf_tensorboard_errors",
    "pytorch", "pt_tensors", "pt_autograd", "pt_nn_module", "pt_loss_optim_data", "pt_training_loop", "pt_zero_grad", "pt_train_eval", "pt_saving_errors",
    "concept_mapping", "same_model_three_views", "framework_choice",
]
