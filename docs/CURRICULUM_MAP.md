# Curriculum Map

Status legend: ✅ built · 🟡 partial · ⬜ planned (declared in the spec, not yet a page — never shown as an empty page)

## Foundations Academy (spec §28)

| # | Module | Route | Status | Lessons (planned) |
|---|---|---|---|---|
| 0 | Start Here | `foundations.start` | ✅ | how to use · roadmap · self-check · AI/ML/DL · model/training/prediction/inference · data+math+optimization · terminology |
| 1 | Data Foundations | `foundations.data` | ✅ | data · dataset · observation · feature · target · modalities · variable types · dtype · shape/axis/rank · anatomy capstone (+ Dataset Anatomy Lab) |
| 2 | Python Foundations for DL | `foundations.python` | ✅ | variables, types, lists/tuples/dicts, indexing/slicing, control flow, functions, imports, NumPy ndarray/shape/dtype, vectorization, broadcasting, pandas, random/seed/reproducibility, list vs array vs tensor |
| 3 | Mathematical Foundations | `foundations.math` | ✅ | notation, variables/constants/coefficients, functions, domain/range, linear/nonlinear, exponent/exponential/log, Σ, average, absolute value |
| 4 | Linear Algebra | `foundations.linalg` | ✅ | scalar, vector, dot product, matrix, transpose, add/scale/multiply, Hadamard, identity, tensor rank/shape/axes, batch & channel dims (+ Tensor Shape Explorer, Vector/Matrix Lab) |
| 5 | Calculus for DL | `foundations.calculus` | ✅ | change, slope, secant/tangent, derivative, partial, gradient, direction, chain rule, computational chain, Jacobian, Hessian (+ Slope/Derivative/Gradient/Chain-rule labs) |
| 6 | Probability & Statistics | `foundations.prob` | ✅ | mean/variance/std, distributions, random variable, probability, conditional, expectation, noise, sampling, correlation/covariance, likelihood, Bernoulli/Categorical/Normal, entropy, cross-entropy |
| 7 | ML Foundations | `foundations.ml` | ✅ | supervised/unsupervised/semi/self, regression/classification types, X/y, model/algorithm/architecture, fit/inference, params/hyperparams, objective/loss/metric, baseline, generalization, linear & logistic regression bridges |
| 8 | Data Preparation | `foundations.prep` | ✅ | inspection, missing, duplicates, outliers, label errors, encoding, scaling/normalization/standardization, shuffling, splitting, stratification, leakage (feature/target/preprocessing), imbalance, augmentation, tensors, batching, pipelines (+ Split, Leakage, Scaling labs) |
| 9 | From Regression to Neuron | `foundations.neuron` | ✅ | y=β₀+β₁x → z=wx+b → multi-input → a=f(z); weight, bias, weighted sum, activation, neuron/unit/node/perceptron (+ Neuron Lab) |
| 10 | NN Architecture | `foundations.architecture` | ✅ | input/hidden/output, connections, depth/width, dense, parameter count (+ Network Builder, Parameter Counter) |
| 11 | Activation Foundations | `foundations.activations` | ✅ | nonlinearity, linear, sigmoid, tanh, ReLU, LeakyReLU, ELU, SELU, Swish, GELU, softmax, derivatives, saturation, dead ReLU, output suitability (+ Activation Lab) |
| 12 | Forward Pass | `foundations.forward` | ✅ | weighted sum, bias, activation, layer output, logits, probability, threshold, walkthrough, animation |
| 13 | Error, Loss & Objective | `foundations.loss` | ✅ | error/residual/loss/cost/objective/metric, MSE, MAE, BCE, CCE, sparse CCE, loss/activation compatibility (+ Loss Lab) |
| 14 | Backpropagation | `foundations.backprop` | ✅ | 16 sub-pages per spec §7 (+ Chain Rule / Gradient labs) |
| 15 | Optimization | `foundations.optim` | ✅ | loss surface, minima, saddle, GD/SGD/mini-batch, LR, momentum, Nesterov, RMSprop, Adam, schedules, plateau, convergence (+ GD Lab, Optimizer Race, LR Lab) |
| 16 | Training Loop | `foundations.training_loop` | ✅ | init → batch → forward → loss → backward → step → next → epoch end → validation → stop (+ Training Loop Simulator) |
| 17 | Batch / Epoch / Iteration | `foundations.batch_epoch` | ✅ | definitions · batch-size effects (+ Epoch/Batch Simulator) |
| 18 | Model Evaluation | `foundations.eval` | ✅ | accuracy, precision, recall, F1, confusion matrix, ROC/AUC, threshold, MSE/RMSE/MAE, metric choice (+ Confusion Matrix Lab, Threshold Lab) |
| 19 | Generalization | `foundations.generalization` | ✅ | train vs val, under/overfitting, bias/variance, complexity, dataset size (+ Overfitting Lab, Curves Diagnostic Lab) |
| 20 | Regularization | `foundations.regularization` | ✅ | L1, L2/weight decay, dropout, early stopping, augmentation, simplification, batch norm (+ Regularization, Dropout labs) |
| 21 | Frameworks Academy | `foundations.frameworks` | ✅ | 30 lessons: from Python to framework · ecosystem map · Keras (parent + layers/models, compile, fit + fit() visualizer, evaluate/predict, callbacks/saving, summary deconstruction, output explorer, errors, code-lab standard) · TensorFlow (parent + tensors, GradientTape, tf.data, custom loop/saving, TensorBoard + errors) · PyTorch (parent + tensors, Autograd visualizer, nn.Module, loss/optim/data, training-loop visualizer, zero_grad experiment, train/eval experiment, saving + errors) · concept mapping · same model three views · framework choice (+ TF Tensor Explorer, GradientTape Lab, PyTorch Tensor Explorer) |
| 22 | Framework Visual Gallery | `foundations.gallery` | ✅ | orientation (notebook/Colab cells) · Keras/TF gallery (tensor, summary, fit log, evaluate/predict, History plot, TensorBoard) · PyTorch gallery (tensor, print(model), console log, CPU vs GPU) · error gallery (shape, dtype/device, warning vs error) · before/during/after — all real outputs in labelled "Educational recreation" frames with the six reading questions |
| 23 | Ready for Deep Learning | `foundations.ready` | ✅ | concept map + framework map · prerequisite check + 20-question diagnostic · six auto-checked challenges · ready status with return list |

## Official Course — Weeks 01–15 (spec §29) — all mandatory

| Week | Title | Route | Status |
|---|---|---|---|
| 01 | مقدمة في التعلم العميق | `course.w01` | ✅ overview + core concepts + applications, pre/post-test |
| 02 | أساسيات التعلم الآلي | `course.w02` | ✅ overview + pre-test · model types (regression/classification with sklearn) · evaluation: split/metrics/baselines/generalization · ML → DL experiment + post-test |
| 03 | Keras و TensorFlow مع تمهيد PyTorch | `course.w03` | ✅ overview · first network (tensor review, 11 questions, full Keras file) · under fit(): GradientTape/tf.data/devices · PyTorch equivalent mini-lab + five errors + post-test |
| 04 | الشبكات العصبية الأمامية | `course.w04` | ✅ overview · forward flow (animation, NumPy = Keras) · depth/width sweep (7 architectures + interactive) + post-test |
| 05 | خوارزميات التحسين | `course.w05` | ✅ overview · GD/η/SGD (4 learning rates on Keras) · momentum/RMSprop/Adam race · loss surface, curve reading, LR schedules + post-test |
| 06 | دوال التنشيط | `course.w06` | ✅ overview · functions & derivatives (vanishing over 8 layers, dead ReLU) · choosing activations (output/hidden rules, live comparison) + post-test |
| 07 | تطبيق عملي عام | `course.w07` | ✅ overview · build pipeline (house prices: problem → baselines → MLP) · evaluate/diagnose/interpret/error analysis/report + post-test |
| 08 | CNN: المفاهيم الأساسية | `course.w08` | ✅ overview · image as tensor & convolution (+ Convolution Lab) · padding/stride/pooling (+ lab) · layers, architectures, shape calculation (+ CNN Shape Calculator) + post-test |
| 09 | تطبيقات CNN | `course.w09` | ✅ overview · preprocessing, CNN build/train/evaluate (real CNN on synthetic images) · overfitting & augmentation experiment · feature maps, error analysis, shape debugging + post-test |
| 10 | RNN | `course.w10` | ✅ overview · sequences & hidden state (+ RNN Unrolling Lab) · BPTT and vanishing/exploding (interactive) · time-series application (SimpleRNN vs naive on two series) + post-test |
| 11 | LSTM | `course.w11` | ✅ overview · data prep: multivariate windows, multi-step targets, padding/Masking · cell state & gates (+ LSTM Gates Lab) · train/evaluate/diagnose vs RNN and naive + post-test |
| 12 | GRU | `course.w12` | ✅ overview · update/reset gates (hand GRU == Keras GRU) (+ GRU Gates Lab) · RNN vs LSTM vs GRU with seeds · applications map + post-test |
| 13 | GPU و Google Colab | `course.w13` | ✅ overview · CPU vs GPU, parallelism, VRAM & batch size (+ GPU/Batch Memory Lab) · Colab runtime, GPU detection, nvidia-smi, OOM diagnostics, performance + post-test |
| 14 | مشروع تطبيقي | `course.w14` | ✅ overview · project guide (14 stages with checklist, project type by data) · notebook template (tabular/images/sequences) + post-test |
| 15 | عرض ومناقشة المشاريع | `course.w15` | ✅ overview · presentation & discussion (10 slides, ten questions, improvement proposals) · final rubric (10 criteria, weights, self-assessment) + course post-test |

## Official axes → weeks (spec §30)

1. أساسيات التعلم الآلي → 01–02
2. الشبكات العصبية الأمامية → 03–04, 07
3. خوارزميات التحسين → 05–06
4. CNN → 08–09
5. RNN → 10
6. LSTM → 11
7. GRU → 12 (then GPU/Colab 13, project 14–15)

## Labs (spec §41)

| Lab | Route | Status |
|---|---|---|
| Dataset Anatomy | `labs.dataset_anatomy` | ✅ |
| Epoch/Batch Simulator | `labs.epoch_batch_simulator` | ✅ (detailed stepper + autoplay animation player) |
| Data Split · Data Leakage · Scaling | `labs.data_split_lab` · `labs.data_leakage_lab` · `labs.scaling_lab` | ✅ |
| Tensor Shape Explorer · Vector/Matrix | `labs.tensor_shape_explorer` · `labs.vector_matrix` | ✅ |
| Derivative · Gradient · Chain Rule | `labs.derivative_lab` · `labs.gradient_lab` · `labs.chain_rule_lab` | ✅ |
| Neuron · Network Builder · Parameter Counter | `labs.neuron_lab` · `labs.network_builder` · `labs.parameter_counter` | ✅ |
| Activation · Loss · Gradient Descent · Optimizer Race · Learning Rate | `labs.activation_lab` · `labs.loss_lab` · `labs.gradient_descent_lab` · `labs.optimizer_race` · `labs.learning_rate_lab` | ✅ |
| Training Loop Simulator · Overfitting · Regularization · Dropout · Curves Diagnostic | `labs.training_loop_simulator` · `labs.overfitting_lab` · `labs.regularization_lab` · `labs.dropout_lab` · `labs.curves_diagnostic_lab` | ✅ |
| Confusion Matrix · Threshold | `labs.confusion_matrix_lab` · `labs.threshold_lab` | ✅ |
| TF Tensor Explorer · GradientTape Lab · PyTorch Tensor Explorer | `labs.tf_tensor_explorer` · `labs.gradient_tape_lab` · `labs.torch_tensor_explorer` | ✅ |
| CNN Convolution · Padding/Stride · CNN Shape Calculator | `labs.cnn_convolution_lab` · `labs.padding_stride_lab` · `labs.cnn_shape_calculator` | ✅ |
| RNN Unrolling · LSTM Gates · GRU Gates | `labs.rnn_unrolling_lab` · `labs.lstm_gates_lab` · `labs.gru_gates_lab` | ✅ |
| GPU/Batch Memory Concept | `labs.gpu_batch_memory_lab` | ✅ |
