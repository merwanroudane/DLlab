# Diagnostics Standards (spec §32–§40, §80)

Problems are explained **inside the lesson where they can occur**, not only in a separate center.
Example — Learning Rate: what is it → math → animation → too small → too large → oscillation →
divergence → NaN → diagnosis → fix.

## The 18-point problem framework (spec §33)

1. Problem name
2. Description
3. Symptoms
4. What the researcher sees (exact log lines, curves, error text)
5. Possible causes
6. Root causes
7. Diagnosis workflow
8. Evidence (what confirms it)
9. Interactive simulation
10. Mathematical explanation
11. Code experiment
12. Wrong vs correct
13. Fixes
14. Trade-offs
15. Common misdiagnosis
16. Related problems
17. Checklist
18. Diagnostic challenge

Component: `components/diagnostics.py` (Phase 2) renders a `Problem` dataclass with these fields
and reuses `good_vs_bad`, `equation`, `code_lab` and the animation player.

## Debugging order (spec §39) — teach and enforce

data → dtype/shape → labels → split → preprocessing → leakage → tiny-subset overfit test → baseline
→ loss → gradients → learning rate → optimizer → architecture → regularization.
Never change ten hyperparameters at once.

## Problem catalogues to cover

- **Data (spec §34):** missing values, wrong/noisy labels, outliers, duplicates, wrong dtype,
  wrong scaling/normalization/standardization/encoding, small dataset, non-representative data,
  class imbalance, train/test mismatch, data/target/preprocessing leakage, distribution shift,
  sequence padding issues, image preprocessing issues.
- **Shape (spec §35):** shape, rank, dimension, batch/feature/output/target dimension; real errors
  like `ValueError: Shapes (32, 1) and (32,) are incompatible` → explain, expected vs given, diagram,
  wrong code, correct code, prevention.
- **Training (spec §36):** loss not decreasing / stuck / oscillating / diverging / NaN, accuracy
  stuck, slow learning, unstable validation, not learning, vanishing/exploding gradients, dead ReLU,
  saturated sigmoid, bad init, bad LR, bad batch size, wrong optimizer, too many/few epochs, wrong
  loss, wrong output activation.
- **Generalization (spec §37):** overfitting, underfitting, high bias, high variance, memorization,
  poor generalization, train/validation gap — curves + animation + diagnosis + fixes.

## Loss Curve Diagnostic Lab (spec §38)

Cases: healthy · overfitting · underfitting · LR too low · LR too high · divergence · noisy ·
plateau. Challenge mode shows an unlabeled curve, asks the learner, then `Reveal Diagnosis`.

## Good vs bad pairs to repeat (spec §40)

correct split vs leakage · scaled vs unscaled · good LR vs bad LR · healthy curves vs overfitting ·
correct output/loss pairing vs wrong · correct tensor shape vs wrong.

## Quality gate (spec §80)

Symptoms? Root causes? Diagnosis workflow? Evidence? Fix? Before/after? Similar problems /
confusions? Checklist?
