# Animation Standards (spec §19, §78)

Animations explain a **dynamic** concept; they are never decoration.

## Every animation must answer

- What am I looking at?
- What changed?
- Why did it change?
- Old value → new value
- Which equation was applied?
- What is the conclusion?

## Controls (when the animation has more than one step)

Play · Pause · Previous · Next · Restart · speed 0.5× / 1× / 2× · step counter · current action ·
synchronized explanation (text that changes with the step).

## Two implementations

1. **Native stepper (current, Phase 1):** Streamlit buttons + slider drive a step index; each step
   renders values, equations, a highlighted pipeline stage, and charts. Used by the Epoch/Batch
   Simulator (`labs/epoch_batch_simulator.py`) and `CodeLab.step_run`. Fully functional and testable
   headlessly. Lacks autoplay.
2. **JS step player (`components/animation_player.py`, implemented):** a `st.components.v2`
   inline component. Python builds `Frame(html, caption_ar, action, values, equation, highlight)`
   objects; the player provides Play/Pause/Prev/Next/Restart/0.5×/1×/2×, a step counter, an
   optional pipeline strip with the highlighted stage, a synchronized RTL caption, an equation line
   and a before/after values table — all in the browser without reruns. It reports the current
   step back via `setStateValue("step")` only on pause / manual step / end, so Python can render
   extra detail (`st.latex`, tables) for the frame the learner stopped on. Honours
   `prefers-reduced-motion` (autoplay off, stepping works). First use: Epoch/Batch Simulator
   section 3 (60 frames = 10 steps × 6 stages). Next uses: backpropagation flow, CNN kernel
   sliding, RNN unrolling, LSTM/GRU gates, `model.fit()` visualizer, PyTorch training loop.

   ```python
   from components.animation_player import Frame, animation_player, caption
   step = animation_player("key", [Frame(svg, caption("…`w`…"), action="Forward", highlight=1)],
                           title_ar="…", stages=["Batch", "Forward", …], interval_ms=1000)
   ```

## Failure animations (spec §19, §33)

Where useful, the same player shows the failure case next to the healthy case: learning rate too
high (oscillation / divergence / NaN), vanishing gradient across layers, dead ReLU, exploding loss.

## Accessibility (spec §66)

Pause is always available; no flashing; meaning is never carried by colour alone (icon + label);
`@media (prefers-reduced-motion: reduce)` disables transitions in `base.css`.

## Quality gate (spec §78)

Does it explain the concept? Can it pause? Restart? Is each step clear? Are values visible? Is the
explanation synchronized? Is there a failure animation when useful?
