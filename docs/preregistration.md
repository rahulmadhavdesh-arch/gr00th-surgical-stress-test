# Surgical Stress Test of GR00T-H N1.7 (locked before any run)
Date 2026-11-16. Rahul Madhav Deshpande.

Question: how much do NVIDIA GR00T-H's predicted surgical actions change when the endoscope view is degraded by realistic OR artifacts?

Model: nvidia/GR00T-H-N1.7, PyTorch, fp16 on Kaggle T4 (sdpa, ViT fp32 as in repo). fp16 vs fp32 check on 20 frames before the main run: max relative action diff must be < 2% of clean-vs-GT error, else run fp32 on fewer frames (logged).
Data: Open-H-Embodiment, primary embodiment = one dVRK set (prefer JHU; fallback any dVRK, then CMR Versius), 40 episodes, 10 evenly spaced timesteps each = 400 frames. Secondary embodiment if time allows (reported separately).
Caveat stated up front: these episodes are likely in GR00T-H's training data. Clean accuracy is optimistic; the measured quantity is sensitivity of the output to the corruption, which is still meaningful.

Corruptions (5 severities each, s=1..5), applied to all camera views identically:
- SMOKE: atmospheric scattering I = J*t + A*(1-t), t from smooth fractal density field, A near-white.
- BLOOD: dark-red translucent pooled regions with darkening, placed in lower/central field.
- FOG: lens condensation: radial blur + contrast loss + haze veil.
- GLARE: saturated specular blobs plus bloom.
- OCCLUSION: opaque tissue-coloured patch covering 5,10,20,30,40% of frame.
- Generic controls: GAUSS noise and BRIGHTNESS shift matched in mean pixel change (PSNR) to SMOKE at each severity.

Stochasticity: flow-matching head is random. Each frame uses a fixed seed across all conditions. Noise floor = drift between two different seeds on the clean frame.

Metrics (translation part of first predicted step and full 8-step chunk, both arms):
- Drift D = || a_corrupt - a_clean || (same seed), reported in mm if units allow, else normalized.
- Relative drift R = D / E_clean, with E_clean = || a_clean - a_GT ||.
- Gripper flip rate (sign change of gripper command).
- GT error under corruption.
Stats: per-frame values, mean and 95% bootstrap CI over episodes (cluster = episode, 2000 resamples).

Pre-registered reading (descriptive, no kill on outcome):
- FRAGILE to artifact X if at s=3: mean D > 2x noise floor (CI excluding) AND R > 0.5.
- ROBUST to X if at s=5: mean D < 2x noise floor.
- SURGICAL-SPECIFIC if SMOKE drift at matched PSNR exceeds GAUSS and BRIGHTNESS drift (CI excluding).
Either outcome is reportable. Only technical failure (model won't load, data unreachable) stops the project.
No threshold changes after results. Changes go in docs/deviations.md.
