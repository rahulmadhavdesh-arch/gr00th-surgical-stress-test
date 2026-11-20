# Deviations and clarifications relative to the preregistration

The thresholds in `preregistration.md` were not changed. This file records how open points were resolved, and which design changes were made before the final run. The only edit to the preregistration after locking is the file name in its last line. Entries marked "superseded" were replaced by later entries.

## Version history in brief

- v1: first implementation, written from the public GR00T-H repository and dataset layout.
- v2: design changes made after an internal code review, before any model run (V2 items below).
- v3: QUICK pilot flag, dataset discovery by walking the Hugging Face tree, mock test hooks (V3 items).
- v4: a small pilot (18 frames) was run. Its summary numbers were seen before the v5 changes below were made.
- v5: diagnostics, extra exploratory metrics and robustness changes (V5 items). The final results in `results/` come from v5 with the full sample (40 episodes x 10 frames).

## v1 decisions

### Model and precision
1. `Gr00tPolicy` hard-codes `model.to(bfloat16)` and casts inputs to bf16. A T4 has no bf16 hardware, so the notebook subclasses it (`DTypePolicy`) and runs fp16. Weights are loaded once (bf16 checkpoint) and then cast. bf16 to fp16 is exact inside the fp16 range.
2. The preregistration says "ViT fp32 as in repo". The repository casts the whole model to one dtype, so there is no separate fp32 ViT at inference. Everything runs in fp16. The fp16 vs fp32 check covers the whole model.
3. The fp16 check rule: on 20 frames spread evenly over the 400, `max_i ||a16_i - a32_i|| / mean_i E_clean_i < 0.02`, using the 8-step translation metric defined below. A NaN also fails the check. If fp32 does not fit a 15 GB T4, bfloat16 is the reference and fallback (logged in `fp_check.json`). On failure, the main run uses the reference dtype on 5 frames per episode, as the preregistration allows.
4. Attention is sdpa because flash-attn is not installed (FlashAttention-2 does not run on sm75). The notebook logs the implementation seen in the model configs.

### Data
5. Datasets are downloaded with `hf_hub_download` on exact files. `snapshot_download(allow_patterns)` lists the whole repository tree first, which is very large.
6. "Prefer JHU" is resolved as: the first of the JHU dVRK paths used in the GR00T-H training configuration that exists and has at least 40 episodes with at least 60 frames. NephFat (standard 3-view layout) is tried first. Fallbacks follow the preregistered order (other dVRK, then CMR Versius). The choice is printed and stored in `selection.json`.
7. Episodes: 40 evenly spaced by index. If a download fails, a spare episode replaces it. Timesteps: 10 evenly spaced in `[0, L-9]` so that 8 ground-truth steps exist.
8. The secondary embodiment ("if time allows") is not run.
9. The checkpoint's action horizon is 50 for the dVRK configurations. The preregistered "8-step chunk" is implemented as the first 8 of 50 steps.
10. The policy returns decoded absolute poses (xyz plus quaternion) and an absolute jaw angle, so translation is the first 3 numbers of each `*_pose` key. Units are metres on dVRK. The code multiplies by 1000 when the median absolute ground-truth xyz is under 5, and otherwise assumes millimetres (stored in `fp_check.json` as `unit_scale`). State is passed uncorrupted. GR00T-H trains with state dropout 1.0, so the model effectively sees vision and language only.
11. (superseded by V2-6) All camera views received the same corruption.

### Metrics
12. D for a chunk is the mean over the steps of the per-step Euclidean distance over both arms (6 numbers per step).
13. R = D / E_clean is the ratio of means (mean D over mean E_clean, also inside each bootstrap resample). Per-frame ratios become unstable when E_clean is near zero, and the preregistration does not specify the averaging.
14. Noise floor: mean D between the clean seed-1 and clean seed-2 predictions of the same frame. "2x noise floor" uses the point estimate of that mean.
15. FRAGILE: lower 95% CI bound of mean D at s=3 above 2x floor, and R at s=3 above 0.5. ROBUST: mean D at s=5 below 2x floor. A third label, INTERMEDIATE, is printed when neither holds. The preregistration defines only the first two.
16. SURGICAL-SPECIFIC is evaluated for SMOKE only, because the matched controls exist only for SMOKE: paired per-frame difference of D (SMOKE minus GAUSS, SMOKE minus BRIGHTNESS), episode-cluster bootstrap, both lower CI bounds above 0. The headline uses s=3, and the severities where it holds are listed as well.
17. (superseded by V2-13) Gripper flip threshold from data percentiles.
18. Ground-truth error under corruption uses the same distance as D, against dataset action rows t to t+7.
19. (superseded by V2-5 for the matching resolution) GAUSS sigma and BRIGHTNESS offset are bisected per frame and per view so that the MSE to the clean view equals SMOKE's at the same severity. BRIGHTNESS is an additive shift with clipping.

### Corruption parameters
20. The preregistration gives names and OCCLUSION areas only. Parameters are fixed in `gr00th_stress.py` (SMOKE beta 0.3/0.6/1.0/1.5/2.2, blood coverage 6/12/20/32/45 percent, fog and glare tables, occlusion 5/10/20/30/40 percent area). They were set by looking at a synthetic frame and a real tile, never at model output. Random fields use the frame seed, so the severities of one frame share the same field.

## v2 changes (before any model run)

### Code fixes
- V2-1. `corrupt_views` takes the real `(1,T,H,W,3)` observation arrays and returns `(1,T,H,W,3)`.
- V2-2. `infer` re-raises CUDA OOM when called with `raise_oom=True`. A reference run containing any NaN counts as a failed reference: `model_check` falls back to bf16, and aborts if that is NaN too. The main loop aborts if frame 0 has NaN clean predictions.
- V2-3. No `nansum`. NaN propagates through D, E and flips. Analysis counts NaN (frame, condition) pairs per condition, prints the count in the result block, and excludes those pairs from means and bootstraps.
- V2-4. `LeRobotEpisodeLoader(..., require_stats=False)`. Each backend failure logs `repr(e)`.
- V2-5. GAUSS and BRIGHTNESS are matched to SMOKE by MSE measured after `cv2.resize(..., (414,236), INTER_AREA)` (the model input size), on the full-resolution corrupted image. Native and model-resolution PSNR are both logged.

### Design decisions
- V2-6. The primary condition corrupts the endoscope view only (the first view whose key contains "endo"). Wrist cameras stay clean. A secondary condition, reported separately, corrupts all views at severities 3 and 5 for all 7 kinds (conditions named `KIND|s|all`).
- V2-7. Corruption realism rework, parameters set by eye on a real Open-H tile and a synthetic frame. BLOOD is a gravity-biased irregular pool with a darker core, a wet sheen and thin drips, masked off low-saturation (tool-like) pixels. GLARE is many small specular spots gated onto bright pixels, plus mild bloom. OCCLUSION is a shaded-metal instrument shaft crossing the view, with width solved so the mask area equals 5/10/20/30/40 percent. SMOKE keeps I = J*t + A*(1-t) with a grey-white A and adds domain-warped plume structure.
- V2-8. Windows are the first 8 and all 50 predicted steps (D8, D50). R = D / E_clean for the same window, ratio of means. The FRAGILE and ROBUST thresholds are unchanged and applied to both windows. The 8-step window is the headline.
- V2-9. Added plots of drift against per-frame matched PSNR as well as against severity. The noise floor stays, and clean-vs-GT error is printed beside it for scale.
- V2-10. The BRIGHTNESS control stays. Brightness and contrast jitter are part of GR00T-H's training augmentation (0.12 and 0.15), so low drift under BRIGHTNESS is expected and is not evidence about smoke physics.
- V2-11. Data access uses `loader.get_sparse_episode(pos, frame_indices, video_indices=[t], allow_padding=True)` per sampled timestep. Loader indexing is by list position in `episodes.jsonl`, not by `episode_index`; the data stage stores both. Backend order is torchcodec, decord, opencv.
- V2-12. The hero frame is chosen after the run as the median frame by mean percentile rank of D8 at s=3 across the 5 artifacts, and each panel prints that frame's percentile for its artifact. Every endoscope frame is saved as JPEG so the choice can be made.
- V2-13. Gripper flip is a sign change of the model's normalized gripper command relative to 0, clean vs corrupted, same seed. Reported at step 0 and as any step within the first 8.
- V2-14. The printed SURGICAL-SPECIFIC line reads "smoke exceeds Gaussian noise and brightness at matched MSE (model-input resolution)".

### Logged and not done
- V2-15. A corrupted-vs-clean comparison at different seeds was not run. ROBUST means "below sampling spread under shared noise", and the result block says so.
- V2-16. Extra controls (uniform haze, blur) were not added. The control claim is limited to Gaussian noise and brightness.
- V2-17. The frames are training data (NephFat is in the GR00T-H mix). The notebook prints a line to that effect, and the write-up states it.

## v3 changes
- V3-1. The notebook no longer prints a `hero_fid` key that did not exist. It prints the hero rank.
- V3-2. `stage_visual` takes the display frame as the 5th sampled frame of the hero episode. This matters only for QUICK runs. The frame is a visual check and is not used in any analysis.
- V3-3. The corruption self-check reports model-resolution PSNR error for the controls and the real occlusion shaft mask area. Print only.
- V3-4. `MOCK=1` branches (fake HF API, mock loader, mock policy) allow CPU testing. They are inert unless `MOCK=1`.
- A QUICK pilot flag (6 episodes x 3 frames) was added. Pilot output is a code and timing check, not evidence. The shipped default is `QUICK = False`.
- A first pilot failed at the data stage because hard-coded dataset paths had no `meta/info.json`. Discovery now walks the Hugging Face tree to find real LeRobot roots and keeps the preregistered candidate order by keyword.

## v5 changes (diagnostics and robustness)
- V5-1. The GR00T-H repository was audited for action decoding. `decode_action` returns absolute xyz (reference state xyz plus relative output, base frame, quaternion xyzw for `jhu_imerse_dvrk`) in the units of the dataset action column, which is the same quantity as the ground-truth action rows. The ground-truth rows, arm and key order, and frame indexing match. No alignment, frame, unit or denormalization bug was found.
- V5-2. The earlier "GT displacement" scale is the distance of ground-truth steps from ground-truth step 0. It is not a valid zero-motion baseline for the model, which predicts relative to the current state. v5 reports a hold-state baseline and an oracle baseline separately.
- V5-3. GR00T-H is trained with state dropout 1.0 for all embodiments, so the policy is vision-only by design. The state-zeroed ablation is expected to match clean at equal seed and serves as a check.
- V5-4. New stage `diag` (60-frame subset): per-frame hold-state baseline, clean error (single seed and mean of 4 seeds), the standalone-script metric (MSE and MAE over the concatenated action vector, first 8 steps), endoscope-blank, all-cameras-blank and state-zeroed ablations (mean-gray images, state zeroed at the model input only), and per-axis correlation and regression slope of predicted against ground-truth displacement. Diagnostics are not part of the preregistered analysis.
- V5-5. Notebook cleaned for publication. No change to computation.
- V5-6. Synthetic selftest figures were removed from the repository. The selftest writes to a temporary directory, stamps every figure "SYNTHETIC SELFTEST", and downloads nothing.
- V5-7. SANITY is diagnostic only and never skips the sweep. It compares the 4-seed ensemble clean error with the hold-state error at 50 steps. NaN is reported as N/A. The ratio of blank-endoscope drift to noise floor is printed in the DIAG block and the result block.
- V5-8. Locked verdict rules are unchanged. New EXPLORATORY metrics, labelled as such in the result block and `summary.json`: (a) ensemble-mean drift (mean over seeds {s, s+1, s+2, s+3} for clean and corrupted; the five artifacts at severity 3, primary condition; floor from a 2 vs 2 split of the clean seeds); (b) vision-use fraction, artifact drift divided by blank-endoscope drift at the same seed (ratio of means, episode bootstrap CI). The full run makes 69 inferences per frame.
- V5-9. The result block reports the hold-state baseline error instead of the oracle ground-truth displacement. The oracle value stays in the diagnostics.
- V5-10. The checkpoint's state-dropout configuration (`action_head.dropout_prob_by_embodiment`) is logged at model load, and the state-zeroed drift is compared with it.
- V5-11. The shipped default is `QUICK = False` (40 episodes x 10 frames). Stage order: data, model_check, diag, visual, main, video, analysis, figures. QUICK runs write to `outputs_quick`. Diagnostics run on a 60-frame subset. Resume in `diag` and `main` is keyed by (config hash, selection hash, frame id), and rows from another key are moved aside.
- V5-12. Robustness: OOM retry in `infer_full` with a NaN fallback, a `clean_rep` condition (same seed twice, expected drift 0), a check that corrupted and blank views differ from the clean view, a log of the hash of the processor's pixel input, GR00T-H pinned to a commit, and the model image size read from the processor.
- V5-13. The preregistration text is unchanged from the locked version except the file name of this document. The statement "all camera views identically" is superseded by V2-6, and v1 items 17 and 19 by V2-13 and V2-5.
