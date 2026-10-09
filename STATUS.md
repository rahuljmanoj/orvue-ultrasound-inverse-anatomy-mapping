# STATUS.md

State of the inverse-mapping project. Updated at the end of every session (plan: `PLAN_inverse_mapping.md`).

| | |
|---|---|
| Repository | rahuljmanoj/orvue-ultrasound-inverse-anatomy-mapping, `main` after the S7 + clinical-version merge (branch `s7`, 2026-10-09) |
| Simulator code | copied in from rahuljmanoj/orvue-ultrasound-simulator `e789389`; frozen; see `UPSTREAM.md` |
| Environment | conda env `orvue-robot`, Python 3.11.16 (`C:/Users/rahul/miniconda3/envs/orvue-robot/python.exe`); import check passed (Prep P2) |
| Last update | 2026-10-09, after S7 and the clinical version |

## Sessions

| Session | State | Evidence / notes |
|---|---|---|
| Prep P1 data folders | Done | `paths.SWEEPS_DIR` (`output/sweeps`), `paths.RESULTS_DIR` (`output/results`); created on demand |
| Prep P2 environment | Done | numpy 2.4.6, scipy 1.17.1, cv2 5.0.0 (`cv2.aruco` with `ArucoDetector`), scikit-image 0.26.0, matplotlib 3.11.2, reportlab 5.0.1, pytest 9.1.1, pyrealsense2 2.58.4 (optional); nothing missing. PyVista / VTK blocked at Prep; pyvista 0.49.0 / vtk 9.7.1 import and render since S3 (optional extra) |
| Prep P3 probe | Done | `mapping/probe.py`: `PROBE` (every field stated), `make_simulator` (persistence 0, not overridable), `probe_metadata`; `tests/mapping/test_probe.py` (4 tests; the previous-pose test fails with persistence 0.3, checked) |
| Prep P4 CLAUDE.md | Done | "Inverse mapping" section; README "Every file" and layout updated |
| Prep P5 baseline | Done | Tests below; frames and timings below; `tracking.markers` and `tracking.tracker` import without a camera (0.09 s) |
| S0 skeleton, config, sweep format | Done | `mapping/config.py`, `mapping/sweep_io.py` implemented; `poses`, `acquisition`, `recon`, `render`, `evaluate`, `experiments`, `errors` are docstring placeholders; `tests/mapping/test_sweep_io.py` (11 tests) + 2 in `test_probe.py`. Details below |
| S1 scripted sweep, acquisition, playback | Done | `mapping/poses.py`, `mapping/acquisition.py`, app `mapping/run_scripted.py` (menu 4 `scripted`); `tests/mapping/test_poses_acquisition.py` (16), `test_run_scripted.py` (3). Details below |
| S2 reconstruction (oracle labels) | Done | `mapping/recon.py`, script `mapping/reconstruct_sweep.py` (menu 5 `recon`); `tests/mapping/test_recon.py` (9). Details below |
| S3 live view of the reconstruction | Done | `mapping/render.py`, `mapping/live3d.py` (optional PyVista), `run_scripted.py` (live reconstruction, slices, coverage, 3D outputs); `tests/mapping/test_render.py` (8). Details below |
| S4 evaluation and "complete" | Done | `mapping/evaluate.py`, script `mapping/evaluate_sweep.py` (menu 6 `evaluate`), key c in `run_scripted.py`; `tests/mapping/test_evaluate.py` (10). Details below |
| Default spacing 0.25 mm | Done | `SweepConfig` / `--spacing` default 0.5 -> 0.25 mm; tests that check the default updated (401 frames per lane), tests that only need a sweep pinned to 0.5 mm (speed) |
| S5 sweep-strategy experiments | Done | `mapping/experiments.py`, script `mapping/run_experiments.py` (menu 7 `experiments`); `tests/mapping/test_experiments.py` (5); full grid 156 runs in 16 min. Details below |
| S6 mouse sweeps | Done | `poses.MousePose`, app `mapping/run_mouse.py` (menu 5 `mouse`); `tests/mapping/test_mouse.py` (14). Details below |
| Clinical version | Done (awaiting review) | `clinical/` (window with B-MODE and INVERSE MAPPING tabs, AR overlay), `poses.ScriptedPose`, one window by default (case drop-down, calibration inside), developer menu `dev`; `tests/clinical/test_clinical.py` (11). Details below |
| S7 camera-tracked probe, error study | Done | `poses.TrackedPose`, `errors.py`, app `run_tracked.py` (menu 6), error study (menu 10); `tests/mapping/test_errors.py` (7), `test_tracked.py` (8), 1 in `test_experiments.py`. Details below |

## Prep frames and timing (2026-10-08)

Normal case, `make_simulator("normal")`, frames 501 x 301 (image uint8, labels int8). PNGs (B-mode and labels coloured
with `COL_TAB`) in `output/prep_check/` (not in git); B-mode and labels agree by eye.

| Pose (x, y, yaw) | Labels in the frame (% of pixels) |
|---|---|
| (50, 50, 0) | fat 68.3, bile 9.8, vein blood 8.1, liver 8.0, GB wall 3.2, vein wall 0.9, duct wall 0.8, art. blood 0.6, art. wall 0.3 |
| (25, 70, 90) | bile 46.9, fat 45.3, GB wall 7.8, art. wall 0.1 |
| (60, 30, -25) | liver 70.4, vein blood 12.4, fat 12.0, bile 1.8, duct wall 1.1, vein wall 0.9, art. wall 0.7, art. blood 0.6, GB wall 0.2 |

Mean over 20 frames (the three poses in turn, after one warm-up): `render()` **78.4 ms** (sd 2.5),
`labels_image()` **13.4 ms** (sd 1.9). An oracle-only sweep of 1000 frames takes ~13 s of label computation.

## S0 sweep format (2026-10-08)

Test lane: normal case, x = 50, y 30 -> 54.5 mm at 0.5 mm, yaw 0, 50 frames, noise seeded.

| | Actual | `estimate_size_mb` | Save | Load |
|---|---|---|---|---|
| With images | 4.99 MB | 5.09 MB (+2%) | 239 ms | 39 ms |
| Labels only | 0.078 MB | 0.091 MB (+16%) | 43 ms | 9 ms |

Compressed bytes per frame on 50-frame lanes of 3 cases x 4 lanes: images 96-106 kB in every case; labels
1.4-1.6 kB over the anatomy, 150-200 B near the region edges. The estimate uses 100 kB + 1.5 kB per frame, so a
labels-only estimate is an upper bound (up to ~4x high away from the anatomy, a few hundred kB absolute).
800-frame sweep: ~81 MB with images, ~1.4 MB labels only.

S0 deviations from the prompt:
- `mapping/probe.py` (Prep) extended: `FRAME_SHAPE`, `simulator_settings(**kw)`, and `make_simulator` sets
  `sim.case` / `sim.settings`, so `make_metadata(sim, ...)` records the effective simulator settings (the
  `BModeSimulator` constructor defaults plus overrides, read from its signature; the copied file is unchanged).
- Added beyond the prompt: `make_metadata()`, `paths.UPSTREAM_PATH` (a test checks `SIMULATOR_SOURCE_COMMIT`
  against `UPSTREAM.md`), a `format_version` array in the file, input checks in `FrameRecord` / `Sweep` / configs.
- The simulator commit is a constant (`sweep_io.SIMULATOR_SOURCE_COMMIT`) checked against `UPSTREAM.md` by a
  test, not parsed from `UPSTREAM.md` at run time.
- `SweepConfig` region as `region_x_mm` / `region_y_mm` bounds (0, 100), like `GridConfig`.

## S1 scripted sweep (2026-10-08)

| Yaw | Lanes | Overlap requested / actual | Frames (0.5 mm) | Planned coverage | Simulated time (10 mm/s) |
|---|---|---|---|---|---|
| 0 | 4 (x = 15, 38.3, 61.7, 85) | 20% / 22.2% | 804 (201 per lane) | 100% | 47.0 s |
| 0 + 90 | 4 + 4 | 20% / 22.2% | 1608 | 100% | 96.1 s |
| 45 | 6 (clipped to contact, 30-119 mm long) | 20% / 25.7% | 900 | 95.3% | 59.6 s |
| 0, 45, 90, 135 | 20 | | 3408 | 100% (45 / 135 alone 95.3%) | 232.3 s |

Overlap 40% -> 5 lanes (41.7%), 50% -> 6 lanes (53.3%), 0% -> 4 lanes (22.2%), as in the plan. At 45 deg the
region corners beyond the clipped lane ends stay uncovered (4.7%).

Timing (headless, normal case): with B-mode, 1000 frames in 75.8 s (76 ms/frame), so playback runs at ~0.78x real
time at 10 mm/s; labels only ~4.2x real time (the app paces it at the playback speed). Revealing the anatomy
(`top_view` at 0.25 mm) takes 4.7 s once; drawing the window ~10 ms. Trigger logic over a full 0 + 90 sweep
(19 235 samples) 0.2 s.

S1 deviations from the prompt:
- `Acquirer(sim, sweep_cfg, acq_cfg)` instead of `Acquirer(sim, cfg)`: spacing and angle trigger are in
  `SweepConfig`, `store_images` in `AcquisitionConfig`. With `store_images=False` the capture is
  `labels_image(T)` only (no `render()`), as `AcquisitionConfig` says.
- The first pose of every lane is always captured (after a lift-off there is no previous frame in the lane),
  so a 100 mm lane gives 201 frames, not 200.
- Lane clipping is computed geometrically (face centre on the region, which is exactly what `in_contact`
  tests for these flat poses) for every yaw, not only 45; tests check `in_contact` for every lane end and
  every captured frame.
- Transitions move at the sweep speed and rotate at 45 deg/s (`poses.ROTATION_DEG_S`), drawn grey as lift-off.
- Trigger tolerance 1e-4 mm / 1e-3 deg because `pose_from_xy_yaw` returns float32 poses (rounding ~1e-5 mm).
- Menu: new group INVERSE MAPPING, step 4 `scripted` (runs yaw 0 + 90); the later steps moved from 4-8 to 5-9.
- One window shows the top view and the status panel; the B-mode is a second window. With `--no-images` the
  second window shows the oracle labels in colour.
- Extra: `pose_at(t)`, `lane_at(t)`, `coverage()`, `summary()`; the app's state and drawing are in
  `ScriptedPlayback` (tested without windows). Saved metadata adds `complete`, `planned`, `expected_frames`.

## S2 reconstruction (2026-10-08)

Normal case, oracle labels, exact poses, 0.5 mm voxels (200 x 200 x 100), yaw 0, 20% overlap. "Interior" =
all 26 neighbours have the same ground-truth label.

| Sweep | Frames | Volume observed | Correct (observed) | Correct (interior) | Insert |
|---|---|---|---|---|---|
| Whole region, 0.25 mm | 1604 | 100.0% | 99.23% | 100.000% (3 485 721 voxels) | 10.2 ms/frame |
| Test block x, y 25-75 mm, 0.25 mm | 402 | 25.5% | 98.48% | 100.000% (845 235) | 9.5 ms/frame |
| Whole region, 1.0 mm (every 4th) | 401 | 53.5% | 99.18% | 100.000% | |
| ... + `fill_small_holes(1)` | | 99.5% | 98.81% | 99.998% | fill 0.6 s |

Mismatches are all at tissue boundaries (whole region, 0.25 mm): fat->liver 5615, liver->fat 3867,
bile->GB wall 2602, fat->GB wall 2416, GB wall->bile 2133, GB wall->fat 1767, vein blood->vein wall 1477,
GB wall->liver 1181 voxels; none in the interior. After filling the 1 mm sweep, 53 fat->liver and 12
stone->bile interior voxels are wrong. Slice PNGs checked by eye: reconstruction and ground truth agree, x right,
y / depth down, liver anterior to the GB.

Timing: insert 9-10 ms/frame (target 15 ms) after vectorising with one `np.unique` per frame (first version
with `np.add.at`: 27 ms); elevation splat 1 mm (5 planes) 46 ms/frame; ground truth of the whole grid
(`Anatomy.labels` at 4 M voxel centres) 1.3 s; `fill_small_holes(1)` 0.6 s.

S2 deviations from the prompt:
- The ground-truth test sweeps the block x, y 25-75 mm (402 frames) to keep the test at ~12 s; the
  whole-region figures above come from the same code on a 1604-frame sweep (not in the test suite).
- `ground_truth()` and `interior_mask()` live in `recon.py` (used by the test and the script); S4 builds the
  evaluation on them.
- Votes saturate at 65535 instead of wrapping; hits are uint32. `insert_batch()` added (incremental = batch
  is tested on votes, hits, intensity sums and the result).
- `reconstruct_sweep` also takes `--max-gap`, `--splat`, `--out`, defaults to the newest sweep (menu step 5
  `recon`), prints the accuracy against the ground truth, and draws unobserved voxels hatched (liver is grey).
- Menu: 5 `recon` added; later steps now 6-10 (numbers right-aligned).

## S3 live view (2026-10-08)

Timing (normal case, 0.5 mm voxels, whole region): slice update 5-6 ms from the compounder (only the three
slices are computed; target 50 ms); full `result()` 52 ms; `surface_meshes` 184-200 ms for 171 780 triangles
(bile 96 084, venous blood 46 708, arterial blood 21 452, stone 5 532, lymph node 1 156; target 300 ms;
142 ms without smoothing); `snapshot_3d` 1.3 s; STL export 22 ms; browser page 174 ms, 3.1 MB. Browser page
checked with a headless Edge screenshot (cases reconstruction / reconstruction vs truth / ground truth). In the
app: the reconstruction is updated with every captured frame (~10 ms, B-mode render ~76 ms), slices every 0.5 s,
live 3D every 3 s (~250 ms mesh extraction blocks the loop briefly).

S3 deviations from the prompt:
- Added on request: browser 3D view (key b, `output/viewer3d/recon_<case>.html`, viewer3d template reused
  unchanged; title of the page is the template's) and a live PyVista window (`mapping/live3d.py`, `--live3d` /
  key p, menu step 4 starts with it).
- Key a (revealed anatomy) replaced by v (cycles black box / coverage / revealed). Extra keys: g truth contours
  (slices; truth also in the live window), PgUp / PgDn or [ / ] move the crosshair in depth.
- `LabelCompounder.result_slice()` and `column_hits()` added to `recon.py` so the live views do not need the
  full `result()`.
- Surfaces are clamped to the grid bounds after smoothing (the portal vein is cut at z = 50 mm).
- The snapshot draws bile at 45% opacity (stones visible inside); ground truth at 12%.
- 3D outputs: snapshot `output/results/snapshot_<case>_<time>.png`; STL `output/export/recon_<case>_<time>/
  recon_<group>.stl`.
- Timing test uses the ground-truth volume as a perfect whole-region reconstruction (same size), not a
  rendered 1604-frame sweep.
- pyvista 0.49.0 / vtk 9.7.1 added to `requirements.txt` and as extra `view3d` (already installed).

## S4 evaluation (2026-10-08)

Full scripted sweeps of the normal case (oracle labels only; B-mode not needed for the evaluation), 20% overlap,
0.5 mm spacing, 0.5 mm voxels, through the app's "complete" path (`ScriptedPlayback.complete`, hole filling
max gap 1). Sweeps `output/sweeps/normal_yaw0_ov20_sp0.5_v10_labels_20261008-154537.npz` and
`normal_yaw0-90_..._20261008-154616.npz`; reports `output/results/normal_20261008-154542/` and
`normal_20261008-154621/`.

| Sweep | Frames | Observed (before fill) | Correct | Detected / missed / not covered | Topology GB - CBD |
|---|---|---|---|---|---|
| yaw 0 | 804 | 100.0% (100.0%) | 99.0% | 14 / 0 / 0 | ok (connected) |
| yaw 0 + 90 | 1608 | 100.0% (100.0%) | 98.9% | 14 / 0 / 0 | ok (connected) |

| Class | Dice yaw 0 | Dice yaw 0 + 90 | MSD yaw 0 / 0 + 90 (mm) | HD95 (mm) |
|---|---|---|---|---|
| bile (3) | 0.985 | 0.983 | 0.16 / 0.18 | 0.50 |
| arterial blood (7) | 0.928 | 0.910 | 0.12 / 0.16 | 0.50 |
| venous blood (9) | 0.982 | 0.975 | 0.11 / 0.15 | 0.50 |
| stone (4) | 0.952 | 0.953 | 0.17 / 0.17 | 0.50 |
| lymph node (10) | 0.923 | 0.927 | 0.17 / 0.16 | 0.50 |
| GB wall (2) | 0.896 | 0.894 | 0.18 / 0.18 | 0.50 |
| duct wall (5) | 0.848 | 0.766 | 0.08 / 0.12 | 0.50 |
| artery wall (6) | 0.789 | 0.743 | 0.11 / 0.13 | 0.50 |
| vein wall (8) | 0.736 | 0.645 | 0.13 / 0.18 | 0.50 |

| Structure | Recall yaw 0 | Recall yaw 0 + 90 | Status (both) |
|---|---|---|---|
| gallbladder | 98.5% | 98.6% | detected |
| cystic_duct | 94.5% | 95.0% | detected |
| chd_cbd | 96.4% | 94.6% | detected |
| proper_hepatic_artery | 96.1% | 90.9% | detected |
| left_hepatic_artery | 91.1% | 89.8% | detected |
| right_hepatic_artery | 91.8% | 92.6% | detected |
| cystic_artery | 82.7% | 82.7% | detected |
| portal_vein | 98.3% | 97.1% | detected |
| left_portal_vein | 96.8% | 97.0% | detected |
| cystic_artery_superficial | 77.2% | 75.2% | detected |
| cystic_artery_deep | 74.9% | 75.1% | detected |
| gallstone_1 | 95.4% | 94.7% | detected |
| gallstone_2 | 92.1% | 92.8% | detected |
| calot_lymph_node | 92.1% | 92.5% | detected |

Coverage 100% for every class and structure in both. Timing: sweep + reconstruction 16 s (yaw 0) / 31 s
(0 + 90) with labels only; complete (fill + ground truth + instances + metrics + report with figures) ~7 s;
evaluation alone ~1.3 s; instance masks 0.6 s.

Observation: yaw 0 + 90 is slightly worse than yaw 0 alone for the thin walls (vein wall Dice 0.645 vs 0.736),
not better. Checked: ties are not the cause (0.02% of voxels tied). The frames lie exactly on voxel faces
(y = 0 mod 0.5 mm at 0.5 mm spacing from the region edge), so every voxel is sampled 0.25 mm from its centre,
on its lower y face (yaw 0) and lower x face (yaw 90); walls thinner than a voxel then disagree between the two.
To examine in S5 (spacing 0.25 mm, a half-voxel offset of the lanes, voxel size).

S4 deviations from the prompt / decisions:
- One-voxel shift test (decided with the user): the standard symmetric MSD is ~0.45-0.7 voxel for a one-voxel
  shift (measured 0.29-0.32 mm for the lumen classes at 0.5 mm; surfaces parallel to the shift do not move), so
  the test checks HD95 = 1 voxel (+/- 30%) and MSD 0.4-0.8 voxel; a separate slab test checks MSD ~ full shift.
- Structure recall uses "effective" voxels: the instance's geometric voxels whose true label is its own label
  (a stone inside the GB lumen counts for the stone, not the GB). Statuses: detected / missed / not covered
  (coverage < 20 %) / no voxels. Wall coverage and recall reported alongside.
- Topology "not covered" when either lumen is < 20 % observed.
- Report folder also holds `slices_errors.png` (centre slices, wrong voxels magenta). `snapshot_3d` got
  per-face colouring for the distance overlay; `SliceView` got an error mode (key e toggles it after c).
- Key c also opens the report folder in Explorer; menu step 6 `evaluate` (with `--fill`) added, later steps
  now 7-11.

## S5 sweep-strategy experiments (2026-10-08)

Full grid: spacing 0.25 / 0.5 / 1 / 2 mm x overlap 20 / 40 / 50 % x orientations [0] / [90] / [0+90] / [0+45+90+135]
x cases normal, parallel_cystic_duct, anterior_cystic_artery at 0.5 mm voxels (144 runs) + extras for the default
(0.5 mm, 20 %, [0] and [0+90]): 0.25 mm voxels and the 0.5 mm grid shifted by half a voxel (12 runs). 144 cached
single-orientation oracle sweeps (148 524 frames, 113 MB in `output/cache/experiments/`), 298 716 frame insertions.
Estimate printed first: 93 min serial, ~11 min with 10 workers (below 30 min, so the full grid ran after the
reduced grid); actual 16.0 min (the 0.25 mm voxel runs are slower than modelled). Results:
`output/results/experiments_20261008-160406/` (results.csv, summary.md, 5 plots).

Means over the 3 cases (thin = cystic duct + the 3 cystic artery branches; min = worst structure in any case):

| Spacing | Overlap | Orientations | Frames | Scan (s) | Lumen Dice | Wall Dice | Thin recall | Min recall | Lumen HD95 | Detected |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 20 | 0 | 1604 | 47 | 0.968 | 0.856 | 87.6% | 80.8% | 0.50 | 42/42 |
| 0.5 | 20 | 0 | 804 | 47 | 0.953 | 0.817 | 82.6% | 74.9% | 0.50 | 42/42 |
| 1 | 20 | 0 | 404 | 47 | 0.932 | 0.780 | 66.9% | 50.7% | 0.50 | 42/42 |
| 2 | 20 | 0 | 204 | 47 | 0.862 | 0.663 | 42.1% | 18.9% | 0.96 | 33/42 |
| 0.5 | 40 | 0 | 1005 | 57 | 0.952 | 0.795 | 81.9% | 74.2% | 0.50 | 42/42 |
| 0.5 | 50 | 0 | 1206 | 67 | 0.952 | 0.795 | 82.4% | 75.2% | 0.50 | 42/42 |
| 0.5 | 20 | 90 | 804 | 47 | 0.948 | 0.725 | 83.4% | 74.3% | 0.50 | 42/42 |
| 0.5 | 20 | 0+90 | 1608 | 96 | 0.949 | 0.763 | 82.8% | 75.1% | 0.50 | 42/42 |
| 1 | 20 | 0+90 | 808 | 96 | 0.949 | 0.768 | 80.5% | 70.7% | 0.50 | 42/42 |
| 0.25 | 20 | 0+90 | 3208 | 96 | 0.969 | 0.853 | 88.3% | 80.8% | 0.50 | 42/42 |
| 0.5 | 20 | 0+45+90+135 | 3408 | 232 | 0.967 | 0.842 | 87.4% | 81.7% | 0.50 | 42/42 |
| 0.25 | 20 | 0+45+90+135 | 6796 | 232 | 0.976 | 0.880 | 90.3% | 83.8% | 0.50 | 42/42 |
| 0.5 (grid offset) | 20 | 0 | 804 | 47 | 0.975 | 0.867 | 91.9% | 86.6% | 0.50 | 42/42 |
| 0.5 (grid offset) | 20 | 0+90 | 1608 | 96 | 0.980 | 0.895 | 92.3% | 86.0% | 0.50 | 42/42 |
| 0.5 (0.25 mm voxels) | 20 | 0 | 804 | 47 | 0.959 | 0.841 | 80.4% | 69.4% | 0.28 | 42/42 |

Topology GB - CBD ok in every run. All 48 strategies in `summary.md`.

**Recommendation (rule in `experiments.recommend`): spacing 0.25 mm, 20 % overlap, yaw 0 only, 0.5 mm voxels**:
1604 frames, 47 s at 10 mm/s; lumen Dice 0.968, thin recall 87.6%, worst structure 80.8%, all detected, topology ok.
Rule: cheapest scan (then fewest frames) among strategies that detect every structure with correct topology in all
cases and are within 0.01 of the best lumen Dice (0.976) and 3 points of the best thin recall (90.3%). The best
quality, [0+45+90+135] at 0.25 mm, needs 5x the scan time (232 s) and 4x the frames for +0.008 Dice / +2.7 points.

Findings:
- Spacing matters most for thin structures: thin recall 87.6 / 82.6 / 66.9 / 42.1% at 0.25 / 0.5 / 1 / 2 mm (yaw 0);
  2 mm misses cystic artery branches (33/42 detected). Large lumens change little (gallbladder Dice 0.99 -> 0.97).
- Overlap adds nothing with exact poses (0.953 / 0.952 / 0.952 lumen Dice at 20 / 40 / 50 %) and costs 21-43 %
  more scan time; it is expected to matter once pose errors are added (S7).
- A second orientation (0+90) at the same spacing does not improve the 0.5 mm grid, but four orientations do; at
  coarse spacing, more orientations compensate (0+90 at 1 mm ~ 0 at 0.5 mm with the same 800 frames).
- The S4 voxel-face observation is confirmed: shifting the grid by half a voxel (frames at voxel centres) gives
  lumen Dice 0.975 / thin recall 91.9% at 0.5 mm spacing, better than 0.25 mm spacing on the face-aligned grid.
  Scripted sweeps can use this (grid origin at -voxel / 2); hand-held sweeps have random frame phase, so the
  effect averages out there. 0.25 mm voxels with 0.5 mm spacing help HD95 (0.28 mm) but not recall.
- Scan time is at a fixed 10 mm/s; at a real frame rate f the speed limit is spacing x f (0.25 mm at 20 frames/s:
  5 mm/s, i.e. 94 s for the recommended sweep), relevant for S6.

S5 deviations from the prompt / decisions:
- Holes are filled up to round(spacing / voxel) - 1 voxels before evaluating (sparse sweeps interpolated between
  frames; `observed_raw` in the CSV is before filling), so coarse spacings are judged on what they reconstruct, not
  only on the voxel layers they hit.
- Per-structure Dice / MSD / HD95 are local: the structure's effective voxels against the reconstructed voxels of its
  label within 2 voxels of it, other structures of the same label excluded. Neighbours' boundary errors still count
  (e.g. cystic duct local Dice ~0.65 in the normal case at 94% recall: GB wall voxels next to the duct reconstructed
  as bile). The class-level metrics (S4) are in the CSV as well.
- Extras added: half-voxel grid offset for the default (to test the S4 observation); 0.25 mm voxels for both default
  orientation sets ([0] and [0+90], the plan's default and its second pass).
- The 30-minute check uses the wall-clock estimate with the worker count (the runs are parallel).
- `--resummarize FOLDER` rebuilds summary.md and the plots from results.csv; the recommendation tolerances were set
  to 0.01 Dice / 3 points after the first full run (at 0.005 / 2 points the rule simply picked the best-quality
  strategy); the best-quality strategy is reported alongside.
- Menu step 7 `experiments`; later steps now 8-12. `paths.EXPERIMENTS_CACHE_DIR` added.

## S6 mouse sweeps (2026-10-09)

App `python -m orvue_us_inverse mouse` (menu 5). `MousePose` (poses.py) on wall-clock time; `MouseSession`
(run_mouse.py) holds the state without windows. One pose per loop is offered to the `Acquirer` (default spacing
0.25 mm); every captured frame goes into the reconstruction. Capture rate = 1 / (EMA of the capture-loop period);
before it is measured 1 / 0.12 s with B-mode images, 1 / 0.04 s with labels only. Max speed = spacing x capture
rate (e.g. 0.25 mm x 20 frames/s = 5 mm/s; with B-mode images ~8-10 frames/s -> ~2-2.5 mm/s). Strokes = button
press to release; u removes the last stroke from the sweep and the reconstruction (`LabelCompounder.remove_batch`,
exact inverse of the insertion; tested equal to a fresh reconstruction of the remaining frames). Coverage = columns
with hits (black = none, inferno heat = more hits); holes = unscanned areas enclosed by scanned columns
(`ndimage.binary_fill_holes`), drawn cyan (a colour the heat map never uses; first version red, confusable with
lightly scanned columns); gaps = consecutive frames of a stroke > 2 x spacing apart, red lines; colour key under the
map. c runs the S4 evaluation (`evaluate.complete_report`, shared with the scripted app; report in
`output/results/<case>_mouse_<time>/` incl. the distance-coloured 3D overlay); its hole filling closes 1-voxel gaps
in the 3D volume, not the holes on the map. Single window (second review): sweep (left) | latest B-mode (middle) |
3D reconstruction (right; PyVista off-screen, `live3d.Embedded3D`, drag rotates ~7 ms / step, wheel zooms) with
status, speed meter and keys underneath; no slice window (the slice view is kept internally for the report's error
slices). The 3D panel is rebuilt (~0.13 s) only while the button is up (after each stroke, undo, reset, g or c),
never during a stroke; g shows the true anatomy translucent; after c it shows the evaluated volume. b browser 3D
view, 3 snapshot + STL. The 3D outputs live in the `render.Recon3DOutputs` mixin used by both apps. 3D view presets
(buttons Iso / Top / Axial / Sagittal, key v): isometric x right / y towards the viewer / depth down, top = probe's
view, axial from the feet (radiology convention: patient R on the left), sagittal from the patient's right; box
faces labelled patient R / L, cranial / caudal, anterior (surface) / posterior (a label is shown when its face is
edge-on or obliquely in front, so labels never sit over the anatomy), orientation triad L / Ca / P. Probe angle:
the mouse wheel over the sweep or q / e turn it by 5 deg, 0 / 9 set 0 / 90 deg, any time.

S6 deviations from the prompt / decisions:
- Key conflict: q is both "rotate" and "quit" in the prompt; q / e rotate, Esc (or closing the window) quits. e is
  rotate, so the S4 error toggle is not on a key here; after c the slices show the errors.
- Added after the first review: cyan holes + colour key; browser view (b), snapshot + STL (3), 3D view in the mouse
  app. Second review: one window with sweep, B-mode and 3D (the separate B-mode, slice and PyVista windows, p and
  --live3d removed from the mouse app). Third review: the angle lock ("locked while recording unless l", from the
  prompt) removed with the l key: only the user turns the probe, so a lock protected against nothing; turning is
  allowed any time (frames are captured every 1 deg of turning) and the mouse wheel over the sweep turns the probe.
  3D view presets and anatomical face labels added (also in the scripted app's live window). Fourth review: B-mode
  on / off in the window (caption button, key i; --no-images = start off): off computes only labels_image (~13 ms
  vs ~78 ms per frame), the panel then shows the labels with a tissue key, the capture-rate measurement restarts
  at each switch. Because a sweep can now mix frames with and without B-mode, the sweep format went to version 2
  (`images` + `image_frames` for the frames that have one; format 1 still loads) and the recon intensity volume
  uses only the frames with an image (tested).
- The Python OpenCV bindings have no `getMouseWheelDelta`; `run_mouse.wheel_delta` decodes the upper 16 bits.
- The live 3D camera (both apps) now looks so that x runs right, y towards the viewer and depth down (the first
  version copied the anatomy viewer's isometric preset, which shows x running to the left).
- Coverage holes are enclosed unscanned areas (not every unscanned column); covered % is over the 100 x 100 mm
  region; moving faster than the limit leaves unscanned rows between frames, which show as holes and as red gaps.
- Lane guides: bands of the orientation in --yaw nearest the probe's yaw (mod 180), current lane = nearest band
  containing the probe centre.
- `Acquirer.truncate`, `LabelCompounder.remove_batch` and `evaluate.complete_report` added; `ScriptedPlayback.complete`
  now uses `complete_report`.
- Menu: 5 `mouse`; later steps now 6-13.
- The interactive window was checked headless (screenshot with synthetic strokes); mouse handling in a live window by
  eye only.

How to scan the region well by hand (also in README):
1. Snap the yaw first (0), start at one edge of a lane band, press and hold, and move slowly along the band to the
   far edge; release. Keep the speed meter green (below 80 % of the maximum).
2. Do the next band the same way in the opposite direction (serpentine); follow the highlighted band so lanes
   overlap by the guide's 20 %.
3. After a pass, look at the coverage map: cyan holes, red gap marks and black patches show what was missed; rescan
   just those spots (short strokes), or u to redo a bad stroke.
4. Optionally a second pass at 90 (key 9) over the hilum; it adds little with exact poses (S5) but helps where the
   first pass had gaps.
5. The 3D panel shows the result after each stroke; press c: the report tells you which structures were not
   covered or missed.

## S7 camera-tracked probe and tracking-error study (2026-10-09)

Built: `poses.TrackedPose` (filtered pose from `ProbeTracker.get_state()`, None while tracking is invalid);
`errors.py` (`PoseErrorModel`: jitter, bias, latency, scale; `LatencyBuffer`; `filter_lag`); app `run_tracked.py`
(menu 6 `tracked`): the mouse-sweep window with the pose from the camera (hold SPACE to record), m switches to the
mouse and back (the app starts with the mouse when the camera cannot be opened), a tracking status line (TRACKING
OK / board held / LOST, reference markers of 4, probe ID 0, reprojection errors reference / probe / upright, frame
rate, filter lag at the current speed, injected error), `--inject` errors live; error study (`experiments.ErrorGrid`,
`run_experiments --errors`, menu 10).

Tracking filter lag (one-euro, TrackerConfig defaults, 30 frames/s): 65 ms at 10 mm/s (0.65 mm behind), 85 ms at
5 mm/s (0.42 mm), 111 ms at 2 mm/s, 47 ms at 20 mm/s; the same within 1 ms end to end on synthetic camera frames
(raw tracked position on synthetic frames: +-0.04 mm). The D405 camera latency adds to this and is not measured here
(needs the hardware).

Error study: default strategy (0.25 mm, 20 %, yaw 0), 4 position jitters x 4 yaw jitters x 3 latencies x 2 biases
x 3 cases = 288 runs in 16.1 min (estimate 11). Needs: cystic duct and CBD detected with local Dice >= 0.5 and the
GB - CBD connection kept, in every case. Results `output/results/errors_20261009-122109/`.

| Error (one at a time) | Cystic duct Dice | CBD Dice | Duct / CBD HD95 (mm) | Topology ok | Needs met |
|---|---|---|---|---|---|
| none | 0.955 | 0.972 | 0.50 / 0.50 | 3/3 | yes |
| jitter 0.5 mm | 0.871 | 0.887 | 0.50 / 0.71 | 3/3 | yes |
| jitter 1 mm | 0.725 | 0.792 | 0.80 / 1.00 | 3/3 | yes |
| jitter 2 mm | 0.481 | 0.525 | 1.12 / 1.53 | 0/3 | NO |
| yaw jitter 2 deg | 0.945 | 0.975 | 0.50 / 0.50 | 3/3 | yes |
| latency 100 ms | 0.905 | 0.960 | 0.50 / 0.50 | 3/3 | yes |
| bias 1 mm | 0.743 | 0.763 | 0.91 / 1.00 | 3/3 | yes |

**Tracking accuracy needed (robust: every combination within the limits passes, 48 combinations): position jitter
<= 0.5 mm (1 sigma per axis), yaw jitter <= 2 deg, latency <= 100 ms at 10 mm/s, bias <= 1 mm.** One factor at a
time 1 mm of jitter passes, but combined with yaw jitter, latency or bias it breaks the GB - CBD connection in 14 of
24 combinations (1-2 of 3 cases); 0.5 mm passes in all 24. Yaw jitter up to 2 deg and latency up to 100 ms
(1 mm along the sweep) cost little; a bias moves the reconstruction rigidly (Dice drops, connections survive).

S7 deviations / decisions:
- Review fix (2026-10-09, reported with the real camera: tracking OK, but SPACE recorded nothing and showed no
  B-mode): `TrackedPose` passed the full tracked pose on, and with the committed calibration the face comes out
  2-3 mm off the surface, so the acquirer rejected every frame as "not in contact" (|z| <= 1 mm). It now uses x, y and
  yaw with the probe upright at z = 0, as the simulator's own tracked mode does (bmode.demo, contact on); the tracked
  face z and tilt are shown in the tracking line. The SPACE detection also accepts a recent space key event as proof
  of focus (the window-title focus check alone may not match OpenCV's window); REC and "probe outside the region"
  are shown. Regression test: the real ProbeTracker with the committed calibration on synthetic frames captures.
- On request: the tracked app keeps the mouse as a switchable pose source (m), and is the mouse-sweep window (one
  window, clinical style) with a tracking line.
- Space "held": OpenCV reports key presses but not releases, so the held state is read with GetAsyncKeyState
  (Windows, only while the window has the focus); elsewhere space toggles recording.
- Latency in the study uses each sweep's planned path (exact); live it uses a buffer of the recent poses. Bias is
  along x; scale and tilt jitter are in the model but not in the grid.
- Tolerance rule: besides "one factor at a time" (optimistic) the summary gives the largest box of error levels in
  which every combination passes (`experiments.robust_box`); the first summary of the run reported only the former,
  rewritten with `--resummarize` (now also for error folders).
- Bug found and fixed in the S5 per-structure local metrics: voxels a structure shares with a same-label neighbour
  (cystic duct / GB neck, duct / CBD junction) were excluded from its reconstruction but kept in its truth, which
  understated the cystic duct's local Dice (0.65 -> 0.955 at zero error) and inflated its HD95 (5 mm -> 0.5 mm). The
  S5 recommendation is unaffected (it used class-level lumen Dice and structure recall); the per-structure columns of
  the S5 results.csv (experiments_20261008-160406) predate the fix (re-run the strategy study to refresh them).
- `MouseSession` got hooks (`update_pose`, `measure`, `report_extra`, `angle_hint`, MODE / TITLE / key rows) and draws
  the last known probe position in red when the pose is lost; `Acquirer.break_stretch()`; `mouse_filename(mode=)`.
- Menu: 6 `tracked`, 10 tracking-error study (`experiments --errors`); later steps now 11-15.

## Clinical version (2026-10-09)

On request: the developer apps stay as they are (`python -m orvue_us_inverse dev`); a clinical version with a
single window is the default (`python -m orvue_us_inverse`, `clinical/app.py`): B-MODE and INVERSE MAPPING tabs
(inverse mapping is an imaging mode next to B-mode; Doppler / Elastography placeholders), case drop-down and probe
calibration in the header.

- B-MODE tab: `bmode.demo`'s loop body as `BModeTab` (same `simulator_window` / `clinical_view` layout and keys);
  the IMAGING MODE tabs are drawn by the clinical module (`modes_y=None` to `simulator_window`), so `bmode.py` is
  unchanged.
- INVERSE MAPPING tab (`clinical/mapping_tab.py`, decided with the user: 3D-centred): sources scripted / mouse /
  camera probe in one window (`MappingSession(TrackedSession)`), record button, live B-mode at the probe (also
  before recording), structures found (volume per group; after Complete the evaluation per structure), large 3D
  view, camera view with the AR overlay, sweep map, status, speed. Lane guides only for the scripted sweep (the user:
  the 20 % guides do not matter when the probe moves freely).
- `poses.ScriptedPose`: the scripted plan played in an app, one captured frame per loop (advance() jumps to the next
  triggering sample), so no frame is skipped; a full plan gives exactly `expected_frames()`.
- AR overlay (`clinical/ar.py`): shallowest reconstructed structure per (x, y) column, colours of the 3D view,
  depth-coded (bright / opaque at the surface, darker / transparent at 50 mm), outlines; projected onto the camera
  image as the phantom's z = 0 plane (homography from `T_cam_phantom`, K, dist). Refreshed with the 3D view.
- `TrackedSession.latched`: recording switched on by a button (no key held).
- Checked headless with the synthetic camera scene through a real `ProbeTracker` (screenshots of both tabs, a
  scripted part sweep plus camera, complete). Not yet checked with the D405 and the physical probe.
- Review round 1 (2026-10-09, on request): SPACE switches recording on / off like the RECORD button for every source
  (camera no longer hold-to-record; mouse recording without a button held, the probe follows the mouse over the map);
  CASE drop-down in the header of both tabs (`CaseDropdown`; a case change starts a new mapping, not while
  recording); the menu items 2 and 3 were the same window, so the clinical menu is gone: `python -m orvue_us_inverse`
  opens the window directly, and "Calibrate probe" in its header runs the calibration (`CalibrationRoutine`) in the
  same window (`CalibrationView`).
- Review round 2: the mouse wheel turns the probe (5 deg, like q / e) in the B-MODE tab too.
- Tests: `tests/clinical/test_clinical.py` (11).

## Tests

`python -m pytest tests` after the clinical version, review round 1 (2026-10-09): **175 passed, 1 xfailed** in 287 s
(S7: 165 passed; the clinical version adds 10 in `tests/clinical/test_clinical.py`). After S7: 165 passed, 1 xfailed in 239 s (S6: 149 passed, 1 xfailed; S7 adds
7 in `tests/mapping/test_errors.py` incl. the filter lag on synthetic camera frames, 8 in `test_tracked.py`, 1 in
`test_experiments.py`; anatomy + tracking 58 passed, 1 xfailed, as upstream). The default 0.25 mm
spacing made some sweeps in the tests longer; slowest: S5 2-run grid ~26 s, S2 block sweep ~12 s, S6 coverage ~12 s. Slowest mapping
tests: the S2 ground-truth block sweep (~12 s), the S0 50-frame lane (~4.7 s), S2 incremental vs batch (~4.7 s). The xfail is the known tracking limit
`300mm_tilt15` with ID 0 + ID 5 (face z error ~1.6 mm). `viewer3d/` is copied but untested here.

## Decisions

- Standalone repository: the simulator was copied in, not imported; copied files are frozen baselines, changes
  logged in `UPSTREAM.md` (2026-10-08).
- Probe for this project: 30 mm wide, 50 mm deep, 0.1 mm pixels, frames 501 x 301; simulator always with
  `persistence=0`.
- Recognition: oracle labels first (per-frame ground truth from the simulator).
- Surface distance: standard symmetric MSD + HD95; shift test on HD95 (decided at S4, 2026-10-08).
- Default frame spacing 0.25 mm (S5 recommendation, decided 2026-10-08): `SweepConfig.frame_spacing_mm` and the
  app's `--spacing` default changed from 0.5 to 0.25 mm; 20 % overlap, yaw 0 and 0.5 mm voxels were already the
  defaults. A full yaw-0 sweep is now 1604 frames (yaw 0 + 90: 3208); with B-mode images the app plays at
  ~0.4x real time and a saved sweep is ~160 MB (yaw 0) / ~330 MB (0 + 90); --no-images keeps it ~2-5 MB. The
  half-voxel grid offset stays an experiment option (not a default). The S0-S4 figures in this file were
  measured at 0.5 mm.
- 3D views (decided at S3, 2026-10-08): OpenCV slice views, matplotlib off-screen snapshot and STL export
  (always available, tested headless), plus a browser 3D page (viewer3d template, key b) and an optional live
  PyVista window (`--live3d` / key p; pyvista now imports and renders here, kept optional with fallback).

## Known issues and risks

- Oracle labels come from the centre plane on a 0.2 mm grid, repeated to 0.1 mm (up to 0.1 mm quantisation);
  the B-mode image integrates 5 elevation planes over ±1 mm.
- `render()` noise is unseeded: poses and labels are reproducible, images are not.
- The portal vein reaches ~55.4 mm and the left portal vein ~51.9 mm, below the 50 mm grid.
- Walls of 0.3-0.6 mm (artery branches 0.3 mm, duct 0.5 mm, vein 0.4 mm) are thinner than a 0.5 mm voxel:
  expect low wall Dice even with perfect poses.
- Frame times: `labels_image()` ~13 ms, `render()` ~78 ms (Prep); `render()` adds unseeded noise, so tests that
  compare images seed `sim._rng`.
- Tracking validated on synthetic images only; `get_pose()` is filtered (adds lag); `H_MM` = 0.0; calibration
  from 2026-10-06 (`config/calibration.json`).
- Physical dummy probe face width (38 mm vs 30 mm) not recorded; check before tracked sweeps.
- Tracked sweeps use the tracked x, y, yaw with the probe upright on the surface (z = 0), like the simulator, so
  lifting the probe does not stop the capture; release SPACE to stop.
- The 3D viewer and the browser 3D view load three.js from a CDN (needs internet).
- PyVista / VTK were blocked by Windows Application Control at Prep and work since S3: if the block returns,
  the live 3D window reports it and the app continues (snapshot / browser view still work).
- The live PyVista window shares the thread with the OpenCV windows (non-blocking `interactive_update`);
  checked off-screen in the tests, the interactive window only by eye.

## Next step

With the hardware: calibrate (menu 2), check tracking (menu 1), run a tracked sweep (menu 6) and compare its report
with a mouse sweep; measure the camera latency; re-check a tracked sweep after the contact fix (REC shows while
recording; face z is shown for information). The plan's sessions S0-S7 are complete. Try the clinical window
with the D405 (`python -m orvue_us_inverse`): calibration from the header, camera-probe recording (SPACE), the AR
overlay on the real board.
