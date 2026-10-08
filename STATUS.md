# STATUS.md

State of the inverse-mapping project. Updated at the end of every session (plan: `PLAN_inverse_mapping.md`).

| | |
|---|---|
| Repository | rahuljmanoj/orvue-ultrasound-inverse-anatomy-mapping, `main` after the S4 merge (branch `s4`, 2026-10-08) |
| Simulator code | copied in from rahuljmanoj/orvue-ultrasound-simulator `e789389`; frozen; see `UPSTREAM.md` |
| Environment | conda env `orvue-robot`, Python 3.11.16 (`C:/Users/rahul/miniconda3/envs/orvue-robot/python.exe`); import check passed (Prep P2) |
| Last update | 2026-10-08, after S4 |

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
| S5-S7 | Not started | Sweeps and reports in `output/` (gitignored) |

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

## Tests

`python -m pytest tests` after S4 (2026-10-08): **128 passed, 1 xfailed** in 126 s (S3: 118 passed, 1 xfailed; S4 adds
10 in `tests/mapping/test_evaluate.py`; anatomy + tracking 58 passed, 1 xfailed, as upstream). Slowest mapping
tests: the S2 ground-truth block sweep (~12 s), the S0 50-frame lane (~4.7 s), S2 incremental vs batch (~4.7 s). The xfail is the known tracking limit
`300mm_tilt15` with ID 0 + ID 5 (face z error ~1.6 mm). `viewer3d/` is copied but untested here.

## Decisions

- Standalone repository: the simulator was copied in, not imported; copied files are frozen baselines, changes
  logged in `UPSTREAM.md` (2026-10-08).
- Probe for this project: 30 mm wide, 50 mm deep, 0.1 mm pixels, frames 501 x 301; simulator always with
  `persistence=0`.
- Recognition: oracle labels first (per-frame ground truth from the simulator).
- Surface distance: standard symmetric MSD + HD95; shift test on HD95 (decided at S4, 2026-10-08).
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
- Physical dummy probe face width (38 mm vs 30 mm) not recorded; check before S7.
- The 3D viewer and the browser 3D view load three.js from a CDN (needs internet).
- PyVista / VTK were blocked by Windows Application Control at Prep and work since S3: if the block returns,
  the live 3D window reports it and the app continues (snapshot / browser view still work).
- The live PyVista window shares the thread with the OpenCV windows (non-blocking `interactive_update`);
  checked off-screen in the tests, the interactive window only by eye.

## Next step

Look at the S4 reports in `output/results/` (report.md, overlay_3d.png, structures.png, slices_errors.png) and try
key c in `python -m orvue_us_inverse scripted`; then S5 (`PLAN_inverse_mapping.md`): sweep-strategy
experiments, including the voxel-face sampling observation above.
