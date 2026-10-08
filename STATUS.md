# STATUS.md

State of the inverse-mapping project. Updated at the end of every session (plan: `PLAN_inverse_mapping.md`).

| | |
|---|---|
| Repository | rahuljmanoj/orvue-ultrasound-inverse-anatomy-mapping, `main` after the S1 merge (branch `s1`, 2026-10-08) |
| Simulator code | copied in from rahuljmanoj/orvue-ultrasound-simulator `e789389`; frozen; see `UPSTREAM.md` |
| Environment | conda env `orvue-robot`, Python 3.11.16 (`C:/Users/rahul/miniconda3/envs/orvue-robot/python.exe`); import check passed (Prep P2) |
| Last update | 2026-10-08, after S1 |

## Sessions

| Session | State | Evidence / notes |
|---|---|---|
| Prep P1 data folders | Done | `paths.SWEEPS_DIR` (`output/sweeps`), `paths.RESULTS_DIR` (`output/results`); created on demand |
| Prep P2 environment | Done | numpy 2.4.6, scipy 1.17.1, cv2 5.0.0 (`cv2.aruco` with `ArucoDetector`), scikit-image 0.26.0, matplotlib 3.11.2, reportlab 5.0.1, pytest 9.1.1, pyrealsense2 2.58.4 (optional); nothing missing. PyVista / VTK not imported (blocked) |
| Prep P3 probe | Done | `mapping/probe.py`: `PROBE` (every field stated), `make_simulator` (persistence 0, not overridable), `probe_metadata`; `tests/mapping/test_probe.py` (4 tests; the previous-pose test fails with persistence 0.3, checked) |
| Prep P4 CLAUDE.md | Done | "Inverse mapping" section; README "Every file" and layout updated |
| Prep P5 baseline | Done | Tests below; frames and timings below; `tracking.markers` and `tracking.tracker` import without a camera (0.09 s) |
| S0 skeleton, config, sweep format | Done | `mapping/config.py`, `mapping/sweep_io.py` implemented; `poses`, `acquisition`, `recon`, `render`, `evaluate`, `experiments`, `errors` are docstring placeholders; `tests/mapping/test_sweep_io.py` (11 tests) + 2 in `test_probe.py`. Details below |
| S1 scripted sweep, acquisition, playback | Done | `mapping/poses.py`, `mapping/acquisition.py`, app `mapping/run_scripted.py` (menu 4 `scripted`); `tests/mapping/test_poses_acquisition.py` (16), `test_run_scripted.py` (3). Details below |
| S2-S7 | Not started | No saved sweeps in the repository (`output/` is gitignored) |

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

## Tests

`python -m pytest tests` after S1 (2026-10-08): **101 passed, 1 xfailed** in 73 s (S0: 82 passed, 1 xfailed; S1 adds
16 in `tests/mapping/test_poses_acquisition.py` and 3 in `test_run_scripted.py`; anatomy + tracking 58 passed,
1 xfailed, as upstream). Slowest mapping tests: the S0 50-frame lane (~4.7 s) and the S1 real-capture lane (~2.4 s). The xfail is the known tracking limit
`300mm_tilt15` with ID 0 + ID 5 (face z error ~1.6 mm). `viewer3d/` is copied but untested here.

## Decisions

- Standalone repository: the simulator was copied in, not imported; copied files are frozen baselines, changes
  logged in `UPSTREAM.md` (2026-10-08).
- Probe for this project: 30 mm wide, 50 mm deep, 0.1 mm pixels, frames 501 x 301; simulator always with
  `persistence=0`.
- Recognition: oracle labels first (per-frame ground truth from the simulator).
- S3 / S4 3D: no PyVista / VTK. Proposed: OpenCV slice views, matplotlib off-screen snapshots, STL export.
  **Open: confirm before S3.**

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
- The 3D viewer loads three.js from a CDN (needs internet).

## Next step

Watch a full sweep (`python -m orvue_us_inverse scripted` / `--yaw 0 90`), save one with s, then S2
(`PLAN_inverse_mapping.md`): reconstruction from oracle labels.
