# STATUS.md

State of the inverse-mapping project. Updated at the end of every session (plan: `PLAN_inverse_mapping.md`).

| | |
|---|---|
| Repository | rahuljmanoj/orvue-ultrasound-inverse-anatomy-mapping, `main` after the Prep merge (branch `prep`, 2026-10-08) |
| Simulator code | copied in from rahuljmanoj/orvue-ultrasound-simulator `e789389`; frozen; see `UPSTREAM.md` |
| Environment | conda env `orvue-robot`, Python 3.11.16 (`C:/Users/rahul/miniconda3/envs/orvue-robot/python.exe`); import check passed (Prep P2) |
| Last update | 2026-10-08, after the Prep session |

## Sessions

| Session | State | Evidence / notes |
|---|---|---|
| Prep P1 data folders | Done | `paths.SWEEPS_DIR` (`output/sweeps`), `paths.RESULTS_DIR` (`output/results`); created on demand |
| Prep P2 environment | Done | numpy 2.4.6, scipy 1.17.1, cv2 5.0.0 (`cv2.aruco` with `ArucoDetector`), scikit-image 0.26.0, matplotlib 3.11.2, reportlab 5.0.1, pytest 9.1.1, pyrealsense2 2.58.4 (optional); nothing missing. PyVista / VTK not imported (blocked) |
| Prep P3 probe | Done | `mapping/probe.py`: `PROBE` (every field stated), `make_simulator` (persistence 0, not overridable), `probe_metadata`; `tests/mapping/test_probe.py` (4 tests; the previous-pose test fails with persistence 0.3, checked) |
| Prep P4 CLAUDE.md | Done | "Inverse mapping" section; README "Every file" and layout updated |
| Prep P5 baseline | Done | Tests below; frames and timings below; `tracking.markers` and `tracking.tracker` import without a camera (0.09 s) |
| S0-S7 | Not started | No sweeps yet |

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

## Tests

`python -m pytest tests` after Prep (2026-10-08): **69 passed, 1 xfailed** in 54 s (65 passed, 1 xfailed at `ff697fe`
plus 4 in `tests/mapping/test_probe.py`; anatomy + tracking 58 passed, 1 xfailed, as upstream). The xfail is the known tracking limit
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

S0 (`PLAN_inverse_mapping.md`).
