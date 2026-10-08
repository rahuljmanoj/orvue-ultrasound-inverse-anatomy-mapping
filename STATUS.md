# STATUS.md

State of the inverse-mapping project. Updated at the end of every session (plan: `PLAN_inverse_mapping.md`).

| | |
|---|---|
| Repository | rahuljmanoj/orvue-ultrasound-inverse-anatomy-mapping, `main` at `ff697fe` (2026-10-08) |
| Simulator code | copied in from rahuljmanoj/orvue-ultrasound-simulator `e789389`; frozen; see `UPSTREAM.md` |
| Environment | conda env `orvue-robot`, Python 3.11.16 (versions in `CLAUDE.md` > Running) |
| Last update | 2026-10-08, after the `import-simulator` merge |

## Sessions

| Session | State | Evidence / notes |
|---|---|---|
| Prep P1 data folders | Not started | `output/` is gitignored; `SWEEPS_DIR`, `RESULTS_DIR` not in `paths.py` yet |
| Prep P2 environment | Partly | Env and versions recorded in `CLAUDE.md`; import check not run; PyVista / VTK blocked (Windows Application Control) |
| Prep P3 probe | Not started | No `mapping/probe.py`; `LinearProbe` defaults are 30 x 50 mm, 0.1 mm, but `persistence` defaults to 0.3 |
| Prep P4 CLAUDE.md | Partly | Simulator conventions, frames, read-only rules and `persistence=0` rule in; no inverse-mapping section |
| Prep P5 baseline | Partly | Tests pass (below); frames and timings not recorded |
| S0-S7 | Not started | No `mapping/` sub-package, no `tests/mapping/`, no sweeps |

## Tests

`python -m pytest tests` at `9a3e911` (merged as `ff697fe`): **65 passed, 1 xfailed** (anatomy + tracking 58 passed,
1 xfailed, as upstream; template and `test_standalone.py` the rest). The xfail is the known tracking limit
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
- S5 runtime is unknown until `labels_image()` is timed in Prep (full `render()` ~100 ms/frame per upstream).
- Tracking validated on synthetic images only; `get_pose()` is filtered (adds lag); `H_MM` = 0.0; calibration
  from 2026-10-06 (`config/calibration.json`).
- Physical dummy probe face width (38 mm vs 30 mm) not recorded; check before S7.
- The 3D viewer loads three.js from a CDN (needs internet).

## Next step

Run the Prep session (`PLAN_inverse_mapping.md` > Preparation), then S0.
