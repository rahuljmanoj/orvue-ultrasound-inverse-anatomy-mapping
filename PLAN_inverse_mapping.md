# Inverse Mapping — Session Plan and Claude Code Prompts

Oct 7, 2026 · @Rahul Manoj · revised Oct 8, 2026 for the standalone repository (simulator copied in, see `UPSTREAM.md`); changes listed in Section 9

## 1. Goal and principles

The aim is to rebuild the hidden anatomy in 3D from a series of tracked ultrasound frames — an inverse mapping from 2D images back to 3D structure — and to measure how close the result comes to the ground truth. The probe moves over a 100 × 100 mm "black box" whose anatomy is hidden; the reconstruction grows in real time as the area is scanned; when the scan is declared complete, the result is compared with the true anatomy and a report is produced.

Four principles shape the plan:

1. **Reconstruction and recognition are separate problems.** Reconstruction places each frame's pixels at their 3D positions and fuses overlapping samples into a voxel grid. Recognition decides what each pixel or voxel is. They are built and evaluated separately so that errors can be traced to one or the other.
2. **Recognition starts with oracle labels.** The simulator already knows the tissue under every pixel. Using that per-frame ground-truth label image as the "recognition" gives the best possible result, so every remaining error is due to reconstruction: coverage, frame spacing, voxel size, slice thickness and, later, tracking error. Classical and learned recognition can replace the oracle later and be measured against this upper bound.
3. **Pose sources are interchangeable.** The same pipeline runs from a scripted sweep, then from mouse control, then from the camera-tracked dummy probe. Scripted sweeps come first because they are reproducible: they let you measure how overlap, frame spacing and orientation affect the result before any hand motion is involved.
4. **One small, testable step per Claude Code session.** Each session has a narrow scope, its own tests. You review the result before starting the next session, so errors are caught where they are introduced.

### How to use this document

Do the preparation below first: it fixes the probe geometry (30 mm wide, 50 mm deep) in `mapping/probe.py`, checks the environment and adds the inverse-mapping conventions to `CLAUDE.md`. Every session ends by updating `STATUS.md`. Then run the sessions in order: for each one, paste the common preamble from Section 5 followed by that session's prompt from Section 6 or 7. Section 2 answers the open questions about sweeps, Sections 3 and 4 give the architecture and roadmap, and Section 8 lists what to check between sessions. Revisit the prompts for S5–S7 after S4, since earlier results may change their details.

## Preparation (before S0)

The simulator code was copied into this repository on 2026-10-08 (`UPSTREAM.md`: source commit `e789389`, package `orvue_us_inverse`; tests 65 passed, 1 xfailed at `ff697fe`). Preparation only adds what inverse mapping needs on top.

| Step | What | Done when |
| --- | --- | --- |
| P1 | Data folders for sweeps and results: `paths.SWEEPS_DIR` (`output/sweeps`) and `paths.RESULTS_DIR` (`output/results`); `output/` is already gitignored | Both in `paths.py` |
| P2 | Python environment `orvue-robot` (Python 3.11.16): import check of the required packages; no PyVista / VTK (blocked by Windows Application Control) | Import check passes; versions recorded in `STATUS.md` |
| P3 | Probe geometry in one place, `mapping/probe.py`: 30 mm wide, 50 mm deep, 0.1 mm pixels, every field stated; `make_simulator()` with `persistence=0` | A rendered frame and its labels are 501 × 301 pixels |
| P4 | Inverse-mapping section in `CLAUDE.md` (the file exists; simulator conventions are already in it) | Section written; README "Every file" updated |
| P5 | Baseline: tests pass, three test frames rendered, `render()` and `labels_image()` timed, tracking modules import without a camera | `STATUS.md` updated |

Physical setup: if the dummy probe was built with a 38 mm face, rebuild or relabel it as 30 mm so the tracked footprint matches the simulated array. The printed tracking board and marker layout are unaffected. (Not recorded in either repository; check it before S7.)

Prompt for the preparation session:

```text
PREPARATION SESSION for inverse mapping (this repository). The simulator code is already in the package
(UPSTREAM.md, copied from e789389) and the tests pass (65 passed, 1 xfailed at ff697fe). Do only the
preparation below; no inverse-mapping code yet. Stop and ask me whenever a step needs a decision.

P1 Data folders
- Add SWEEPS_DIR = output/sweeps and RESULTS_DIR = output/results to paths.py (created on demand; output/ is
  already gitignored).

P2 Python environment
- Report the interpreter (expected: conda env orvue-robot, Python 3.11.16) and check imports: numpy, scipy,
  cv2 (including cv2.aruco), skimage, matplotlib, reportlab, pytest; pyrealsense2 (optional extra). Do not
  import or install pyvista / vtk (blocked on this machine). Report anything missing; install nothing without
  asking.

P3 Probe geometry
- Create src/orvue_us_inverse/mapping/__init__.py and mapping/probe.py: PROBE = LinearProbe(width_mm=30.0,
  depth_mm=50.0, px_mm=0.1, ...) with every other field written out at its current LinearProbe default, so a
  change of the defaults cannot change this project; make_simulator(case="normal", **kw) returning
  BModeSimulator(build_case(case), PROBE, persistence=0.0, **kw) (persistence cannot be overridden); and
  probe_metadata() returning every LinearProbe field as a dict (for sweep files). Read simulation/bmode.py to
  confirm the field names; do not modify it.
- tests/mapping/__init__.py and tests/mapping/test_probe.py: image and label shapes 501 x 301; persistence is 0
  (the image at a pose does not depend on the previous frame's pose); PROBE fields equal the values above.
- Smoke check: render the normal case at (50, 50, yaw 0), (25, 70, 90) and (60, 30, -25); save frames and
  colour-coded label images as PNG to output/prep_check/; report the mean render() time and the mean
  labels_image() time over 20 frames each (labels_image alone is what oracle-only sweeps need, S5).

P4 CLAUDE.md
- Add an "Inverse mapping" section: purpose (reconstruction vs recognition, oracle labels); the sub-package
  src/orvue_us_inverse/mapping/ ("mapping/") and tests/mapping/; mapping/probe.py as the only probe source;
  SWEEPS_DIR and RESULTS_DIR; PLAN_inverse_mapping.md (session plan) and STATUS.md (state, updated at the end
  of every session); tests headless; no PyVista / VTK. Add every new file to the README "Every file" table.

P5 Baseline
- Run python -m pytest tests and report the counts. Confirm tracking/markers.py and tracking/tracker.py import
  without a camera. Update STATUS.md: environment, frame times, test counts, Prep done.
```

## 2. Sweep strategy

A sweep moves the probe so that its image planes fill a volume. Four quantities decide whether the volume is filled well: frame spacing, lane overlap, orientation and depth. The defaults below are a sound starting point; the scripted experiments in session S5 then measure their actual effect. All numbers assume the 30 mm wide, 50 mm deep probe.

### 2.1 Geometry of one frame

Each B-mode frame is a vertical slab: 30 mm wide along the array (lateral, u), 50 mm deep (along the beam, n), and about 1.2 mm thick across the slice (elevation, v, FWHM of the simulated beam). With 0.1 mm pixels it is 501 × 301 pixels. Moving the probe along v, perpendicular to the image plane, stacks slabs into a volume. Moving it along u, within the plane, adds almost nothing new.

The oracle label image comes from the centre plane only (elevation 0) and is computed on a 0.2 mm grid, then repeated to the 0.1 mm pixels; the B-mode image integrates 5 elevation planes over ±1 mm. Label positions therefore carry up to 0.1 mm of quantisation, well below the 0.5 mm voxel.

### 2.2 Frame spacing

Frames must be closer together than the slice is thick, or thin structures fall between them. Rule: spacing ≤ the voxel size and ≤ half the slice thickness. With 0.5 mm voxels and a 1.2 mm slice, **0.5 mm spacing** is the default; 0.25 mm is the fine option. Acquisition is triggered by distance travelled (every 0.5 mm, or every 1° of rotation), not by time, so spacing stays uniform whatever the speed.

For hand-held sweeps the speed limit follows from the frame rate: max speed = spacing × frame rate. At 0.5 mm and 15–25 frames per second that is roughly 8–12 mm/s.

### 2.3 Lanes and overlap

One lane covers a 30 mm wide band, so the 100 mm region needs several parallel lanes, swept back and forth (serpentine path). Overlap is defined as (30 − lane stride) / 30. Because 100 mm is not a multiple of 30 mm, lanes are distributed evenly so that the outer lanes' image edges touch the region edges exactly; the actual overlap is therefore at least the requested value.

| Requested overlap | Lanes | Lane centres (mm) | Actual overlap |
| --- | --- | --- | --- |
| 0–20% (default 20%) | 4 | 15, 38.3, 61.7, 85 | 22% |
| 40% | 5 | 15, 32.5, 50, 67.5, 85 | 42% |
| 50% | 6 | 15, 29, 43, 57, 71, 85 | 53% |

Why overlap matters even though the simulator is perfect:

- With perfect poses and a uniform beam, zero overlap would leave no gaps. In practice, any yaw error moves the image edges most (15 mm × the angle error, about 0.26 mm per degree), so adjacent lanes can separate and leave seams.
- Real probes have weaker resolution at the image edges; overlap means every voxel is also seen nearer the centre of some frame.
- Overlap gives each voxel several observations, which makes voting (and later, intensity compounding) more robust.

Default: **20% requested, 22% actual: four lanes** for a single orientation.

### 2.4 Orientation

You do not have to keep the same orientation, but you can. A single orientation (yaw 0°, sweeping along y) gives complete coverage. The reconstruction is then anisotropic: sharp within the image plane (0.1 mm pixels, about 0.3–0.6 mm beam resolution) but coarser across sweeps (set by frame spacing and slice thickness). Structures that run across the image plane are captured as a sequence of cross-sections and reconstruct well; thin structures lying along the sweep direction are captured in fewer frames.

A second pass at 90° (sweeping along x) makes the result more isotropic and fills shadows behind stones from a different direction. Default: **yaw 0° first; add 90° as a second pass.** Session S5 measures whether 0° + 90°, or four orientations, is worth the extra scan time.

In the scripted and mouse sweeps, yaw is held constant within a lane. Rotating during a lane is allowed later but makes coverage uneven.

### 2.5 Depth

The probe now images 50 mm deep, the full depth of the anatomy volume, so every structure in the 100 × 100 × 50 mm volume can be reached. Only the deepest part of the portal vein lies below the modelled volume: the main portal vein reaches about 55.4 mm (its end at (50, 20, 49) mm, 6.0 mm lumen radius plus 0.4 mm wall) and the left portal vein about 51.9 mm. Reconstruction and evaluation use z from 0 to 50 mm, and only voxels actually observed count in the comparison.

### 2.6 Defaults and cost

| Setting | Default | Notes |
| --- | --- | --- |
| Probe | 30 mm wide, 50 mm deep | Frames of 501 × 301 pixels at 0.1 mm |
| Voxel size | 0.5 mm | 100 × 100 × 50 mm = 4.0 million voxels |
| Frame spacing | 0.5 mm | Distance-triggered; 1° rotation trigger |
| Overlap | 20% requested (4 lanes) | Actual 22% |
| Orientations | 0°, then 0° + 90° |  |
| Frames | 800 per orientation | 4 lanes × 200 frames |
| Simulator settings | persistence = 0 | Temporal averaging would smear the reconstruction |

### 2.7 Questions the scripted experiments will answer (S5)

| Factor | Values tested | Expected effect |
| --- | --- | --- |
| Frame spacing | 0.25, 0.5, 1, 2 mm | Thin structures (cystic artery, cystic duct) break up above about 1 mm |
| Overlap | 20%, 40%, 50% requested (22%, 42%, 53% actual) | Little effect with perfect poses; matters once pose error is added (S7) |
| Orientations | 0°; 90°; 0°+90°; 0°, 45°, 90°, 135° | More orientations give more isotropic surfaces at a proportional time cost |
| Voxel size | 0.5, 0.25 mm | Finer voxels help the smallest vessels; 8× memory |
| Pose error (S7) | Jitter, bias, latency | Sets the tracking accuracy the method needs |

## 3. Architecture and data format

The pipeline runs pose source → acquisition → recorder → reconstruction → rendering, with evaluation at the end. Each stage is its own module, so any one can be replaced: the pose source moves from scripted to mouse to camera, and later a real probe can replace the simulator as the image source.

### 3.1 Package layout

New code lives in the sub-package `src/orvue_us_inverse/mapping/` ("`mapping/`" below). Apps and scripts are modules in it, run with `python -m orvue_us_inverse.mapping.<module>` and added as menu steps (`COMMANDS` + `MENU` in `__main__.py`). Tests go in `tests/mapping/`; generated data under `output/` (`paths.SWEEPS_DIR`, `paths.RESULTS_DIR`). The simulator files copied from the simulator repository (`simulation/`, `tracking/`, `ui/`, `viewer3d/`) are never modified; any agreed exception is logged in `UPSTREAM.md`.

| Module (`mapping/`) | Responsibility | Session |
| --- | --- | --- |
| `probe.py` | The probe (30 × 50 mm, 0.1 mm), `make_simulator()` with persistence 0, probe metadata | Prep |
| `config.py` | Grid bounds, voxel size, sweep and acquisition settings (dataclasses) | S0 |
| `sweep_io.py` | Frame record and sweep container; save and load | S0 |
| `poses.py` | Pose sources: scripted sweep (S1), mouse (S6), camera tracker (S7) | S1, S6, S7 |
| `acquisition.py` | Distance-triggered frame capture from the simulator | S1 |
| `recon.py` | Voxel grid and label compounding (oracle labels), coverage, hole filling | S2 |
| `render.py` | Live slice views, surface meshes, off-screen 3D snapshot (no PyVista) | S3 |
| `evaluate.py` | Ground-truth voxelisation, metrics, report | S4 |
| `experiments.py` | Headless parameter studies | S5, S7 |
| `errors.py` | Pose-error injection (jitter, bias, latency, scale) | S7 |
| `run_scripted.py`, `run_mouse.py`, `run_tracked.py` | Interactive apps | S1, S6, S7 |
| `reconstruct_sweep.py`, `evaluate_sweep.py`, `run_experiments.py` | Offline scripts | S2, S4, S5 |

### 3.2 Sweep data format

Every sweep is saved to disk, so it can be reconstructed again with other settings, voxel sizes or injected errors without re-scanning. A sweep holds metadata plus one record per frame.

| Field | Type | Content |
| --- | --- | --- |
| `index`, `t` | int, float | Frame number; simulated time in seconds |
| `T_true` | 4×4 float | Pose used to generate the image |
| `T_measured` | 4×4 float | Pose used for reconstruction; equal to `T_true` unless errors are injected |
| `labels` | int8 image | Oracle tissue labels (501 × 301) |
| `image` | uint8 image, optional | B-mode frame (501 × 301), for later classical recognition |
| Metadata | JSON | Case name, probe geometry from `mapping/probe.py` (30 mm width, 50 mm depth, pixel size), simulator settings, sweep settings, timestamp; provenance: this repository's git commit and the simulator source commit from `UPSTREAM.md` (`e789389`) |

Storage: compressed `.npz` per sweep in `paths.SWEEPS_DIR` (`output/sweeps`). A 600-frame sweep with images and labels is typically tens of megabytes; images can be omitted.

### 3.3 Two clocks

The scripted sweep runs on simulated time, not wall-clock time. If rendering a frame takes longer than real time, playback slows down rather than dropping frames. The acquired poses and oracle labels are therefore identical on any computer, which keeps experiments reproducible. B-mode images are not: `render()` draws its electronic noise from an unseeded generator, so keep the saved sweep when an image-based result must be repeated. Mouse and camera modes run on wall-clock time, where the speed meter matters.

### 3.4 Reconstruction method (oracle labels)

For every frame, each pixel's 3D position is computed from `T_measured` and the probe geometry (the same mapping as `BModeSimulator.plane_points`). The pixel's label is added to a per-voxel class histogram in the nearest voxel (pixel-nearest-neighbour insertion). The reconstructed label of a voxel is the class with the most votes; voxels never hit stay "unobserved". An optional final step fills small gaps surrounded by observed voxels, but never extends into unscanned areas. A mean-intensity volume is accumulated alongside, for later classical recognition.

## 4. Session roadmap

Eight sessions, each ending with passing tests. S0–S4 give the complete scripted demo with evaluation; S5 answers the sweep-strategy questions; S6 and S7 add the mouse and the camera-tracked probe.

| Session | Goal | What you can see or run afterwards | Done when |
| --- | --- | --- | --- |
| Prep | Data folders, environment check, probe 30 × 50 mm, `CLAUDE.md` | Three test frames of 501 × 301 pixels | Simulator tests pass; baseline report written |
| S0 | Package skeleton, configuration and sweep data format | Nothing visual; a sweep file can be written and read | Save/load round-trip tests pass |
| S1 | Scripted sweep generator, distance-triggered acquisition, live playback | The probe moving over the hidden 100 mm box with planned lanes, coverage trace and B-mode; saved sweeps | Lane coverage and frame-spacing tests pass |
| S2 | Reconstruction from oracle labels | Reconstructed label volume and orthogonal slice images from a saved sweep | Interior voxels ≥ 99.5% correct with perfect poses |
| S3 | Live 3D rendering | Surfaces growing in a 3D window during the scripted sweep, coverage heat map, optional ground-truth overlay | Off-screen rendering tests pass; update time ≤ 300 ms |
| S4 | Evaluation and "complete" | Press c: metrics table, 3D error overlay and report saved | Metric tests pass (perfect = 1.0, known shifts give known errors) |
| S5 | Sweep-strategy experiments | CSV and plots of quality versus spacing, overlap, orientation and voxel size; a recommendation | Experiment grid runs headless; report generated |
| S6 | Mouse sweeps | Hold-to-scan with lane guides, speed meter and coverage holes; evaluation on completion | Speed-gating and recording logic tests pass |
| S7 | Camera-tracked probe and pose-error injection | Live scanning with the dummy probe; error study showing the tracking accuracy needed | Zero-error equivalence and latency tests pass |

## 5. Common preamble

Paste this at the start of every session, followed by that session's prompt.

```text
CONTEXT (same for every session of this task)
Read CLAUDE.md first (conventions, frames, read-only rules), then STATUS.md (where the project stands).
Package orvue_us_inverse in src/. The simulator was copied in from the simulator repository (UPSTREAM.md) and
is frozen:
- simulation/anatomy.py: geometry (Tube / Blob primitives, Anatomy.labels(P), build_case(name), CASES, TISSUES
  labels 0-10: 0 liver, 1 fat, 2 GB wall, 3 bile, 4 stone, 5 duct wall, 6 artery wall, 7 arterial blood,
  8 vein wall, 9 venous blood, 10 lymph node).
- simulation/bmode.py: BModeSimulator with render(T, return_labels=True), plane_points(T), labels_image(T),
  pose_from_xy_yaw(x, y, yaw_deg, z=0), in_contact(T); top_view(anatomy). labels_image is computed on a 0.2 mm
  grid from the centre plane and repeated to 501 x 301; render() noise is unseeded (labels are reproducible,
  images are not).
- mapping/probe.py: THE probe for this project: 30 mm wide, 50 mm deep, 0.1 mm pixels -> 501 x 301 images.
  Always create the simulator with make_simulator(case) (persistence=0); never rely on LinearProbe defaults.
- tracking/markers.py and tracking/tracker.py: camera tracking (used from S7 only).
- viewer3d/: browser 3D anatomy viewer (three.js from a CDN, needs internet); geometry.py has marching-cubes
  meshes and STL/VTP export.
Frames: phantom frame in mm, right-handed: +x right (patient's left), +y down (caudal), +z depth. Active region
x, y in [0, 100]; imaged depth z in [0, 50]. Probe pose T is 4x4; columns: u lateral (along the array),
v elevation (across the slice), n beam direction, o centre of the probe face.

TASK: "inverse mapping" = reconstruct the 3D segmented anatomy from tracked 2D frames and compare it with the
ground truth. Reconstruction (placing frame pixels in 3D) and recognition (labelling) are separate. In this
task, recognition uses ORACLE labels: the simulator's per-pixel ground-truth label image of each frame.
Everything you need for this session is in CLAUDE.md, STATUS.md, this preamble and the session prompt below.

RULES
- Do NOT modify the copied simulator files (simulation/, tracking/, ui/, viewer3d/). If one must change, stop
  and ask; log an agreed change in UPSTREAM.md.
- New code: mapping/ (src/orvue_us_inverse/mapping/); apps are modules run with
  python -m orvue_us_inverse.mapping.<app> and added to the menu; tests in tests/mapping/; every file location
  in paths.py; generated data under output/.
- Work only within this session's scope. Do not start the next session's work. If something in the scope is
  unclear or seems wrong, stop and ask me rather than guessing.
- Small, readable modules with docstrings and type hints; core logic testable without a display
  (tests must run headless; OpenCV windows only in apps). No PyVista / VTK: blocked on this machine.
- Use the existing environment (orvue-robot); ask before installing anything. Keep requirements.txt and
  pyproject.toml up to date.
- Keep README ("Every file" table), CLAUDE.md and STATUS.md in step.
- End of session: run python -m pytest tests and show the results; summarise what you built, any deviations
  from the prompt, and anything I should check by eye; update STATUS.md (session state, test counts, timings,
  deviations); confirm the copied simulator files are unchanged (git diff --stat on simulation/, tracking/,
  ui/, viewer3d/).
```

## 6. Session prompts S0–S3

### S0 — Branch, skeleton and data format

```text
SESSION S0: package skeleton, configuration and sweep data format. No visual output this session.

1. Create these modules in mapping/ (probe.py exists from Prep). Implement config.py and sweep_io.py fully; the
   others contain only a module docstring describing their future role and the session that implements them:
   config.py, sweep_io.py, poses.py (S1), acquisition.py (S1), recon.py (S2), render.py (S3),
   evaluate.py (S4), experiments.py (S5), errors.py (S7).
2. config.py (dataclasses with defaults):
   - GridConfig: x, y bounds [0, 100] mm, z bounds [0, 50] mm, voxel_mm 0.5; derived shape and voxel centres.
   - SweepConfig: yaw_list_deg [0.0], overlap_pct 20.0, frame_spacing_mm 0.5, angle_trigger_deg 1.0,
     speed_mm_s 10.0, serpentine True, region 100 x 100 mm. Probe width and depth are read from
     mapping/probe.py, not duplicated.
   - AcquisitionConfig: store_images True; the simulator always comes from mapping.probe.make_simulator().
3. sweep_io.py:
   - FrameRecord: index, t, T_true (4x4), T_measured (4x4), labels (int8, 501 x 301), image (uint8 or None).
   - Sweep: metadata dict + list of FrameRecord; append(frame); save(path) as compressed .npz (stack arrays;
     metadata as JSON string, including case name, simulator settings, sweep config, timestamp); Sweep.load(path).
   - Metadata also stores the probe fields from probe_metadata() and provenance: this repository's git commit
     (git rev-parse HEAD; None if git is not available) and the simulator source commit e789389 from UPSTREAM.md.
   - estimate_size_mb(n_frames, store_images) helper.
4. tests/mapping/test_sweep_io.py: save/load round-trip equality (with and without images), metadata
   preserved (including probe fields and provenance), size estimate within 30% of the actual file for a
   50-frame sweep. Use 50 real frames from make_simulator("normal") along a short lane (seed sim._rng), not
   random arrays: labels compress far better than speckle, so random data would calibrate the estimate wrong.
5. README.md "Every file" table and the CLAUDE.md inverse-mapping section: the new modules with their sessions
   (no separate README).
```

### S1 — Scripted sweep, acquisition and live playback

```text
SESSION S1: scripted sweep generator, distance-triggered acquisition, live playback app, saving sweeps.

1. mapping/poses.py: ScriptedSweep(sweep_cfg)
   - For each yaw in yaw_list_deg: the lateral axis is u = (cos yaw, sin yaw), the sweep direction is v
     (perpendicular). Lanes are parallel lines along v. With probe width W = 30 mm (from mapping/probe.py):
     lane stride = W * (1 - overlap); number of lanes n = ceil((100 - W) / stride) + 1; then distribute lane
     centres EVENLY so the outer lanes' image edges touch the region edges exactly (actual overlap >= requested;
     report it). Examples for yaw 0: overlap 20% -> 4 lanes, centres x = 15, 38.3, 61.7, 85 (actual 22%);
     40% -> 5 lanes, 15 ... 85 step 17.5 (actual 42%); 50% -> 6 lanes, 15 ... 85 step 14 (actual 53%).
   - Each lane sweeps the probe centre along v across the full region (0..100 mm), serpentine between lanes.
     Transitions between lanes and between orientations are NOT recorded (lift-off).
   - For yaw values other than 0/90 (e.g. 45), clip lanes to positions where sim.in_contact(T) is true and
     report the resulting coverage.
   - Expose poses as a function of simulated time at speed_mm_s (fine time step, e.g. 0.05 mm), plus
     planned_lanes() for drawing. Poses via BModeSimulator.pose_from_xy_yaw.
2. mapping/acquisition.py: Acquirer(sim, cfg)
   - feed(T, t, recording): capture a frame when the probe has moved >= frame_spacing_mm (face-centre distance)
     or rotated >= angle_trigger_deg since the last captured frame, and only if recording and in_contact.
     Capture = sim.render(T, return_labels=True) -> FrameRecord with T_true = T_measured = T.
   - Simulated time decoupled from wall clock: if rendering is slow, playback slows; no frames are dropped.
3. App mapping/run_scripted.py (python -m orvue_us_inverse.mapping.run_scripted, plus a menu step; OpenCV
   windows; do not reuse or change the simulator window in simulation/bmode.py):
   - Top view of the 100 x 100 mm region at 4 px/mm: black box by default (anatomy hidden); key a toggles a
     revealed anatomy view (simulation.bmode.top_view or an equivalent projection). Draw planned lanes (thin),
     the probe footprint (30 mm line) at its current pose, and the trace of captured frames (one line per
     frame) so coverage builds up visibly.
   - B-mode window showing the latest captured frame (501 x 301).
   - Status text: case, orientation, lane i/n, frames captured, expected total, actual overlap, sim time.
   - Keys: space pause/resume, + / - playback speed, r restart, s save sweep to SWEEPS_DIR/<case>_<settings>_<time>.npz,
     q quit. CLI: --case normal --yaw 0 90 --overlap 20 --spacing 0.5 --speed 10 --no-images.
4. tests/mapping/test_poses_acquisition.py:
   - Lane union covers x (or y) 0..100 mm completely for overlap 0/20/40/50 at yaw 0 and 90; lane counts and
     actual overlaps match the examples above.
   - All recorded poses satisfy in_contact; captured frames are spaced frame_spacing_mm +/- 10% within a lane;
     the number of frames per lane is about lane length / spacing (about 200 at 0.5 mm).
   - No frames captured during lane transitions.
   - A short sweep saves and loads with the S0 format.
At the end tell me which commands to run to watch a full sweep for yaw 0 and for yaw 0 + 90.
```

### S2 — Reconstruction from oracle labels

```text
SESSION S2: reconstruction of a label volume from a sweep (oracle labels), offline and incremental.

1. mapping/recon.py:
   - VoxelGrid(grid_cfg): shape, world<->index conversion, voxel centres (100 x 100 x 50 mm at 0.5 mm =
     200 x 200 x 100 voxels).
   - pixel_points(T, probe geometry): 3D points of all pixels, same mapping as BModeSimulator.plane_points
     (o + s*u + d*n, s in [-15, 15] mm, d in [0, 50] mm, 0.1 mm pixels), computed from the probe geometry
     stored in the sweep metadata, so reconstruction does not need a simulator instance.
   - LabelCompounder(grid, n_classes=11): per-voxel class vote counts (uint16) and hit count; insert(frame)
     using T_measured, nearest-voxel binning (np.bincount / np.add.at, vectorised); optional elevation splat
     thickness (default 0 = pixel-nearest-neighbour only). Also accumulate a mean-intensity volume when images
     are present.
   - result(): int8 label volume, -1 = unobserved; argmax of votes with a documented deterministic tie rule.
   - fill_small_holes(max_gap_voxels=1): fill unobserved voxels only when surrounded by observed voxels on
     both sides along at least one axis (majority label); never extend into unscanned regions.
   - Performance target: insert() < 15 ms per frame at 0.5 mm voxels on a laptop CPU; print timings in tests.
2. Script mapping/reconstruct_sweep.py <sweep.npz> [--voxel 0.5] [--fill]: reconstruct and save
   recon_<name>.npz (labels, hits, intensity, grid) plus three orthogonal slice PNGs through the centre,
   coloured by class, with the ground-truth slice beside each for visual comparison.
3. tests/mapping/test_recon.py:
   - pixel_points equals sim.plane_points for random poses (simulator from mapping.probe.make_simulator).
   - Ground-truth test: sweep at 0.25 mm spacing over the normal case with exact poses; compare the
     reconstructed label with Anatomy.labels at the voxel centre, within observed voxels:
       interior voxels (all 26 neighbours share one ground-truth label): >= 99.5% correct;
       all observed voxels: >= 95% correct (voxels straddling a tissue boundary may legitimately differ).
     Report the actual figures and the class confusion for the mismatches.
   - Incremental insertion gives exactly the same result as batch insertion.
   - Unobserved voxels stay -1; hole filling never fills outside the observed bounding region.
Report the voxel accuracy, the percentage of the 100 x 100 x 50 mm volume observed, and the timings.
```

### S3 — Live view of the reconstruction

PyVista / VTK cannot run on this machine (Windows Application Control), so the plan no longer uses them. Live views are OpenCV slice images; 3D is a matplotlib off-screen snapshot, plus STL export for the browser viewer or any mesh viewer. (Proposed 2026-10-08; confirm before running S3.)

```text
SESSION S3: live view of the reconstruction during the scripted sweep. No PyVista / VTK (blocked on this
machine). May work, try again. 

1. mapping/render.py:
   - SliceView (OpenCV image, no window in the module): three orthogonal slices through a movable crosshair
     (depth slice seen from above with x right / y down; an x-z slice and a y-z slice with depth downwards) of the
     current label volume, coloured by class: bile lumen 3 green #639922, arterial blood 7 red #E24B4A, venous
     blood 9 blue #378ADD, stone 4 amber #FAC775, lymph node 10 purple #AFA9EC, walls and liver / fat in muted
     greys; unobserved voxels dark. Optional ground-truth contours (Anatomy.labels on the grid).
   - surface_meshes(labels, grid): marching cubes (skimage.measure.marching_cubes) per group above, light
     smoothing, empty groups skipped; vertices in phantom mm. Import viewer3d.geometry helpers if they fit;
     do not edit them.
   - snapshot_3d(meshes, path, gt=None): matplotlib (Agg) off-screen PNG; isometric, equal axes, x right, y down,
     depth downwards, scanning surface at z = 0 on top; ground truth translucent when given.
   - export_stl(meshes, folder): one STL per group in output/export/ (openable in the browser viewer or any
     mesh viewer).
2. Coverage map in the OpenCV top view: per-column hit count (voxel hits summed over depth) as a heat map; key
   v cycles black box / coverage / revealed anatomy.
3. Integrate into mapping/run_scripted.py: the reconstruction is updated incrementally with every captured
   frame; a slice window updated every 0.5 s (mouse click / arrow keys move the crosshair); key 3 writes the
   3D snapshot and STL files.
4. tests/mapping/test_render.py (headless, matplotlib Agg): meshes from a small reconstructed volume lie inside
   the grid; slice images have the expected shapes and colours; updating twice is idempotent; snapshot PNG
   written. Measure the slice update (target <= 50 ms) and full mesh extraction for a normal-case
   reconstruction at 0.5 mm (target <= 300 ms); report both.
Tell me the command to run a full scripted sweep and watch the reconstruction grow. Is the browser based 3d viewer  also an option
```

## 7. Session prompts S4–S7

Revisit these after S4: results from the first sessions may change details such as default spacing or the metrics that matter most.

### S4 — Evaluation and "complete"

```text
SESSION S4: comparison with ground truth, triggered by "complete".

1. mapping/evaluate.py:
   - Ground-truth label volume: Anatomy.labels at the voxel centres of the same grid.
   - Ground-truth instance masks (new code, simulation/anatomy.py unchanged): for every Tube, its lumen (tube.sd < 0) and
     wall (0 <= sd < wall) on the grid; for every Blob, its interior. Keep the structure names.
   - Evaluation mask = observed voxels (after optional hole filling). Report every metric inside this mask, plus
     the coverage of each structure (% of its ground-truth voxels observed).
   - Per tissue class (bile lumen 3, arterial blood 7, venous blood 9, stone 4, lymph node 10, walls 2/5/6/8):
     Dice, IoU, precision, recall; mean surface distance and 95th-percentile Hausdorff distance in mm via
     distance transforms of the boundaries (scipy.ndimage).
   - Per structure (instance): recall of its lumen voxels with the correct class; detected = recall >= 50%.
   - Topology: are the gallbladder lumen and the CHD/CBD lumen connected through bile-lumen voxels in the
     reconstruction (26-connectivity), as they are in the ground truth?
   - Report: RESULTS_DIR/<case>_<time>/ with report.json, report.md (tables), and figures: a 3D overlay
     (ground truth translucent, reconstruction solid, surface coloured by distance to the ground-truth surface
     in mm, off-screen matplotlib snapshot from S3) and a per-structure bar chart of coverage and recall.
2. In run_scripted.py: key c = complete: stop acquisition, finalise (hole filling), run the evaluation, print the
   summary table, open the report folder path, and show the slice view with error colouring and save the 3D overlay PNG.
3. Script mapping/evaluate_sweep.py <sweep.npz> [--voxel 0.5] [--fill] for offline evaluation.
4. tests/mapping/test_evaluate.py:
   - Reconstruction equal to ground truth -> Dice 1.0, surface distances 0, all structures detected, topology OK.
   - Reconstruction shifted by one voxel along x -> mean surface distance about one voxel size (+/- 30%).
   - Removing the cystic duct voxels from the bile class -> topology check reports disconnection.
   - Structures outside the observed region are reported as "not covered", not as missed.
Run the full scripted sweep (yaw 0, then yaw 0 + 90) on the normal case and show me both summary tables.
```

### S5 — Sweep-strategy experiments

```text
SESSION S5: headless parameter study to choose the sweep strategy.

1. mapping/experiments.py and script mapping/run_experiments.py:
   - Grid: frame spacing [0.25, 0.5, 1.0, 2.0] mm; overlap [20, 40, 50] % requested (22, 42, 53 % actual with
     the 30 mm probe); orientations [[0], [90], [0, 90], [0, 45, 90, 135]]; voxel [0.5] (plus 0.25 for the
     default strategy only); cases [normal, parallel_cystic_duct, anterior_cystic_artery].
   - Generate each sweep once and cache it (sweeps are reusable across voxel sizes); reconstruct and evaluate
     with S2 and S4 code; run in parallel processes; no windows. Simulator always from
     mapping.probe.make_simulator().
   - Oracle-only runs need labels, not images: acquire with sim.labels_image(T) instead of sim.render(T)
     (store_images False) and estimate runtime from the Prep timings. The [0] and [90] sweeps are parts of
     [0, 90] and [0, 45, 90, 135]: generate each orientation once and reuse it.
   - Before running, print the estimated number of frames and runtime. If it exceeds 30 minutes, run a reduced
     grid first and ask me before running the full grid.
   - Outputs in RESULTS_DIR/experiments_<time>/: results.csv (one row per run: settings, frames, scan time at
     10 mm/s, per-structure coverage, recall, Dice, MSD, HD95, topology), plots (Dice and HD95 versus spacing per
     structure; versus overlap; versus orientation set; scan time versus quality), and summary.md with a
     recommended default strategy and the reasoning.
2. tests: a tiny grid (2 runs) completes headless and writes the CSV with the expected columns.
Show me the summary table and the recommendation.
```

### S6 — Mouse sweeps

```text
SESSION S6: hand-guided sweeps with the mouse over the hidden region.

1. mapping/poses.py: MousePose source (wall-clock time).
2. App mapping/run_mouse.py (OpenCV top view + B-mode + slice view from S3; menu step):
   - Black-box top view; mouse position = probe face centre; hold the left button to record.
   - Yaw: q/e rotate by 5 degrees; 0 and 9 snap to 0 and 90 degrees. Yaw is locked while recording unless key l
     unlocks it.
   - Lane guides from the chosen strategy (default from S5, CLI --overlap --yaw), drawn as thin bands; the
     current lane highlighted.
   - Speed meter: maximum speed = frame_spacing_mm x measured capture rate (frames/s). Green below 80% of the
     maximum, amber up to 100%, red above. Frames are still captured when too fast, but gaps larger than
     2 x frame spacing are marked on the coverage map.
   - Coverage map with unscanned holes highlighted; percentage covered shown.
   - Keys: c complete (S4 evaluation), s save sweep, u undo the last stroke, r reset, a reveal anatomy, q quit.
3. tests: synthetic mouse event sequences check recording gating (button), yaw lock, speed classification,
   gap detection, and undo.
Give me a short how-to for scanning the region well by hand.
```

### S7 — Camera-tracked probe and pose-error injection

```text
SESSION S7: the camera-tracked dummy probe as pose source, and a study of tracking error.

1. mapping/poses.py: TrackedPose source wrapping ProbeTracker.get_pose() (tracking/tracker.py); returns
   None when tracking is invalid (no capture then). get_pose() is the one-euro-filtered pose, so the filter adds
   its own lag on top of the camera latency: measure it and report it.
2. mapping/errors.py: PoseErrorModel producing T_measured from T_true:
   Gaussian jitter (position sigma mm, yaw/tilt sigma deg), constant bias (offset mm, yaw deg), latency
   (T_measured = the true pose from latency_ms earlier), and scale error (%). Default: no error.
   Explain in the docstring: when the simulator renders from the tracked pose, image and pose are always
   consistent, so real tracking error is invisible unless it is injected; with a real probe, the image comes from
   the true position and tracking error degrades the reconstruction.
3. App mapping/run_tracked.py: same interface as run_mouse.py, pose from the tracker; record while space is held;
   tracking status (valid, reference markers seen, reprojection error) shown; c complete, s save.
   Optional --inject jitter=0.5,yaw=0.5,latency=50 applies the error model live.
4. Extend mapping/run_experiments.py with an error grid on the default strategy: position jitter [0, 0.5, 1, 2] mm,
   yaw jitter [0, 0.5, 1, 2] deg, latency [0, 33, 100] ms at 10 mm/s, bias [0, 1] mm. Plots of Dice, HD95 and
   topology versus error, and summary.md stating the tracking accuracy needed to keep the cystic duct and CBD
   reconstructed and connected.
5. tests: zero error gives identical reconstructions; latency shifts poses by speed x latency; bias shifts the
   reconstruction by the bias (within one voxel).
Tell me the order of commands to: calibrate the tracker, run a tracked sweep, and run the error study.
```

## 8. Working between sessions

### Review checklist after each session

- The tests pass, and the summary lists what each test checks. Skim one test to confirm it tests something meaningful, not just that code runs.
- The copied simulator files (`simulation/`, `tracking/`, `ui/`, `viewer3d/`) are unchanged (`git diff --stat`), and `UPSTREAM.md` logs any agreed exception.
- `STATUS.md` was updated: session state, test counts, timings, deviations.
- Run the app or script the session produced and look at it. The visual checks catch convention errors that tests can miss: lanes in the right place, x to the right and y downwards in every view, surfaces where the revealed anatomy says they should be.
- Read the "deviations" part of the summary. If Claude Code changed the plan, decide whether to accept it before moving on, and update this document if you do.
- Note the timings it reports. If frame insertion or surface updates are slower than the targets, deal with it before adding the next layer.
- Keep the saved sweeps from each session: they let later sessions be compared on identical data.

### Things to watch for

| Symptom | Likely cause |
| --- | --- |
| Reconstruction mirrored or rotated relative to the revealed anatomy | Axis or yaw convention mismatch between poses, pixel\_points and the grid |
| Stripes or gaps between frames | Frame spacing larger than intended, or capture missing frames during slow rendering |
| Smeared or doubled structures | Persistence not 0, or pose and image out of step |
| Seams between lanes | Lane overlap computed incorrectly, or yaw changing within a lane |
| Small vessels broken into pieces | Spacing or voxel size too coarse; expected at 1–2 mm spacing |
| Evaluation punishes unscanned areas | Metrics computed outside the observed mask |
| `ImportError` or blocked DLL for vtk / pyvista | PyVista / VTK are blocked by Windows Application Control; use the S3 slice views and matplotlib snapshots |
| Low Dice for walls (labels 2, 5, 6, 8) even with perfect poses | Walls of 0.3–0.6 mm are thinner than a 0.5 mm voxel; judge walls by surface distance, lumens by Dice |

### After S7

Natural next steps are classical recognition (detecting anechoic structures from the B-mode intensity instead of oracle labels, evaluated against the oracle upper bound), Doppler-based vessel labelling, and replacing the simulator with a real probe as the image source using the same sweep format.


## 9. Revision log

**Oct 8, 2026: standalone repository.** The simulator code was copied into this repository (package `orvue_us_inverse`; source commit `e789389`; details in `UPSTREAM.md`). Changes to this plan:

- Layout: new code in `src/orvue_us_inverse/mapping/`, apps as modules with menu steps, tests in `tests/mapping/`, data in `output/sweeps` and `output/results` via `paths.py`. Replaces the flat `inverse_mapping/`, root apps and root data folders.
- Read-only files: all copied simulator files (`simulation/`, `tracking/`, `ui/`, `viewer3d/`), with exceptions logged in `UPSTREAM.md`. Replaces the three flat file names.
- Preparation rewritten: the template, `CLAUDE.md` with the simulator conventions and the passing tests already exist; the probe goes in `mapping/probe.py` with every field stated.
- `STATUS.md` added and updated at the end of every session; `README_inverse_mapping.md` dropped (one README rule).
- S3 and the S4 3D overlay: PyVista / VTK are blocked by Windows Application Control. Replaced by OpenCV slice views, matplotlib off-screen snapshots and STL export (proposed; confirm before S3).
- Facts corrected from the code: the portal vein reaches about 55.4 mm (not 53 mm); oracle labels come from the centre plane on a 0.2 mm grid; B-mode noise is unseeded, so only poses and labels are reproducible.
- S5: oracle-only sweeps use `labels_image()` instead of `render()`, and orientation sweeps are reused across orientation sets.
- S7: `get_pose()` returns the filtered pose; its lag is measured.
- Things to watch: rows for blocked VTK and for thin walls below the voxel size.

**Oct 9, 2026: after S7, clinical version and review fixes.**

- Beyond the plan, on request: a clinical version (`clinical/`): `python -m orvue_us_inverse` opens one window, the
  Ultrasound Imaging Simulator, with B-MODE and INVERSE MAPPING tabs (scripted / mouse / camera-probe sources), a case
  selector, the probe calibration and an AR overlay of the reconstruction on the camera view; the session apps stay
  as developer tools (`python -m orvue_us_inverse dev`). Merged with S7 (PR #13).
- AR overlay review: the overlay is a surface map (vertical projection to z = 0), not a perspective rendering; the
  shallowest-structure fill alone hid covered vessels (cystic artery under the gallbladder), so bile and artery
  outlines are drawn at any depth (dashed when covered, 2.5 mm band); tested on the normal case's ground truth
  (>= 95 % of the cystic artery footprints shown). A fixed chip states that the segmentation is the oracle's
  (simulator labels), until recognition replaces it.
- AR registration tested on synthetic camera poses (straight and 15 deg tilt: < 0.5 px); the clinical window's frame
  time measured (8.9 frames/s recording with B-mode and AR, 15.4 without B-mode; the AR warp costs ~34 ms a frame).
