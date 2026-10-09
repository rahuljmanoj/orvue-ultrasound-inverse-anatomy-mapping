# Ultrasound Inverse Anatomy Mapping

Orvue Surgical: inverse anatomy mapping for ultrasound. Goal: reconstruct the 3D segmented anatomy from tracked
2D B-mode frames (the inverse of the Ultrasound Imaging Simulator), then recognise the anatomy. Focus: the
gallbladder / Calot's triangle anatomy of the Ultrasound Imaging Simulator, whose 8 anatomy cases, B-mode renderer
and probe tracking provide the frames, the poses and the ground truth.

The repository is standalone: the simulator code (`simulation/`, `tracking/`, `ui/`, `viewer3d/`) was copied into
this package from [orvue-ultrasound-simulator](https://github.com/rahuljmanoj/orvue-ultrasound-simulator) at commit
`e789389`. The copied files are frozen baselines; their origin, the old -> new path table and every change since
the copy are in [UPSTREAM.md](UPSTREAM.md).

Built from the Orvue Surgical Python project template: an installable package in `src/`, one entry point with a
menu, every file location in `paths.py`, committed configuration in `config/`, generated PDFs in `docs/` and
run-time output in the gitignored `output/`.

## Contents
- [Start a new project from this template](#start-a-new-project-from-this-template)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Scanning by hand (mouse sweep)](#scanning-by-hand-mouse-sweep)
- [Project layout](#project-layout)
- [Rules](#rules)
- [Tests](#tests)

## Start a new project from this template
1. On GitHub: **Use this template → Create a new repository**, then clone it.
2. Rename the package and the title (once):
   ```
   python scripts/rename_project.py <package_name> "<Project Title>"
   ```
3. Create or pick a conda environment, install, and run the tests (see [Setup](#setup)).
4. Replace `core/example.py`, `config/settings.json` and `tests/test_template.py` with the project's own; update
   this README (description, every-file table), `CLAUDE.md` and `reports/manual.py`.
5. In PyCharm: set the interpreter to the conda environment; if the project name shows a different name in
   brackets, use **Refactor → Rename → Rename module** on the project folder.

## Setup
Python 3.10+, conda environment `orvue-robot`. From this repository folder:
```
pip install -r requirements.txt
pip install -e .
```
`pip install -e .` installs the package in editable mode: imports and `python -m orvue_us_inverse` work from any
folder and code changes take effect without reinstalling. Re-run it only after changing `pyproject.toml`, moving
the folder or creating a new environment. The Intel RealSense D405 camera tracking needs the optional extra
`pip install -e .[camera]` (pyrealsense2, also in `requirements.txt`); everything else, including the tests,
runs without it.

## Quick start
```
python -m orvue_us_inverse              menu (steps in the order of use, 0 = exit)
python -m orvue_us_inverse viewer       check the probe tracking (D405)
python -m orvue_us_inverse calibrate    probe calibration -> config/calibration.json
python -m orvue_us_inverse sim [case]   Ultrasound Imaging Simulator ([--track] [--cam-view] [--mouse])
python -m orvue_us_inverse scripted     scripted sweep with the reconstruction growing live ([--case X]
                                        [--yaw 0 90] [--overlap 20] [--spacing 0.25] [--speed 10] [--no-images]
                                        [--live3d]); s saves to output/sweeps/, 3 snapshot + STL, b browser 3D,
                                        c complete: evaluate -> output/results/<case>_<time>/ report
python -m orvue_us_inverse mouse        hand-guided sweep, one window: sweep | B-mode | 3D reconstruction; hold
                                        the left button to record, i B-mode on / off, speed meter, coverage holes,
                                        u undo, c evaluate ([--case X] [--yaw 0] [--overlap 20] [--spacing 0.25]
                                        [--no-images] = start with B-mode off)
python -m orvue_us_inverse recon        reconstruct a sweep ([sweep.npz, default newest] [--voxel 0.5] [--fill])
                                        -> output/results/recon_<name>.npz + 3 slice PNGs vs ground truth
python -m orvue_us_inverse evaluate     evaluate a sweep against the ground truth ([sweep.npz] [--voxel 0.5]
                                        [--fill]) -> output/results/<case>_<time>/ (report.md, .json, figures)
python -m orvue_us_inverse experiments  sweep-strategy study, headless, parallel ([--workers N] [--quick]
                                        [--yes]) -> output/results/experiments_<time>/ (results.csv, plots,
                                        summary.md); sweeps cached in output/cache/experiments/
python -m orvue_us_inverse run          example step -> output/logs/example.txt
python -m orvue_us_inverse test         all tests
python -m orvue_us_inverse board        -> docs/print/tracking_board.pdf
python -m orvue_us_inverse anatomy      3D anatomy viewer in the browser ([--case X] [--export])
python -m orvue_us_inverse manual       -> docs/Ultrasound Inverse Anatomy Mapping - User Manual.pdf
```
Frames and ground truth in code: `mapping/probe.py` is the only probe source; `make_simulator` always uses
`persistence=0`, so each frame belongs to its own pose:
```python
from orvue_us_inverse.mapping.probe import make_simulator
sim = make_simulator("normal")
img, lab = sim.render(sim.pose_from_xy_yaw(50, 60, 0), return_labels=True)   # 501 x 301 uint8 / int8 labels
lab = sim.labels_image(sim.pose_from_xy_yaw(50, 60, 0))                      # labels only (oracle), ~6x faster
```
The console script `orvue-us-inverse` does the same as `python -m orvue_us_inverse`.

## Scanning by hand (mouse sweep)

`python -m orvue_us_inverse mouse` (menu 5). The mouse position is the probe face centre; hold the left button to
record. One window: the sweep (left), the latest frame (middle), the 3D reconstruction (right; drag to rotate, wheel
to zoom), status and speed underneath. The middle panel's button "B-mode ON / OFF" (or key i) switches the B-mode
simulation: off computes only the oracle labels (~13 ms instead of ~78 ms per frame), shown in tissue colours with a
key, and about doubles the speed limit; the reconstruction and the evaluation use the labels either way. Switch it
on where you want to see the ultrasound image; `--no-images` starts with it off.

1. Set the probe angle first (key 0), start at one end of a lane band, hold the button and move slowly along the band to
   the other end; release. Keep the speed meter green: the maximum is frame spacing x capture rate (0.25 mm x
   ~20 frames/s with labels only = 5 mm/s; with B-mode images about half).
2. Scan the next band the same way in the opposite direction, following the highlighted band (20 % overlap).
3. Check the coverage map (key under it): black = not scanned, dark purple -> yellow = imaged once -> many times,
   cyan = hole (unscanned area enclosed by scanned area), red lines = speed gaps. Rescan holes, gaps and black
   patches over the anatomy with short strokes, or press u to undo a bad stroke.
4. Optionally add a 90 degree pass (key 9) where the first pass had gaps. The mouse wheel over the sweep or q / e
   turn the probe in 5 degree steps, any time (keep the angle constant within a stroke for straight lanes).
5. Watch the 3D panel: it is rebuilt after every stroke (g adds the true anatomy translucent). The buttons Iso /
   Top / Axial (from the feet) / Sagittal (from the patient's right), or v, set the view; the box faces are
   labelled patient R / L, cranial / caudal, anterior (surface) / posterior. b opens the browser 3D view, 3
   writes a snapshot and STL files.
6. Press c: the report shows which structures were detected, missed or not covered; c also fills 1-voxel gaps
   between frames in the 3D volume (not the holes on the map). s saves the sweep, Esc quits.

## Project layout
```
config/                  committed configuration (settings.json, calibration.json)
docs/                    generated PDFs (manual), figures/, images/, print/ (tracking board)
scripts/                 helper scripts outside the package
src/orvue_us_inverse/    the package: core/, reports/, mapping/ (inverse mapping) and the copied simulator:
                         simulation/, tracking/, ui/, viewer3d/
tests/                   pytest tests, helpers/ for synthetic test data, mapping/ for inverse mapping
output/                  generated at run time, gitignored: captures/, logs/, export/, viewer3d/, cache/,
                         sweeps/, results/
```

### Every file

| File | Purpose / use |
|---|---|
| `.gitignore` | Ignores PyCharm, Python caches, packaging output and `output/` |
| `pyproject.toml` | Package metadata, dependencies, console script, pytest options (`src` layout) |
| `requirements.txt` | Pinned versions of the working environment |
| `README.md` | This file: the only README |
| `CLAUDE.md` | Conventions for Claude Code sessions |
| `UPSTREAM.md` | Origin of the copied simulator code (repository, commit), old -> new paths, change log |
| `PLAN_inverse_mapping.md` | Session plan for the inverse mapping (goal, principles, sessions, Claude Code prompts) |
| `STATUS.md` | Project state, updated at the end of every session (sessions done, evidence, known issues) |
| `config/settings.json` | Example settings, read by `core/example.py` |
| `config/calibration.json` | Probe calibration of the current probe build, written by `tracking/calibrate.py` |
| `docs/.gitkeep` | Keeps `docs/` in git until the first PDF is generated |
| `scripts/rename_project.py` | Turns the template into a new project (package name and title) |
| `src/orvue_us_inverse/__init__.py` | Package docstring and `__version__` |
| `src/orvue_us_inverse/__main__.py` | Entry point: menu and `python -m orvue_us_inverse <command>` |
| `src/orvue_us_inverse/paths.py` | Every file and folder location |
| `src/orvue_us_inverse/assets/orvue_logo.jpg` | Orvue Surgical logo (PDF header) |
| `src/orvue_us_inverse/core/__init__.py` | Example area sub-package |
| `src/orvue_us_inverse/core/example.py` | Example module: reads the settings, writes a log |
| `src/orvue_us_inverse/reports/__init__.py` | Reports sub-package |
| `src/orvue_us_inverse/reports/pdf_template.py` | Orvue Surgical PDF template (`Doc`), used by every generated PDF |
| `src/orvue_us_inverse/reports/manual.py` | Builds the user manual PDF from the code |
| `src/orvue_us_inverse/mapping/__init__.py` | Inverse-mapping sub-package (reconstruction, recognition) |
| `src/orvue_us_inverse/mapping/probe.py` | The only probe source: `PROBE` (every field stated), `FRAME_SHAPE`, `make_simulator` (persistence 0), `simulator_settings`, `probe_metadata` (Prep) |
| `src/orvue_us_inverse/mapping/config.py` | Settings dataclasses: `GridConfig` (bounds, 0.5 mm voxels, shape, centres), `SweepConfig`, `AcquisitionConfig` (S0) |
| `src/orvue_us_inverse/mapping/sweep_io.py` | `FrameRecord`, `Sweep` (save / load compressed .npz), `make_metadata` (probe, settings, provenance), `estimate_size_mb` (S0) |
| `src/orvue_us_inverse/mapping/poses.py` | `ScriptedSweep`: serpentine lanes per yaw, overlap, poses on simulated time, coverage (S1); `MousePose`: mouse-driven pose, angle, speed (S6); camera (S7) to follow |
| `src/orvue_us_inverse/mapping/run_mouse.py` | App: hand-guided mouse sweep in one window (sweep, B-mode, 3D) with lane guides, speed meter, coverage holes, speed gaps, undo, complete (S6) |
| `src/orvue_us_inverse/mapping/acquisition.py` | `Acquirer`: distance / angle-triggered capture of frames into a `Sweep` (S1) |
| `src/orvue_us_inverse/mapping/run_scripted.py` | App: scripted sweep over the hidden box, top view with lanes and coverage trace, B-mode, save (S1) |
| `src/orvue_us_inverse/mapping/recon.py` | `VoxelGrid`, `pixel_points`, `LabelCompounder` (votes, hits, intensity), `fill_small_holes`, ground-truth helpers (S2) |
| `src/orvue_us_inverse/mapping/reconstruct_sweep.py` | Script: saved sweep -> label volume and centre slices vs ground truth in `output/results/` (S2) |
| `src/orvue_us_inverse/mapping/render.py` | `SliceView` (3 slices through a crosshair), `surface_meshes`, `snapshot_3d` (matplotlib), `export_stl`, browser 3D page (S3) |
| `src/orvue_us_inverse/mapping/live3d.py` | Optional live PyVista 3D window of the reconstruction (`pip install -e .[view3d]`) (S3) |
| `src/orvue_us_inverse/mapping/evaluate.py` | Ground-truth instance masks, class metrics (Dice, IoU, precision, recall, MSD, HD95), structure detection, bile topology, report (S4) |
| `src/orvue_us_inverse/mapping/evaluate_sweep.py` | Script: saved sweep -> reconstruction -> evaluation report in `output/results/<case>_<time>/` (S4) |
| `src/orvue_us_inverse/mapping/experiments.py` | Sweep-strategy study: grid, cached oracle sweeps, parallel runs, per-structure metrics, CSV, plots, summary and recommendation (S5) |
| `src/orvue_us_inverse/mapping/run_experiments.py` | Script: runtime estimate, full or reduced grid, prints the summary table and recommendation (S5) |
| `src/orvue_us_inverse/mapping/errors.py` | Pose-error injection (jitter, bias, latency, scale); placeholder until S7 |
| `src/orvue_us_inverse/simulation/__init__.py` | Simulation sub-package (copied) |
| `src/orvue_us_inverse/simulation/anatomy.py` | Virtual anatomy: tissue table (labels 0-10), tubes / blobs, the 8 cases, `validate` (copied, read-only) |
| `src/orvue_us_inverse/simulation/bmode.py` | B-mode simulator `BModeSimulator`, simulator window and demo (copied, read-only) |
| `src/orvue_us_inverse/tracking/__init__.py` | Tracking sub-package (copied) |
| `src/orvue_us_inverse/tracking/markers.py` | ArUco marker layout, frames, OpenCV boards, pose helpers (copied; layout constants read-only) |
| `src/orvue_us_inverse/tracking/tracker.py` | `ProbeTracker`: D405 / file capture, probe pose, filters, calibration (copied, read-only) |
| `src/orvue_us_inverse/tracking/calibrate.py` | Probe calibration (yaw offset, face position) -> `config/calibration.json` (copied) |
| `src/orvue_us_inverse/tracking/viewer.py` | Live tracking check window, overlay, camera zoom, CSV log (copied) |
| `src/orvue_us_inverse/tracking/board.py` | Writes `docs/print/tracking_board.pdf` (copied) |
| `src/orvue_us_inverse/ui/__init__.py` | UI sub-package (copied) |
| `src/orvue_us_inverse/ui/clinical.py` | Shared clinical window style for the simulator and tracking windows (copied) |
| `src/orvue_us_inverse/viewer3d/__init__.py` | 3D anatomy viewer sub-package (copied) |
| `src/orvue_us_inverse/viewer3d/__main__.py` | `python -m orvue_us_inverse.viewer3d` (copied) |
| `src/orvue_us_inverse/viewer3d/geometry.py` | Triangle meshes of the anatomy, mesh cache, STL / VTP export (copied) |
| `src/orvue_us_inverse/viewer3d/info.py` | Structure names, colours and descriptions for the viewer (copied) |
| `src/orvue_us_inverse/viewer3d/viewer.py` | Builds and opens the three.js viewer page, headless screenshots (copied) |
| `src/orvue_us_inverse/viewer3d/template.html` | three.js viewer page template (copied) |
| `tests/__init__.py` | Makes `tests` a package (for `tests.helpers`) |
| `tests/helpers/__init__.py` | Helpers for tests (synthetic data) |
| `tests/helpers/synthetic_scene.py` | Synthetic D405-like camera frames of the board and probe markers (copied) |
| `tests/test_template.py` | Tests of paths, settings, example, entry point and manual |
| `tests/test_standalone.py` | No file under `src/` or `tests/` refers to the simulator's original package |
| `tests/test_anatomy.py` | Anatomy tests: case geometry and collisions, renderer ground-truth labels (copied) |
| `tests/test_tracking.py` | Tracking tests on synthetic images: conventions, filters, calibration (copied) |
| `tests/mapping/__init__.py` | Inverse-mapping tests package |
| `tests/mapping/test_probe.py` | Probe fields, frame and label shapes 501 x 301, persistence 0, simulator settings |
| `tests/mapping/test_sweep_io.py` | Sweep save / load round trip (with and without images), metadata and provenance, size estimate, configs |
| `tests/mapping/test_poses_acquisition.py` | Lane layout and overlap, coverage, triggers over full sweeps, no capture in transitions, save / load |
| `tests/mapping/test_run_scripted.py` | Scripted-sweep app without windows: stepping, drawing, file name, save; menu step |
| `tests/mapping/test_evaluate.py` | Perfect reconstruction, one-voxel shift, cut cystic duct, not covered vs missed, report files, script, app complete |
| `tests/mapping/test_experiments.py` | Grid and sweep reuse, runtime estimate, a 2-run grid headless with CSV columns, plots, summary, cache reuse |
| `tests/mapping/test_mouse.py` | Mouse sweeps with synthetic events: recording gating, turning (keys, wheel), window routing, 3D view presets, speed classes, capture rate, gaps, undo, coverage holes, 3D panel, complete |
| `tests/mapping/test_render.py` | Slice shapes / colours / idempotence, meshes inside the grid, timing, snapshot, STL, browser page, app outputs, PyVista off-screen |
| `tests/mapping/test_recon.py` | `pixel_points` = `plane_points`, accuracy against the ground truth, incremental = batch, hole filling, script |

## Rules
- Absolute imports only (`from orvue_us_inverse.core.example import scaled`); no `sys.path` manipulation.
- Every file location goes in `paths.py`; run-time output under `output/`; committed configuration under `config/`.
- One sub-package per area (`core/`, `simulation/`, `tracking/` ...); each module runnable with `python -m orvue_us_inverse.<area>.<module>`;
  add new steps to `COMMANDS` and `MENU` in `__main__.py`.
- The copied simulator files are frozen baselines (`simulation/anatomy.py`, `simulation/bmode.py`,
  `tracking/markers.py` layout constants and `tracking/tracker.py` read-only); log any change in `UPSTREAM.md`.
- One README.md: add, move or remove the row in "Every file" whenever a tracked file is added, moved or removed.
- Documentation as PDF built with `reports.pdf_template.Doc` from the code; no Word documents. Keep README, manual
  and CLAUDE.md in step.

## Tests
```
python -m pytest tests
```
from the repository folder (pyproject sets `-q` and puts `src` and `.` on the path). Tests must run without hardware.
