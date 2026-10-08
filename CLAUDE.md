# CLAUDE.md

Conventions a new session must know.

## Project
Ultrasound Inverse Anatomy Mapping for Orvue Surgical. Goal ("inverse mapping"): reconstruct the 3D segmented
anatomy from tracked 2D B-mode frames, then recognise the anatomy. Stages: (1) reconstruction (tracked frames ->
3D volume / segmented label volume), (2) recognition (which structures: gallbladder, cystic duct, CBD, cystic
artery, RHA, Calot's node ...). Focus: the gallbladder / Calot's triangle anatomy of the Ultrasound Imaging
Simulator.

The repository is standalone. The simulator (anatomy, B-mode renderer, probe tracking, clinical UI, 3D viewer) was
copied into this package from the simulator repository (rahuljmanoj/orvue-ultrasound-simulator, commit `e789389`);
origin, path table and every change since the copy are in `UPSTREAM.md`. Nothing here imports the simulator's
own package (`tests/test_standalone.py` checks it); the sibling folder `../Ultrasound_Simulator` is not used and is
never edited from this project. Typical use:
`from orvue_us_inverse.simulation.anatomy import build_case`;
`from orvue_us_inverse.simulation.bmode import BModeSimulator`;
`sim = BModeSimulator(build_case("normal"), persistence=0.0)`;
`img, lab = sim.render(T, return_labels=True)` gives the B-mode image and its ground-truth labels for a pose `T`.

## Layout
One installable package `orvue_us_inverse` in `src/` (`pyproject.toml`; installed with `pip install -e .`). Absolute
imports only (`from orvue_us_inverse.simulation.bmode import BModeSimulator`); no `sys.path` manipulation. Every file
location comes from `orvue_us_inverse.paths`.

```
config/settings.json         settings (committed)
config/calibration.json      probe calibration of the current probe build (committed, copied from the simulator)
docs/                        generated PDFs, figures/, images/, print/tracking_board.pdf
scripts/rename_project.py    turns the template into a new project
src/orvue_us_inverse/        the package (below)
tests/                       test_template.py, test_standalone.py, test_anatomy.py, test_tracking.py,
                             helpers/synthetic_scene.py, mapping/test_probe.py,
                             mapping/test_sweep_io.py, mapping/test_poses_acquisition.py,
                             mapping/test_run_scripted.py
output/                      run-time output, gitignored: captures/, logs/, export/, viewer3d/, cache/, sweeps/,
                             results/
PLAN_inverse_mapping.md      session plan for the inverse mapping
STATUS.md                    project state, updated at the end of every session
UPSTREAM.md                  origin of the copied simulator code and its change log
```

## Files (`src/orvue_us_inverse/`)
- `__main__.py`: single entry point; menu (steps in the order of use, 0 = exit) or
  `python -m orvue_us_inverse <command>` (`viewer`, `calibrate`, `sim`, `scripted`, `run`, `test`, `board`,
  `anatomy`, `manual`); console script `orvue-us-inverse`. Runs `python -m <module>` subprocesses from the repository folder
  (stdin = DEVNULL, 1 s pause so the D405 is released). New steps: `COMMANDS` + `MENU`.
- `paths.py`: REPO_ROOT, PACKAGE_DIR, LOGO_PATH, SETTINGS_PATH, CALIBRATION_PATH, UPSTREAM_PATH, DOCS_DIR, FIGURES_DIR,
  IMAGES_DIR, PRINT_DIR, MANUAL_PDF, BOARD_PDF, OUTPUT_DIR, CAPTURES_DIR, LOGS_DIR, EXPORT_DIR, VIEWER3D_OUT_DIR,
  CACHE_DIR, SWEEPS_DIR, RESULTS_DIR.
- `core/example.py`: example area module (`load_settings`, `scaled`, `main`).
- `mapping/`: inverse mapping (see Inverse mapping): `probe.py`, `config.py`, `sweep_io.py`, `poses.py`,
  `acquisition.py`, app `run_scripted.py`; placeholders `recon.py`, `render.py`, `evaluate.py`, `experiments.py`,
  `errors.py`.
- Copied from the simulator (`UPSTREAM.md`):
  - `simulation/anatomy.py`: tissue table `TISSUES` (labels 0-10), `Tube` / `Blob`, `Anatomy`, `build_case`,
    `CASES`, `CONNECTED`, `validate()`, `calot_triangle()`. All geometry lives here.
  - `simulation/bmode.py`: `LinearProbe`, `BModeSimulator` (labels -> scatterers -> PSF -> TGC -> log compression),
    `clinical_view`, `SliderPanel`, `top_view`, the all-in-one window `simulator_window`, `demo()`, `main()`.
  - `ui/clinical.py`: shared clinical window style (palette, header, panel, chips, sliders, mode tabs, button,
    patient directions, `phantom_map`, `compose_tracking`).
  - `tracking/markers.py`: SINGLE SOURCE OF TRUTH for the marker layout, frames, OpenCV boards and pose helpers.
  - `tracking/tracker.py`: `ProbeTracker` (background D405 / file capture, pose, filters, calibration offsets).
  - `tracking/viewer.py`: live tracking check window, `annotate` overlay, `CameraZoom` (shared `ZOOM`), CSV logger.
  - `tracking/calibrate.py`: yaw-offset + face-position calibration, writes `config/calibration.json`.
  - `tracking/board.py`: writes `docs/print/tracking_board.pdf` (the sheet to print).
  - `viewer3d/`: 3D anatomy viewer in the browser: `geometry.py` (marching-cubes meshes, cache, STL/VTP export),
    `info.py` (names, colours, descriptions), `viewer.py` (HTML payload, open page, headless Edge screenshots),
    `template.html` (three.js page), `__main__.py`.
- `reports/pdf_template.py`: the Orvue Surgical PDF template (`Doc`); every generated PDF uses it.
- `reports/manual.py`: writes the user manual PDF from the code.
- `assets/orvue_logo.jpg` (window headers, PDFs, 3D viewer).

## Frames and conventions (from the simulator; keep unchanged)
- Phantom frame (mm, right-handed): origin at the top-left corner of the printed 100 x 100 mm region on the
  surface; +x to the right on the print = patient's left (medial); +y down the print = caudal; +z into the table
  = depth / posterior. z = 0 is the gel surface (the printed sheet lies at z = +H_MM). Anatomy volume
  `size_xyz = (100, 100, 50)`.
- The liver lies ANTERIOR to the gallbladder (smaller z) over the GB bed and hilum; liver edge at y ~ 52 mm.
- Probe frame: origin at the centre of the contact face; +x along the array (away from the orientation mark, which
  is at the -x end = image left = screen dot); +y across the face (elevation); +z out of the face into tissue
  (beam axis). The marker platform is at z = -PLATFORM_HEIGHT_MM.
- Probe pose `T` (4x4, phantom <- probe), columns: `T[:3,0]` = u lateral, `T[:3,1]` = v elevation,
  `T[:3,2]` = n axial (down), `T[:3,3]` = o face centre. `pose_from_xy_yaw` sets u = (cos, sin, 0),
  v = (-sin, cos, 0), n = (0, 0, 1).
- Image: `render(T)` returns 501 x 301 (rows = depth nz, columns = lateral nx) at `px_mm = 0.1` for the default
  probe (50 mm depth, 30 mm width); `labels_image` / `return_labels=True` give the int8 labels on the same grid.
- Yaw: `atan2(u_y, u_x)` in degrees, from +x toward +y (counter-clockwise in phantom x-y, which is clockwise when
  seen from the camera above: `tracking.tracker.YAW_SIGN_CCW_FROM_CAMERA = -1`). `tracking.markers.xy_yaw(T)`
  returns `(x, y, yaw_deg)` for `pose_from_xy_yaw`.
- Orientation dot: the amber screen dot marks the probe's -x end = image left; every map marks the same end with an
  amber dot; image-side labels from `ui.clinical.image_sides(yaw)`. ID 0 is mounted with its printed +x arrow
  pointing AWAY from the probe's orientation mark.
- Tracker interface: the simulator only needs `T_phantom_probe` (`= inv(T_cam_phantom) @ T_cam_probe`).
- Calibration: `T = T_raw @ Rz(yaw_offset_deg) @ Trans(dx, dy, dz)` (offsets in the probe frame).
- `Tube.sd(P)` = distance to the lumen surface (negative inside); the wall is `0 <= sd < wall`.

### Markers and board (`tracking/markers.py`)
- Dictionary `cv2.aruco.DICT_4X4_50`; corner refinement subpixel.
- Reference board: IDs 1-4, `REF_MARKER_SIZE_MM = 24.0`, `REF_QUIET_ZONE_MM = 5.0`, `REGION_MM = 100.0`,
  top-left corners in the phantom frame `REF_MARKERS_TL = {1: (-30, 0), 2: (106, 0), 3: (106, 76), 4: (-30, 76)}`;
  sheet `SHEET_TL = (-35, -5)`, `SHEET_SIZE = (170, 110)`.
- Probe: `PROBE_MARKER_ID = 0`, `PROBE_MARKER_SIZE_MM = 30.0` (40 x 40 mm platform), centred over the face at
  z = -PLATFORM_HEIGHT_MM, marker 'right' = probe +x. Optional `SECOND_MARKER_ID = 5`, 20 mm, centred at
  x = SECOND_MARKER_OFFSET_X_MM, used only when `USE_SECOND_MARKER`.
- Layout constants are fixed (they match the printed sheet). `H_MM`, `PLATFORM_HEIGHT_MM`,
  `SECOND_MARKER_OFFSET_X_MM`, `USE_SECOND_MARKER` are the user's measured values: change only when told.

## Simulator interfaces (exact signatures at `e789389`)

`orvue_us_inverse.simulation.bmode`
```python
class LinearProbe:  # dataclass
    LinearProbe(f0_mhz: float = 7.5, width_mm: float = 30.0, depth_mm: float = 50.0, px_mm: float = 0.1,
                fnum: float = 3.0, cycles: float = 2.5, elev_sigma_mm: float = 0.5, focus_mm: float = 20.0,
                rayleigh_mm: float = 10.0, c_mm_us: float = 1.54)

class BModeSimulator:
    def __init__(self, anatomy, probe=LinearProbe(), dyn_range_db=50.0, gain_db=0.0, noise_db=-42.0,
                 tgc_db_cm_mhz=0.5, specular_gain=4.0, n_elev=5, compound_deg=(-7.0, 0.0, 7.0), band_mm=4.0,
                 persistence=0.3, gamma=1.25, point_gain=5.0, sri=0.6)
    def render(self, T, return_labels=False)      # uint8 image (nz x nx = 501 x 301), or (img, labels)
    def plane_points(self, T, elev=0.0, coarse=False)   # phantom-frame points of the image plane
    def labels_image(self, T)                      # tissue labels on the image plane
    @staticmethod
    def pose_from_xy_yaw(x, y, yaw_deg, z=0.0)     # probe perpendicular to the surface
    def in_contact(self, T, tol_mm=1.0, max_tilt_deg=30.0)
    def set_tgc(self, db_cm_mhz)
    def set_focus(self, focus_mm)
    def ground_truth_overlay(self, img, lab)

def top_view(anatomy, px_mm=0.25, depth_mm=50.0, dz=0.5, palette=COL_TAB)
def draw_depth_legend(img, depth_mm=50.0, colour_lab=5)
def draw_calot_triangle(img, anatomy, px_per_mm, colour=(255, 255, 255))
def clinical_view(us, sim, status_lines=(), sliders=None, panel_w=200, scale_w=70, top_h=56, bottom_h=16, logo_h=40)
def demo(case='normal', tracker=None, cam_view=False, camera_control=True, scene_source=camera_scene)
def main(argv=None)
CONTROL_RANGES = {"gain": (-30.0, 30.0), "dr": (30.0, 90.0), "pers": (0.0, 0.9), "tgc": (0.0, 1.5)}
```

`orvue_us_inverse.simulation.anatomy`
```python
Tube(name, pts, r_lumen, wall, lumen_label, wall_label)   # .sd(P), .bbox()
Blob(name, centre, radii, label)                          # .inside(P)
Anatomy(name='normal', description='', size_xyz=(100.0, 100.0, 50.0), fat_mean=2.5, fat_amp=0.8,
        liver_edge_y0=52.0, liver_edge_slope=-0.1, liver_edge_wedge=0.5, tubes=[], blobs=[])
    def labels(self, P)        # tissue label (int8) for each point P (N, 3) in the phantom frame
    def tube(self, name)
def build_case(name='normal')
def validate(an, res=0.4, min_gap=0.0)      # [] when nothing intersects
def calot_triangle(an)                      # (x, y) polygon on the surface
```

`orvue_us_inverse.tracking.markers`
```python
def dictionary(); def detector(); def reference_board(); def probe_board(use_second_marker=None)
def board_pose(board, corners, ids, K, dist)              # T_cam_board or None (IPPE)
def board_pose_refined(board, corners, ids, K, dist)      # + solvePnPRefineLM
def phantom_probe_pose(gray, K, dist, det=None, ref=None, prb=None)   # T_phantom_probe or None
def xy_yaw(T_phantom_probe)                   # (x, y, yaw_deg)
```

`orvue_us_inverse.tracking.tracker`
```python
ProbeTracker(source='realsense', use_second_marker=None, config=None, intrinsics=None, autostart=True)
    # source: "realsense", or a video file / image folder path (FileSource, intrinsics.json next to it)
    start(); stop(); get_state() -> TrackState; get_frame(); get_pose(); get_xy_yaw()
    process_frame(self, image, timestamp=None); reload_calibration(); reset_filters()
TrackerConfig(..., calibration_path=CALIBRATION_FILE, loop_files=False)
Calibration(yaw_offset_deg=0.0, dx=0.0, dy=0.0, dz=0.0)
TrackState(...)   # .T_phantom_probe, .valid, .to_row() (CSV_FIELDS), .to_dict()
YAW_SIGN_CCW_FROM_CAMERA = -1; CALIBRATION_FILE = paths.CALIBRATION_PATH  # config/calibration.json
```

## Tissue labels (`anatomy.TISSUES`: impedance MRayl, echogenicity, attenuation dB/cm/MHz)

| Label | Tissue | Z | Echo | Atten. |
|---|---|---|---|---|
| 0 | liver | 1.65 | 0.70 | 0.5 |
| 1 | fat | 1.38 | 1.10 | 0.6 |
| 2 | gb_wall | 1.70 | 1.60 | 0.8 |
| 3 | bile | 1.52 | 0.003 | 0.05 |
| 4 | stone | 6.00 | 2.50 | 12.0 |
| 5 | duct_wall | 1.70 | 1.50 | 0.8 |
| 6 | art_wall | 1.75 | 1.80 | 1.0 |
| 7 | art_blood | 1.61 | 0.008 | 0.15 |
| 8 | vein_wall | 1.68 | 1.20 | 0.8 |
| 9 | vein_blood | 1.61 | 0.008 | 0.15 |
| 10 | lymph_node | 1.60 | 0.20 | 0.6 |

## Anatomy cases (`anatomy.CASES`)
normal, parallel_cystic_duct, medial_spiral_insertion, short_cystic_duct, anterior_cystic_artery,
caterpillar_hump, inflamed_obese (5 mm GB wall, 9 mm fat, impacted Hartmann stone), choledocholithiasis (CBD stone).
- The cystic artery ends inside the GB wall at the neck and splits into `cystic_artery_superficial` (posterior)
  and `cystic_artery_deep` (anterior, liver bed); caterpillar_hump has `cystic_branch_1` from the RHA instead.
- Calot's node (label 10) rests on the anterior side of the cystic artery (absent in short_cystic_duct and
  caterpillar_hump).
- Structures that touch are listed in `CONNECTED`.

## Inverse mapping
- Purpose: two separate problems, built and evaluated separately. Reconstruction places each frame's pixels at
  their 3D positions (pose `T`) and fuses them into a voxel grid; recognition decides what each pixel / voxel is.
  Recognition starts with oracle labels (the simulator's per-frame ground truth, `labels_image(T)`), so every
  remaining error is due to reconstruction; classical / learned recognition is measured against that later.
- Code in the sub-package `src/orvue_us_inverse/mapping/` ("mapping/"); tests in `tests/mapping/`.
- `mapping/probe.py` is the only probe source: `PROBE` (`LinearProbe` with every field stated: f0 7.5 MHz,
  30 mm wide, 50 mm deep, 0.1 mm pixels, fnum 3, 2.5 cycles, elevation sigma 0.5 mm, focus 20 mm, Rayleigh 10 mm,
  c 1.54 mm/us), `make_simulator(case="normal", **kw)` (`BModeSimulator` with `PROBE` and `persistence=0`; passing
  `persistence` raises `TypeError`), `probe_metadata()` (every field as a dict, stored with each sweep). Never
  build a `LinearProbe` or `BModeSimulator` directly in `mapping/`. Also `FRAME_SHAPE` (501, 301) and
  `simulator_settings(**kw)`; the simulator from `make_simulator` carries `.case` and `.settings` (for metadata).
- Modules (`mapping/`) and the session that implements them:

  | Module | Role | Session |
  |---|---|---|
  | `probe.py` | Probe, `make_simulator`, probe metadata | Prep |
  | `config.py` | `GridConfig` (x, y 0-100, z 0-50 mm, 0.5 mm voxels -> shape (200, 200, 100) indexed [ix, iy, iz], voxel i centred at lo + (i + 0.5) * voxel), `SweepConfig` (yaw list, overlap %, spacing, angle trigger, speed, serpentine, region; probe width / depth from `PROBE`), `AcquisitionConfig` (`store_images`) | S0 |
  | `sweep_io.py` | `FrameRecord` (index, t, T_true, T_measured, int8 labels, uint8 image or None), `Sweep` (metadata + frames; `save` / `load` .npz), `make_metadata`, `estimate_size_mb` | S0 |
  | `poses.py` | `ScriptedSweep(cfg)`: lanes (`Lane`, `planned_lanes()`), `segments` (lanes + unrecorded transitions), `samples(step_mm=0.05)` / `pose_at(t)` -> `Sample(t, T, recording, lane)`, `lane_at(t)`, `expected_frames()`, `coverage()`, `summary()`; mouse (S6), camera (S7) | S1 |
  | `acquisition.py` | `Acquirer(sim, sweep_cfg, acq_cfg)`: `feed(T, t, recording)` -> `FrameRecord` or None; `.sweep` | S1 |
  | `run_scripted.py` | App `python -m orvue_us_inverse.mapping.run_scripted` (menu step `scripted`); `ScriptedPlayback` holds the state and draws without windows | S1 |
  | `recon.py` | Voxel grid, label compounding, coverage | S2 |
  | `render.py` | Slice views, surfaces, off-screen 3D snapshots | S3 |
  | `evaluate.py` | Ground-truth voxels, metrics, report | S4 |
  | `experiments.py` | Headless parameter studies | S5, S7 |
  | `errors.py` | Pose-error injection | S7 |

- Sweep file (`sweep_io`, format 1): one compressed .npz with `index` (N,), `t` (N,) simulated seconds,
  `T_true`, `T_measured` (N, 4, 4), `labels` (N, 501, 301) int8, `images` (N, 501, 301) uint8 only when stored
  (all frames or none), `metadata` (JSON: case, simulator settings, probe fields, frame shape, sweep and
  acquisition configs, UTC timestamp, provenance = repository commit or None + simulator source `e789389`),
  `format_version`. Size ~0.1 MB/frame with images, ~1.5 kB/frame labels only (800 frames: ~81 MB / ~1.4 MB).
- Scripted sweep: for each yaw, u = (cos, sin), v = (-sin, cos); lanes along v at lateral positions p . u;
  n = ceil((E - W) / (W (1 - overlap))) + 1 lanes for the region extent E along u, centres spread evenly so the
  outer image edges touch the region edges (20% -> 4 lanes at 15, 38.3, 61.7, 85, actual 22.2%). Lanes are
  clipped to where the face centre is on the region (= `in_contact`); serpentine (lane k along +v when k is
  even). Transitions (lift-off) move at the sweep speed and rotate at 45 deg/s; never recorded. Samples every
  0.05 mm from each lane start; capture every `frame_spacing_mm` of face-centre travel or `angle_trigger_deg`,
  plus the first pose of every lane: 201 frames per 100 mm lane at 0.5 mm. Poses from `pose_from_xy_yaw` are
  float32, so the trigger has a 1e-4 mm tolerance.
- Frames 501 x 301 (depth x lateral). Timing (Prep, 2026-10-08): `render()` ~78 ms/frame, `labels_image()`
  ~13 ms/frame; oracle-only sweeps use `labels_image()` alone.
- Data: sweeps in `paths.SWEEPS_DIR` (`output/sweeps`), reconstructions / metrics / reports in
  `paths.RESULTS_DIR` (`output/results`); both created on demand, gitignored.
- `PLAN_inverse_mapping.md` is the session plan (one narrow, tested step per session); `STATUS.md` is the state,
  updated at the end of every session (results, test counts, decisions, known issues, next step).
- Tests are headless (no windows, no camera, no browser). No PyVista / VTK (blocked by Windows Application
  Control on this machine): do not import or install them.

## Rules
- Read-only (frozen baselines from the simulator; use and import, do not edit without asking):
  `simulation/anatomy.py`, `simulation/bmode.py`, the layout constants in `tracking/markers.py`,
  `tracking/tracker.py`. The other copied files are baselines too. Any change to a copied file is logged in
  `UPSTREAM.md` (date, file, change, reason). `reports/pdf_template.py` is the shared PDF template.
- New work goes in new modules / sub-packages of `orvue_us_inverse` that import the copied ones.
- Frames for reconstruction: always create the simulator with `persistence=0` (the default 0.3 blends each frame
  with the previous ones, so a frame would not belong to its own pose).
- Keep the frame / orientation conventions and the tissue labels unchanged.
- Nothing may import the simulator's original package name (checked by `tests/test_standalone.py`).
- New file locations go in `paths.py`; generated run-time files go under `output/`.
- One README.md only; documentation as PDF built with `reports.pdf_template.Doc`; keep README, manual and CLAUDE.md
  in step.
- README "Project layout > Every file" lists every tracked file with its purpose: add, move or remove the row whenever
  a file is added, moved or removed.
- Company name: "Orvue Surgical".

## Running
- Environment: conda env `orvue-robot`, Python 3.11.16 (`C:/Users/rahul/miniconda3/envs/orvue-robot/python.exe`).
  numpy 2.4.6, scipy 1.17.1, opencv-contrib-python 5.0.0.93, matplotlib 3.11.2, scikit-image 0.26.0,
  pyrealsense2 2.58.4 (optional extra `camera`), reportlab 5.0.1, pytest 9.1.1. The 3D viewer needs a browser
  (three.js from jsDelivr).
- Install once: `pip install -r requirements.txt` and `pip install -e .`.
- `python -m orvue_us_inverse` (menu) or
  `python -m orvue_us_inverse viewer | calibrate | sim | scripted | run | test | board | anatomy | manual`.
- `python -m orvue_us_inverse scripted [--case X] [--yaw 0 90] [--overlap 20] [--spacing 0.5] [--speed 10]
  [--no-images]` (keys space pause, + / - playback speed, a anatomy, r restart, s save, q / Esc quit).
- `python -m orvue_us_inverse sim [case] [--track] [--cam-view] [--mouse]` (keys q/e yaw, n/p case, g ground truth,
  c contact, m camera / mouse, t camera view, z zoom fit, v 3D viewer, Esc quit).
- Tests: `python -m pytest tests` from the repository root (no hardware needed; ~1 min).
