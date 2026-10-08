# UPSTREAM.md

Origin of the simulator code in this repository and every change made to it since the copy.

## Source
- Repository: https://github.com/rahuljmanoj/orvue-ultrasound-simulator (local sibling folder `../Ultrasound_Simulator`)
- Commit: `e78938935ebc83fbb8245818d2f3717916b38a1c` (`e789389`), 2026-10-08 11:53:51 +0200
- Copied: 2026-10-08, plain copy of the committed files (`git archive e789389`), no git history.
- Package renamed: `orvue_us_sim` -> `orvue_us_inverse`. Nothing in this repository imports `orvue_us_sim`
  (checked by `tests/test_standalone.py`).

## Rule
The copied files are frozen baselines. They are not synchronised with the upstream repository in either
direction. Any later change to a copied file is logged in [Change log](#change-log) below with the date, the file,
what changed and why. New work goes in new modules that import the copied ones.

## Old -> new paths

| Upstream (`e789389`) | Here |
|---|---|
| `src/orvue_us_sim/simulation/__init__.py` | `src/orvue_us_inverse/simulation/__init__.py` |
| `src/orvue_us_sim/simulation/anatomy.py` | `src/orvue_us_inverse/simulation/anatomy.py` |
| `src/orvue_us_sim/simulation/bmode.py` | `src/orvue_us_inverse/simulation/bmode.py` |
| `src/orvue_us_sim/tracking/__init__.py` | `src/orvue_us_inverse/tracking/__init__.py` |
| `src/orvue_us_sim/tracking/markers.py` | `src/orvue_us_inverse/tracking/markers.py` |
| `src/orvue_us_sim/tracking/tracker.py` | `src/orvue_us_inverse/tracking/tracker.py` |
| `src/orvue_us_sim/tracking/calibrate.py` | `src/orvue_us_inverse/tracking/calibrate.py` |
| `src/orvue_us_sim/tracking/viewer.py` | `src/orvue_us_inverse/tracking/viewer.py` |
| `src/orvue_us_sim/tracking/board.py` | `src/orvue_us_inverse/tracking/board.py` |
| `src/orvue_us_sim/ui/__init__.py` | `src/orvue_us_inverse/ui/__init__.py` |
| `src/orvue_us_sim/ui/clinical.py` | `src/orvue_us_inverse/ui/clinical.py` |
| `src/orvue_us_sim/viewer3d/__init__.py` | `src/orvue_us_inverse/viewer3d/__init__.py` |
| `src/orvue_us_sim/viewer3d/__main__.py` | `src/orvue_us_inverse/viewer3d/__main__.py` |
| `src/orvue_us_sim/viewer3d/geometry.py` | `src/orvue_us_inverse/viewer3d/geometry.py` |
| `src/orvue_us_sim/viewer3d/info.py` | `src/orvue_us_inverse/viewer3d/info.py` |
| `src/orvue_us_sim/viewer3d/viewer.py` | `src/orvue_us_inverse/viewer3d/viewer.py` |
| `src/orvue_us_sim/viewer3d/template.html` | `src/orvue_us_inverse/viewer3d/template.html` |
| `config/calibration.json` | `config/calibration.json` |
| `tests/test_anatomy.py` | `tests/test_anatomy.py` |
| `tests/test_tracking.py` | `tests/test_tracking.py` |
| `tests/helpers/synthetic_scene.py` | `tests/helpers/synthetic_scene.py` |

Not copied: `__main__.py`, `paths.py` (this project has its own; see below), `reports/` (this project has its own
`pdf_template.py` and `manual.py`; `figures.py` and `tissue_dimensions.py` not needed), `scripts/`, `docs/`,
`tests/test_viewer.py` (the 3D viewer is therefore untested here), `tests/helpers/__init__.py` (upstream: a
docstring only; the existing empty file is kept).

`viewer3d/` was not on the original copy list; it was added (user decision) because `simulation/bmode.py` opens it
from the simulator window (`v` key / 3D viewer button) through a lazy import.

## Changes made during the copy
1. **Package name**: every occurrence of `orvue_us_sim` replaced by `orvue_us_inverse` (61 lines; nothing else in
   the copied files changed; `config/calibration.json` unchanged). Import statements and the matching docstrings /
   usage lines / comments:

   | File | Lines changed |
   |---|---|
   | `simulation/anatomy.py` | docstring title, usage import (2) |
   | `simulation/bmode.py` | docstring title and 2 usage lines; imports of `simulation.anatomy`, `paths.LOGO_PATH`, `tracking.viewer` (3, lazy), `ui.clinical` (3, lazy), `viewer3d.viewer` (lazy), `tracking.tracker` (lazy) (13) |
   | `tracking/markers.py` | docstring title (1) |
   | `tracking/tracker.py` | docstring title and 1 docstring line; imports of `paths.CALIBRATION_PATH`, `tracking.markers` (4) |
   | `tracking/calibrate.py` | docstring title and usage; imports of `tracking.tracker`, `ui.clinical` (lazy), `tracking.viewer` (lazy) (5) |
   | `tracking/viewer.py` | docstring title, usage and 2 lines; imports of `paths`, `ui.clinical`, `tracking.markers`, `tracking.calibrate`, `tracking.tracker` (9) |
   | `tracking/board.py` | docstring title and usage; imports of `paths.BOARD_PDF`, `tracking.markers` (4) |
   | `ui/clinical.py` | docstring title; import of `paths.LOGO_PATH` (2) |
   | `viewer3d/geometry.py` | docstring title; imports of `paths`, `simulation.anatomy`, `simulation.bmode` (lazy) (4) |
   | `viewer3d/info.py` | docstring title; import of `simulation.anatomy` (2) |
   | `viewer3d/viewer.py` | docstring title and 3 usage lines; imports of `paths`, `simulation.anatomy` (6) |
   | `viewer3d/template.html` | the export command shown in the page (1) |
   | `tests/test_anatomy.py` | imports of `simulation.anatomy`, `simulation.bmode` (2) |
   | `tests/test_tracking.py` | docstring line; imports of `tracking.markers`, `tracking.calibrate`, `tracking.tracker` (4) |
   | `tests/helpers/synthetic_scene.py` | docstring line; import of `tracking.markers` (2) |

2. **`paths.py`** (this project's own file): added the locations the copied modules use, same values as upstream:
   `CALIBRATION_PATH` (`config/calibration.json`), `PRINT_DIR` (`docs/print/`), `BOARD_PDF`
   (`docs/print/tracking_board.pdf`), `CAPTURES_DIR` (`output/captures/`), `VIEWER3D_OUT_DIR` (`output/viewer3d/`).
   Already present: `PACKAGE_DIR`, `LOGO_PATH`, `LOGS_DIR`, `EXPORT_DIR`, `CACHE_DIR`.
3. **`pyproject.toml`**: dependency `orvue-us-sim` removed; `scipy`, `opencv-contrib-python`, `matplotlib`,
   `scikit-image` added (same lower bounds as upstream); optional extra `camera = ["pyrealsense2>=2.55"]`;
   `viewer3d/template.html` added to package-data. **`requirements.txt`**: pinned as upstream.
4. **`__main__.py`** (this project's own file): commands `viewer`, `calibrate`, `sim`, `board`, `anatomy` and their
   menu rows. Upstream's `validate` command (anatomy.py has no `main()`) and the `sim` case prompt are not copied.
5. Line endings: committed with LF, as upstream stores them.

## Change log
Later changes to copied files (date, file, change, reason). None yet.
