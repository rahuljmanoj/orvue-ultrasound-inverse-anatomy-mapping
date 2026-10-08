"""
paths.py - every file and folder location used by the package, in one place.

The package lives in src/orvue_us_inverse/ of the repository and is installed in editable mode
(`pip install -e .`), so REPO_ROOT is the repository folder:

    config/settings.json           settings (committed)
    config/calibration.json        probe calibration (committed, rewritten by the calibration)
    docs/                          generated PDFs, figures/, images/, print/tracking_board.pdf
    output/                        generated at run time, not in git: captures/, logs/, export/, viewer3d/, cache/,
                                   sweeps/ (inverse-mapping sweeps), results/ (reconstructions, metrics)

Folders under output/ are created on demand by the code that writes into them.
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PACKAGE_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "orvue_logo.jpg")

REPO_ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.json")
CALIBRATION_PATH = os.path.join(CONFIG_DIR, "calibration.json")
UPSTREAM_PATH = os.path.join(REPO_ROOT, "UPSTREAM.md")         # origin of the copied simulator code

DOCS_DIR = os.path.join(REPO_ROOT, "docs")
FIGURES_DIR = os.path.join(DOCS_DIR, "figures")
IMAGES_DIR = os.path.join(DOCS_DIR, "images")
PRINT_DIR = os.path.join(DOCS_DIR, "print")
MANUAL_PDF = os.path.join(DOCS_DIR, "Ultrasound Inverse Anatomy Mapping - User Manual.pdf")
BOARD_PDF = os.path.join(PRINT_DIR, "tracking_board.pdf")

OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
CAPTURES_DIR = os.path.join(OUTPUT_DIR, "captures")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
EXPORT_DIR = os.path.join(OUTPUT_DIR, "export")
VIEWER3D_OUT_DIR = os.path.join(OUTPUT_DIR, "viewer3d")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SWEEPS_DIR = os.path.join(OUTPUT_DIR, "sweeps")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
EXPERIMENTS_CACHE_DIR = os.path.join(CACHE_DIR, "experiments")   # cached sweeps of the S5 study
