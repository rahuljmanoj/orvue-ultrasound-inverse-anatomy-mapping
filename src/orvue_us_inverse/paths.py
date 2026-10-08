"""
paths.py - every file and folder location used by the package, in one place.

The package lives in src/orvue_us_inverse/ of the repository and is installed in editable mode
(`pip install -e .`), so REPO_ROOT is the repository folder:

    config/settings.json           settings (committed)
    docs/                          generated PDFs, figures/, images/
    output/                        generated at run time, not in git: logs/, export/, cache/
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PACKAGE_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "orvue_logo.jpg")

REPO_ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.json")

DOCS_DIR = os.path.join(REPO_ROOT, "docs")
FIGURES_DIR = os.path.join(DOCS_DIR, "figures")
IMAGES_DIR = os.path.join(DOCS_DIR, "images")
MANUAL_PDF = os.path.join(DOCS_DIR, "Ultrasound Inverse Anatomy Mapping - User Manual.pdf")

OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
EXPORT_DIR = os.path.join(OUTPUT_DIR, "export")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
