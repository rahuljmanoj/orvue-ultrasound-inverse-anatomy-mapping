"""The repository is standalone: no source or test file refers to the simulator package it was copied from."""
import os

from orvue_us_inverse.paths import REPO_ROOT

FORBIDDEN = "orvue_us" + "_sim"          # split so this file does not match itself
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def source_files():
    for top in ("src", "tests"):
        for folder, dirs, files in os.walk(os.path.join(REPO_ROOT, top)):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
            for name in files:
                if not name.endswith((".pyc", ".jpg", ".png", ".pdf")):
                    yield os.path.join(folder, name)


def test_files_found():
    names = {os.path.basename(p) for p in source_files()}
    assert {"bmode.py", "tracker.py", "test_standalone.py"} <= names


def test_no_simulator_package_reference():
    hits = []
    for path in source_files():
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if FORBIDDEN in line:
                    hits.append(f"{os.path.relpath(path, REPO_ROOT)}:{i}: {line.strip()}")
    assert not hits, "references to the simulator package:\n" + "\n".join(hits)
