# Ultrasound Inverse Anatomy Mapping

Orvue Surgical: inverse anatomy mapping for ultrasound. Goal: reconstruct the 3D segmented anatomy from tracked
2D B-mode frames (the inverse of the Ultrasound Imaging Simulator), then recognise the anatomy. Focus: the
gallbladder / Calot's triangle anatomy of the simulator (`orvue_us_sim`, repository
[orvue-ultrasound-simulator](https://github.com/rahuljmanoj/orvue-ultrasound-simulator)), whose 8 anatomy cases,
B-mode renderer and probe tracking provide the frames, the poses and the ground truth.

Built from the Orvue Surgical Python project template: an installable package in `src/`, one entry point with a
menu, every file location in `paths.py`, committed configuration in `config/`, generated PDFs in `docs/` and
run-time output in the gitignored `output/`.

## Contents
- [Start a new project from this template](#start-a-new-project-from-this-template)
- [Setup](#setup)
- [Quick start](#quick-start)
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
Python 3.10+, conda environment `orvue-robot` (shared with the simulator, which this project imports).
Install the simulator first (`pip install -e .` in `../Ultrasound_Simulator`), then from this repository folder:
```
pip install -r requirements.txt
pip install -e .
```
`pip install -e .` installs the package in editable mode: imports and `python -m orvue_us_inverse` work from any
folder and code changes take effect without reinstalling. Re-run it only after changing `pyproject.toml`, moving
the folder or creating a new environment.

## Quick start
```
python -m orvue_us_inverse              menu (steps in the order of use, 0 = exit)
python -m orvue_us_inverse run          example step -> output/logs/example.txt
python -m orvue_us_inverse test         all tests
python -m orvue_us_inverse manual       -> docs/Ultrasound Inverse Anatomy Mapping - User Manual.pdf
```
The console script `orvue-us-inverse` does the same as `python -m orvue_us_inverse`.

## Project layout
```
config/                  committed configuration (settings.json)
docs/                    generated PDFs (manual), figures/, images/
scripts/                 helper scripts outside the package
src/orvue_us_inverse/      the package
tests/                   pytest tests, helpers/ for synthetic test data
output/                  generated at run time, gitignored: logs/, export/, cache/
```

### Every file

| File | Purpose / use |
|---|---|
| `.gitignore` | Ignores PyCharm, Python caches, packaging output and `output/` |
| `pyproject.toml` | Package metadata, dependencies, console script, pytest options (`src` layout) |
| `requirements.txt` | Pinned versions of the working environment |
| `README.md` | This file: the only README |
| `CLAUDE.md` | Conventions for Claude Code sessions |
| `config/settings.json` | Example settings, read by `core/example.py` |
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
| `tests/__init__.py` | Makes `tests` a package (for `tests.helpers`) |
| `tests/helpers/__init__.py` | Helpers for tests (synthetic data) |
| `tests/test_template.py` | Tests of paths, settings, example, entry point and manual |

## Rules
- Absolute imports only (`from orvue_us_inverse.core.example import scaled`); no `sys.path` manipulation.
- Every file location goes in `paths.py`; run-time output under `output/`; committed configuration under `config/`.
- One sub-package per area (`core/` here); each module runnable with `python -m orvue_us_inverse.<area>.<module>`;
  add new steps to `COMMANDS` and `MENU` in `__main__.py`.
- One README.md: add, move or remove the row in "Every file" whenever a tracked file is added, moved or removed.
- Documentation as PDF built with `reports.pdf_template.Doc` from the code; no Word documents. Keep README, manual
  and CLAUDE.md in step.

## Tests
```
python -m pytest tests
```
from the repository folder (pyproject sets `-q` and puts `src` and `.` on the path). Tests must run without hardware.
