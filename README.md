# Project Template

Orvue Surgical template for Python projects: an installable package in `src/`, one entry point with a menu,
every file location in `paths.py`, committed configuration in `config/`, generated PDFs in `docs/` and run-time
output in the gitignored `output/`. Reference project built this way: the Ultrasound Imaging Simulator
(`orvue_us_sim`).

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
Python 3.10+. In the project's conda environment, from the repository folder:
```
pip install -r requirements.txt
pip install -e .
```
`pip install -e .` installs the package in editable mode: imports and `python -m orvue_template` work from any
folder and code changes take effect without reinstalling. Re-run it only after changing `pyproject.toml`, moving
the folder or creating a new environment.

## Quick start
```
python -m orvue_template              menu (steps in the order of use, 0 = exit)
python -m orvue_template run          example step -> output/logs/example.txt
python -m orvue_template test         all tests
python -m orvue_template manual       -> docs/Project Template - User Manual.pdf
```
The console script `orvue-template` does the same as `python -m orvue_template`.

## Project layout
```
config/                  committed configuration (settings.json)
docs/                    generated PDFs (manual), figures/, images/
scripts/                 helper scripts outside the package
src/orvue_template/      the package
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
| `src/orvue_template/__init__.py` | Package docstring and `__version__` |
| `src/orvue_template/__main__.py` | Entry point: menu and `python -m orvue_template <command>` |
| `src/orvue_template/paths.py` | Every file and folder location |
| `src/orvue_template/assets/orvue_logo.jpg` | Orvue Surgical logo (PDF header) |
| `src/orvue_template/core/__init__.py` | Example area sub-package |
| `src/orvue_template/core/example.py` | Example module: reads the settings, writes a log |
| `src/orvue_template/reports/__init__.py` | Reports sub-package |
| `src/orvue_template/reports/pdf_template.py` | Orvue Surgical PDF template (`Doc`), used by every generated PDF |
| `src/orvue_template/reports/manual.py` | Builds the user manual PDF from the code |
| `tests/__init__.py` | Makes `tests` a package (for `tests.helpers`) |
| `tests/helpers/__init__.py` | Helpers for tests (synthetic data) |
| `tests/test_template.py` | Tests of paths, settings, example, entry point and manual |

## Rules
- Absolute imports only (`from orvue_template.core.example import scaled`); no `sys.path` manipulation.
- Every file location goes in `paths.py`; run-time output under `output/`; committed configuration under `config/`.
- One sub-package per area (`core/` here); each module runnable with `python -m orvue_template.<area>.<module>`;
  add new steps to `COMMANDS` and `MENU` in `__main__.py`.
- One README.md: add, move or remove the row in "Every file" whenever a tracked file is added, moved or removed.
- Documentation as PDF built with `reports.pdf_template.Doc` from the code; no Word documents. Keep README, manual
  and CLAUDE.md in step.

## Tests
```
python -m pytest tests
```
from the repository folder (pyproject sets `-q` and puts `src` and `.` on the path). Tests must run without hardware.
