# CLAUDE.md

Conventions a new session must know. (Template: after `scripts/rename_project.py`, replace the Project section and
add the project's frames, interfaces and parameters.)

## Project
Project Template: Orvue Surgical template for Python projects.

## Layout
One installable package `orvue_template` in `src/` (`pyproject.toml`; installed with `pip install -e .`). Absolute
imports only (`from orvue_template.core.example import scaled`); no `sys.path` manipulation. Every file location
comes from `orvue_template.paths`.

```
config/settings.json         settings (committed)
docs/                        generated PDFs, figures/, images/
scripts/rename_project.py    turns the template into a new project
src/orvue_template/          the package (below)
tests/                       test_template.py, helpers/
output/                      run-time output, gitignored: logs/, export/, cache/
```

## Files (`src/orvue_template/`)
- `__main__.py`: single entry point; menu (steps in the order of use, 0 = exit) or
  `python -m orvue_template <command>` (`run`, `test`, `manual`); console script `orvue-template`. Runs
  `python -m <module>` subprocesses from the repository folder (stdin = DEVNULL). New steps: `COMMANDS` + `MENU`.
- `paths.py`: REPO_ROOT, LOGO_PATH, SETTINGS_PATH, DOCS_DIR, FIGURES_DIR, IMAGES_DIR, MANUAL_PDF, OUTPUT_DIR,
  LOGS_DIR, EXPORT_DIR, CACHE_DIR.
- `core/example.py`: example area module (`load_settings`, `scaled`, `main`).
- `reports/pdf_template.py`: the Orvue Surgical PDF template (`Doc`); every generated PDF uses it.
- `reports/manual.py`: writes the user manual PDF from the code.
- `assets/orvue_logo.jpg`.

## Rules
- Stable shared modules (list them here) are read-only: new work goes in new modules / sub-packages that import them.
- New file locations go in `paths.py`; generated run-time files go under `output/`.
- One README.md only; documentation as PDF built with `reports.pdf_template.Doc`; keep README, manual and CLAUDE.md
  in step.
- README "Project layout > Every file" lists every tracked file with its purpose: add, move or remove the row whenever
  a file is added, moved or removed.
- Company name: "Orvue Surgical".

## Running
- Environment: conda env `<env>` (set after creating it), Python 3.11.
- Install once: `pip install -r requirements.txt` and `pip install -e .`.
- `python -m orvue_template` (menu) or `python -m orvue_template run | test | manual`.
- Tests: `python -m pytest tests` from the repository root (no hardware needed).
