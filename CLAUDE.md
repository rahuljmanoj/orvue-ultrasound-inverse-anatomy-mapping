# CLAUDE.md

Conventions a new session must know.

## Project
Ultrasound Inverse Anatomy Mapping for Orvue Surgical. Goal ("inverse mapping"): reconstruct the 3D segmented
anatomy from tracked 2D B-mode frames, then recognise the anatomy. Stages: (1) reconstruction (tracked frames ->
3D volume / segmented label volume), (2) recognition (which structures: gallbladder, cystic duct, CBD, cystic
artery, RHA, Calot's node ...). Focus: the gallbladder / Calot's triangle anatomy of the Ultrasound Imaging
Simulator.

Source of frames, poses and ground truth: the simulator package `orvue_us_sim` (sibling folder
`../Ultrasound_Simulator`, repository rahuljmanoj/orvue-ultrasound-simulator), installed editable in the same env.
Read its CLAUDE.md for the frames (phantom frame mm, z = depth; probe pose `T` phantom <- probe), tissue labels
0-10, the 8 anatomy cases and the exact simulator / tracker signatures. Typical use:
`from orvue_us_sim.simulation.anatomy import build_case`; `from orvue_us_sim.simulation.bmode import BModeSimulator`;
`sim.render(T, return_labels=True)` gives the B-mode image and its ground-truth labels for a pose `T`.
`orvue_us_sim` is read-only here: import it, never edit it from this project (ask first).

## Layout
One installable package `orvue_us_inverse` in `src/` (`pyproject.toml`; installed with `pip install -e .`). Absolute
imports only (`from orvue_us_inverse.core.example import scaled`); no `sys.path` manipulation. Every file location
comes from `orvue_us_inverse.paths`.

```
config/settings.json         settings (committed)
docs/                        generated PDFs, figures/, images/
scripts/rename_project.py    turns the template into a new project
src/orvue_us_inverse/          the package (below)
tests/                       test_template.py, helpers/
output/                      run-time output, gitignored: logs/, export/, cache/
```

## Files (`src/orvue_us_inverse/`)
- `__main__.py`: single entry point; menu (steps in the order of use, 0 = exit) or
  `python -m orvue_us_inverse <command>` (`run`, `test`, `manual`); console script `orvue-us-inverse`. Runs
  `python -m <module>` subprocesses from the repository folder (stdin = DEVNULL). New steps: `COMMANDS` + `MENU`.
- `paths.py`: REPO_ROOT, LOGO_PATH, SETTINGS_PATH, DOCS_DIR, FIGURES_DIR, IMAGES_DIR, MANUAL_PDF, OUTPUT_DIR,
  LOGS_DIR, EXPORT_DIR, CACHE_DIR.
- `core/example.py`: example area module (`load_settings`, `scaled`, `main`).
- `reports/pdf_template.py`: the Orvue Surgical PDF template (`Doc`); every generated PDF uses it.
- `reports/manual.py`: writes the user manual PDF from the code.
- `assets/orvue_logo.jpg`.

## Rules
- `orvue_us_sim` (the simulator) is read-only from this project; `reports/pdf_template.py` is the shared PDF
  template. New work goes in new modules / sub-packages of `orvue_us_inverse`.
- Keep the simulator's frame / orientation conventions and tissue labels unchanged.
- New file locations go in `paths.py`; generated run-time files go under `output/`.
- One README.md only; documentation as PDF built with `reports.pdf_template.Doc`; keep README, manual and CLAUDE.md
  in step.
- README "Project layout > Every file" lists every tracked file with its purpose: add, move or remove the row whenever
  a file is added, moved or removed.
- Company name: "Orvue Surgical".

## Running
- Environment: conda env `orvue-robot`, Python 3.11.16 (`C:/Users/rahul/miniconda3/envs/orvue-robot/python.exe`),
  shared with the simulator (`orvue_us_sim` installed editable from `../Ultrasound_Simulator`).
- Install once: `pip install -r requirements.txt` and `pip install -e .` (simulator installed first).
- `python -m orvue_us_inverse` (menu) or `python -m orvue_us_inverse run | test | manual`.
- Tests: `python -m pytest tests` from the repository root (no hardware needed).
