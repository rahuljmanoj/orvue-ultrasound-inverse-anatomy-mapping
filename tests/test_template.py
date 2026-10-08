"""Template tests: paths, settings, example module, entry point and manual. Replace with the project's own tests."""
import os
import subprocess
import sys

from orvue_template import paths
from orvue_template.__main__ import COMMANDS, MENU
from orvue_template.core.example import load_settings, scaled
from orvue_template.reports.manual import build


def test_paths_inside_repository():
    assert os.path.isfile(os.path.join(paths.REPO_ROOT, "pyproject.toml"))
    assert os.path.isfile(paths.SETTINGS_PATH)
    assert os.path.isfile(paths.LOGO_PATH)


def test_example():
    assert scaled(3.0, {"scale": 2.0}) == 6.0
    assert "scale" in load_settings()


def test_menu_commands_exist():
    assert all(cmd in COMMANDS for _, _, _, _, cmd, _ in MENU)


def test_entry_point_menu_exits():
    out = subprocess.run([sys.executable, "-m", "orvue_template"], stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, cwd=paths.REPO_ROOT, timeout=60)
    assert out.returncode == 0 and "0  Exit" in out.stdout


def test_manual(tmp_path):
    pdf = build(str(tmp_path / "manual.pdf"))
    assert os.path.getsize(pdf) > 1000
