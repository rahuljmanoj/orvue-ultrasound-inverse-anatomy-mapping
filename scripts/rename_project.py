"""
rename_project.py - turn the template into a new project (run once, right after creating the repository).

    python scripts/rename_project.py <package_name> "<Project Title>"
    e.g. python scripts/rename_project.py orvue_us_sim "Ultrasound Imaging Simulator"

Renames src/orvue_template/ to src/<package_name>/ and replaces, in every text file of the repository:
orvue_template -> <package_name>, orvue-template -> <package-name>, "Project Template" -> <Project Title>.
Then: pip install -e . in the project's conda env, python -m pytest tests, and delete this script's row from
the README if you delete the script.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_PKG, OLD_DIST, OLD_TITLE = "orvue_template", "orvue-template", "Project Template"
TEXT_EXT = {".py", ".toml", ".md", ".txt", ".json", ".cfg", ".ini", ".html", ".yml", ".yaml"}
SKIP_DIRS = {".git", ".idea", "output", "__pycache__", ".pytest_cache", "build", "dist"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or not re.fullmatch(r"[a-z][a-z0-9_]*", argv[0]):
        print(__doc__)
        return 2
    pkg, title = argv
    dist = pkg.replace("_", "-")
    this = os.path.abspath(__file__)
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.endswith(".egg-info")]
        for name in files:
            path = os.path.join(d, name)
            if os.path.splitext(name)[1] not in TEXT_EXT or path == this:
                continue
            with open(path, encoding="utf-8") as f:
                text = f.read()
            new = text.replace(OLD_PKG, pkg).replace(OLD_DIST, dist).replace(OLD_TITLE, title)
            if new != text:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(new)
                print("updated", os.path.relpath(path, ROOT))
    os.rename(os.path.join(ROOT, "src", OLD_PKG), os.path.join(ROOT, "src", pkg))
    print(f"renamed src/{OLD_PKG} -> src/{pkg}")
    print("next: pip install -e .   then   python -m pytest tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
