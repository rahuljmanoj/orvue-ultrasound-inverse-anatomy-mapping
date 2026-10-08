"""
orvue_template.core.example - example module: replace with the project's own areas (one sub-package per area).

    python -m orvue_template run              (or: python -m orvue_template.core.example [value])

Reads config/settings.json and writes output/logs/example.txt.
"""
import json
import os
import sys

from orvue_template.paths import LOGS_DIR, SETTINGS_PATH


def load_settings(path=SETTINGS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scaled(value, settings=None):
    """The value times settings["scale"]."""
    settings = load_settings() if settings is None else settings
    return value * settings["scale"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    value = float(argv[0]) if argv else 1.0
    result = scaled(value)
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, "example.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{value} -> {result}\n")
    print(f"{value} -> {result} (written to {path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
