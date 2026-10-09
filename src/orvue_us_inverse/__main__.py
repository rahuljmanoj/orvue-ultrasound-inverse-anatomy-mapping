"""
orvue_us_inverse (python -m orvue_us_inverse) - single entry point of the project.

    python -m orvue_us_inverse                      the Ultrasound Imaging Simulator: one window for B-mode,
                                                    inverse anatomy mapping and the probe calibration
    python -m orvue_us_inverse dev                  developer menu (every step and study)
    python -m orvue_us_inverse <command> [options]  (or the console script: orvue-us-inverse <command> [options])

Developer menu: type a step's number and press Enter. The step runs in its own process and the menu stays in this
console: typing another number closes the running step and starts the new one. 0 exits.

Commands (extra options are passed on to the module):

    viewer      check the probe tracking (D405)      orvue_us_inverse.tracking.viewer
    calibrate   probe calibration                    orvue_us_inverse.tracking.calibrate -> config/calibration.json
    clinical    Ultrasound Imaging Simulator window  orvue_us_inverse.clinical           [--tab bmode|mapping] [--case X] [--no-camera]
    sim         Ultrasound Imaging Simulator         orvue_us_inverse.simulation.bmode   [case] [--track] [--cam-view] [--mouse]
    scripted    scripted sweep, live reconstruction  orvue_us_inverse.mapping.run_scripted [--case X] [--yaw 0 90] [--no-images] [--live3d] ...
    mouse       hand-guided sweep, one window        orvue_us_inverse.mapping.run_mouse [--case X] [--yaw 0] [--no-images]
    tracked     sweep with the tracked dummy probe   orvue_us_inverse.mapping.run_tracked [--inject jitter=0.5,...] [--mouse]
    recon       reconstruct a saved sweep            orvue_us_inverse.mapping.reconstruct_sweep [sweep.npz] [--voxel 0.5] [--fill] -> output/results/
    evaluate    evaluate a saved sweep (report)      orvue_us_inverse.mapping.evaluate_sweep [sweep.npz] [--voxel 0.5] [--fill] -> output/results/
    experiments sweep-strategy study (headless)      orvue_us_inverse.mapping.run_experiments [--workers N] [--quick] [--yes] -> output/results/
    run         example step                         orvue_us_inverse.core.example
    test        all tests                            pytest tests
    board       tracking board PDF                   orvue_us_inverse.tracking.board     -> docs/print/
    anatomy     3D anatomy viewer (browser)          orvue_us_3inverse.viewer3d           [--case X] [--export]
    manual      rebuild the user manual PDF          orvue_us_inverse.reports.manual     -> docs/

Every step runs from the repository folder; the files it reads and writes are fixed in orvue_us_inverse.paths.
Add a step: one entry in COMMANDS and one row in MENU (developer menu).
"""
import os
import subprocess
import sys
import threading
import time

from orvue_us_inverse.paths import REPO_ROOT as ROOT

RELEASE_S = 1.0                 # pause after stopping a step (e.g. so a camera is free for the next one)

# command -> (argv after the python executable, working directory)
COMMANDS = {
    "viewer": (["-m", "orvue_us_inverse.tracking.viewer"], ROOT),
    "calibrate": (["-m", "orvue_us_inverse.tracking.calibrate"], ROOT),
    "clinical": (["-m", "orvue_us_inverse.clinical"], ROOT),
    "sim": (["-m", "orvue_us_inverse.simulation.bmode"], ROOT),
    "scripted": (["-m", "orvue_us_inverse.mapping.run_scripted"], ROOT),
    "mouse": (["-m", "orvue_us_inverse.mapping.run_mouse"], ROOT),
    "tracked": (["-m", "orvue_us_inverse.mapping.run_tracked"], ROOT),
    "recon": (["-m", "orvue_us_inverse.mapping.reconstruct_sweep"], ROOT),
    "evaluate": (["-m", "orvue_us_inverse.mapping.evaluate_sweep"], ROOT),
    "experiments": (["-m", "orvue_us_inverse.mapping.run_experiments"], ROOT),
    "run": (["-m", "orvue_us_inverse.core.example"], ROOT),
    "test": (["-m", "pytest", "tests", "-q"], ROOT),
    "board": (["-m", "orvue_us_inverse.tracking.board"], ROOT),
    "anatomy": (["-m", "orvue_us_inverse.viewer3d"], ROOT),
    "manual": (["-m", "orvue_us_inverse.reports.manual"], ROOT),
}

DEFAULT_COMMAND = "clinical"      # python -m orvue_us_inverse without arguments: the one clinical window

# developer menu (python -m orvue_us_inverse dev), in the order of use:
# (group, number, name, what it does, command, options)
MENU = [
    ("CAMERA TRACKING", "1", "Check tracking", "camera view, phantom map, readouts", "viewer", []),
    ("CAMERA TRACKING", "2", "Calibrate probe", "yaw and face position", "calibrate", []),
    ("SIMULATOR", "3", "Ultrasound Imaging Simulator", "B-mode; m camera / mouse, t camera view", "sim", ["--cam-view"]),
    ("INVERSE MAPPING", "4", "Scripted sweep", "yaw 0 + 90, live reconstruction + 3D; s saves", "scripted",
     ["--yaw", "0", "90", "--live3d"]),
    ("INVERSE MAPPING", "5", "Mouse sweep", "sweep | B-mode | 3D; hold L to record, c evaluates", "mouse", []),
    ("INVERSE MAPPING", "6", "Tracked sweep", "camera-tracked probe, hold SPACE; m: mouse", "tracked", []),
    ("INVERSE MAPPING", "7", "Reconstruct latest sweep", "label volume + slice PNGs -> output/results/", "recon",
     []),
    ("INVERSE MAPPING", "8", "Evaluate latest sweep", "report vs ground truth -> output/results/", "evaluate",
     ["--fill"]),
    ("INVERSE MAPPING", "9", "Sweep-strategy experiments", "headless study (~11 min) -> output/results/",
     "experiments", []),
    ("INVERSE MAPPING", "10", "Tracking-error study", "jitter / latency / bias (~11 min) -> output/results/",
     "experiments", ["--errors"]),
    ("MAIN", "11", "Run example", "writes output/logs/example.txt", "run", []),
    ("TOOLS", "12", "Run all tests", "no camera needed", "test", []),
    ("TOOLS", "13", "Tracking board PDF", "regenerate docs/print/tracking_board.pdf", "board", []),
    ("TOOLS", "14", "3D anatomy viewer", "every case in the browser (three.js)", "anatomy", []),
    ("TOOLS", "15", "User manual PDF", "rebuild the manual from the code", "manual", []),
]
PROMPT = "Type a number and press Enter: "
def command_line(command, options):
    argv, cwd = COMMANDS[command]
    return [sys.executable, *argv, *options], cwd


def shown(cmd):
    return " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) and c.startswith(ROOT) else c for c in cmd[1:])


def run(command, options=()):
    """Run one command to completion (non-menu use); returns its exit code."""
    cmd, cwd = command_line(command, options)
    print(f"[main] {shown(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=cwd)


# ---------------------------------------------------------------- menu
def print_menu(running=None, items=None, title=None, footer=""):
    items = MENU if items is None else items
    line = "=" * 74
    print(f"\n{line}\n  {title or 'Orvue Surgical - Ultrasound Inverse Anatomy Mapping (developer)'}\n{line}")
    group = None
    for g, num, name, what, _, _ in items:
        if g != group:
            print(f"  {g}")
            group = g
        print(f"   {num:>2}  {name:<30} {what}")
    print("  " + "-" * 72)
    print("    0  Exit")
    if footer:
        print(f"\n  {footer}")
    if running:
        print(f"\n  Now running: {running}")
        print("  Type another number to close it and start that step instead.")
    print(flush=True)


class Runner:
    """Runs one menu step at a time in the background; starting another stops the current one."""

    def __init__(self, show_menu=print_menu):
        self.show_menu = show_menu
        self.proc, self.title = None, None
        self.lock = threading.Lock()

    def running(self):
        with self.lock:
            return self.title if self.proc is not None and self.proc.poll() is None else None

    def start(self, title, cmd, cwd):
        self.stop()
        print(f"[main] starting {title}: {shown(cmd)}", flush=True)
        with self.lock:
            # the steps take keys from their own windows; keep the console input for the menu
            self.proc, self.title = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL), title
            proc = self.proc
        threading.Thread(target=self._watch, args=(proc, title), daemon=True).start()

    def stop(self):
        with self.lock:
            proc, title = self.proc, self.title
            self.proc = self.title = None
        if proc is not None and proc.poll() is None:
            print(f"[main] closing {title} ...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(RELEASE_S)

    def _watch(self, proc, title):
        code = proc.wait()
        with self.lock:
            if proc is not self.proc:            # stopped or replaced from the menu: nothing to report
                return
            self.proc = self.title = None
        print(f"\n[main] {title} finished" + ("" if code == 0 else f" (exit code {code})") + ".", flush=True)
        self.show_menu()
        print(PROMPT, end="", flush=True)


def menu(items=None, title=None, footer=""):
    items = MENU if items is None else items

    def show(running=None):
        print_menu(running, items, title, footer)

    runner = Runner(show)
    show()
    try:
        while True:
            try:
                choice = input(PROMPT).strip()
            except EOFError:                     # no console input (e.g. started from a pipe)
                choice = "0"
            if choice == "":
                show(runner.running())
                continue
            if choice == "0":
                return 0
            item = next((m for m in items if m[1] == choice), None)
            if item is None:
                print(f"  '{choice}' is not on the menu: type 0-{len(items)}.")
                continue
            _, num, name, what, command, options = item
            title = f"{num} {name} ({what})"
            if runner.running() == title:
                print(f"  {title} is already running.")
                continue
            cmd, cwd = command_line(command, options)
            runner.start(title, cmd, cwd)
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        runner.stop()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("[main] Ultrasound Imaging Simulator (developer menu: python -m orvue_us_inverse dev)", flush=True)
        return run(DEFAULT_COMMAND)
    if argv[0] == "dev":
        return menu()
    if argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] not in COMMANDS:
        print(f"[main] unknown command {argv[0]!r}; choose from: {', '.join(COMMANDS)}")
        return 2
    return run(argv[0], argv[1:])


if __name__ == "__main__":
    sys.exit(main())
