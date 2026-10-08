"""
orvue_us_inverse.viewer3d.viewer - interactive 3D viewer of the simulator's virtual anatomy (browser, three.js / WebGL).

    python -m orvue_us_inverse.viewer3d --case normal        build (cached) and open the viewer in the browser
    python -m orvue_us_inverse.viewer3d --case normal --export   write the case's meshes as STL + VTP to output/export/
    python -m orvue_us_inverse.viewer3d --screenshot out.png --state "liver=0"     headless screenshot (Microsoft Edge)

The meshes come from anatomy.py itself (viewer3d.geometry); this module packs them (quantised and
compressed, identical structures shared between cases) into one self-contained HTML page with the
JavaScript viewer (template.html). three.js is loaded from the jsDelivr CDN, so the first load needs
internet access.

Camera presets (also used by the page and the tests): offset = direction from the target to the camera.
    1 isometric (default, parallel projection)      offset (-1, -1, -1), up (0, 0, -1): surface at the top
    2 probe's view from above                       offset ( 0,  0, -1), up (0, -1, 0): x right, y down
    3 side view looking caudally                    offset ( 0, -1,  0), up (0, 0, -1)
    4 view from the patient's right                 offset (-1,  0,  0), up (0, 0, -1)
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import time
import webbrowser
import zlib

import numpy as np

from orvue_us_inverse import paths  # noqa: E402
from orvue_us_inverse.simulation.anatomy import CASES, build_case  # noqa: E402

from . import geometry as geo  # noqa: E402
from . import info  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "template.html")
OUT_DIR = paths.VIEWER3D_OUT_DIR
EXPORT_DIR = paths.EXPORT_DIR
TARGET = (50.0, 50.0, 25.0)               # centre of the 100 x 100 x 50 mm volume
PRESETS = {
    "1": dict(name="Isometric", offset=(-1.0, -1.0, -1.0), up=(0.0, 0.0, -1.0)),
    "2": dict(name="Probe's view from above", offset=(0.0, 0.0, -1.0), up=(0.0, -1.0, 0.0)),
    "3": dict(name="Side view looking caudally", offset=(0.0, -1.0, 0.0), up=(0.0, 0.0, -1.0)),
    "4": dict(name="View from the patient's right", offset=(-1.0, 0.0, 0.0), up=(0.0, 0.0, -1.0)),
}
DEFAULT_PRESET = "1"
DEFAULT_PARALLEL = True
TOGGLES = {                               # name: (default, key, label)
    "labels": (True, "l", "Labels"),
    "liver": (True, "v", "Liver"),
    "fat": (False, "f", "Fat layer"),
    "gb_translucent": (True, "g", "Gallbladder translucent"),
    "lumens": (False, "u", "Lumens inside translucent walls"),
    "calot": (False, "c", "Calot's triangle"),
    "surface": (True, "b", "Phantom surface + 100 mm region"),
    "probe": (False, "k", "Probe plane"),
}


# ---------------------------------------------------------------- camera maths (shared with the page and tests)
def view_basis(preset):
    """(right, screen_up, forward) unit vectors of a preset camera; screen_up = towards the top of the screen."""
    p = PRESETS[str(preset)]
    f = -np.asarray(p["offset"], float)
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(p["up"], float))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


def project(points, preset, parallel=True, target=TARGET, distance=400.0):
    """Screen coordinates (x to the right, y DOWNWARDS, like pixel rows) of world points for a preset camera,
    up to scale. Parallel projection by default; perspective divides by the depth along the view axis."""
    r, u, f = view_basis(preset)
    d = np.atleast_2d(np.asarray(points, float)) - np.asarray(target, float)
    sx, sy, depth = d @ r, -(d @ u), d @ f + distance
    if not parallel:
        sx, sy = sx / depth, sy / depth
    return np.stack([sx, sy], 1)


# ---------------------------------------------------------------- payload
def _pack(arr):
    return base64.b64encode(zlib.compress(np.ascontiguousarray(arr).tobytes(), 9)).decode("ascii")


def encode_mesh(v, f):
    """Positions quantised to uint16 over the mesh bounds (resolution < 2 um), faces uint16 / uint32."""
    v = np.asarray(v, np.float64)
    lo, hi = v.min(0), v.max(0)
    scale = np.where(hi > lo, (hi - lo) / 65535.0, 1.0)
    q = np.round((v - lo) / scale).astype(np.uint16)
    big = len(v) > 65535
    return dict(nv=int(len(v)), nf=int(len(f)), lo=lo.tolist(), scale=scale.tolist(), pos=_pack(q),
                idx=_pack(np.asarray(f, np.uint32 if big else np.uint16)), idx32=bool(big))


def case_entry(an, meshes, quality="display", cache=True):
    """Structures of one case (meshes stored once in `meshes`, keyed by their geometry hash)."""
    structures = []
    for it in geo.case_meshes(an, cache=cache, quality=quality):
        v, f = it["mesh"]
        if len(f) == 0:
            continue
        if it["key"] not in meshes:
            meshes[it["key"]] = encode_mesh(v, f)
        name, prim = it["name"], it["prim"]
        hex_c, opacity = info.colour(name, prim)
        if it["kind"] == "tube":
            anchor = geo.tube_anchor(prim)
        elif it["kind"] == "blob":
            anchor = np.asarray(prim.centre)
        elif it["kind"] == "calot":
            anchor = v.mean(0)
        elif it["kind"] == "liver":
            anchor = np.array([20.0, 20.0, 12.0])
        else:
            anchor = np.array([92.0, 95.0, 1.0])
        structures.append(dict(id=it["id"], name=name, kind=it["kind"], part=it["part"], mesh=it["key"],
                               pretty=info.pretty_name(name), desc=info.description(name, prim),
                               colour=hex_c, opacity=opacity, cls=info.colour_class(name, prim),
                               anchor=[float(a) for a in anchor]))
    return dict(name=an.name, description=info.case_description(an.name), structures=structures)


def build_payload(cases=None, quality="display", cache=True):
    cases = list(CASES) if cases is None else list(cases)
    meshes = {}
    entries = [case_entry(build_case(c), meshes, quality, cache) for c in cases]
    w, d = geo.probe_size()
    return dict(cases=entries, meshes=meshes, presets=PRESETS, default_preset=DEFAULT_PRESET,
                default_parallel=DEFAULT_PARALLEL, target=TARGET, toggles=TOGGLES,
                probe=dict(width=w, depth=d), region=[100.0, 100.0, 50.0])


def write_html(path, payload, start_case="normal"):
    with open(TEMPLATE, encoding="utf-8") as fh:
        html = fh.read()
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")      # safe inside <script>
    logo = paths.LOGO_PATH
    logo_uri = ("data:image/jpeg;base64," + base64.b64encode(open(logo, "rb").read()).decode("ascii")
                if os.path.exists(logo) else "")
    html = (html.replace("/*__PAYLOAD__*/null", data).replace("__START_CASE__", start_case)
            .replace("__LOGO__", logo_uri))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


# ---------------------------------------------------------------- open from other tools
DEFAULT_HTML = os.path.join(OUT_DIR, "anatomy_viewer.html")
SOURCES = [os.path.join(paths.PACKAGE_DIR, "simulation", "anatomy.py"),
           os.path.join(paths.PACKAGE_DIR, "simulation", "bmode.py"), TEMPLATE,
           os.path.join(HERE, "geometry.py"), os.path.join(HERE, "info.py"), os.path.abspath(__file__)]


def ensure_page(path=DEFAULT_HTML, quality="display"):
    """Build the page if it is missing or older than the anatomy / viewer sources; returns its path."""
    newest = max(os.path.getmtime(p) for p in SOURCES if os.path.exists(p))
    if not os.path.exists(path) or os.path.getmtime(path) < newest:
        write_html(path, build_payload(quality=quality))
    return path


def open_page(state="", path=DEFAULT_HTML):
    """Open the viewer in the default browser with a start state (URL hash such as
    'case=normal&probe=1&px=55&py=70&pyaw=30'). Windows drops '#...' from file links passed to the browser, so
    a small redirect page next to the viewer carries the state."""
    page = ensure_page(path)
    launcher = os.path.join(os.path.dirname(page), "open_viewer.html")
    target = os.path.basename(page) + ("#" + state if state else "")
    with open(launcher, "w", encoding="utf-8") as fh:
        fh.write('<!doctype html><meta charset="utf-8"><title>Opening the 3D anatomy viewer</title>'
                 f'<meta http-equiv="refresh" content="0; url={target}">'
                 f'<script>location.replace({json.dumps(target)});</script>')
    webbrowser.open("file:///" + os.path.abspath(launcher).replace("\\", "/"))
    return page


# ---------------------------------------------------------------- headless screenshot
def find_edge():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if os.path.exists(p):
            return p
    return shutil.which("msedge")


def screenshot(html_path, png_path, state="", size=(1600, 1000), wait_ms=12000):
    """Render the page in headless Microsoft Edge (software WebGL) and save a PNG. state: URL hash such as
    'case=normal&liver=0&view=2'."""
    edge = find_edge()
    if edge is None:
        raise RuntimeError("Microsoft Edge not found for headless screenshots")
    url = "file:///" + os.path.abspath(html_path).replace("\\", "/") + ("#" + state if state else "")
    png_path = os.path.abspath(png_path)
    if os.path.exists(png_path):
        os.remove(png_path)
    profile = os.path.join(OUT_DIR, "edge_profile")
    cmd = [edge, "--headless=new", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
           "--hide-scrollbars", f"--user-data-dir={profile}", f"--window-size={size[0]},{size[1]}",
           f"--virtual-time-budget={wait_ms}", f"--screenshot={png_path}", url]
    subprocess.run(cmd, capture_output=True, timeout=180)
    if not os.path.exists(png_path):
        raise RuntimeError("headless Edge did not write the screenshot")
    return png_path


def render_check(html_path, state="", wait_ms=12000):
    """Load the page in headless Edge and return its DOM after rendering (for checks: the page sets
    data-ready="1" on <body> once the case is built; errors are written into #status)."""
    edge = find_edge()
    if edge is None:
        raise RuntimeError("Microsoft Edge not found")
    url = "file:///" + os.path.abspath(html_path).replace("\\", "/") + ("#" + state if state else "")
    cmd = [edge, "--headless=new", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
           f"--user-data-dir={os.path.join(OUT_DIR, 'edge_profile')}", "--window-size=1400,900",
           f"--virtual-time-budget={wait_ms}", "--dump-dom", url]
    return subprocess.run(cmd, capture_output=True, timeout=180, text=True, encoding="utf-8").stdout


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="3D anatomy viewer for the Ultrasound Imaging Simulator's virtual anatomy.")
    ap.add_argument("--case", default="normal", choices=list(CASES), help="case shown first")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "anatomy_viewer.html"), help="HTML file to write")
    ap.add_argument("--quality", default="display", choices=list(geo.QUALITY), help="mesh resolution")
    ap.add_argument("--no-open", action="store_true", help="write the page but do not open the browser")
    ap.add_argument("--no-cache", action="store_true", help="rebuild all meshes")
    ap.add_argument("--export", action="store_true", help="write the case's meshes (full quality) as STL + VTP")
    ap.add_argument("--screenshot", help="save a headless screenshot (PNG) instead of opening the browser")
    ap.add_argument("--state", default="", help='view state for --screenshot, e.g. "liver=0&view=2"')
    a = ap.parse_args(argv)

    if a.export:
        paths = geo.export_case(build_case(a.case), EXPORT_DIR, cache=not a.no_cache)
        print(f"[viewer] exported {len(paths)} files to {os.path.join(EXPORT_DIR, a.case)}")
        return 0
    t0 = time.time()
    payload = build_payload(quality=a.quality, cache=not a.no_cache)
    path = write_html(a.out, payload, a.case)
    print(f"[viewer] {path} ({os.path.getsize(path) / 1e6:.1f} MB, built in {time.time() - t0:.1f} s)")
    if a.screenshot:
        state = "&".join(s for s in (f"case={a.case}", a.state) if s)
        print("[viewer] screenshot:", screenshot(path, a.screenshot, state))
    elif not a.no_open:
        webbrowser.open("file:///" + os.path.abspath(path).replace("\\", "/"))
    return 0
