"""
orvue_us_inverse.mapping.run_scripted - live playback of a scripted sweep over the hidden 100 x 100 mm box, with the
reconstruction growing as frames are captured (S1, S3).

    python -m orvue_us_inverse.mapping.run_scripted [--case normal] [--yaw 0 90] [--overlap 20] [--spacing 0.5]
                                                     [--speed 10] [--no-images] [--live3d]
    python -m orvue_us_inverse scripted [options]

Windows:
  "Scripted sweep"           top view of the region at 4 px/mm (+x right, +y down); key v cycles black box (anatomy
                             hidden) / coverage (hits summed over depth, heat map) / revealed anatomy
                             (simulation.bmode.top_view); planned lanes (thin), the probe footprint at its pose (30 mm
                             line, amber dot at the orientation-mark end, grey during lift-off), one line per captured
                             frame; status panel.
  "Scripted sweep - B-mode"  the latest captured frame, 501 x 301 (oracle labels in colour with --no-images).
  "Scripted sweep - slices"  three orthogonal slices of the reconstruction through a crosshair (render.SliceView),
                             refreshed every 0.5 s; click, arrow keys (x, y) and PgUp / PgDn or [ / ] (depth) move it;
                             g adds the ground-truth contours.
  "Reconstruction 3D"        optional live PyVista window (--live3d or key p), surfaces refreshed every 3 s.

Keys: space pause / resume, + / - playback speed, v top view, r restart, s save the sweep
(output/sweeps/<case>_<settings>_<time>.npz), 3 3D snapshot PNG (output/results/) + STL per structure
(output/export/recon_<case>_<time>/), b browser 3D view (output/viewer3d/recon_<case>.html, three.js),
p live 3D window on / off, g ground-truth contours, q / Esc quit.

The sweep runs on simulated time (poses every 0.05 mm, mapping.poses.ScriptedSweep); frames are captured by
distance (mapping.acquisition.Acquirer) and inserted into the reconstruction (mapping.recon.LabelCompounder) as they
arrive. Rendering is synchronous, so when it is slower than real time the playback slows down; no frame is dropped.
"""
import argparse
import os
import time
import webbrowser

import cv2
import numpy as np

from orvue_us_inverse.mapping.acquisition import Acquirer
from orvue_us_inverse.mapping.config import AcquisitionConfig, GridConfig, SweepConfig
from orvue_us_inverse.mapping.poses import STEP_MM, ScriptedSweep
from orvue_us_inverse.mapping.probe import FRAME_SHAPE, PROBE, make_simulator, probe_metadata
from orvue_us_inverse.mapping.recon import LabelCompounder, VoxelGrid, ground_truth
from orvue_us_inverse.mapping.render import (SliceView, export_stl, snapshot_3d, surface_meshes,
                                             write_browser_view)
from orvue_us_inverse.paths import EXPORT_DIR, RESULTS_DIR, SWEEPS_DIR, VIEWER3D_OUT_DIR
from orvue_us_inverse.simulation.anatomy import CASES, COL_TAB
from orvue_us_inverse.simulation.bmode import top_view
from orvue_us_inverse.ui import clinical as ui

WIN_TOP = "Scripted sweep"
WIN_BMODE = "Scripted sweep - B-mode"
WIN_SLICES = "Scripted sweep - slices"
PX_PER_MM = 4.0
MARGIN = 40                       # px around the region (footprints of oblique lanes reach outside it)
PANEL_W = 330
KEYS_H = 68                       # two rows of keys
HELP_H = 28                       # help strip under the slices
TRACE_BGR = [(235, 170, 70), (80, 200, 255), (140, 230, 120), (200, 120, 230)]     # per orientation
PLAY_STEPS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
MAX_WALL_STEP_S = 0.1             # simulated time per loop is at most this much wall time x playback speed
VIEWS = ("hidden", "coverage", "revealed")
SLICE_PERIOD_S = 0.5
LIVE3D_PERIOD_S = 3.0
# cv2.waitKeyEx codes (Windows)
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN, KEY_PGUP, KEY_PGDN = 2424832, 2490368, 2555904, 2621440, 2162688, 2228224


def sweep_filename(case: str, cfg: SweepConfig, store_images: bool, stamp: str | None = None) -> str:
    """<case>_<settings>_<time>.npz, e.g. normal_yaw0-90_ov20_sp0.5_v10_20261008-141500.npz."""
    yaws = "-".join(f"{y:g}" for y in cfg.yaw_list_deg)
    settings = f"yaw{yaws}_ov{cfg.overlap_pct:g}_sp{cfg.frame_spacing_mm:g}_v{cfg.speed_mm_s:g}"
    if not store_images:
        settings += "_labels"
    return f"{case}_{settings}_{stamp or time.strftime('%Y%m%d-%H%M%S')}.npz"


class ScriptedPlayback:
    """Sweep and reconstruction state, stepping on simulated time and drawing; no windows (main() shows them)."""

    def __init__(self, case: str, cfg: SweepConfig, acq_cfg: AcquisitionConfig, grid_cfg: GridConfig | None = None):
        self.case, self.cfg, self.acq_cfg = case, cfg, acq_cfg
        self.sim = make_simulator(case)
        self.plan = ScriptedSweep(cfg)
        self.samples = list(self.plan.samples(STEP_MM))
        self.expected = self.plan.expected_frames()
        self.size = int(round(100 * PX_PER_MM)) + 2 * MARGIN
        self.grid = VoxelGrid(grid_cfg or GridConfig())
        self.slice_view = SliceView(self.grid)
        self._revealed_img = None
        self._gt = None
        self._gt_meshes = None
        self.view = 0                                # index into VIEWS
        self.play = 1.0
        self.paused = False
        self.message = ""
        self.restart()

    # ---- state
    def restart(self) -> None:
        self.acq = Acquirer(self.sim, self.cfg, self.acq_cfg)
        self.comp = LabelCompounder(self.grid, probe=probe_metadata())
        self.i = 0                                   # next sample to feed
        self.sample = self.samples[0]
        self.latest = None                           # latest FrameRecord
        self.trace = np.zeros((self.size, self.size, 3), np.uint8)
        self.trace_mask = np.zeros((self.size, self.size), bool)
        self.wall_rate = None                        # simulated s per wall s, smoothed
        self._coverage = (-1, None)                  # (frames when computed, image)
        self.observed_pct = 0.0

    @property
    def done(self) -> bool:
        return self.i >= len(self.samples)

    @property
    def t(self) -> float:
        return self.sample.t

    def advance(self, sim_dt: float) -> int:
        """Feed the samples up to t + sim_dt, stopping after the first captured frame (so every frame is shown);
        a captured frame is inserted into the reconstruction. Returns the number of frames captured (0 or 1)."""
        target = self.t + sim_dt
        while not self.done and self.samples[self.i].t <= target + 1e-12:
            s = self.samples[self.i]
            self.i += 1
            self.sample = s
            frame = self.acq.feed(s.T, s.t, s.recording)
            if frame is not None:
                self.latest = frame
                self.comp.insert(frame)
                self._draw_trace(frame.T_true, s.lane.yaw_deg)
                return 1
        return 0

    def save(self, folder: str = SWEEPS_DIR) -> str:
        sweep = self.acq.sweep
        sweep.metadata.update(complete=self.done, planned=self.plan.summary(), expected_frames=self.expected)
        path = sweep.save(os.path.join(folder, sweep_filename(self.case, self.cfg, self.acq_cfg.store_images)))
        self.message = f"saved {os.path.basename(path)} ({len(sweep)} frames)"
        return path

    # ---- ground truth and 3D outputs
    def gt_volume(self) -> np.ndarray:
        if self._gt is None:
            self._gt = ground_truth(self.sim.an, self.grid)
        return self._gt

    def gt_meshes(self) -> dict:
        if self._gt_meshes is None:
            self._gt_meshes = surface_meshes(self.gt_volume(), self.grid)
        return self._gt_meshes

    def toggle_gt_contours(self) -> None:
        v = self.slice_view
        if v.gt is None:
            v.gt = self.gt_volume()
            v.show_gt = True
        else:
            v.show_gt = not v.show_gt

    def meshes(self) -> dict:
        return surface_meshes(self.comp.result(), self.grid)

    def export_3d(self, results: str = RESULTS_DIR, export: str = EXPORT_DIR) -> tuple[str, list[str]]:
        """3D snapshot PNG (ground truth translucent) and one STL per structure; returns (png, stl paths)."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        meshes = self.meshes()
        title = f"{self.case}: {len(self.acq.sweep)} frames, reconstruction (solid) vs ground truth (translucent)"
        png = snapshot_3d(meshes, os.path.join(results, f"snapshot_{self.case}_{stamp}.png"), gt=self.gt_meshes(),
                          title=title)
        stls = export_stl(meshes, os.path.join(export, f"recon_{self.case}_{stamp}"))
        self.message = f"3D snapshot {os.path.basename(png)}; {len(stls)} STL files in output/export/"
        return png, stls

    def browser_view(self, folder: str = VIEWER3D_OUT_DIR) -> str:
        path = write_browser_view(self.meshes(), os.path.join(folder, f"recon_{self.case}.html"), gt=self.gt_meshes(),
                                  title=f"{self.case}, {len(self.acq.sweep)} frames")
        self.message = f"browser view {os.path.basename(path)}"
        return path

    # ---- drawing: top view
    def px(self, xy) -> tuple[int, int]:
        return int(round(MARGIN + xy[0] * PX_PER_MM)), int(round(MARGIN + xy[1] * PX_PER_MM))

    def _footprint(self, T: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        o, u = T[:2, 3], T[:2, 0]
        h = PROBE.width_mm / 2
        return self.px(o - h * u), self.px(o + h * u)       # (orientation-mark end = probe -x, other end)

    def _draw_trace(self, T: np.ndarray, yaw: float) -> None:
        colour = TRACE_BGR[self.cfg.yaw_list_deg.index(yaw) % len(TRACE_BGR)]
        a, b = self._footprint(T)
        cv2.line(self.trace, a, b, colour, 1, cv2.LINE_AA)
        m = np.zeros(self.trace_mask.shape, np.uint8)
        cv2.line(m, a, b, 255, 1, cv2.LINE_AA)
        self.trace_mask |= m > 0

    def cycle_view(self) -> None:
        self.view = (self.view + 1) % len(VIEWS)
        if VIEWS[self.view] == "revealed" and self._revealed_img is None:
            self._revealed_img = top_view(self.sim.an, px_mm=1.0 / PX_PER_MM)

    def coverage_image(self) -> np.ndarray:
        """Region-sized heat map (rows y, cols x) of the hits summed over depth; log scale, black = never imaged."""
        n = len(self.acq.sweep)
        if self._coverage[0] != n:
            hits = self.comp.column_hits().T.astype(np.float64)          # (ny, nx)
            top = np.log1p(hits.max()) if hits.max() > 0 else 1.0
            v = (255 * np.log1p(hits) / top).astype(np.uint8)
            img = cv2.applyColorMap(v, cv2.COLORMAP_INFERNO)
            img[hits == 0] = 0
            r = int(round(100 * PX_PER_MM))
            self._coverage = (n, cv2.resize(img, (r, r), interpolation=cv2.INTER_NEAREST))
        return self._coverage[1]

    def top_image(self) -> np.ndarray:
        """Top view (black box / coverage / revealed anatomy) with planned lanes, trace and the probe footprint."""
        n = self.size
        img = np.full((n, n, 3), 28, np.uint8)
        r0, r1 = MARGIN, MARGIN + int(round(100 * PX_PER_MM))
        mode = VIEWS[self.view]
        if mode == "revealed" and self._revealed_img is not None:
            img[r0:r1, r0:r1] = self._revealed_img
        elif mode == "coverage":
            img[r0:r1, r0:r1] = self.coverage_image()
        else:
            img[r0:r1, r0:r1] = 0
        if mode != "coverage":
            m = self.trace_mask
            alpha = 0.3 if mode == "revealed" else 1.0       # revealed: the anatomy shows through the trace
            img[m] = (alpha * self.trace[m] + (1 - alpha) * img[m]).astype(np.uint8)
        cv2.rectangle(img, (r0 - 1, r0 - 1), (r1, r1), ui.BORDER, 1)
        cur = self.sample.lane
        for ln in self.plan.planned_lanes():
            colour = ui.AMBER if ln is cur else ui.DIM
            cv2.line(img, self.px(ln.start), self.px(ln.end), colour, 1, cv2.LINE_AA)
        a, b = self._footprint(self.sample.T)
        rec = self.sample.recording
        cv2.line(img, a, b, ui.GREEN if rec else ui.GREY, 3, cv2.LINE_AA)
        cv2.circle(img, a, 5, ui.AMBER, -1, cv2.LINE_AA)
        if not rec:
            ui.text(img, "lift-off", (b[0] + 6, b[1] + 4), ui.GREY, 0.42)
        ui.text(img, "x", (r1 - 12, r0 - 10), ui.GREY, 0.45)
        cv2.arrowedLine(img, (r1 - 60, r0 - 14), (r1 - 20, r0 - 14), ui.GREY, 1, cv2.LINE_AA, tipLength=0.25)
        ui.text(img, "y", (r0 - 30, r1 - 8), ui.GREY, 0.45)
        cv2.arrowedLine(img, (r0 - 16, r1 - 60), (r0 - 16, r1 - 20), ui.GREY, 1, cv2.LINE_AA, tipLength=0.25)
        ui.text(img, "0", (r0 - 14, r0 - 4), ui.DIM, 0.4)
        ui.text(img, "100 mm", (r1 - 50, r1 + 18), ui.DIM, 0.4)
        ui.text(img, {"hidden": "anatomy hidden", "coverage": "coverage: hits over depth (log)",
                      "revealed": "anatomy revealed"}[mode], (r0, r1 + 18), ui.GREY, 0.42)
        return img

    def status_rows(self) -> list[tuple]:
        lane = self.sample.lane
        yaws = self.cfg.yaw_list_deg
        if lane is not None:
            orient = f"yaw {lane.yaw_deg:g}  ({yaws.index(lane.yaw_deg) + 1}/{len(yaws)})"
            lane_s = f"{lane.k + 1}/{lane.n}"
        else:
            orient, lane_s = "lift-off", "---"
        ov = ", ".join(f"{self.plan.overlap_pct[y]:.1f}" for y in yaws)
        state = "done" if self.done else ("paused" if self.paused else "running")
        rate = "---" if self.wall_rate is None else f"{self.wall_rate:.2f}x real time"
        return [("case", self.case), ("orientation", orient), ("lane", lane_s),
                ("frames", f"{len(self.acq.sweep)} / {self.expected}"),
                ("overlap", f"{self.cfg.overlap_pct:g}% req, {ov}% act"),
                ("spacing", f"{self.cfg.frame_spacing_mm:g} mm, {self.cfg.speed_mm_s:g} mm/s"),
                ("sim time", f"{self.t:.1f} / {self.plan.duration_s:.1f} s"),
                ("playback", f"x{self.play:g}  ({rate})"),
                ("state", state, ui.GREEN if state == "running" else ui.AMBER),
                ("images", "B-mode + labels" if self.acq_cfg.store_images else "labels only"),
                ("observed", f"{self.observed_pct:.1f}% of the volume")]

    def compose(self) -> np.ndarray:
        """The "Scripted sweep" window: header, top view, status panel, key bar."""
        top = self.top_image()
        h = ui.HEADER_H + self.size + KEYS_H
        img = np.full((h, self.size + PANEL_W, 3), ui.BG, np.uint8)
        ui.header(img, "Scripted sweep", "inverse mapping")
        img[ui.HEADER_H:ui.HEADER_H + self.size, :self.size] = top
        x0, y = self.size + 16, ui.HEADER_H + 34
        ui.panel(img, (x0, y, PANEL_W - 30, self.size - 50), "STATUS")
        y = ui.rows(img, x0 + 10, y + 26, self.status_rows(), value_x=100)
        for line in ui.wrap(self.message, PANEL_W - 50, 0.42)[:4]:
            ui.text(img, line, (x0 + 10, y + 6), ui.GREY, 0.42)
            y += 18
        ui.key_bar(img, [("space", "pause"), ("+/-", "speed"), ("v", "top view"), ("r", "restart"),
                         ("s", "save sweep"), ("q", "quit")], h - KEYS_H)
        ui.key_bar(img, [("3", "3D snapshot + STL"), ("b", "browser 3D"), ("p", "live 3D"), ("g", "truth contours")],
                   h - KEYS_H // 2)
        return img

    # ---- drawing: B-mode and slices
    def bmode_image(self) -> np.ndarray:
        f = self.latest
        if f is None:
            img = np.zeros((*FRAME_SHAPE, 3), np.uint8)
            ui.text(img, "no frame yet", (90, 250), ui.GREY, 0.5)
            return img
        img = cv2.cvtColor(f.image, cv2.COLOR_GRAY2BGR) if f.image is not None else COL_TAB[f.labels].copy()
        cv2.circle(img, (8, 8), 4, ui.AMBER, -1, cv2.LINE_AA)                 # orientation mark = image left
        cap = f"frame {f.index}  t {f.t:.2f} s" + ("" if f.image is not None else "  (labels)")
        ui.text(img, cap, (18, 14), ui.WHITE, 0.4)
        return img

    def slice_image(self) -> np.ndarray:
        """Slices of the current reconstruction (only the three slices are computed) and a help strip."""
        self.observed_pct = 100.0 * np.count_nonzero(self.comp.hits) / self.grid.n_voxels
        img = self.slice_view.update(self.comp)
        out = np.full((img.shape[0] + HELP_H, img.shape[1], 3), ui.BG, np.uint8)
        out[:img.shape[0]] = img
        c = self.slice_view.crosshair_mm()
        gt = "on" if (self.slice_view.show_gt and self.slice_view.gt is not None) else "off"
        ui.text(out, f"crosshair ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) mm   click / arrows / PgUp PgDn or [ ]: move   "
                     f"g: truth contours ({gt})", (4, img.shape[0] + 18), ui.GREY, 0.42)
        return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scripted sweep over the hidden anatomy with live reconstruction.")
    p.add_argument("--case", default="normal", choices=sorted(CASES))
    p.add_argument("--yaw", type=float, nargs="+", default=[0.0], help="orientations in degrees, e.g. 0 90")
    p.add_argument("--overlap", type=float, default=20.0, help="requested lane overlap in percent")
    p.add_argument("--spacing", type=float, default=0.5, help="frame spacing in mm")
    p.add_argument("--speed", type=float, default=10.0, help="probe speed in mm/s (simulated time)")
    p.add_argument("--no-images", action="store_true", help="store oracle labels only (no B-mode rendering)")
    p.add_argument("--live3d", action="store_true", help="open the live PyVista 3D window at the start")
    return p.parse_args(argv)


def open_live3d(app: ScriptedPlayback):
    """Live PyVista window, or None (with a message) when PyVista / VTK cannot be used."""
    try:
        from orvue_us_inverse.mapping.live3d import Live3D
        live = Live3D()
        live.update(app.meshes(), gt=app.gt_meshes() if app.slice_view.show_gt else None)
        app.message = "live 3D window open (p closes it)"
        return live
    except Exception as e:                       # ImportError, blocked DLL, no OpenGL ...
        app.message = f"live 3D window not available ({type(e).__name__}: {e}); use 3 or b"
        print(f"[run_scripted] {app.message}", flush=True)
        return None


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = SweepConfig(yaw_list_deg=args.yaw, overlap_pct=args.overlap, frame_spacing_mm=args.spacing,
                      speed_mm_s=args.speed)
    app = ScriptedPlayback(args.case, cfg, AcquisitionConfig(store_images=not args.no_images))
    print("[run_scripted] " + "\n[run_scripted] ".join(app.plan.summary()), flush=True)
    for w in (WIN_TOP, WIN_BMODE, WIN_SLICES):
        cv2.namedWindow(w, cv2.WINDOW_AUTOSIZE)
    dirty = {"slices": True}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
            if app.slice_view.click(x, y):
                dirty["slices"] = True

    cv2.setMouseCallback(WIN_SLICES, on_mouse)
    live = open_live3d(app) if args.live3d else None
    live_frames = len(app.acq.sweep)
    last = time.perf_counter()
    last_slices = last_live = 0.0
    sim_dt = None
    while True:
        now = time.perf_counter()
        elapsed, last = now - last, now
        if not app.paused and not app.done:
            if sim_dt is not None:                       # simulated s per wall s over the previous loop
                rate = sim_dt / max(elapsed, 1e-6)
                app.wall_rate = rate if app.wall_rate is None else 0.9 * app.wall_rate + 0.1 * rate
            t0 = app.t
            app.advance(min(elapsed, MAX_WALL_STEP_S) * app.play)
            sim_dt = app.t - t0
            if app.done:
                app.message = f"sweep complete: {len(app.acq.sweep)} frames. s saves, 3 / b show the 3D result."
        else:
            sim_dt = None
        if dirty["slices"] or now - last_slices >= SLICE_PERIOD_S:
            cv2.imshow(WIN_SLICES, app.slice_image())
            last_slices, dirty["slices"] = now, False
        if live is not None:
            if live.closed:
                live = None
            else:
                if now - last_live >= LIVE3D_PERIOD_S and len(app.acq.sweep) != live_frames:
                    live.update(app.meshes(), gt=app.gt_meshes() if app.slice_view.show_gt else None)
                    live_frames, last_live = len(app.acq.sweep), now
                live.process()
        cv2.imshow(WIN_TOP, app.compose())
        cv2.imshow(WIN_BMODE, app.bmode_image())
        code = cv2.waitKeyEx(1)
        key = code & 0xFF if 0 <= code < 256 else -1
        move = {KEY_LEFT: (-1, 0, 0), KEY_RIGHT: (1, 0, 0), KEY_UP: (0, -1, 0), KEY_DOWN: (0, 1, 0),
                KEY_PGUP: (0, 0, -1), KEY_PGDN: (0, 0, 1)}.get(code)
        if key == ord("["):
            move = (0, 0, -1)
        elif key == ord("]"):
            move = (0, 0, 1)
        if move:
            app.slice_view.move(*move)
            dirty["slices"] = True
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            app.paused = not app.paused
        elif key in (ord("+"), ord("=")):
            app.play = next((p for p in PLAY_STEPS if p > app.play), app.play)
        elif key in (ord("-"), ord("_")):
            app.play = next((p for p in reversed(PLAY_STEPS) if p < app.play), app.play)
        elif key == ord("v"):
            if VIEWS[(app.view + 1) % len(VIEWS)] == "revealed" and app._revealed_img is None:
                app.message = "computing the anatomy top view ..."
                cv2.imshow(WIN_TOP, app.compose())
                cv2.waitKey(1)
            app.cycle_view()
            app.message = ""
        elif key == ord("g"):
            app.toggle_gt_contours()
            dirty["slices"] = True
            if live is not None:
                live.update(app.meshes(), gt=app.gt_meshes() if app.slice_view.show_gt else None)
        elif key == ord("r"):
            app.restart()
            sim_dt = None
            dirty["slices"] = True
            app.message = "restarted"
        elif key == ord("s"):
            print(f"[run_scripted] {app.save()}", flush=True)
        elif key == ord("3"):
            png, stls = app.export_3d()
            print(f"[run_scripted] {png}\n[run_scripted] " + "\n[run_scripted] ".join(stls), flush=True)
        elif key == ord("b"):
            path = app.browser_view()
            print(f"[run_scripted] {path}", flush=True)
            webbrowser.open("file:///" + os.path.abspath(path).replace(os.sep, "/"))
        elif key == ord("p"):
            if live is None:
                live = open_live3d(app)
                live_frames, last_live = len(app.acq.sweep), time.perf_counter()
            else:
                live.close()
                live = None
                app.message = "live 3D window closed"
        try:
            if cv2.getWindowProperty(WIN_TOP, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    if live is not None:
        live.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
