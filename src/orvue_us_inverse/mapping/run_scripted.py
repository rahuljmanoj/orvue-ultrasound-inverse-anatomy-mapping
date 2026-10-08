"""
orvue_us_inverse.mapping.run_scripted - live playback of a scripted sweep over the hidden 100 x 100 mm box (S1).

    python -m orvue_us_inverse.mapping.run_scripted [--case normal] [--yaw 0 90] [--overlap 20] [--spacing 0.5]
                                                     [--speed 10] [--no-images]
    python -m orvue_us_inverse scripted [options]

Two windows:
  "Scripted sweep"           top view of the region at 4 px/mm (+x right, +y down): black box (anatomy hidden) or,
                             key a, the revealed anatomy (simulation.bmode.top_view); planned lanes (thin), the
                             probe footprint at its pose (30 mm line, amber dot at the orientation-mark end, grey
                             during lift-off), one line per captured frame (coverage builds up); status panel.
  "Scripted sweep - B-mode"  the latest captured frame, 501 x 301 (oracle labels in colour with --no-images).

Keys: space pause / resume, + / - playback speed, a anatomy hidden / revealed, r restart,
s save the sweep to output/sweeps/<case>_<settings>_<time>.npz, q / Esc quit.

The sweep runs on simulated time (poses every 0.05 mm, mapping.poses.ScriptedSweep); frames are captured by
distance (mapping.acquisition.Acquirer). Rendering is synchronous, so when it is slower than real time the playback
slows down; no frame is dropped. The status panel shows the achieved rate (simulated s per wall-clock s).
"""
import argparse
import os
import time

import cv2
import numpy as np

from orvue_us_inverse.mapping.acquisition import Acquirer
from orvue_us_inverse.mapping.config import AcquisitionConfig, SweepConfig
from orvue_us_inverse.mapping.poses import STEP_MM, ScriptedSweep
from orvue_us_inverse.mapping.probe import FRAME_SHAPE, PROBE, make_simulator
from orvue_us_inverse.paths import SWEEPS_DIR
from orvue_us_inverse.simulation.anatomy import CASES, COL_TAB
from orvue_us_inverse.simulation.bmode import top_view
from orvue_us_inverse.ui import clinical as ui

WIN_TOP = "Scripted sweep"
WIN_BMODE = "Scripted sweep - B-mode"
PX_PER_MM = 4.0
MARGIN = 40                       # px around the region (footprints of oblique lanes reach outside it)
PANEL_W = 330
KEYS_H = 36
TRACE_BGR = [(235, 170, 70), (80, 200, 255), (140, 230, 120), (200, 120, 230)]     # per orientation
PLAY_STEPS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
MAX_WALL_STEP_S = 0.1             # simulated time per loop is at most this much wall time x playback speed


def sweep_filename(case: str, cfg: SweepConfig, store_images: bool, stamp: str | None = None) -> str:
    """<case>_<settings>_<time>.npz, e.g. normal_yaw0-90_ov20_sp0.5_v10_20261008-141500.npz."""
    yaws = "-".join(f"{y:g}" for y in cfg.yaw_list_deg)
    settings = f"yaw{yaws}_ov{cfg.overlap_pct:g}_sp{cfg.frame_spacing_mm:g}_v{cfg.speed_mm_s:g}"
    if not store_images:
        settings += "_labels"
    return f"{case}_{settings}_{stamp or time.strftime('%Y%m%d-%H%M%S')}.npz"


class ScriptedPlayback:
    """Sweep state, stepping on simulated time and drawing; no windows (main() shows the images)."""

    def __init__(self, case: str, cfg: SweepConfig, acq_cfg: AcquisitionConfig):
        self.case, self.cfg, self.acq_cfg = case, cfg, acq_cfg
        self.sim = make_simulator(case)
        self.plan = ScriptedSweep(cfg)
        self.samples = list(self.plan.samples(STEP_MM))
        self.expected = self.plan.expected_frames()
        self.size = int(round(100 * PX_PER_MM)) + 2 * MARGIN
        self._revealed_img = None
        self.revealed = False
        self.play = 1.0
        self.paused = False
        self.message = ""
        self.restart()

    # ---- state
    def restart(self) -> None:
        self.acq = Acquirer(self.sim, self.cfg, self.acq_cfg)
        self.i = 0                                   # next sample to feed
        self.sample = self.samples[0]
        self.latest = None                           # latest FrameRecord
        self.trace = np.zeros((self.size, self.size, 3), np.uint8)
        self.trace_mask = np.zeros((self.size, self.size), bool)
        self.wall_rate = None                        # simulated s per wall s, smoothed

    @property
    def done(self) -> bool:
        return self.i >= len(self.samples)

    @property
    def t(self) -> float:
        return self.sample.t

    def advance(self, sim_dt: float) -> int:
        """Feed the samples up to t + sim_dt, stopping after the first captured frame (so every frame is shown).
        Returns the number of frames captured (0 or 1)."""
        target = self.t + sim_dt
        while not self.done and self.samples[self.i].t <= target + 1e-12:
            s = self.samples[self.i]
            self.i += 1
            self.sample = s
            frame = self.acq.feed(s.T, s.t, s.recording)
            if frame is not None:
                self.latest = frame
                self._draw_trace(frame.T_true, s.lane.yaw_deg)
                return 1
        return 0

    def save(self, folder: str = SWEEPS_DIR) -> str:
        sweep = self.acq.sweep
        sweep.metadata.update(complete=self.done, planned=self.plan.summary(), expected_frames=self.expected)
        path = sweep.save(os.path.join(folder, sweep_filename(self.case, self.cfg, self.acq_cfg.store_images)))
        self.message = f"saved {os.path.basename(path)} ({len(sweep)} frames)"
        return path

    # ---- drawing
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

    def toggle_revealed(self) -> None:
        if self._revealed_img is None:
            self._revealed_img = top_view(self.sim.an, px_mm=1.0 / PX_PER_MM)
        self.revealed = not self.revealed

    def top_image(self) -> np.ndarray:
        """Top view with planned lanes, coverage trace and the probe footprint."""
        n = self.size
        img = np.full((n, n, 3), 28, np.uint8)
        r0, r1 = MARGIN, MARGIN + int(round(100 * PX_PER_MM))
        img[r0:r1, r0:r1] = self._revealed_img if (self.revealed and self._revealed_img is not None) else 0
        m = self.trace_mask
        alpha = 0.3 if self.revealed else 1.0           # revealed: the anatomy shows through the trace
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
                ("images", "B-mode + labels" if self.acq_cfg.store_images else "labels only")]

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
        for line in ui.wrap(self.message, PANEL_W - 50, 0.42)[:3]:
            ui.text(img, line, (x0 + 10, y + 6), ui.GREY, 0.42)
            y += 18
        ui.key_bar(img, [("space", "pause"), ("+/-", "speed"), ("a", "anatomy"), ("r", "restart"),
                         ("s", "save"), ("q", "quit")], h - KEYS_H)
        return img

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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scripted sweep over the hidden anatomy with live playback.")
    p.add_argument("--case", default="normal", choices=sorted(CASES))
    p.add_argument("--yaw", type=float, nargs="+", default=[0.0], help="orientations in degrees, e.g. 0 90")
    p.add_argument("--overlap", type=float, default=20.0, help="requested lane overlap in percent")
    p.add_argument("--spacing", type=float, default=0.5, help="frame spacing in mm")
    p.add_argument("--speed", type=float, default=10.0, help="probe speed in mm/s (simulated time)")
    p.add_argument("--no-images", action="store_true", help="store oracle labels only (no B-mode rendering)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = SweepConfig(yaw_list_deg=args.yaw, overlap_pct=args.overlap, frame_spacing_mm=args.spacing,
                      speed_mm_s=args.speed)
    app = ScriptedPlayback(args.case, cfg, AcquisitionConfig(store_images=not args.no_images))
    print("[run_scripted] " + "\n[run_scripted] ".join(app.plan.summary()), flush=True)
    cv2.namedWindow(WIN_TOP, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_BMODE, cv2.WINDOW_AUTOSIZE)
    last = time.perf_counter()
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
        else:
            sim_dt = None
            if app.done:
                app.message = f"sweep complete: {len(app.acq.sweep)} frames. s saves, r restarts."
        cv2.imshow(WIN_TOP, app.compose())
        cv2.imshow(WIN_BMODE, app.bmode_image())
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            app.paused = not app.paused
        elif key in (ord("+"), ord("=")):
            app.play = next((p for p in PLAY_STEPS if p > app.play), app.play)
        elif key in (ord("-"), ord("_")):
            app.play = next((p for p in reversed(PLAY_STEPS) if p < app.play), app.play)
        elif key == ord("a"):
            app.message = "computing the anatomy top view ..." if app._revealed_img is None else ""
            cv2.imshow(WIN_TOP, app.compose())
            cv2.waitKey(1)
            app.toggle_revealed()
            app.message = ""
        elif key == ord("r"):
            app.restart()
            sim_dt = None
            app.message = "restarted"
        elif key == ord("s"):
            print(f"[run_scripted] {app.save()}", flush=True)
        try:
            if cv2.getWindowProperty(WIN_TOP, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
