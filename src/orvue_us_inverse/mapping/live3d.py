"""
orvue_us_inverse.mapping.live3d - live 3D window of the reconstruction with PyVista / VTK (S3, optional).

    live = Live3D()                         # opens the window (off_screen=True for tests / screenshots)
    live.update(meshes, gt=gt_meshes)       # replace the surfaces (render.surface_meshes); gt translucent
    live.process()                          # call often: handles mouse rotate / zoom and redraws
    live.closed                             # True once the user closed the window
    live.close()

PyVista is an optional extra (pip install -e .[view3d]); importing this module raises ImportError without it, and
the apps fall back to the matplotlib snapshot (render.snapshot_3d). Camera as the 3D anatomy viewer's isometric
preset: looking from (-1, -1, -1) with up = -z, so the scanning surface z = 0 is on top and depth goes down.
"""
import numpy as np
import pyvista as pv

from orvue_us_inverse.mapping.render import GROUPS

REGION = (100.0, 100.0, 50.0)
GT_OPACITY = 0.18
BILE_OPACITY = 0.6                  # stones inside the gallbladder stay visible


def _polydata(v: np.ndarray, f: np.ndarray) -> "pv.PolyData":
    faces = np.hstack([np.full((len(f), 1), 3, np.int64), f.astype(np.int64)]).ravel()
    return pv.PolyData(np.asarray(v, np.float32), faces)


class Live3D:
    """A PyVista window that shows the latest surfaces; non-blocking (interactive_update)."""

    def __init__(self, title: str = "Reconstruction 3D", window_size=(900, 700), off_screen: bool = False):
        self.p = pv.Plotter(title=title, window_size=window_size, off_screen=off_screen)
        self.p.set_background("#101012")
        rx, ry, rz = REGION
        self.p.add_mesh(pv.Box(bounds=(0, rx, 0, ry, 0, rz)), style="wireframe", color="#6e6e73", line_width=1)
        self.p.add_mesh(pv.Plane(center=(rx / 2, ry / 2, 0), direction=(0, 0, 1), i_size=rx, j_size=ry),
                        color="#3a3a3f", opacity=0.25)
        self.p.add_text("x right, y down the print, depth down; scanning surface on top", font_size=9,
                        color="#aaaaaa", position="lower_left")
        self._names: list[str] = []
        c = np.array([rx / 2, ry / 2, rz / 2])
        self.p.camera_position = [tuple(c + 260 * np.array([-1.0, -1.0, -1.0]) / np.sqrt(3)), tuple(c), (0, 0, -1)]
        self.off_screen = off_screen
        if not off_screen:
            self.p.show(interactive_update=True, auto_close=False)

    @property
    def closed(self) -> bool:
        return bool(getattr(self.p, "_closed", False)) or self.p.render_window is None

    def update(self, meshes: dict, gt: dict | None = None) -> None:
        """Replace the shown surfaces."""
        for name in self._names:
            self.p.remove_actor(name, render=False)
        self._names = []
        for tag, src, base in (("gt", gt or {}, GT_OPACITY), ("rec", meshes, 1.0)):
            for name, (v, f) in src.items():
                if len(f) == 0:
                    continue
                key = f"{tag}_{name}"
                opacity = base * (BILE_OPACITY if name == "bile" and tag == "rec" else 1.0)
                colour = GROUPS.get(name, ((), "#B4B2A9"))[1]
                self.p.add_mesh(_polydata(v, f), color=colour, opacity=opacity, name=key, smooth_shading=True,
                                render=False)
                self._names.append(key)
        self.p.render()

    def process(self) -> None:
        """Handle window events (rotate / zoom) and redraw; cheap, call every loop."""
        if not self.off_screen and not self.closed:
            self.p.update(stime=1, force_redraw=False)

    def screenshot(self, path: str) -> str:
        self.p.screenshot(path)
        return path

    def close(self) -> None:
        if not self.closed:
            self.p.close()
