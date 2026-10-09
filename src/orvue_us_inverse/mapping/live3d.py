"""
orvue_us_inverse.mapping.live3d - live 3D window of the reconstruction with PyVista / VTK (S3, optional).

    live = Live3D()                         # opens the window (off_screen=True for tests / screenshots)
    live.update(meshes, gt=gt_meshes)       # replace the surfaces (render.surface_meshes); gt translucent
    live.set_view("axial")                  # camera preset (VIEWS)
    live.process()                          # call often: handles mouse rotate / zoom and redraws
    live.closed                             # True once the user closed the window
    view = Embedded3D((420, 480))           # off-screen, for a panel of an OpenCV window: view.image() (BGR),
                                            # view.orbit(dx, dy) on drag, view.zoom(f) on the wheel
    live.close()

PyVista is an optional extra (pip install -e .[view3d]); importing this module raises ImportError without it, and
the apps fall back to the matplotlib snapshot (render.snapshot_3d).

Anatomical orientation (phantom frame, CLAUDE.md): +x = patient's left, -x = patient's right; +y = caudal (feet),
-y = cranial (head); +z = posterior (depth), z = 0 = anterior (the scanning surface). Every face of the 100 x 100 x
50 mm box can carry its label; a label is shown when its face is seen edge-on (the label lands outside the box) or
obliquely from the front, and hidden when the face looks straight at the camera or away from it (the label would sit
over the anatomy); the set is recomputed after set_view and orbit. An orientation triad in the lower-left corner
shows the axes (L = patient's left, Ca = caudal, P = posterior).

Camera presets (VIEWS: direction from the box centre to the camera, view-up), all with the box fitted:
    isometric   from above, in front and slightly right: x to the right, y towards the viewer, depth down
    top         probe's view from above: x right, y down (like the sweep top views)
    axial       from the feet looking cranially (radiology convention): patient's left on the right, depth down
    sagittal    from the patient's right: cranial left, caudal right, depth down
"""
import numpy as np
import pyvista as pv

from orvue_us_inverse.mapping.render import GROUPS

REGION = (100.0, 100.0, 50.0)
# name: (label for buttons, centre -> camera direction, view-up)
VIEWS = {
    "isometric": ("Iso", (-0.4, 1.0, -0.9), (0.0, 0.0, -1.0)),
    "top": ("Top", (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    "axial": ("Axial", (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "sagittal": ("Sagittal", (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
}
DEFAULT_VIEW = "isometric"
GT_OPACITY = 0.18
BILE_OPACITY = 0.6                  # stones inside the gallbladder stay visible
LABEL_OFFSET_MM = 6.0
FIT_ZOOM = 0.88                     # margin around the fitted box for the labels
FACE_NORMALS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], float)


def orientation_labels(region=REGION) -> tuple[np.ndarray, list[str]]:
    """Face-centre positions just outside the box and their anatomical labels."""
    rx, ry, rz = region
    c = np.array([rx / 2, ry / 2, rz / 2])
    o = LABEL_OFFSET_MM
    pts = np.array([[-o, c[1], c[2]], [rx + o, c[1], c[2]],          # x: patient's right / left
                    [c[0], -o, c[2]], [c[0], ry + o, c[2]],          # y: cranial / caudal
                    [c[0], c[1], -o], [c[0], c[1], rz + o]])         # z: anterior (surface) / posterior
    labels = ["patient R", "patient L", "cranial", "caudal", "anterior (surface)", "posterior"]
    return pts, labels


def visible_labels(to_camera: np.ndarray) -> list[bool]:
    """Which face labels to show for a unit direction from the box centre to the camera (FACE_NORMALS order)."""
    c = FACE_NORMALS @ (np.asarray(to_camera, float) / np.linalg.norm(to_camera))
    return [bool(abs(k) < 0.3 or 0 < k < 0.85) for k in c]


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
        self.p.add_axes(xlabel="L", ylabel="Ca", zlabel="P", line_width=3, labels_off=False,
                        viewport=(0.0, 0.0, 0.28, 0.28))
        self._names: list[str] = []
        self.view = DEFAULT_VIEW
        self.set_view(DEFAULT_VIEW)
        self.off_screen = off_screen
        if not off_screen:
            self.p.show(interactive_update=True, auto_close=False)

    @property
    def closed(self) -> bool:
        return bool(getattr(self.p, "_closed", False)) or self.p.render_window is None

    def set_view(self, name: str) -> None:
        """Camera preset from VIEWS, box fitted."""
        _, d, up = VIEWS[name]
        d = np.asarray(d, float)
        c = np.array(REGION) / 2
        self.p.camera_position = [tuple(c + 260 * d / np.linalg.norm(d)), tuple(c), up]
        self.p.reset_camera()
        self.p.camera.Zoom(FIT_ZOOM)
        self.view = name
        self.refresh_labels()

    def refresh_labels(self) -> None:
        """Show the face labels that suit the current camera (visible_labels)."""
        cam = self.p.camera
        to_cam = np.asarray(cam.position) - np.asarray(cam.focal_point)
        pts, labels = orientation_labels()
        keep = visible_labels(to_cam)
        self.p.remove_actor("orientation_labels", render=False)
        if any(keep):
            self.p.add_point_labels(pts[keep], [lb for lb, k in zip(labels, keep) if k], name="orientation_labels",
                                    font_size=11, text_color="#e6e6e6", shape_color="#2a2a30", shape_opacity=0.75,
                                    show_points=False, always_visible=True, margin=2, render=False)
        self.shown_labels = [lb for lb, k in zip(labels, keep) if k]

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


class Embedded3D(Live3D):
    """Off-screen 3D view rendered into an image for a panel of an OpenCV window (mouse sweep): drag to orbit, wheel
    to zoom, set_view for the presets. image() re-renders only after a change."""

    ORBIT_DEG_PER_PX = 0.4

    def __init__(self, size=(420, 480)):
        self._img = None
        super().__init__(title="", window_size=size, off_screen=True)

    def set_view(self, name: str) -> None:
        super().set_view(name)
        self._img = None

    def update(self, meshes: dict, gt: dict | None = None) -> None:
        super().update(meshes, gt)
        self._img = None

    def orbit(self, dx_px: float, dy_px: float) -> None:
        cam = self.p.camera
        cam.Azimuth(-dx_px * self.ORBIT_DEG_PER_PX)
        cam.Elevation(dy_px * self.ORBIT_DEG_PER_PX)
        cam.OrthogonalizeViewUp()
        self.p.reset_camera_clipping_range()
        self.refresh_labels()
        self._img = None

    def zoom(self, factor: float) -> None:
        self.p.camera.Zoom(factor)
        self._img = None

    def image(self) -> np.ndarray:
        """The current view as a BGR uint8 image (window_size)."""
        if self._img is None:
            self.p.render()                                  # screenshot() alone returns the last rendered frame
            rgb = self.p.screenshot(return_img=True)
            self._img = np.ascontiguousarray(rgb[..., :3][..., ::-1])
        return self._img
