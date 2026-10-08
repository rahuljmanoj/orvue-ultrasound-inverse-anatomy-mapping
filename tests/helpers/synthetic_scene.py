"""Synthetic D405-like camera images of the reference board and the probe markers.

Every marker is generated with cv2.aruco.generateImageMarker at the geometry in orvue_us_inverse.tracking.markers
and projected through a pinhole camera with cv2.warpPerspective (rendered at 2x and area-downsampled
for anti-aliasing), then mildly blurred and noised.
"""
import math

import cv2
import numpy as np

from orvue_us_inverse.tracking import markers as tm  # noqa: E402

W, H = 1280, 720
K = np.array([[645.0, 0.0, 640.0], [0.0, 645.0, 360.0], [0.0, 0.0, 1.0]])   # ~D405 colour at 1280x720
DIST = np.zeros(5)
SS = 2                      # supersampling factor
PX_PER_MM = 12              # texture resolution


def pose_from_xy_yaw(x, y, yaw_deg, z=0.0):
    """Same convention as BModeSimulator.pose_from_xy_yaw (probe perpendicular to the surface)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    T = np.eye(4)
    T[:3, 0] = [c, s, 0]
    T[:3, 1] = [-s, c, 0]
    T[:3, 2] = [0, 0, 1]
    T[:3, 3] = [x, y, z]
    return T


def rot(axis, deg):
    v = np.zeros(3)
    v["xyz".index(axis)] = math.radians(deg)
    return cv2.Rodrigues(v)[0]


def camera_pose(height_mm=300.0, tilt_x_deg=0.0, tilt_y_deg=0.0, roll_deg=0.0, target=(50.0, 50.0, None)):
    """T_cam_phantom for a camera looking down (+z) at target from height_mm, image x ~ +x, image y ~ +y."""
    R = rot("z", roll_deg) @ rot("x", tilt_x_deg) @ rot("y", tilt_y_deg)      # phantom -> camera
    tz = tm.H_MM if target[2] is None else target[2]
    view = R.T @ np.array([0.0, 0.0, 1.0])                                    # camera +z in phantom
    C = np.array([target[0], target[1], tz]) - height_mm * view
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ C
    return T


def _marker_tile(mid, size_mm, quiet_mm, scale=1.0):
    """White tile with the marker's black square centred, at PX_PER_MM. scale != 1 draws a wrong-size marker."""
    n = int(round(size_mm * scale * PX_PER_MM))
    q = int(round(quiet_mm * PX_PER_MM))
    m = cv2.aruco.generateImageMarker(tm.dictionary(), mid, n, borderBits=1)
    return cv2.copyMakeBorder(m, q, q, q, q, cv2.BORDER_CONSTANT, value=255), q


def _draw_plane(canvas, tex, origin, ex, ey, T_cam_phantom, Kss):
    """Warp a texture whose pixel (u, v) lies at origin + (u+0.5)/s*ex + (v+0.5)/s*ey (phantom frame)."""
    R, t = T_cam_phantom[:3, :3], T_cam_phantom[:3, 3]
    Hm = Kss @ np.column_stack([R @ ex, R @ ey, R @ origin + t])
    s = PX_PER_MM
    A = np.array([[1 / s, 0, 0.5 / s], [0, 1 / s, 0.5 / s], [0, 0, 1]])
    Ht = Hm @ A
    size = (canvas.shape[1], canvas.shape[0])
    warped = cv2.warpPerspective(tex, Ht, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(tex.shape, 255, np.uint8), Ht, size, flags=cv2.INTER_NEAREST)
    canvas[mask > 0] = warped[mask > 0]


def render(T_cam_phantom, T_phantom_probe=None, ref_ids=None, use_second_marker=None,
           ref_scale=None, probe_scale=None, blur=0.6, noise=2.0, seed=0):
    """Grey image of the scene. ref_ids: reference markers to draw (default all). ref_scale / probe_scale:
    {id: factor} to draw a marker at the wrong size (centred on its true position)."""
    ref_ids = sorted(tm.REF_MARKERS_TL) if ref_ids is None else ref_ids
    use_second = tm.USE_SECOND_MARKER if use_second_marker is None else use_second_marker
    ref_scale, probe_scale = ref_scale or {}, probe_scale or {}
    Kss = K.copy()
    Kss[:2] *= SS
    Kss[0, 2] += (SS - 1) / 2
    Kss[1, 2] += (SS - 1) / 2
    canvas = np.full((H * SS, W * SS), 90, np.uint8)            # table
    ex, ey = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])

    # printed sheet (white, at z = H_MM) with the reference markers
    sx, sy = tm.SHEET_TL
    sw, sh = tm.SHEET_SIZE
    sheet = np.full((int(sh * PX_PER_MM), int(sw * PX_PER_MM)), 235, np.uint8)
    _draw_plane(canvas, sheet, np.array([sx, sy, tm.H_MM]), ex, ey, T_cam_phantom, Kss)
    for mid in ref_ids:
        f = ref_scale.get(mid, 1.0)
        size = tm.REF_MARKER_SIZE_MM
        tile, _ = _marker_tile(mid, size, 2.0, f)
        tlx, tly = tm.REF_MARKERS_TL[mid]
        c = size / 2
        half = tile.shape[0] / PX_PER_MM / 2
        _draw_plane(canvas, tile, np.array([tlx + c - half, tly + c - half, tm.H_MM]), ex, ey, T_cam_phantom, Kss)

    # probe markers, planar at z = -PLATFORM_HEIGHT_MM in the probe frame
    if T_phantom_probe is not None:
        R, p = T_phantom_probe[:3, :3], T_phantom_probe[:3, 3]
        items = [(tm.PROBE_MARKER_ID, tm.PROBE_MARKER_SIZE_MM, 5.0, 0.0)]
        if use_second:
            items.append((tm.SECOND_MARKER_ID, tm.SECOND_MARKER_SIZE_MM, 4.0, tm.SECOND_MARKER_OFFSET_X_MM))
        for mid, size, quiet, cx in items:
            tile, _ = _marker_tile(mid, size, quiet, probe_scale.get(mid, 1.0))
            half = tile.shape[0] / PX_PER_MM / 2
            o = p + R @ np.array([cx - half, -half, -tm.PLATFORM_HEIGHT_MM])
            _draw_plane(canvas, tile, o, R[:, 0], R[:, 1], T_cam_phantom, Kss)

    img = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def true_region_outline_px(T_cam_phantom, z):
    R = tm.REGION_MM
    P = np.array([[0, 0, z], [R, 0, z], [R, R, z], [0, R, z]], dtype=np.float64)
    rvec = cv2.Rodrigues(T_cam_phantom[:3, :3])[0]
    return cv2.projectPoints(P, rvec, T_cam_phantom[:3, 3], K, DIST)[0].reshape(-1, 2)
