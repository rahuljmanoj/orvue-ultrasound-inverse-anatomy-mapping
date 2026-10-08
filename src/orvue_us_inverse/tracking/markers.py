"""
orvue_us_inverse.tracking.markers - ArUco marker layout for the ultrasound simulator tracking setup.

This file is the single source of truth: the printable PDF (docs/print/tracking_board.pdf) was generated
from these numbers, and the tracker builds its OpenCV boards from them.

FRAMES (all millimetres, right-handed)
  Phantom frame: origin at the top-left corner of the printed 100 x 100 mm region,
                 +x to the right on the print, +y down the print, +z into the table
                 (= into tissue). z = 0 is the GEL SURFACE, so the printed sheet lies at z = +H_MM.
  Probe frame:   origin at the centre of the probe's contact face, +x along the array
                 (away from the orientation mark, which is at the -x end = the image's left side,
                 shown by the screen dot), +y across the face, +z out of the face into
                 tissue. The marker platform is above the face, i.e. at z = -PLATFORM_HEIGHT_MM.

BEFORE USE: measure and set H_MM, PLATFORM_HEIGHT_MM and SECOND_MARKER_OFFSET_X_MM for your build,
and check that the printed 100 mm scale bar measures exactly 100.0 mm.
"""
import numpy as np
import cv2

DICTIONARY_ID = cv2.aruco.DICT_4X4_50

# ---------------- reference markers on the printed sheet ----------------
REF_MARKER_SIZE_MM = 24.0            # black square edge, as printed
REF_QUIET_ZONE_MM = 5.0              # minimum white margin around each marker (kept clear on the print)
REGION_MM = 100.0                    # active region edge
# The sheet is cut to 170 x 110 mm: x from -35 to 135, y from -5 to 105 (phantom frame).
SHEET_TL = (-35.0, -5.0)
SHEET_SIZE = (170.0, 110.0)
# top-left corner of each marker's black square in the phantom frame (x, y);
# markers sit in the 35 mm side strips left and right of the region
REF_MARKERS_TL = {
    1: (-30.0, 0.0),                 # left, top
    2: (106.0, 0.0),                 # right, top
    3: (106.0, 76.0),                # right, bottom
    4: (-30.0, 76.0),                # left, bottom
}
H_MM = 0.0                           # SET THIS: gel surface height above the printed sheet (mm)

# ---------------- probe markers ----------------
PROBE_MARKER_ID = 0
PROBE_MARKER_SIZE_MM = 30.0          # on the 40 x 40 mm platform (5 mm quiet zone each side)
SECOND_MARKER_ID = 5
SECOND_MARKER_SIZE_MM = 20.0         # on the end of the arm (28 x 28 mm cut tile)
PLATFORM_HEIGHT_MM = 15.0            # SET THIS: contact face to marker surface (mm)
SECOND_MARKER_OFFSET_X_MM = 80.0     # SET THIS: centre of ID 5 along probe +x from the face centre
USE_SECOND_MARKER = False


def square_corners(tl_x, tl_y, size, z):
    """Corners of a marker in OpenCV order (TL, TR, BR, BL) as printed, in a frame whose
    x/y axes match the printed marker's right/down directions."""
    return np.array([[tl_x, tl_y, z],
                     [tl_x + size, tl_y, z],
                     [tl_x + size, tl_y + size, z],
                     [tl_x, tl_y + size, z]], dtype=np.float32)


def dictionary():
    return cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)


def reference_board():
    """Board of the four corner markers, defined directly in the phantom frame."""
    ids = sorted(REF_MARKERS_TL)
    obj = [square_corners(*REF_MARKERS_TL[i], REF_MARKER_SIZE_MM, H_MM) for i in ids]
    return cv2.aruco.Board(obj, dictionary(), np.array(ids, dtype=np.int32))


def probe_board(use_second_marker=None):
    """Board of the probe marker(s), defined in the probe frame (marker 'right' = probe +x).
    use_second_marker=None follows USE_SECOND_MARKER."""
    if use_second_marker is None:
        use_second_marker = USE_SECOND_MARKER
    s0, s5 = PROBE_MARKER_SIZE_MM, SECOND_MARKER_SIZE_MM
    z = -PLATFORM_HEIGHT_MM
    obj = [square_corners(-s0 / 2, -s0 / 2, s0, z)]
    ids = [PROBE_MARKER_ID]
    if use_second_marker:
        obj.append(square_corners(SECOND_MARKER_OFFSET_X_MM - s5 / 2, -s5 / 2, s5, z))
        ids.append(SECOND_MARKER_ID)
    return cv2.aruco.Board(obj, dictionary(), np.array(ids, dtype=np.int32))


def detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary(), params)


def board_pose(board, corners, ids, K, dist):
    """4x4 pose of a board in the camera frame (T_cam_board), or None if not enough markers."""
    if ids is None or len(ids) == 0:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    if obj is None or len(obj) < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def board_pose_refined(board, corners, ids, K, dist):
    """board_pose() followed by a Levenberg-Marquardt refinement of the reprojection error.
    IPPE alone can be ~1-2 deg off in tilt when a planar board is seen nearly face-on."""
    T = board_pose(board, corners, ids, K, dist)
    if T is None:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    rvec, tvec = cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].reshape(3, 1).copy()
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def board_markers_seen(board, ids):
    """IDs of this board's markers among the detected ids (sorted list)."""
    if ids is None:
        return []
    return sorted(set(board.getIds().ravel().tolist()) & set(np.asarray(ids).ravel().tolist()))


def board_reprojection_error(board, corners, ids, T_cam_board, K, dist):
    """RMS distance (px) between the detected corners of the board's markers and their
    projection through T_cam_board, or None if none of the board's markers were detected."""
    if ids is None or len(ids) == 0 or T_cam_board is None:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    if obj is None or len(obj) == 0:
        return None
    rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0]
    px, _ = cv2.projectPoints(obj.reshape(-1, 3).astype(np.float64), rvec, T_cam_board[:3, 3], K, dist)
    d = px.reshape(-1, 2) - img.reshape(-1, 2)
    return float(np.sqrt((d ** 2).sum(1).mean()))


def phantom_probe_pose(gray, K, dist, det=None, ref=None, prb=None):
    """Probe pose in the phantom frame (the 4x4 the simulator's render() expects), or None."""
    det = det or detector()
    ref = ref or reference_board()
    prb = prb or probe_board()
    corners, ids, _ = det.detectMarkers(gray)
    T_cam_phantom = board_pose(ref, corners, ids, K, dist)
    T_cam_probe = board_pose(prb, corners, ids, K, dist)
    if T_cam_phantom is None or T_cam_probe is None:
        return None
    return np.linalg.inv(T_cam_phantom) @ T_cam_probe


def region_outline_px(T_cam_phantom, K, dist, z=0.0):
    """Project the 100 x 100 mm region outline (at height z in the phantom frame; 0 = gel surface,
    H_MM = printed sheet) into the camera image. Draw it on the live video as a registration check:
    it should lie exactly on the printed outline (z = H_MM) or on the gel edges (z = 0)."""
    R = REGION_MM
    P = np.array([[0, 0, z], [R, 0, z], [R, R, z], [0, R, z]], dtype=np.float64)
    rvec = cv2.Rodrigues(T_cam_phantom[:3, :3])[0]
    px, _ = cv2.projectPoints(P, rvec, T_cam_phantom[:3, 3], K, dist)
    return px.reshape(-1, 2)


def xy_yaw(T_phantom_probe):
    """(x, y, yaw_deg) of the probe in the phantom frame, for pose_from_xy_yaw()."""
    u = T_phantom_probe[:3, 0]
    return float(T_phantom_probe[0, 3]), float(T_phantom_probe[1, 3]), float(np.degrees(np.arctan2(u[1], u[0])))
