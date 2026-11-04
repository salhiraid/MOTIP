"""3D box -> 2D box projection used by the 3D-annotated datasets (nuScenes, Argoverse 2, generic 3D).

Conventions (nuScenes and Argoverse 2 agree): a cuboid has centre c, size (length, width, height) along its local
(x, y, z) axes and an orientation matrix R (box -> reference frame).  Cameras use the OpenCV convention (x right,
y down, z forward) and pinhole intrinsics K (lens distortion is ignored, as in both devkits' default projection).
"""
import numpy as np

# 12 edges of a cuboid whose corners are ordered as in `box_corners`
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def quat_to_rot(q):
    """quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_to_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_corners(center, length, width, height, rot):
    """8 corners (8, 3) in the reference frame. Local order: front-top-left, front-top-right, front-bottom-right,
    front-bottom-left, then the same four at the back."""
    x = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1.0])
    y = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1.0])
    z = height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1.0])
    return (rot @ np.stack([x, y, z])).T + np.asarray(center, dtype=np.float64)


def to_frame(points, rot, trans):
    """points expressed in frame A -> frame B, where (rot, trans) is the pose of B in A (B->A): p_B = R^T (p_A - t)."""
    return (np.asarray(points) - np.asarray(trans)) @ np.asarray(rot)


def project_box(corners_cam, K, width, height, near=0.1):
    """Axis-aligned 2D box of a cuboid given in camera coordinates.
    Edges are clipped to the plane z = `near` first, so boxes crossing the camera plane still project correctly.
    Returns None if the box is entirely behind the camera, else (x1, y1, x2, y2, frac_inside) where the box is clipped
    to the image and frac_inside = clipped area / unclipped area."""
    pts = []
    for i, j in EDGES:
        a, b = corners_cam[i], corners_cam[j]
        za, zb = a[2], b[2]
        if za < near and zb < near:
            continue
        if za < near:
            a = a + (near - za) / (zb - za) * (b - a)
        elif zb < near:
            b = b + (near - zb) / (za - zb) * (a - b)
        pts += [a, b]
    if not pts:
        return None
    p = np.asarray(pts)
    uv = (np.asarray(K) @ p.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x1, y1 = uv.min(0)
    x2, y2 = uv.max(0)
    full = max(1e-9, (x2 - x1) * (y2 - y1))
    cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2, ((cx2 - cx1) * (cy2 - cy1)) / full


def project_matrix_box(corners_world, P, width, height):
    """Same as `project_box` for a 3x4 projection matrix P (world -> pixels); corners behind the camera (w <= 0)
    invalidate the box (used for the generic importer, where only P is known)."""
    h = np.concatenate([corners_world, np.ones((len(corners_world), 1))], 1) @ np.asarray(P).T
    if (h[:, 2] <= 1e-6).any():
        return None
    uv = h[:, :2] / h[:, 2:3]
    x1, y1 = uv.min(0)
    x2, y2 = uv.max(0)
    full = max(1e-9, (x2 - x1) * (y2 - y1))
    cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2, ((cx2 - cx1) * (cy2 - cy1)) / full
