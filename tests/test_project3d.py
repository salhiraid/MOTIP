# 3D -> 2D projection helpers with analytic expectations (independent of the dataset converters).
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.converters.project3d import quat_to_rot, yaw_to_rot, box_corners, project_box, project_matrix_box, to_frame  # noqa

K = np.array([[60.0, 0, 48], [0, 60.0, 32], [0, 0, 1]])


def test_quaternion_rotation_known_values():
    R = quat_to_rot([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])                # 90 degrees about z
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12) and np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(quat_to_rot([2, 0, 0, 0]), np.eye(3))                      # unnormalised input
    assert np.allclose(yaw_to_rot(np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_to_frame_inverts_pose():
    R, t = yaw_to_rot(0.7), np.array([3.0, -2.0, 1.0])
    p = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(to_frame(p @ R.T + t, R, t), p)


def test_box_in_front_projects_to_the_extreme_corners():
    c = box_corners([0.0, 0.0, 10.0], 2.0, 1.0, 4.0, np.eye(3))                  # camera frame: x 2, y 1, z (depth) 4
    x1, y1, x2, y2, frac = project_box(c, K, 96, 64)
    # nearest face z = 8: x in [-1, 1] -> u = 48 +- 60/8 ; y in [-.5, .5] -> v = 32 +- 60*.5/8
    assert np.allclose([x1, y1, x2, y2], [48 - 7.5, 32 - 3.75, 48 + 7.5, 32 + 3.75]) and frac == 1.0


def test_box_crossing_the_camera_plane_is_clipped_not_dropped():
    c = box_corners([0.0, 0.0, 0.5], 2.0, 1.0, 2.0, np.eye(3))                   # depth from -0.5 to 1.5
    x1, y1, x2, y2, frac = project_box(c, K, 96, 64, near=0.1)
    # the intersection with z = 0.1 is a 2 x 1 rectangle -> u = 48 +- 600 (clipped to the image), v = 32 +- 300
    assert (x1, y1, x2, y2) == (0.0, 0.0, 96.0, 64.0) and 0 < frac < 0.01


def test_box_behind_camera_or_outside_image_is_none_and_partial_has_fraction():
    behind = box_corners([0.0, 0.0, -5.0], 2.0, 1.0, 2.0, np.eye(3))
    assert project_box(behind, K, 96, 64) is None
    outside = box_corners([50.0, 0.0, 10.0], 2.0, 1.0, 2.0, np.eye(3))
    assert project_box(outside, K, 96, 64) is None
    half = box_corners([-7.0, 0.0, 10.0], 4.0, 1.0, 1.0, np.eye(3))               # x in [-9, -5]: left part is cut off
    x1, _, x2, _, frac = project_box(half, K, 96, 64)
    assert x1 == 0.0 and 0.2 < frac < 0.9


def test_matrix_projection_matches_pinhole_and_rejects_boxes_behind():
    c = box_corners([0.0, 0.0, 10.0], 2.0, 1.0, 4.0, np.eye(3))
    P = np.hstack([K, np.zeros((3, 1))])
    a, b = project_box(c, K, 96, 64), project_matrix_box(c, P, 96, 64)
    assert np.allclose(a, b)
    assert project_matrix_box(box_corners([0.0, 0.0, 1.0], 2.0, 1.0, 4.0, np.eye(3)), P, 96, 64) is None
