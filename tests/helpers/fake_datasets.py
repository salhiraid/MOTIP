"""Tiny synthetic datasets written in the NATIVE formats of the public tracking datasets (from their documentation),
all describing the same scene, so the converters can be checked end to end against a known ground truth.

Scene: W x H = 96 x 64, N = 12 frames.  Track A = car, B = truck/bus, C = pedestrian (static).
Image-plane datasets use the boxes below directly; 3D datasets (nuScenes, Argoverse 2, generic) place cuboids so that
the expected 2D boxes follow from an INDEPENDENT corner-by-corner projection coded here (not project3d.py).
"""
import itertools
import json
import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

W, H, N = 96, 64, 12


def box_a(f): return (5.0 + 3 * f, 10.0, 20.0, 12.0)          # car
def box_b(f): return (40.0 - 2 * f, 25.0, 18.0, 14.0)         # truck / bus
def box_c(f): return (70.0, 30.0, 8.0, 16.0)                  # pedestrian (static)
IGN = (0.0, 50.0, 30.0, 12.0)                                  # an ignore region


def make_images(d, n=N, pattern="{:08d}.jpg", start=1, size=(W, H)):
    os.makedirs(d, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(d, pattern.format(i + start))
        Image.new("RGB", size, (30 + 5 * i, 90, 160)).save(p)
        paths.append(p)
    return paths


# ------------------------------------------------------------------------------------------------ image-plane formats
def bdd100k(root):
    for split, vid in (("train", "trainvid"), ("val", "valvid")):
        d = os.path.join(root, "images", "track", split, vid)
        names = [os.path.basename(p) for p in make_images(d, pattern=vid + "-{:07d}.jpg")]
        frames = []
        for i, name in enumerate(names):
            f = i + 1
            lb = lambda id_, cat, b, crowd=False: {"id": id_, "category": cat, "attributes": {"crowd": crowd},
                                                    "box2d": {"x1": b[0], "y1": b[1], "x2": b[0] + b[2], "y2": b[1] + b[3]}}
            frames.append({"name": name, "videoName": vid, "frameIndex": i, "labels": [
                lb("1001", "car", box_a(f)), lb("77", "truck", box_b(f)), lb("5", "pedestrian", box_c(f)),
                lb("9", "car", IGN, crowd=True)]})
        os.makedirs(os.path.join(root, "labels", "box_track_20", split), exist_ok=True)
        json.dump(frames, open(os.path.join(root, "labels", "box_track_20", split, vid + ".json"), "w"))


def detrac(root):
    for split, imgf, annf, name in (("train", "DETRAC-train-data", "DETRAC-Train-Annotations-XML", "MVI_20011"),
                                    ("val", "DETRAC-test-data", "DETRAC-Test-Annotations-XML", "MVI_39031")):
        make_images(os.path.join(root, imgf, name), pattern="img{:05d}.jpg")
        os.makedirs(os.path.join(root, annf), exist_ok=True)
        seq = ET.Element("sequence", name=name)
        ir = ET.SubElement(seq, "ignored_region")
        ET.SubElement(ir, "box", left=str(IGN[0]), top=str(IGN[1]), width=str(IGN[2]), height=str(IGN[3]))
        for f in range(1, N + 1):
            fe = ET.SubElement(seq, "frame", density="3", num=str(f))
            tl = ET.SubElement(fe, "target_list")
            for tid, vt, b in ((1, "car", box_a(f)), (2, "bus", box_b(f)), (3, "others", box_c(f))):
                t = ET.SubElement(tl, "target", id=str(tid))
                ET.SubElement(t, "box", left=str(b[0]), top=str(b[1]), width=str(b[2]), height=str(b[3]))
                ET.SubElement(t, "attribute", vehicle_type=vt, orientation="0", speed="1")
        ET.ElementTree(seq).write(os.path.join(root, annf, name + ".xml"))


def visdrone(root):
    for split, folder, name in (("train", "VisDrone2019-MOT-train", "uav0000013_00000_v"),
                                ("val", "VisDrone2019-MOT-val", "uav0000086_00000_v")):
        make_images(os.path.join(root, folder, "sequences", name), pattern="{:07d}.jpg")
        os.makedirs(os.path.join(root, folder, "annotations"), exist_ok=True)
        with open(os.path.join(root, folder, "annotations", name + ".txt"), "w") as fh:
            for f in range(1, N + 1):
                for tid, cat, b in ((1, 4, box_a(f)), (2, 6, box_b(f)), (3, 1, box_c(f)), (0, 0, IGN)):
                    fh.write(f"{f},{tid},{b[0]},{b[1]},{b[2]},{b[3]},1,{cat},0,0\n")
                fh.write(f"{f},9,2,2,10,10,0,4,0,0\n")             # score 0 -> not considered -> ignore region


def uavdt(root):
    gt = os.path.join(root, "UAV-benchmark-MOTD_v1.0", "GT")
    os.makedirs(gt)
    for name in ("M0101", "M0203"):                                 # M0203 is in the official test list -> val
        make_images(os.path.join(root, "UAV-benchmark-M", name), pattern="img{:06d}.jpg")
        with open(os.path.join(gt, f"{name}_gt_whole.txt"), "w") as fh:
            for f in range(1, N + 1):
                for tid, cat, b in ((1, 1, box_a(f)), (2, 2, box_b(f))):
                    fh.write(f"{f},{tid},{b[0]},{b[1]},{b[2]},{b[3]},0,1,{cat}\n")
        with open(os.path.join(gt, f"{name}_gt_ignore.txt"), "w") as fh:
            for f in range(1, N + 1):
                fh.write(f"{f},0,{IGN[0]},{IGN[1]},{IGN[2]},{IGN[3]},1,1,1\n")


def kitti(root):
    for sid in ("0000", "0001"):                                    # sequence 1 is a CenterTrack val sequence
        make_images(os.path.join(root, "training", "image_02", sid), pattern="{:06d}.png", start=0)
        os.makedirs(os.path.join(root, "training", "label_02"), exist_ok=True)
        with open(os.path.join(root, "training", "label_02", sid + ".txt"), "w") as fh:
            for f in range(1, N + 1):
                for tid, typ, b in ((0, "Car", box_a(f)), (1, "Truck", box_b(f)), (2, "Pedestrian", box_c(f)),
                                    (-1, "DontCare", IGN)):
                    fh.write(f"{f - 1} {tid} {typ} 0 0 0 {b[0]} {b[1]} {b[0] + b[2]} {b[1] + b[3]} 1 1 1 0 0 0 0\n")


def mot_format(root):
    """Flat MOT-challenge layout (FastTracker-style): class column = 3 car, 7 truck, 1 person, 9 something unmapped."""
    for name in ("cctv1", "cctv2", "cctv3", "cctv4"):
        d = os.path.join(root, name)
        make_images(os.path.join(d, "img1"), pattern="{:06d}.jpg")
        os.makedirs(os.path.join(d, "gt"))
        with open(os.path.join(d, "gt", "gt.txt"), "w") as fh:
            for f in range(1, N + 1):
                for tid, c, b in ((1, 3, box_a(f)), (2, 7, box_b(f)), (3, 1, box_c(f)), (4, 9, IGN)):
                    fh.write(f"{f},{tid},{b[0]},{b[1]},{b[2]},{b[3]},1,{c},1.0\n")
                fh.write(f"{f},5,1,1,5,5,0,3,1.0\n")                 # conf 0 row -> skipped
        open(os.path.join(d, "seqinfo.ini"), "w").write(
            f"[Sequence]\nname={name}\nimDir=img1\nframeRate=15\nseqLength={N}\nimWidth={W}\nimHeight={H}\nimExt=.jpg\n")


def cityflow(root):
    for folder, scene in (("train", "S01"), ("validation", "S02")):
        d = os.path.join(root, folder, scene, "c001")
        frames = os.path.join(root, "_tmp_frames", scene)
        make_images(frames, pattern="{:08d}.jpg")
        os.makedirs(os.path.join(d, "gt"))
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-framerate", "10", "-i",
                        os.path.join(frames, "%08d.jpg"), "-c:v", "mjpeg", "-q:v", "1", os.path.join(d, "vdo.avi")], check=True)
        with open(os.path.join(d, "gt", "gt.txt"), "w") as fh:
            for f in range(1, N + 1):
                for tid, b in ((101, box_a(f)), (202, box_b(f))):
                    fh.write(f"{f},{tid},{b[0]},{b[1]},{b[2]},{b[3]},1,-1,-1,-1\n")


def waymo(root):
    import pandas as pd
    import io
    for folder, seg in (("training", "segA"), ("validation", "segB")):
        os.makedirs(os.path.join(root, folder, "camera_image"))
        os.makedirs(os.path.join(root, folder, "camera_box"))
        imgs, boxes = [], []
        for i in range(N):
            f = i + 1
            ts = 1_000_000 + 100_000 * i
            for cam in (1, 2):
                buf = io.BytesIO()
                Image.new("RGB", (W, H), (30 + 5 * i, 90, 160)).save(buf, "JPEG")
                imgs.append({"key.segment_context_name": seg, "key.frame_timestamp_micros": ts, "key.camera_name": cam,
                             "[CameraImageComponent].image": buf.getvalue()})
            for oid, typ, b in (("a", 1, box_a(f)), ("b", 1, box_b(f)), ("c", 2, box_c(f)), ("s", 3, IGN)):
                boxes.append({"key.segment_context_name": seg, "key.frame_timestamp_micros": ts, "key.camera_name": 1,
                              "key.camera_object_id": oid, "[CameraBoxComponent].box.center.x": b[0] + b[2] / 2,
                              "[CameraBoxComponent].box.center.y": b[1] + b[3] / 2, "[CameraBoxComponent].box.size.x": b[2],
                              "[CameraBoxComponent].box.size.y": b[3], "[CameraBoxComponent].type": typ})
        pd.DataFrame(imgs).to_parquet(os.path.join(root, folder, "camera_image", seg + ".parquet"))
        pd.DataFrame(boxes).to_parquet(os.path.join(root, folder, "camera_box", seg + ".parquet"))


# ------------------------------------------------------------------------------------------------------- 3D formats
FX, CX, CY = 60.0, 48.0, 32.0
# cuboids in the EGO frame (x forward, y left, z up): (l, w, h), centre(f)
def ego_a(f): return (4.0, 2.0, 1.5), (10.0 + 0.2 * f, 1.0 - 0.05 * f, 0.0)
def ego_b(f): return (8.0, 2.5, 3.0), (16.0 - 0.1 * f, -1.5, 0.5)
def ego_c(f): return (0.6, 0.6, 1.8), (7.0, 3.0, 0.0)


def project_expected(size, centre):
    """Independent projection of an axis-aligned (yaw 0) ego-frame cuboid into the forward camera at the ego origin:
    camera x = -ego y, camera y = -ego z, camera z = ego x."""
    l, w, h = size
    us, vs = [], []
    for sx, sy, sz in itertools.product((-1, 1), repeat=3):
        xe, ye, ze = centre[0] + sx * l / 2, centre[1] + sy * w / 2, centre[2] + sz * h / 2
        us.append(FX * (-ye) / xe + CX)
        vs.append(FX * (-ze) / xe + CY)
    x1, y1, x2, y2 = max(0, min(us)), max(0, min(vs)), min(W, max(us)), min(H, max(vs))
    return (x1, y1, x2 - x1, y2 - y1)


def rot_to_quat(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1)
        return [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0] * 4
    q[1 + i] = 0.25 * s
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


CAM_TO_EGO = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]])        # columns: camera x/y/z axes expressed in ego axes
Z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])                  # ego yaw of +90 degrees


def nuscenes(root):
    ver = os.path.join(root, "v1.0-mini")
    os.makedirs(ver)
    T = {k: [] for k in ("scene", "sample", "sample_data", "sensor", "calibrated_sensor", "ego_pose", "sample_annotation",
                         "instance", "category", "log", "visibility")}
    T["sensor"].append({"token": "sens", "channel": "CAM_FRONT", "modality": "camera"})
    T["calibrated_sensor"].append({"token": "cs", "sensor_token": "sens", "translation": [0, 0, 0],
                                   "rotation": rot_to_quat(CAM_TO_EGO), "camera_intrinsic": [[FX, 0, CX], [0, FX, CY], [0, 0, 1]]})
    T["visibility"].append({"token": "4", "level": "v80-100"})
    for tok, name in (("c_car", "vehicle.car"), ("c_truck", "vehicle.truck"), ("c_ped", "human.pedestrian.adult")):
        T["category"].append({"token": tok, "name": name})
    for tok, cat in (("iA", "c_car"), ("iB", "c_truck"), ("iC", "c_ped")):
        T["instance"].append({"token": tok, "category_token": cat})
    # scene 0: ego translated; scene 1: ego translated AND yawed by 90 degrees (global != ego orientation)
    for s, (ego_t, ego_R) in enumerate((([100.0, 200.0, 0.0], np.eye(3)), ([-30.0, 5.0, 0.0], Z90))):
        T["log"].append({"token": f"log{s}", "location": f"loc{s}"})
        T["scene"].append({"token": f"sc{s}", "name": f"scene-{s:04d}", "log_token": f"log{s}",
                           "first_sample_token": f"s{s}_1", "last_sample_token": f"s{s}_{N}"})
        for f in range(1, N + 1):
            st = f"s{s}_{f}"
            T["sample"].append({"token": st, "scene_token": f"sc{s}", "timestamp": f * 500_000,
                                "next": f"s{s}_{f + 1}" if f < N else "", "prev": f"s{s}_{f - 1}" if f > 1 else ""})
            T["ego_pose"].append({"token": f"ep{st}", "translation": ego_t, "rotation": rot_to_quat(ego_R)})
            fn = f"samples/CAM_FRONT/n{s}-{f:03d}.jpg"
            make_images(os.path.join(root, "samples", "CAM_FRONT"), n=1, pattern=f"n{s}-{f:03d}.jpg", start=0)
            T["sample_data"].append({"token": f"sd{st}", "sample_token": st, "ego_pose_token": f"ep{st}",
                                     "calibrated_sensor_token": "cs", "filename": fn, "is_key_frame": True,
                                     "width": W, "height": H})
            for inst, fn_ in (("iA", ego_a), ("iB", ego_b), ("iC", ego_c)):
                (l, w, h), c = fn_(f)
                g = ego_R @ np.array(c) + np.array(ego_t)               # ego -> global
                # global box orientation: the ego yaw (the cuboid is axis-aligned in the ego frame)
                T["sample_annotation"].append({"token": f"an{st}{inst}", "sample_token": st, "instance_token": inst,
                                               "visibility_token": "4", "translation": g.tolist(), "size": [w, l, h],
                                               "rotation": rot_to_quat(ego_R), "num_lidar_pts": 20, "num_radar_pts": 0})
    for k, v in T.items():
        json.dump(v, open(os.path.join(ver, k + ".json"), "w"))


def argoverse2(root):
    import pandas as pd
    for split, log in (("train", "logT"), ("val", "logV")):
        ld = os.path.join(root, split, log)
        cdir = os.path.join(ld, "sensors", "cameras", "ring_front_center")
        os.makedirs(os.path.join(ld, "calibration"))
        os.makedirs(cdir)
        q_cam = rot_to_quat(CAM_TO_EGO)
        pd.DataFrame([{"sensor_name": "ring_front_center", "fx_px": FX, "fy_px": FX, "cx_px": CX, "cy_px": CY,
                       "width_px": W, "height_px": H}]).to_feather(os.path.join(ld, "calibration", "intrinsics.feather"))
        pd.DataFrame([{"sensor_name": "ring_front_center", "qw": q_cam[0], "qx": q_cam[1], "qy": q_cam[2], "qz": q_cam[3],
                       "tx_m": 0.0, "ty_m": 0.0, "tz_m": 0.0}]).to_feather(
            os.path.join(ld, "calibration", "egovehicle_SE3_sensor.feather"))
        poses, anns = [], []
        for f in range(1, N + 1):
            t_l = f * 100_000_000                                   # lidar / annotation time (ns)
            t_c = t_l + 20_000_000                                  # the camera frame is 20 ms later: the ego has moved
            for t, x in ((t_l, 3.0 * f), (t_c, 3.0 * f + 0.5)):     # ego drives along +x in the city frame
                poses.append({"timestamp_ns": t, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "tx_m": x, "ty_m": 0.0, "tz_m": 0.0})
            Image.new("RGB", (W, H), (30 + 5 * f, 90, 160)).save(os.path.join(cdir, f"{t_c}.jpg"))
            for uid, cat, fn_ in (("uuidA", "REGULAR_VEHICLE", ego_a), ("uuidB", "BOX_TRUCK", ego_b), ("uuidC", "PEDESTRIAN", ego_c)):
                (l, w, h), c = fn_(f)
                anns.append({"timestamp_ns": t_l, "track_uuid": uid, "category": cat, "length_m": l, "width_m": w, "height_m": h,
                             "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "tx_m": c[0], "ty_m": c[1], "tz_m": c[2],
                             "num_interior_pts": 15})
        pd.DataFrame(poses).to_feather(os.path.join(ld, "city_SE3_egovehicle.feather"))
        pd.DataFrame(anns).to_feather(os.path.join(ld, "annotations.feather"))


def ego_moved(size, centre, dx=0.5):
    """AV2: the annotation is in ego(t_lidar); the camera frame is taken after the ego advanced by dx along +x, so the
    object is dx closer in the camera-time ego frame."""
    return size, (centre[0] - dx, centre[1], centre[2])


def boxes3d(root):
    os.makedirs(root)
    P = np.array([[FX, 0, CX, 0.0], [0, FX, CY, 0.0], [0, 0, 1.0, 0.0]]) @ np.block(
        [[CAM_TO_EGO.T, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])   # ego frame -> camera -> pixels
    make_images(os.path.join(root, "frames"), n=N, pattern="{:05d}.jpg", start=0)
    boxes = []
    for f in range(1, N + 1):
        for oid, cls, fn_ in (("a", "car", ego_a), ("b", "truck", ego_b)):
            (l, w, h), c = fn_(f)
            boxes.append({"frame": f - 1, "id": oid, "class": cls, "center": list(c), "size": [l, w, h], "yaw": 0.0})
    json.dump({"name": "i24like", "split": "train", "fps": 30,
               "cameras": {"camA": {"P": P.tolist(), "width": W, "height": H, "frames_dir": "frames", "ext": ".jpg"}},
               "boxes": boxes}, open(os.path.join(root, "scene.json"), "w"))
