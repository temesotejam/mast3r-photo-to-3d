"""Estimate camera-to-world poses from RGB frames and approximate axial depth.

The first camera defines world coordinates (OpenCV axes: right, down, forward).
Translations inherit the depth units. Intrinsics must describe the RGB images.
This is adjacent-frame visual odometry: depth-scale drift and pose errors accumulate.
It requires overlapping views of a textured, rigid background, including outside
the selected moving parts. Motion of the entire visible scene is ambiguous.
"""
import cv2
import numpy as np


class _NoBackgroundPose(ValueError):
    """The current frame pair does not constrain a reliable camera pose."""


def _inside(points, box, pad=0):
    x0, y0, x1, y1 = box
    return ((points[:, 0] >= x0 - pad) & (points[:, 0] <= x1 + pad)
            & (points[:, 1] >= y0 - pad) & (points[:, 1] <= y1 + pad))


def _move_boxes(boxes, src, dst, shape):
    height, width = shape
    moved = []
    for box in boxes:
        hits = _inside(src, box)
        # A median translation is robust to mismatches and follows the marked
        # part without fitting an unstable transform to a tiny set of corners.
        shift = np.median(dst[hits] - src[hits], axis=0) if hits.sum() >= 3 else np.zeros(2)
        x0, y0, x1, y1 = box + np.tile(shift, 2)
        moved.append(np.array([x0, y0, x1, y1]).clip([0, 0, 0, 0], [width, height, width, height]))
    return moved


def _lk_matches(prev, cur, boxes):
    background = np.full(prev.shape, 255, dtype=np.uint8)
    for x0, y0, x1, y1 in np.array(boxes, dtype=np.int32).reshape(-1, 4):
        cv2.rectangle(background, (x0, y0), (x1, y1), 0, -1)
    corners = []
    for mask, limit in [(background, 1600), (255 - background, 350)]:
        if not mask.any():
            continue
        points = cv2.goodFeaturesToTrack(prev, limit, 0.012, 6, mask=mask, blockSize=5)
        if points is not None:
            corners.append(points)
    if not corners:
        raise _NoBackgroundPose('追跡できる模様がありません。')
    src = np.concatenate(corners)
    params = dict(winSize=(23, 23), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.015))
    dst, valid, _ = cv2.calcOpticalFlowPyrLK(prev, cur, src, None, **params)
    src, dst = src[valid.ravel() != 0], dst[valid.ravel() != 0]
    if not len(src):
        raise _NoBackgroundPose('隣接フレームの対応点を検出できません。')
    back, valid, _ = cv2.calcOpticalFlowPyrLK(cur, prev, dst, None, **params)
    keep = (valid.ravel() != 0) & (np.linalg.norm((back - src).reshape(-1, 2), axis=1) < 1.5)
    return src[keep].reshape(-1, 2), dst[keep].reshape(-1, 2)


def _orb_matches(prev, cur):
    detector = cv2.ORB_create(nfeatures=3000, fastThreshold=12)
    k0, d0 = detector.detectAndCompute(prev, None)
    k1, d1 = detector.detectAndCompute(cur, None)
    if d0 is None or d1 is None or min(len(d0), len(d1)) < 2:
        raise _NoBackgroundPose('背景の特徴点を検出できません。')
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(d0, d1, k=2)
    reverse = {m.queryIdx: m.trainIdx for m in matcher.match(d1, d0)}
    good = [a for a, b in pairs if a.distance < 0.76 * b.distance and reverse.get(a.trainIdx) == a.queryIdx]
    return (np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1, 2),
            np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 2))


def _relative_pose(src, dst, depth, calibration, boxes):
    h, w = depth.shape
    # LK can return points near/outside the frame when a feature moves out of view.
    inside = ((src >= 0).all(axis=1) & (src < [w - 1, h - 1]).all(axis=1)
              & (dst >= 0).all(axis=1) & (dst < [w - 1, h - 1]).all(axis=1))
    src, dst = src[inside], dst[inside]
    moved = _move_boxes(boxes, src, dst, depth.shape)
    static = np.ones(len(src), dtype=bool)
    for old, new in zip(boxes, moved):
        # Allow for part rotation and inaccuracies in the box propagation.
        pad = max(5.0, 0.08 * max(old[2] - old[0], old[3] - old[1]))
        static &= ~_inside(src, old, pad) & ~_inside(dst, new, pad)
    src, dst = src[static], dst[static]
    pixels = np.rint(src).astype(np.int32)
    z = depth[pixels[:, 1], pixels[:, 0]]
    keep = np.isfinite(z) & (z > 0)
    src, dst, z = src[keep], dst[keep], z[keep]
    if len(src) < 15:
        raise _NoBackgroundPose('有効な深度を持つ背景対応点が15点未満です。')
    rays = np.column_stack((src, np.ones(len(src)))) @ np.linalg.inv(calibration).T
    xyz = np.ascontiguousarray(rays * (z / rays[:, 2])[:, None], dtype=np.float64)
    ok, r, t, inliers = cv2.solvePnPRansac(
        xyz, dst, calibration, None, iterationsCount=350, reprojectionError=3.0,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < max(12, 0.55 * len(src)):
        raise _NoBackgroundPose('背景点の過半数が同一のカメラ運動に一致しません。')
    ids = inliers.ravel()
    hull = cv2.convexHull(src[ids])
    if cv2.contourArea(hull) < w * h * 0.025:
        raise _NoBackgroundPose('背景の対応点が狭い領域に集中しています。')
    pts, uv = xyz[ids], dst[ids]
    r, t = cv2.solvePnPRefineLM(pts, uv, calibration, None, r, t)
    rotation = cv2.Rodrigues(r)[0]
    projected = cv2.projectPoints(pts, r, t, calibration, None)[0].reshape(-1, 2)
    errors = np.linalg.norm(projected - uv, axis=1)
    positive = (pts @ rotation[2] + t[2, 0]) > 0
    if not np.isfinite(errors).all() or np.median(errors) > 2.2 or positive.mean() < 0.90:
        raise _NoBackgroundPose('背景点の再投影誤差が大きく、カメラ姿勢を確定できません。')
    # solvePnP maps previous-camera points into the current camera. Returning
    # its inverse lets callers compose camera-to-first-frame poses in time order.
    relative = np.eye(4)
    relative[:3, :3] = rotation.T
    relative[:3, 3] = -(rotation.T @ t).ravel()
    return relative, moved


def estimate_camera_poses(frames, depths, intrinsics, parts):
    """Return a list of 4x4 camera-to-first-camera transforms.

    frames: equal-sized uint8 RGB images; depths: aligned HxW camera-Z maps
    (may have lower resolution, but must cover the same field of view).
    intrinsics: shared 3x3 pinhole calibration. Each part's first-frame box is
    [left, top, right, bottom] in full-resolution pixels. Unmarked moving
    objects are treated as outliers; a dominant moving background cannot be
    distinguished from true camera motion. Use fixed-camera mode for tripod
    captures without enough stationary background.
    """
    if not len(frames) or len(frames) != len(depths):
        raise ValueError('フレーム数と深度画像の枚数が一致する必要があります。')
    calibration = np.array(intrinsics, dtype=np.float64)
    if (calibration.shape != (3, 3) or not np.isfinite(calibration).all()
            or calibration[0, 0] <= 0 or calibration[1, 1] <= 0
            or not np.allclose(calibration[2], [0, 0, 1])):
        raise ValueError('有効な3×3のカメラ内部パラメータが必要です。')
    shape = np.asarray(frames[0]).shape
    if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 24:
        raise ValueError('24画素以上のRGBフレームが必要です。')
    h, w = shape[:2]
    scale = min(1.0, 960 / max(w, h))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    sx, sy = size[0] / w, size[1] / h
    calibration = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]]) @ calibration
    boxes = []
    for part in parts:
        box = np.asarray(part['box'], dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            raise ValueError('移動部品の範囲を左上・右下の画素座標で指定してください。')
        boxes.append(box * [sx, sy, sx, sy])

    def gray(frame):
        frame = np.asarray(frame)
        if frame.shape != shape or frame.dtype != np.uint8:
            raise ValueError('同一サイズの8ビットRGBフレームが必要です。')
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA) if size != (w, h) else image

    poses = [np.eye(4)]
    previous = gray(frames[0])
    for i in range(1, len(frames)):
        current = gray(frames[i])
        depth = np.asarray(depths[i - 1], dtype=np.float32)
        if depth.ndim != 2 or min(depth.shape) == 0:
            raise ValueError('2次元の有効な深度画像が必要です。')
        if depth.shape != previous.shape:
            depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
        for use_orb in (False, True):
            try:
                tracks = _orb_matches(previous, current) if use_orb else _lk_matches(previous, current, boxes)
                relative, new_boxes = _relative_pose(*tracks, depth, calibration, boxes)
                break
            except (_NoBackgroundPose, cv2.error) as error:
                if use_orb:
                    why = str(error) if isinstance(error, _NoBackgroundPose) else '幾何検証に失敗しました。'
                    raise ValueError(f'カメラ追跡失敗（フレーム{i}→{i + 1}）。{why}'
                                     '静止した背景を広く含め、フレーム間隔を短くしてください。') from error
        poses.append(poses[-1] @ relative)
        previous, boxes = current, new_boxes
    return poses
