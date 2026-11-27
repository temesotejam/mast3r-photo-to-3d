"""Track and fuse separate rigid links using RGB, axial depth and a pinhole camera."""

import cv2
import numpy as np


def _xyz(pixels, depth, K):
    uv = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    h, w = depth.shape
    positions = np.rint(uv).astype(int)
    inside = ((positions[:, 0] >= 0) & (positions[:, 0] < w)
              & (positions[:, 1] >= 0) & (positions[:, 1] < h))
    z = depth[positions[:, 1].clip(0, h - 1), positions[:, 0].clip(0, w - 1)]
    z = np.where(inside & np.isfinite(z) & (z > 0), z, np.nan)
    return np.column_stack(((uv[:, 0] - K[0, 2]) * z / K[0, 0],
                            (uv[:, 1] - K[1, 2]) * z / K[1, 1], z))


def _transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def _inverse(pose):
    result = np.eye(4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ pose[:3, 3]
    return result


def _mask_sample(mask, limit):
    y, x = np.nonzero(mask)
    if len(x) > limit:
        indices = np.linspace(0, len(x) - 1, limit, dtype=int)
        x, y = x[indices], y[indices]
    return np.column_stack((x, y))


def _inside(mask, uv):
    h, w = mask.shape
    x, y = np.rint(uv).astype(int).T
    return ((x >= 0) & (y >= 0) & (x < w) & (y < h)
            & (mask[y.clip(0, h - 1), x.clip(0, w - 1)] > 0))


def _component(mask, support):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    counts = np.bincount(labels[support > 0], minlength=n)
    # Ignore small islands even if they include a feature seed.
    score = np.where(stats[:, cv2.CC_STAT_AREA] >= 16, counts, 0)
    score[0] = 0
    if score.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    return (labels == score.argmax()).astype("uint8")


def _segment(rgb, depth, support, rect, first=False):
    """Use a small ROI so foreground/background fitting is per link."""
    h, w = depth.shape
    xa, ya, xb, yb = rect
    bw, bh = xb - xa, yb - ya
    if min(bw, bh) < 5:
        return np.zeros((h, w), np.uint8)
    pad = max(4, int(0.18 * max(bw, bh)))
    x, y = max(0, xa - pad), max(0, ya - pad)
    r, b = min(w, xb + pad), min(h, yb + pad)
    roi = np.ascontiguousarray(rgb[y:b, x:r])
    seeds = support[y:b, x:r] > 0
    labels = np.zeros(roi.shape[:2], dtype=np.uint8)
    labels[ya-y:yb-y, xa-x:xb-x] = cv2.GC_PR_BGD
    if first:
        labels[ya-y:yb-y, xa-x:xb-x] = cv2.GC_PR_FGD
    else:
        labels[seeds] = cv2.GC_PR_FGD
    labels[[0, -1], :] = 0
    labels[:, [0, -1]] = 0
    if np.any(labels == cv2.GC_PR_FGD) and np.any(labels == 0):
        cv2.grabCut(roi, labels, None, np.zeros((1, 65)), np.zeros((1, 65)),
                    2, cv2.GC_INIT_WITH_MASK)
    candidate = (labels & 1).astype(np.uint8)
    # A failed segmentation is preferable to promoting the whole box to geometry.
    chosen = _component(candidate, seeds)
    result = np.zeros_like(support, dtype=np.uint8)
    result[y:b, x:r] = chosen
    return result


def _box(rect, w, h):
    coords = np.asarray(rect, dtype=float)
    if coords.shape != (4,) or not np.isfinite(coords).all():
        raise ValueError("Each link requires an x0,y0,x1,y1 rectangle.")
    if coords.min() >= 0 and coords.max() <= 1:
        coords = coords * [w, h, w, h]
    x, y, r, b = np.rint(coords).astype(int)
    x, y, r, b = max(0, x), max(0, y), min(w, r), min(h, b)
    if min(r - x, b - y) < 5:
        raise ValueError("Link rectangle is empty or too small.")
    return x, y, r, b


def _sample_features(gray, mask, depth, K, pose):
    # Store canonical 3D coordinates for optical-flow correspondences.
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=220, qualityLevel=0.008,
                                      minDistance=3, mask=mask.astype("uint8"))
    if corners is None:
        return np.empty((0, 2), np.float32), np.empty((0, 3), np.float32)
    uv = corners.reshape(-1, 2)
    xyz = _xyz(uv, depth, K)
    keep = np.isfinite(xyz).all(axis=1)
    return uv[keep], _transform(xyz[keep], _inverse(pose)).astype("float32")


def _flow(last, gray):
    _, previous, uv, local = last
    if len(uv) < 6:
        return None
    params = dict(winSize=(19, 19), maxLevel=3)
    now, ok, _ = cv2.calcOpticalFlowPyrLK(previous, gray, uv, None, **params)
    if now is None:
        return None
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(gray, previous, now, None, **params)
    if back is None:
        return None
    good = (ok.ravel() > 0) & (ok2.ravel() > 0) & (np.linalg.norm(uv - back, axis=1) < 1.25)
    return local[good], now[good]


def _describe_matches(keys, scene_features, matcher):
    uv, desc = scene_features
    if desc is None or len(desc) < 2:
        return []
    matches = []
    for old_desc, local in keys:
        forward = matcher.knnMatch(old_desc, desc, k=2)
        reverse = {m.queryIdx: m.trainIdx for m in matcher.match(desc, old_desc)}
        keep = [m for pair in forward if len(pair) == 2 for m, n in [pair]
                if m.distance < 0.77 * n.distance and reverse[m.trainIdx] == m.queryIdx]
        if len(keep) >= 4:
            matches.append((local[[m.queryIdx for m in keep]], uv[[m.trainIdx for m in keep]]))
    return matches


def _distinctive_hue(hsv, mask):
    """Use paint colour as an identity cue only when it separates the link from its surroundings."""
    hue, sat, val = cv2.split(hsv)
    chromatic = (sat > 55) & (val > 34)
    fg = mask.astype(bool)
    ring = (cv2.dilate(mask, np.ones((23, 23), np.uint8)) > 0) & ~fg
    counts = np.bincount(hue[fg & chromatic], minlength=180)
    contrast = np.bincount(hue[ring & chromatic], minlength=180)
    window = np.ones(23)
    within = np.convolve(np.pad(counts, 11, mode="wrap"), window, mode="valid") / max(1, fg.sum())
    outside = np.convolve(np.pad(contrast, 11, mode="wrap"), window, mode="valid") / max(1, ring.sum())
    mode = int(np.argmax(within - outside))
    return mode if within[mode] > 0.38 and within[mode] - outside[mode] > 0.24 else None


def _hue_region(hsv, hue):
    delta = np.abs(hsv[:, :, 0].astype(int) - hue)
    core = (np.minimum(delta, 180 - delta) < 21) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 22)
    # Printed detail, highlights and small neutral patches inside the link are legitimate.
    dilated = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return dilated


def _rigid(source, target):
    p, q = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - p).T @ (target - q))
    vt[-1] *= np.sign(np.linalg.det(vt.T @ u.T))
    pose = np.eye(4)
    pose[:3, :3] = vt.T @ u.T
    pose[:3, 3] = q - pose[:3, :3] @ p
    return pose


def _pose_from_pairs(matches, depth, K, mask, scale):
    if not matches:
        return None
    # A corner can be found by several keyframes and optical flow. Give its
    # image neighborhood a single vote in the pose fit.
    local = np.concatenate([pair[0] for pair in matches]).astype(np.float32)
    uv = np.concatenate([pair[1] for pair in matches]).astype(np.float32)
    if mask is not None:
        keep = _inside(mask, uv)
        local, uv = local[keep], uv[keep]
    _, unique = np.unique(np.rint(uv / 2).astype(int), axis=0, return_index=True)
    local, uv = local[unique], uv[unique]
    if len(local) < 6:
        return None
    ok, rot, shift, inliers = cv2.solvePnPRansac(
        local, uv, K, None, iterationsCount=140, reprojectionError=3.0,
        confidence=0.995, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < max(6, int(0.22 * len(local))):
        return None
    indices = inliers.ravel()
    if cv2.contourArea(cv2.convexHull(uv[indices])) < 12:
        return None
    observed = _xyz(uv[indices], depth, K)
    rot, shift = cv2.solvePnPRefineLM(local[indices], uv[indices], K, None, rot, shift)
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rot)[0]
    pose[:3, 3] = shift.ravel()
    predicted = _transform(local[indices], pose)
    # Reject pixels belonging to a different surface in front of the part.
    agree = (np.isfinite(observed).all(axis=1) & np.isfinite(predicted).all(axis=1)
             & (predicted[:, 2] > 0)
             & (np.abs(observed[:, 2] - predicted[:, 2])
                < np.maximum(0.24 * scale, 0.09 * np.abs(predicted[:, 2]))))
    if np.count_nonzero(agree) < max(6, int(0.55 * len(indices))):
        return None
    # RGB-D registration constrains out-of-plane rotation of thin links, which
    # can be poorly determined by single-view PnP. Use it only when both 3D
    # distances and reprojections agree after outlier rejection.
    local3 = local[indices[agree]]
    target3 = observed[agree]
    pixels = uv[indices[agree]]
    tol = max(0.035 * scale, 0.008 * float(np.median(target3[:, 2])))
    best = np.zeros(len(local3), bool)
    rng = np.random.default_rng(57)
    for _ in range(36):
        sample = rng.choice(len(local3), 3, replace=False)
        trial = _rigid(local3[sample], target3[sample])
        projected = _transform(local3, trial) @ K.T
        close = ((np.linalg.norm(_transform(local3, trial) - target3, axis=1) < tol)
                 & (np.linalg.norm(projected[:, :2] / projected[:, 2:] - pixels, axis=1) < 3.5))
        if close.sum() > best.sum():
            best = close
    if best.sum() >= max(6, len(local3) * 0.60):
        pose = _rigid(local3[best], target3[best])
        pixels = pixels[best]
    return pose, pixels


class _Link:
    def __init__(self, part, shape):
        self.spec = part
        self.name = str(part.get("name", part.get("id", "link")))
        self.box = _box(part["box"], shape[1], shape[0])
        self.voxels = {}
        self.keys = []
        self.last = None
        self.scale = None
        self.hue = None
        self.initial_area = 0
        self.initial_distance = 0
        self.rows = []

    def fuse(self, rgb, depth, K, mask, pose):
        uv = _mask_sample(mask, 4500)
        cam = _xyz(uv, depth, K)
        local = _transform(cam, _inverse(pose))
        valid = (np.isfinite(local).all(axis=1)
                 & (np.linalg.norm(local, axis=1) < self.scale * 1.8))
        local, uv = local[valid], uv[valid]
        if not len(uv):
            return 0, 0
        color = rgb[uv[:, 1], uv[:, 0]]
        cells = np.rint(local / (self.scale / 80)).astype(np.int32)
        unique, inverse = np.unique(cells, return_inverse=True, axis=0)
        sample = np.column_stack((local, color))
        sums = np.zeros((len(unique), 6))
        np.add.at(sums, inverse, sample)
        means = sums / np.bincount(inverse)[:, None]
        before = len(self.voxels)
        for pos, obs in zip(unique, means):
            key = tuple(pos)
            if key in self.voxels:
                old, weight = self.voxels[key]
                self.voxels[key] = old + (obs - old) / (weight + 1), weight + 1
            else:
                self.voxels[key] = obs, 1
        return len(self.voxels) - before, len(local)

    def projected_mask(self, rgb, depth, K, pose, inlier_uv):
        h, w = depth.shape
        # Support from already seen model surfaces, with a current-depth
        # visibility check, leaves room for newly exposed faces nearby.
        saved = list(self.voxels.values())
        points = np.asarray([v[0][:3] for v in saved[::max(1, len(saved) // 2500)]])
        support = np.zeros((h, w), dtype=np.uint8)
        if len(points):
            cam = _transform(points, pose)
            cam = cam[cam[:, 2] > 1e-6]
            proj = cam @ K.T
            uv = proj[:, :2] / proj[:, 2:3]
            keep = (np.abs(uv[:, 0]) < w * 2) & (np.abs(uv[:, 1]) < h * 2)
            cam, uv = cam[keep], uv[keep]
            current = _xyz(uv, depth, K)[:, 2]
            visible = (np.isfinite(current) & (np.abs(current - cam[:, 2])
                       < np.maximum(0.20 * self.scale, 0.06 * cam[:, 2])))
            for px, py in np.rint(uv[visible]).astype(int):
                cv2.circle(support, (px, py), 3, 1, -1)
        for px, py in np.rint(inlier_uv).astype(int):
            cv2.circle(support, (px, py), 4, 1, -1)
        x, y, bw, bh = cv2.boundingRect(support)
        # The rectangle may grow beyond the currently projected silhouette.
        expand = max(4, int(0.20 * max(bw, bh)))
        rect = (max(0, x-expand), max(0, y-expand),
                min(w, x+bw+expand), min(h, y+bh+expand))
        mask = _segment(rgb, depth, support, rect)
        if not len(points) or not mask.any():
            return mask
        # Adjacent links often form a single 2D silhouette. Reject surfaces
        # separated in 3D from the link, while allowing the model to grow
        # around an edge as a previously unseen face turns into view.
        ys, xs = np.nonzero(mask)
        local = _transform(_xyz(np.column_stack((xs, ys)), depth, K), _inverse(pose))
        finite = np.isfinite(local).all(axis=1)
        keep = np.zeros(len(local), bool)
        if finite.any():
            index = cv2.flann_Index(points.astype(np.float32), dict(algorithm=1, trees=2))
            _, distance2 = index.knnSearch(local[finite].astype(np.float32), 1, params=dict(checks=16))
            radius = max(0.11 * self.scale, 0.025 * self.initial_distance)
            keep[finite] = distance2[:, 0] < radius * radius
        mask[ys[~keep], xs[~keep]] = 0
        return mask

    def observe(self, idx, rgb, gray, hsv, depth, K, features, matcher, masks, camera):
        row = {"frame": idx, "status": "unavailable", "part_to_camera": None,
               "part_to_world": None, "inliers": 0, "new_points": 0, "fused_samples": 0}
        self.rows.append(row)
        mask = None
        if masks is not None:
            value = masks.get(idx) if isinstance(masks, dict) else masks[idx]
            if value is not None:
                mask = np.asarray(value, dtype=bool).astype("uint8")
                if mask.shape != depth.shape:
                    raise ValueError(f"Mask size does not match for {self.name}, frame {idx}.")
                if not mask.any():
                    return
        if not self.voxels:
            if idx != 0 and mask is None:
                return
            if mask is None:
                xa, ya, xb, yb = self.box
                seed = np.zeros_like(depth, dtype=np.uint8)
                seed[ya:yb, xa:xb] = 1
                mask = _segment(rgb, depth, seed, self.box, first=True)
                self.hue = _distinctive_hue(hsv, mask)
                if self.hue is not None:
                    mask &= _hue_region(hsv, self.hue)
            xyz = _xyz(_mask_sample(mask, 3000), depth, K)
            xyz = xyz[np.isfinite(xyz).all(axis=1)]
            if len(xyz) < 16:
                return
            pose = np.eye(4)
            pose[:3, 3] = np.median(xyz, axis=0)
            self.initial_area = np.count_nonzero(mask)
            self.initial_distance = pose[2, 3]
            self.scale = max(1e-7, np.linalg.norm(np.percentile(xyz, 95, axis=0)
                                                   - np.percentile(xyz, 5, axis=0)))
            inliers = np.empty((0, 2))
        else:
            saved_matches = _describe_matches(self.keys, features, matcher)
            matches = list(saved_matches)
            if self.last is not None and self.last[0] == idx - 1:
                flowed = _flow(self.last, gray)
                if flowed is not None:
                    matches.insert(0, flowed)
            allowed = _hue_region(hsv, self.hue) if self.hue is not None and mask is None else mask
            result = _pose_from_pairs(matches, depth, K, allowed, self.scale)
            if result is None and self.hue is not None and mask is None:
                # A newly turned face can have a different colour. Reacquire
                # from many mutually matched, stored texture descriptors and a
                # rigid fit before trusting observations away from the paint cue.
                alternative = _pose_from_pairs(saved_matches, depth, K, None, self.scale)
                if alternative is not None and len(alternative[1]) >= 18:
                    prior = [r for r in self.rows[:-1] if r["status"] == "observed"]
                    space = "part_to_camera" if camera is None else "part_to_world"
                    last = prior[-1][space][:3, 3]
                    proposed = alternative[0] if camera is None else camera @ alternative[0]
                    # A weakly identified face must connect to the existing
                    # track. A similar texture on a moving occluder can also
                    # fit a rigid pose but usually jumps to another depth.
                    per_frame = 0.08 * self.scale
                    if len(prior) > 1:
                        shift = np.linalg.norm(last - prior[-2][space][:3, 3])
                        per_frame = max(per_frame, 2.5 * shift / (prior[-1]["frame"] - prior[-2]["frame"]))
                    travel = per_frame * (idx - prior[-1]["frame"]) + 0.04 * self.scale
                    if np.linalg.norm(proposed[:3, 3] - last) <= travel:
                        result = alternative
            if result is None:
                return
            pose, inliers = result
            if mask is None:
                mask = self.projected_mask(rgb, depth, K, pose, inliers)
                if allowed is not None:
                    # Allow nearby new texture on a face reached through a
                    # neighbouring view, even where its paint is different.
                    radius = min(12, max(4, round(0.04 * np.sqrt(self.initial_area))))
                    for x, y in np.rint(inliers).astype(int):
                        cv2.circle(allowed, (x, y), radius, 1, -1)
                    mask &= allowed
            expected_area = self.initial_area * (self.initial_distance / max(pose[2, 3], 1e-6)) ** 2
            if mask.sum() < 0.075 * expected_area:
                return
            # Registration is sufficient to observe pose. Learn points and
            # appearance only on a confident current-frame segmentation.
            if len(inliers) and np.count_nonzero(_inside(mask, inliers)) < max(4, len(inliers)//3):
                mask = np.zeros_like(mask)

        new_points, n = self.fuse(rgb, depth, K, mask, pose)
        row.update(status="observed", part_to_camera=pose, inliers=len(inliers),
                   part_to_world=camera @ pose if camera is not None else None,
                   new_points=new_points, fused_samples=n)
        if n:
            uv, local = _sample_features(gray, mask, depth, K, pose)
            self.last = idx, gray, uv, local
            uv, desc = features
            if desc is not None:
                selected = _inside(mask, uv)
                local = _transform(_xyz(uv[selected], depth, K), _inverse(pose))
                good = np.isfinite(local).all(axis=1) & (np.linalg.norm(local, axis=1) < 1.8 * self.scale)
                if good.sum() >= 5:
                    self.keys.append((desc[selected][good][:180], local[good][:180].astype("float32")))
                    if len(self.keys) > 16:
                        # Keep the first appearance and recent keyframes.
                        self.keys.pop(1)

    def finish(self, cameras, times):
        observed = [i for i, row in enumerate(self.rows) if row["status"] == "observed"]
        for a, b in zip(observed, observed[1:]):
            if b == a + 1:
                continue
            space = "part_to_world" if cameras is not None else "part_to_camera"
            start, end = self.rows[a][space], self.rows[b][space]
            axis = cv2.Rodrigues(start[:3, :3].T @ end[:3, :3])[0]
            for i in range(a + 1, b):
                f = (times[i] - times[a]) / (times[b] - times[a])
                pose = np.eye(4)
                pose[:3, :3] = start[:3, :3] @ cv2.Rodrigues(f * axis)[0]
                pose[:3, 3] = (1 - f) * start[:3, 3] + f * end[:3, 3]
                row = self.rows[i]
                row.update(status="interpolated", part_to_camera=pose)
                if cameras is not None:
                    row.update(part_to_world=pose, part_to_camera=_inverse(cameras[i]) @ pose)
        samples = np.asarray([m for m, _ in self.voxels.values()]).reshape(-1, 6)
        return {"id": self.spec.get("id", self.name), "name": self.name,
                "points": samples[:, :3].astype("float32"),
                "colors": np.rint(samples[:, 3:]).clip(0, 255).astype("uint8"),
                "frames": self.rows, "fused_frames": [r["frame"] for r in self.rows if r["fused_samples"]],
                "new_points_per_frame": [r["new_points"] for r in self.rows]}


def reconstruct_parts(frames, depths, intrinsics, parts, camera_poses=None, progress=None, times=None):
    """Estimate independent part-to-camera transforms in OpenCV camera coordinates.

    Supply uint8 RGB frames, matching positive axial depths (one consistent
    unit), a shared or per-frame K, and first-frame boxes [x0,y0,x1,y1]. Each
    part may supply per-frame binary masks (None = infer, all-zero = invisible).
    Optional camera-to-world 4x4 poses enable world-relative gap interpolation.
    Only observed, segmented frames contribute geometry.
    """
    n = len(frames)
    if not n or n != len(depths) or not parts:
        raise ValueError("A matching sequence of frames and depths and at least one link are required.")
    h, w = np.asarray(frames[0]).shape[:2]
    cameras = np.asarray(camera_poses, dtype=float) if camera_poses is not None else None
    ks = np.broadcast_to(np.asarray(intrinsics, dtype=float), (n, 3, 3))
    times = np.arange(n) if times is None else np.asarray(times, dtype=float)
    if times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Frame timestamps must be finite and strictly increasing.")
    if cameras is not None and cameras.shape != (n, 4, 4):
        raise ValueError("Expected one 4x4 camera-to-world transform per frame.")
    links = [_Link(spec, (h, w)) for spec in parts]
    orb = cv2.ORB_create(nfeatures=3000, edgeThreshold=8, patchSize=21, fastThreshold=7)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    for idx, (rgb, depth, K) in enumerate(zip(frames, depths, ks)):
        if progress is not None:
            progress(f"Tracking rigid parts ({idx + 1}/{n})")
        rgb = np.asarray(rgb, dtype=np.uint8)
        depth = np.asarray(depth, dtype=np.float32)
        if rgb.shape != (h, w, 3) or depth.shape != (h, w):
            raise ValueError("Every RGB and depth frame must share one resolution.")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        locations = np.array([p.pt for p in keypoints], dtype="float32").reshape(-1, 2)
        for link in links:
            link.observe(idx, rgb, gray, hsv, depth, K, (locations, descriptors), matcher,
                         link.spec.get("masks"), cameras[idx] if cameras is not None else None)
    return [link.finish(cameras, times) for link in links]
