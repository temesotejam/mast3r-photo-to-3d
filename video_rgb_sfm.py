"""Sparse RGB-only rigid structure from motion for short, fixed-camera videos.

Tomasi–Kanade factorization initializes the two weak-perspective mirror solutions.
Calibrated perspective bundle adjustment chooses them using image reprojection.
Each object has its own unobservable global scale.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from video_calibration import _features, plane_warps
from video_parts import _distinctive_hue, _hue_region, _inside, _segment


def _track(grays, masks, warps):
    interior = cv2.erode(masks[0], np.ones((3, 3), np.uint8))
    seed = cv2.goodFeaturesToTrack(grays[0], 360, 0.004, 3, mask=interior)
    if seed is None or len(seed) < 12:
        return None
    uv = seed[:, 0].copy()
    tracks = np.full((len(grays), len(uv), 2), np.nan, dtype=np.float32)
    tracks[0] = uv
    for i in range(1, len(grays)):
        # Track locally through time, with a separate template registration
        # that can recover points after they pass behind another object.
        if i in warps:
            homography = warps[i]
            h, w = grays[i].shape
            aligned = cv2.warpPerspective(
                grays[i], homography, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            )
            target, ok, error = cv2.calcOpticalFlowPyrLK(
                grays[0], aligned, uv, None, winSize=(11, 11), maxLevel=1
            )
            if target is not None:
                good = (ok.ravel() > 0) & (error.ravel() < 11) & (np.linalg.norm(target - uv, axis=1) < 4)
                if good.any():
                    tracks[i, good] = cv2.perspectiveTransform(target[good, None], homography)[:, 0]
        idx = np.where(np.isfinite(tracks[i - 1, :, 0]))[0]
        if len(idx):
            previous = tracks[i - 1, idx]
            dest, ok, error = cv2.calcOpticalFlowPyrLK(
                grays[i - 1], grays[i], previous, None, winSize=(15, 15), maxLevel=3
            )
            if dest is not None:
                back, ok2, _ = cv2.calcOpticalFlowPyrLK(
                    grays[i], grays[i - 1], dest, None, winSize=(15, 15), maxLevel=3
                )
                if back is not None:
                    good = (
                        (ok.ravel() > 0)
                        & (ok2.ravel() > 0)
                        & (error.ravel() < 15)
                        & (np.linalg.norm(back - previous, axis=1) < 0.75)
                        & ~np.isfinite(tracks[i, idx, 0])
                    )
                    tracks[i, idx[good]] = dest[good]
        seen = np.flatnonzero(np.isfinite(tracks[i, :, 0]))
        if len(seen):
            tracks[i, seen[~_inside(masks[i], tracks[i, seen])]] = np.nan
    return tracks


def _factor(uv, k):
    """Complete a centered rank-three track matrix, then enforce Euclidean camera rows."""
    valid = np.isfinite(uv[..., 0])
    w = np.nan_to_num(uv) / k[0, 0]
    centers = np.sum(w, axis=1) / valid.sum(axis=1)[:, None]
    values = (w - centers[:, None]).transpose(0, 2, 1).reshape(-1, w.shape[1])
    observed = np.repeat(valid, 2, axis=0)
    matrix = np.where(observed, values, 0.0)
    for _ in range(30):
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        if s[1] < 1e-5:
            return None
        rank3 = (u[:, :3] * s[:3]) @ vt[:3]
        # Robust treatment of mistracked locations. Refit frame translations
        # along with the unseen entries rather than treating them as observations.
        offset = np.nanmedian(np.where(observed, values - rank3, np.nan), axis=1, keepdims=True)
        matrix = rank3 + np.where(observed, np.clip(values - offset - rank3, -0.004, 0.004), 0.0)
    motion = u[:, :3] * np.sqrt(np.maximum(s[:3], 1e-12))
    shape = np.sqrt(np.maximum(s[:3], 1e-12))[:, None] * vt[:3]
    a, b = motion[::2], motion[1::2]

    def coeff(x, y):
        return np.column_stack(
            (
                x[:, 0] * y[:, 0],
                x[:, 1] * y[:, 1],
                x[:, 2] * y[:, 2],
                x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0],
                x[:, 0] * y[:, 2] + x[:, 2] * y[:, 0],
                x[:, 1] * y[:, 2] + x[:, 2] * y[:, 1],
            )
        )

    system = np.vstack((coeff(a, a), coeff(b, b), coeff(a, b)))
    target = np.repeat([1.0, 1.0, 0.0], len(a))
    l = np.linalg.lstsq(system, target, rcond=1e-5)[0]
    metric = np.array([[l[0], l[3], l[4]], [l[3], l[1], l[5]], [l[4], l[5], l[2]]])
    vals, vecs = np.linalg.eigh(metric)
    gauge = vecs * np.sqrt(np.maximum(vals, max(vals[-1], 1e-5) * 0.01))
    first = motion[:2] @ gauge
    first = np.vstack((first, np.cross(*first)))
    u, _, vt = np.linalg.svd(first)
    return (u @ vt @ np.linalg.solve(gauge, shape))[2]


@dataclass
class _PerspectiveFit:
    parameters: np.ndarray
    mean_clipped_error: float
    errors: np.ndarray
    frame_indices: np.ndarray
    point_indices: np.ndarray
    reference_rays: np.ndarray
    reference_center: np.ndarray
    refine: Callable
    project: Callable


def _fit(tracks, intrinsics, z0):
    valid = np.isfinite(tracks[..., 0])
    frame_ids, point_ids = np.nonzero(valid[1:])
    frame_ids += 1
    rays = np.column_stack((tracks[0], np.ones(tracks.shape[1]))) @ np.linalg.inv(intrinsics).T
    reference_center = rays.mean(axis=0)
    moving_views = len(tracks) - 1

    def project(x, frame, corner):
        poses = x[: 6 * moving_views].reshape(moving_views, 6)
        rotations = Rotation.from_rotvec(poses[:, :3]).as_matrix()
        p = rays[corner] * np.exp(x[6 * moving_views :][corner, None]) - reference_center
        cam = np.einsum("nij,nj->ni", rotations[frame - 1], p) + poses[frame - 1, 3:]
        return cam[:, :2] / np.maximum(0.01, cam[:, 2:]) * intrinsics.diagonal()[:2] + intrinsics[:2, 2]

    points = (rays * z0[:, None] - reference_center).astype("float64")
    p0 = []
    for i in range(1, len(tracks)):
        select = valid[i]
        ok, rv, translations, _ = cv2.solvePnPRansac(
            points[select],
            tracks[i, select],
            intrinsics,
            None,
            iterationsCount=120,
            reprojectionError=3,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or not 0.2 < translations[2, 0] < 4:
            rv = np.zeros(3)
            mean = (np.nanmean(tracks[i], axis=0) - intrinsics[:2, 2]) / intrinsics.diagonal()[:2]
            translations = np.r_[mean, 1.0]
        p0.extend((*rv.ravel(), *translations.ravel()))
    start = np.r_[p0, np.log(z0)]
    # Fix the otherwise arbitrary size of each independent moving object.
    scale_anchor = int(np.argmin(abs(z0 - 1)))

    def solve(x, selected, iterations):
        frame, corner = frame_ids[selected], point_ids[selected]
        observed = tracks[frame, corner]

        def error(par):
            return np.r_[
                (project(par, frame, corner) - observed).ravel(),
                intrinsics[0, 0] * par[6 * moving_views + scale_anchor],
            ]

        jac = lil_matrix((2 * len(frame) + 1, len(x)), dtype=bool)
        rows = np.arange(2 * len(frame))
        for d in range(6):
            jac[rows, 6 * np.repeat(frame - 1, 2) + d] = True
        jac[rows, 6 * moving_views + np.repeat(corner, 2)] = True
        jac[-1, 6 * moving_views + scale_anchor] = True
        return least_squares(
            error,
            x,
            jac_sparsity=jac.tocsr(),
            loss="soft_l1",
            f_scale=0.75,
            max_nfev=iterations,
            x_scale="jac",
            bounds=(
                np.r_[np.full(6 * moving_views, -np.inf), np.full(len(rays), np.log(0.4))],
                np.r_[np.full(6 * moving_views, np.inf), np.full(len(rays), np.log(2.5))],
            ),
        ).x

    ids = np.arange(len(frame_ids))
    result = solve(start, ids, 50)
    errs = np.linalg.norm(project(result, frame_ids, point_ids) - tracks[frame_ids, point_ids], axis=1)
    return _PerspectiveFit(
        result,
        np.mean(np.minimum(errs, 4)),
        errs,
        frame_ids,
        point_ids,
        rays,
        reference_center,
        solve,
        project,
    )


def _reconstruct(grays, frames, features, k, part, times):
    n = len(frames)

    def empty_result():
        return {
            "name": part["name"],
            "points": np.empty((0, 3)),
            "colors": np.empty((0, 3)),
            "frames": [{"status": "unavailable", "inliers": 0, "part_to_world": None} for _ in frames],
            "new_points_per_frame": [0] * n,
        }

    x, y, r, b = part["box"]
    rect = np.zeros(grays[0].shape, np.uint8)
    rect[y:b, x:r] = 1
    try:
        mask = _segment(frames[0], None, rect, part["box"], first=True)
    except cv2.error:
        return empty_result()
    if not mask.any():
        return empty_result()
    hsv = [cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) for rgb in frames]
    hue = _distinctive_hue(hsv[0], mask)
    masks = [_hue_region(img, hue) if hue is not None else np.ones_like(rect) for img in hsv]
    masks[0] &= mask
    warps = plane_warps(grays, masks[0], features=features)
    tracks = _track(grays, masks, warps)
    if tracks is None:
        return empty_result()
    # Require several observations of the same physical point, and image support
    # spread over a two-dimensional region before estimating that frame's pose.
    good = np.isfinite(tracks[..., 0]).sum(axis=0) >= 3
    tracks = tracks[:, good]
    selected = []
    for i, tr in enumerate(tracks):
        xy = tr[np.isfinite(tr[:, 0])]
        if len(xy) >= 14 and np.linalg.eigvalsh(np.cov(xy.T))[0] > 12:
            selected.append(i)
    if len(selected) < 3 or selected[0] != 0:
        return empty_result()
    uv = tracks[selected]
    keep = np.isfinite(uv[..., 0]).sum(axis=0) >= 3
    uv = uv[:, keep]
    if uv.shape[1] < 14:
        return empty_result()
    valid = np.isfinite(uv[..., 0])
    if valid.sum(axis=1).min() < 8:
        return empty_result()
    movement = np.linalg.norm(uv - uv[0], axis=2)
    if np.nanmax(np.nanmedian(movement, axis=1)) < 1:
        return empty_result()
    z = _factor(uv, k)
    if z is None:
        return empty_result()
    z = np.clip(z - np.median(z), -0.35, 0.35)
    candidates = [_fit(uv, k, 1 + sign * z) for sign in (-1, 1)]
    fit = min(candidates, key=lambda candidate: candidate.mean_clipped_error)
    frame_ids, point_ids = fit.frame_indices, fit.point_indices
    inliers = fit.errors < 2.25
    if np.sum(inliers) < 24:
        return empty_result()
    parameters = fit.refine(fit.parameters, np.flatnonzero(inliers), 50)
    errors = np.linalg.norm(
        fit.project(parameters, frame_ids, point_ids) - uv[frame_ids, point_ids], axis=1
    )
    agree = errors < 2
    moving_views = len(selected) - 1
    points = fit.reference_rays * np.exp(parameters[6 * moving_views :, None]) - fit.reference_center
    count = np.bincount(point_ids[agree], minlength=len(points))
    point_ok = (count >= 2) & (
        np.linalg.norm(points, axis=1) < max(0.035, 3 * np.median(np.linalg.norm(points, axis=1)))
    )
    if point_ok.sum() < 12:
        return empty_result()
    poses = np.tile(np.eye(4), (len(selected), 1, 1))
    poses[0, :3, 3] = fit.reference_center
    pose_parameters = parameters[: 6 * moving_views].reshape(moving_views, 6)
    poses[1:, :3, :3] = Rotation.from_rotvec(pose_parameters[:, :3]).as_matrix()
    poses[1:, :3, 3] = pose_parameters[:, 3:]
    out = empty_result()
    out["points"] = points[point_ok]
    pixel_x, pixel_y = np.rint(uv[0, point_ok]).astype(int).T
    out["colors"] = frames[0][pixel_y, pixel_x]
    out["new_points_per_frame"][0] = len(pixel_x)
    seen = []
    for j, (i, pose) in enumerate(zip(selected, poses)):
        ids = np.unique(point_ids[agree & (frame_ids == j)])
        ids = ids[point_ok[ids]] if j else np.flatnonzero(point_ok)
        if len(ids) < 10 or (j and not 0.2 < pose[2, 3] < 4):
            continue
        out["frames"][i] = {"status": "observed", "inliers": len(ids), "part_to_world": pose}
        seen.append(i)
    for i, j in itertools.pairwise(seen):
        if j - i - 1 > 3:
            continue
        before, after = (out["frames"][f]["part_to_world"] for f in (i, j))
        d = Rotation.from_matrix(before[:3, :3].T @ after[:3, :3]).as_rotvec()
        for f in range(i + 1, j):
            t = (times[f] - times[i]) / (times[j] - times[i])
            pose = np.eye(4)
            pose[:3, :3] = before[:3, :3] @ Rotation.from_rotvec(t * d).as_matrix()
            pose[:3, 3] = (1 - t) * before[:3, 3] + t * after[:3, 3]
            out["frames"][f] = {"status": "interpolated", "inliers": 0, "part_to_world": pose}
    return out


def reconstruct_rgb_parts(frames, intrinsics, parts, times=None):
    """Return sparse tracked landmarks and part poses; the camera must stay fixed."""
    k = np.asarray(intrinsics, dtype=float)
    times = np.arange(len(frames)) if times is None else np.asarray(times)
    grays = [cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for rgb in frames]
    features = _features(grays)
    return [_reconstruct(grays, frames, features, k, part, times) for part in parts]
