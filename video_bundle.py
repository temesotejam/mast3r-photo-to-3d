"""Refine each rigid part using shared feature tracks across the observed video."""
from itertools import product

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def _project(xyz, K):
    z = np.maximum(xyz[:, 2], 1e-6)
    return xyz[:, :2] / z[:, None] * [K[0, 0], K[1, 1]] + K[:2, 2]


def _inside(uv, mask):
    xy = np.rint(uv).astype(int)
    h, w = mask.shape
    return ((xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            & (mask[xy[:, 1].clip(0, h-1), xy[:, 0].clip(0, w-1)] != 0))


def _landmark_tracks(grays, views, depths, intrinsics, masks, poses):
    """Match subpixel corners directly to multiple views; a failed match never propagates."""
    tracks = []
    flow = dict(winSize=(17, 17), maxLevel=2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    # Anchors spanning the clip provide landmarks for surfaces revealed later.
    anchor_indices = list(range(0, len(views), 3))
    if anchor_indices[-1] != len(views) - 1:
        anchor_indices.append(len(views) - 1)
    for anchor in anchor_indices:
        src = views[anchor]
        mask = cv2.erode(masks[src], np.ones((5, 5), np.uint8))
        corners = cv2.goodFeaturesToTrack(grays[src], 75, 0.012, 5, mask=mask)
        if corners is None:
            continue
        uv = corners.reshape(-1, 2)
        x, y = np.rint(uv).astype(int).T
        z = depths[src][y, x]
        valid = np.isfinite(z) & (z > 0)
        uv, z = uv[valid], z[valid]
        if not len(uv):
            continue
        k = intrinsics[src]
        xy = (uv - k[:2, 2]) / [k[0, 0], k[1, 1]]
        xyz = np.column_stack((xy * z[:, None], z))
        ref = (xyz - poses[src][:3, 3]) @ poses[src][:3, :3]
        observations = [{anchor: p} for p in uv]
        for other, dst in enumerate(views):
            if other == anchor:
                continue
            pose = poses[dst]
            cam = ref @ pose[:3, :3].T + pose[:3, 3]
            seed = _project(cam, intrinsics[dst]).astype(np.float32)
            # Avoid feeding non-projectable or unbounded seeds to optical flow.
            seed = np.clip(seed, [-grays[dst].shape[1], -grays[dst].shape[0]],
                           [2*grays[dst].shape[1], 2*grays[dst].shape[0]]).astype(np.float32)
            found, ok, err = cv2.calcOpticalFlowPyrLK(grays[src], grays[dst], uv, seed, **flow)
            if found is None:
                continue
            back, back_ok, _ = cv2.calcOpticalFlowPyrLK(grays[dst], grays[src], found, uv.copy(), **flow)
            if back is None:
                continue
            keep = ((ok.ravel() != 0) & (back_ok.ravel() != 0)
                    & (np.linalg.norm(back-uv, axis=1) < 0.85)
                    & (err.ravel() < 22) & _inside(found, masks[dst]) & (cam[:, 2] > 0))
            for j in np.flatnonzero(keep):
                observations[j][other] = found[j]
        for xyz, observations_for_point in zip(ref, observations):
            if len(observations_for_point) >= 3:
                tracks.append((xyz, observations_for_point))
    return tracks


def _surface_rotations(grays, masks, ks, views, features, allow_focal_refit):
    """Recover metric rotations of a textured plane without using predicted surface slopes."""
    from video_calibration import plane_warps, _fit_plane

    if len(views) < 5:
        return {}
    source = views[0]
    warps = plane_warps([grays[v] for v in views], masks[source],
                         features=[features[v] for v in views])
    # Discard patches which now map mainly onto another moving part.
    template_corners = cv2.goodFeaturesToTrack(grays[source], 180, .008, 4, mask=masks[source])
    if template_corners is None:
        return {}
    warps = {i: H for i, H in warps.items()
             if np.mean(_inside(cv2.perspectiveTransform(template_corners, H).reshape(-1, 2),
                                masks[views[i]])) >= .2}
    if len(warps) < 4:
        return {}
    ids = np.array(list(warps))
    homographies = np.array(list(warps.values()))
    rays = np.linalg.inv(ks[np.array(views)[ids]]) @ homographies @ ks[source]

    def tangent(n):
        return np.linalg.qr(np.array([[1., 0.], [0., 1.], [-n[0], -n[1]]]))[0]

    def error(n, rays):
        warped = rays @ tangent(n)
        gram = warped.swapaxes(1, 2) @ warped
        trace = gram[:, 0, 0] + gram[:, 1, 1]
        return np.column_stack(((gram[:, 0, 0] - gram[:, 1, 1])/trace,
                                2*gram[:, 0, 1]/trace)).ravel()

    def estimate(rays):
        # Explore tilt signs on both axes to avoid the reflected-plane local minimum.
        starts = product((-1., 0., 1.), repeat=2)
        candidates = [least_squares(error, n, args=(rays,), loss='soft_l1', f_scale=.006,
                                    bounds=(-3., 3.)) for n in starts]
        return min(candidates, key=lambda s: s.cost)

    fit = estimate(rays)
    if allow_focal_refit and np.sqrt(np.mean(error(fit.x, rays)**2)) > .025 and len(warps) >= 7:
        # A model focal can be poorly constrained in a stationary camera. Use
        # an independently identifiable planar calibration for rotation only.
        h, w = grays[source].shape
        center = np.array([[1/w, 0., -(w-1)/(2*w)], [0., 1/w, -(h-1)/(2*w)], [0., 0., 1.]])
        inferred = _fit_plane(center @ homographies @ np.linalg.inv(center))
        if inferred is not None:
            f = inferred[0] * w
            k = np.array([[f, 0., (w-1)/2], [0., f, (h-1)/2], [0., 0., 1.]])
            rays = np.linalg.inv(k) @ homographies @ k
            fit = estimate(rays)
    errors = np.linalg.norm(error(fit.x, rays).reshape(-1, 2), axis=1)
    if (np.median(errors) > .025 or np.max(np.abs(fit.x)) > 2.95
            or np.linalg.svd(fit.jac, compute_uv=False)[-1] < .06):
        return {}
    basis = tangent(fit.x)
    source_basis = np.column_stack((basis, np.cross(*basis.T)))
    rotations = {}
    for i, ray, residual in zip(ids, rays, errors):
        if residual > .045:
            continue
        # On a tangent plane H is a scaled rotation. Its polar factor gives an
        # orthonormal image of the two axes; their cross product gives the third.
        u, _, vt = np.linalg.svd(ray @ basis, full_matrices=False)
        warped = u @ vt
        rotations[i] = np.column_stack((warped, np.cross(*warped.T))) @ source_basis.T
    return rotations


def _refine_window(grays, depths, intrinsics, poses, masks, scale, views, features, allow_focal_refit):
    if len(views) < 3:
        return {}
    tracks = _landmark_tracks(grays, views, depths, intrinsics, masks, poses)
    if len(tracks) < 12:
        return {}

    # Only change views whose image measurements connect to the first view.
    connected = {0}
    while True:
        next_views = set(connected)
        for _, track in tracks:
            if connected.intersection(track):
                next_views.update(track)
        if next_views == connected:
            break
        connected = next_views
    tracks = [(xyz, track) for xyz, track in tracks if connected.intersection(track)]
    if len(tracks) < 12:
        return {}
    views_used = sorted(connected)
    remap = {i: j for j, i in enumerate(views_used)}
    n_views = len(views_used)
    used_poses = np.array([poses[views[i]] for i in views_used])
    Ks = intrinsics[[views[i] for i in views_used]]
    f_idx, p_idx, pixels, measured_z = [], [], [], []
    for p, (_, obs) in enumerate(tracks):
        for k, uv in obs.items():
            x, y = np.rint(uv).astype(int)
            z = depths[views[k]][y, x]
            f_idx.append(remap[k])
            p_idx.append(p)
            pixels.append(uv)
            measured_z.append(z / scale)
    fi, pi = np.array(f_idx), np.array(p_idx)
    pixels, measured_z = np.array(pixels), np.array(measured_z)
    valid_z = np.isfinite(measured_z) & (measured_z > 0)
    measured_z = np.where(valid_z, measured_z, 0)
    # Monocular depth is a soft scale/shape prior. Pixel measurements are the
    # stronger constraint when depths disagree between views of the same point.
    sigma_z = np.maximum(0.02 * measured_z, 0.018)
    focal = Ks[fi][:, (0, 1), (0, 1)]
    principal = Ks[fi, :2, 2]
    initial_pose = np.column_stack((Rotation.from_matrix(used_poses[:, :3, :3]).as_rotvec(),
                                    used_poses[:, :3, 3] / scale))
    planar = _surface_rotations(grays, masks, intrinsics, views, features, allow_focal_refit)
    rotations0 = used_poses[:, :3, :3].copy()
    free = []
    for j, i in enumerate(views_used[1:], start=1):
        if i in planar:
            rotations0[j] = planar[i] @ rotations0[0]
        else:
            free.append(j)
    free = np.array(free, dtype=int)
    num_rot_params = 3 * len(free)
    num_pose_params = num_rot_params + 3 * (n_views - 1)
    guess = np.r_[initial_pose[free, :3].ravel(), initial_pose[1:, 3:].ravel(),
                  (np.array([xyz for xyz, _ in tracks]) / scale).ravel()]

    def unpack(x):
        rotations = rotations0.copy()
        if len(free):
            rotations[free] = Rotation.from_rotvec(x[:num_rot_params].reshape(-1, 3)).as_matrix()
        centers = np.vstack((initial_pose[0, 3:], x[num_rot_params:num_pose_params].reshape(-1, 3)))
        return rotations, centers, x[num_pose_params:].reshape(-1, 3)

    def residual(x):
        rotations, centers, points = unpack(x)
        cam = np.einsum('nij,nj->ni', rotations[fi], points[pi]) + centers[fi]
        xy = cam[:, :2] / np.maximum(cam[:, 2:], 1e-5) * focal + principal
        depth_error = (cam[:, 2] - measured_z) / sigma_z
        return np.column_stack((xy - pixels, np.where(valid_z, depth_error, 0))).ravel()

    # Each observation depends only on one camera pose and one shared landmark.
    r = np.arange(len(fi)) * 3
    sparse = lil_matrix((3 * len(fi), len(guess)), dtype=np.int8)
    movable = fi != 0
    rotation_index = np.full(n_views, -1)
    rotation_index[free] = np.arange(len(free))
    free_rows = rotation_index[fi] >= 0
    for axis in range(3):
        for col in range(3):
            sparse[r[free_rows] + axis, rotation_index[fi[free_rows]] * 3 + col] = 1
            sparse[r[movable] + axis, num_rot_params + (fi[movable] - 1) * 3 + col] = 1
        for col in range(3):
            sparse[r + axis, num_pose_params + pi * 3 + col] = 1
    fit = least_squares(residual, guess, jac_sparsity=sparse.tocsr(), x_scale='jac',
                        loss='soft_l1', f_scale=1.5, max_nfev=65, ftol=0.0005, xtol=0.0001)
    if not np.isfinite(fit.x).all():
        return {}
    rotations, centers, _ = unpack(fit.x)
    before = np.linalg.norm(residual(guess).reshape(-1, 3)[:, :2], axis=1)
    after = np.linalg.norm(residual(fit.x).reshape(-1, 3)[:, :2], axis=1)
    result = {}
    for j in range(1, n_views):
        chosen = fi == j
        good = chosen & (after < 3)
        if (good.sum() < max(10, 0.55 * chosen.sum())
                or cv2.contourArea(cv2.convexHull(pixels[good].astype(np.float32))) < 25
                or np.median(after[chosen]) > np.median(before[chosen]) + 0.6
                or abs(centers[j, 2] - initial_pose[j, 5]) > 0.25 * initial_pose[j, 5]):
            continue
        pose = np.eye(4)
        pose[:3, :3] = rotations[j]
        pose[:3, 3] = centers[j] * scale
        result[views[views_used[j]]] = pose
    return result


def refine_part_poses(grays, depths, intrinsics, rows, masks, scale, features, allow_focal_refit):
    """Refine connected observations in bounded windows. Unobserved poses are never fitted."""
    views = sorted(i for i, m in masks.items() if m.any() and rows[i]["status"] == "observed")
    poses = {i: rows[i]["part_to_camera"] for i in views}
    updates = {}
    # A shared anchor links neighbouring windows. Restricting each solve to 24
    # views bounds both optical-flow matching and the sparse landmark system.
    if len(views) < 3:
        return {}
    n_windows = int(np.ceil((len(views) - 1) / 23))
    boundaries = np.linspace(0, len(views)-1, n_windows+1).round().astype(int)
    for start, end in zip(boundaries, boundaries[1:]):
        window = views[start:end+1]
        refined = _refine_window(grays, depths, intrinsics, poses, masks, scale, window, features,
                                 allow_focal_refit)
        poses.update(refined)
        updates.update(refined)
    return updates
