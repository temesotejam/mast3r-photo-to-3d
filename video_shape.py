"""Check a fused rigid surface against independent observed video frames."""

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree


def surface_consensus(points, depths, intrinsics, rows, masks, scale, occupied=None):
    """Keep surfels unless multiple segmented views place them in empty space.

    A point behind a measured surface can belong to an occluded/newly exposed
    face. Such a view abstains, as do pixels near an uncertain silhouette or
    depth edge. Only use observations with actual segmented foreground.
    """
    points = np.asarray(points, dtype=np.float32)
    keep = np.ones(len(points), dtype=bool)
    observed = [i for i, row in enumerate(rows) if row['status'] == 'observed'
                and masks.get(i) is not None and np.any(masks[i])]
    if len(observed) < 3 or not len(points):
        return keep
    # Dense video can contain hundreds of nearly identical observations.
    # Cover the whole sequence without allowing one long stationary interval
    # to increase the per-point working set without bound.
    if len(observed) > 32:
        observed = [observed[i] for i in np.linspace(0, len(observed) - 1, 32).round().astype(int)]
    ks = np.broadcast_to(intrinsics, (len(rows), 3, 3))
    support = np.zeros(len(points), dtype=np.int16)
    conflicts = np.zeros(len(points), dtype=np.int16)
    for i in observed:
        pose = rows[i]['part_to_camera']
        in_camera = points @ np.asarray(pose[:3, :3], dtype=np.float32).T + np.asarray(pose[:3, 3], dtype=np.float32)
        projection = in_camera @ np.asarray(ks[i], dtype=np.float32).T
        h, w = depths[i].shape
        z = in_camera[:, 2]
        in_front = np.isfinite(projection).all(axis=1) & (z > 0)
        # Bound in homogeneous coordinates before rounding or casting.
        in_image = (in_front & (projection[:, 0] >= 0) & (projection[:, 0] <= (w - 1) * z)
                    & (projection[:, 1] >= 0) & (projection[:, 1] <= (h - 1) * z))
        ids = np.flatnonzero(in_image)
        if not len(ids):
            continue
        x, y = np.rint(projection[ids, :2] / z[ids, None]).astype(int).T
        depth = np.asarray(depths[i], dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        mask = np.asarray(masks[i], dtype=bool)
        # Leave room for registration, segmentation and model depth errors;
        # require consensus across a small pixel neighbourhood before carving.
        near = minimum_filter(np.where(valid, depth, np.inf), size=5)[y, x]
        far = maximum_filter(np.where(valid, depth, -np.inf), size=5)[y, x]
        inside = mask[y, x]
        outside = ~maximum_filter(mask, size=7)[y, x]
        visible = np.isfinite(near) & np.isfinite(far) & valid[y, x]
        tolerance = np.maximum(0.035 * scale, 0.012 * z[ids])
        comparable = visible & (z[ids] >= near - tolerance) & (z[ids] <= far + tolerance)
        support[ids] += inside & comparable
        in_empty_space = z[ids] < near - 1.5 * tolerance
        # At a depth discontinuity the nearest surface may occlude this ray.
        # Do not label points behind it as lying outside the target part.
        outside_and_clear = outside & (z[ids] <= near + tolerance)
        # Near contact can put two links within the depth model's uncertainty.
        # A known competing silhouette then explains a missing target mask.
        if occupied is not None and i in occupied:
            outside_and_clear &= ~np.asarray(occupied[i], dtype=bool)[y, x]
        conflicts[ids] += visible & (in_empty_space | outside_and_clear)
    return ~((conflicts >= 3) & (conflicts > 2 * support))


def _triangulated_support(gray, other_gray, uv, rays, z, transform, k, mask, scale):
    """Intersect each anchor ray with one independently tracked support ray."""
    direction = rays @ transform[:3, :3].T
    shift = transform[:3, 3]
    xyz = direction * z[:, None] + shift
    height, width = mask.shape
    focal, principal = k.diagonal()[:2], k[:2, 2]
    seed = xyz[:, :2] / np.maximum(xyz[:, 2:], 1e-6) * focal + principal
    seed = seed.clip([-width, -height], [2 * width, 2 * height]).astype(np.float32)
    options = dict(winSize=(17, 17), maxLevel=2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    target, ok, error = cv2.calcOpticalFlowPyrLK(gray, other_gray, uv, seed, **options)
    if target is None:
        return None
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(other_gray, gray, target, uv.copy(), **options)
    if back is None:
        return None
    x, y = np.rint(target).astype(int).T
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    good = (inside & (ok.ravel() > 0) & (ok_back.ravel() > 0) & (error.ravel() < 19)
            & (np.linalg.norm(back - uv, axis=1) < .8) & (xyz[:, 2] > 0)
            & mask[y.clip(0, height-1), x.clip(0, width-1)].astype(bool))

    # For candidate axial depth d, (a*d - b)/Z is the pixel reprojection residual.
    xy = (target - principal) / focal
    a = (direction[:, :2] - xy * direction[:, 2:]) * focal
    b = (xy * shift[2] - shift[:2]) * focal
    norm = np.sum(a*a, axis=1)
    depth = np.sum(a*b, axis=1) / np.maximum(norm, 1e-12)
    prior_error = np.linalg.norm(a*z[:, None] - b, axis=1) / np.maximum(xyz[:, 2], 1e-6)
    fit_z = direction[:, 2] * depth + shift[2]
    fit_error = np.linalg.norm(a*depth[:, None] - b, axis=1) / np.maximum(fit_z, 1e-6)
    # A 1-pixel match shift must constrain depth better than the part size.
    good &= ((norm > (xyz[:, 2] / (.14*scale))**2) & (depth > 0) & (fit_z > 0)
             & (np.abs(z-depth) < max(.65*scale, .04*np.median(z))) & (fit_error < 1.5))
    return depth, good, a, b, direction[:, 2], shift[2], prior_error


def _refine_depth(i, grays, depths, ks, masks, poses, scale, views):
    mask = masks[i]
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8))
    corners = cv2.goodFeaturesToTrack(grays[i], 300, .008, 4, mask=interior)
    if corners is None or len(corners) < 6:
        return None
    uv = corners[:, 0]
    x, y = np.rint(uv).astype(int).T
    z = depths[i][y, x]
    finite = np.isfinite(z) & (z > 0)
    uv, z = uv[finite], z[finite]
    if len(z) < 6:
        return None
    k = ks[i]
    rays = np.column_stack(((uv-k[:2, 2])/k.diagonal()[:2], np.ones(len(uv))))
    inverse_pose = np.linalg.inv(poses[i])
    others = [j for j in views if j != i]
    if len(others) > 10:
        others = [others[j] for j in np.linspace(0, len(others)-1, 10).round().astype(int)]
    fits = []
    for j in others:
        fit = _triangulated_support(grays[i], grays[j], uv, rays, z, poses[j] @ inverse_pose,
                                    ks[j], masks[j], scale)
        if fit is not None:
            fits.append(fit)
    if len(fits) < 2:
        return None
    estimates, valid = (np.stack([fit[field] for fit in fits]) for field in (0, 1))
    selected = valid.sum(axis=0) >= 2
    if selected.sum() < 5:
        return None
    # Exclude unsupported columns before masked medians to avoid all-NaN slices.
    valid = valid[:, selected]
    samples = np.where(valid, estimates[:, selected], np.nan)
    refined = np.nanmedian(samples, axis=0)
    dispersion = np.nanmedian(np.abs(samples-refined), axis=0)
    before, after = [], []
    for _, _, a, b, rz, tz, old_error in fits:
        target_z = rz[selected]*refined + tz
        residual = a[selected]*refined[:, None] - b[selected]
        after.append(np.linalg.norm(residual, axis=1) / np.maximum(target_z, 1e-6))
        before.append(old_error[selected])
    baseline = np.nanmedian(np.where(valid, before, np.nan), axis=0)
    updated = np.nanmedian(np.where(valid, after, np.nan), axis=0)
    consistent = (dispersion < .075*scale) & (updated <= baseline)
    if consistent.sum() < 5:
        return None
    anchors, prior = uv[selected][consistent], z[selected][consistent]
    delta = refined[consistent] - prior
    ys, xs = np.nonzero(mask)
    distance, ids = cKDTree(anchors).query(np.column_stack((xs, ys)), min(8, len(anchors)))
    # Interpolate a correction locally, within similar-depth surfaces. Leave
    # silhouettes, large untextured regions or isolated landmarks unsupported.
    measured = depths[i][ys, xs]
    weight = np.exp(-.5*(distance/14)**2 - ((measured[:, None]-prior[ids])/(.14*scale))**2)
    total = weight.sum(axis=1)
    reliable = total > 1.2
    if not reliable.any():
        return None
    correction = (weight[reliable]*delta[ids[reliable]]).sum(axis=1) / total[reliable]
    corrected = measured[reliable] + np.minimum(.9, total[reliable]/4) * correction
    positive = np.isfinite(corrected) & (corrected > 0)
    if not positive.any():
        return None
    result = depths[i].copy()
    result[ys[reliable][positive], xs[reliable][positive]] = corrected[positive]
    return result


def refine_surface_depths(grays, depths, intrinsics, rows, masks, scale):
    """Use verified subpixel parallax to refine learned depths only inside a tracked part."""
    views = sorted(i for i, mask in masks.items() if rows[i]["status"] == "observed" and mask.any())
    if len(views) < 3:
        return depths
    poses = {i: rows[i]["part_to_camera"] for i in views}
    result = depths
    for i in views:
        depth = _refine_depth(i, grays, depths, intrinsics, masks, poses, scale, views)
        if depth is not None:
            if result is depths:
                result = list(depths)
            result[i] = depth
    return result


def visible_color_consensus(points, colors, frames, depths, intrinsics, rows, masks, scale, occupied=None):
    """Prune fused points repeatedly contradicted by the RGB views that can see them.

    Inspired by voxel coloring (Seitz & Dyer). A foreground depth buffer prevents
    the hidden side of a part from being penalized for the front surface's color.
    Geometry, lighting and segmentation uncertainty leave a neutral voting band.
    """
    views = [i for i in sorted(masks) if rows[i]["status"] == "observed" and masks[i].any()]
    if not len(points) or len(views) < 3:
        return np.ones(len(points), dtype=bool)
    if len(views) > 32:
        views = [views[j] for j in np.linspace(0, len(views) - 1, 32).round().astype(int)]
    ks = np.broadcast_to(intrinsics, (len(rows), 3, 3))
    support = np.zeros(len(points), np.int16)
    conflict = np.zeros_like(support)
    colors = np.asarray(colors, dtype=np.float32)
    margin = np.ones((5, 5), np.uint8)
    for i in views:
        pose = rows[i]["part_to_camera"]
        k = ks[i]
        camera = points @ pose[:3, :3].T + pose[:3, 3]
        z = camera[:, 2]
        uv = camera[:, :2] / np.maximum(z[:, None], 1e-7) * k.diagonal()[:2] + k[:2, 2]
        h, w = masks[i].shape
        ids = np.flatnonzero((z > 0) & (uv[:, 0] >= 2) & (uv[:, 0] <= w - 3)
                             & (uv[:, 1] >= 2) & (uv[:, 1] <= h - 3))
        if not len(ids):
            continue
        x, y = np.rint(uv[ids]).astype(np.intp).T
        z = z[ids]
        front = np.full((h, w), np.inf, np.float32)
        np.minimum.at(front, (y, x), z)
        nearest = minimum_filter(front, size=3)[y, x]
        measured = np.asarray(depths[i])
        far = maximum_filter(np.where(np.isfinite(measured) & (measured > 0), measured, -np.inf), size=5)
        visible = (z <= nearest + 0.04 * scale) & (z <= far[y, x] + 0.08 * scale)
        inside = masks[i][y, x] != 0
        possible = masks[i] if occupied is None or i not in occupied else occupied[i]
        # Two competing foreground masks cannot identify which color is visible.
        if occupied is not None and i in occupied:
            visible &= possible[y, x] <= 1
        outside = cv2.dilate(possible.astype(np.uint8), margin)[y, x] == 0

        distance = np.full(len(ids), np.inf, np.float32)
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                rgb_error = frames[i][y + dy, x + dx].astype(np.float32) - colors[ids]
                # Tolerate mild exposure changes and two pixels of reprojection uncertainty.
                rgb_error -= 0.5 * rgb_error.mean(axis=1, keepdims=True)
                distance = np.minimum(distance, (rgb_error**2).mean(axis=1))
        support[ids] += visible & inside & (distance < 24**2)
        conflict[ids] += visible & (outside | (inside & (distance > 36**2)))
    return ~((conflict >= 3) & (conflict > 1 + 1.5 * support))
