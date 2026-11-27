"""Recover rotations of partly visible textured patches by direct alignment to remembered views."""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _warp(pixels, homography):
    return cv2.perspectiveTransform(pixels.astype(np.float32)[:, None], homography)[:, 0]


def _sample(image, pixels):
    """Continuous bilinear RGB sampling for subpixel photometric optimization."""
    height, width = image.shape[:2]
    u, v = np.clip(pixels, [0, 0], [width - 1.001, height - 1.001]).T
    x, y = u.astype(int), v.astype(int)
    a, b = (u - x)[:, None], (v - y)[:, None]
    upper = (1 - a) * image[y, x] + a * image[y, x + 1]
    lower = (1 - a) * image[y + 1, x] + a * image[y + 1, x + 1]
    return (1 - b) * upper + b * lower


def _inside(mask, pixels):
    h, w = mask.shape
    x, y = np.rint(pixels).astype(int).T
    in_view = (x >= 0) & (y >= 0) & (x < w) & (y < h)
    return in_view & (mask[y.clip(0, h - 1), x.clip(0, w - 1)] != 0)


def _projection(points, intrinsics, rotation, translation):
    camera = points @ rotation.T + translation
    return camera[:, :2] / np.maximum(.01, camera[:, 2:]) * intrinsics.diagonal()[:2] + intrinsics[:2, 2]


def _template(images, masks, views, known, normal, intrinsics):
    src = views[0]
    y, x = np.where(cv2.erode(masks[src], np.ones((3, 3), np.uint8)) != 0)
    if len(x) < 60:
        return None
    idx = np.linspace(0, len(x) - 1, min(len(x), 4000)).astype(int)
    pixels = np.column_stack((x[idx], y[idx])).astype(np.float32)
    colors = _sample(images[0], pixels)
    support = np.zeros(len(pixels), int)
    for i, homography in known.items():
        if i == 0:
            continue
        projected = _warp(pixels, homography)
        difference = np.mean(abs(_sample(images[i], projected) - colors), axis=1)
        support += _inside(masks[views[i]], projected) & (difference < 12)
    # Reject pixels on a different face or occluder that the plane homographies cannot explain.
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(intrinsics[src]).T
    denom = rays @ normal
    keep = (support >= max(2, int(np.ceil((len(known) - 1) / 4)))) & (denom > .02)
    if np.count_nonzero(keep) < 60:
        return None
    return rays[keep] / denom[keep, None], pixels[keep], colors[keep]


def _align(image, mask, intrinsics, points, colors, training, rotation, expected_pixels, prior):
    """Refine a temporal seed against actual target RGB, using separate spatial validation tiles."""
    rays = (expected_pixels - intrinsics[:2, 2]) / intrinsics.diagonal()[:2]
    rotated = rotation.apply(points)
    system = np.zeros((len(points), 2, 3))
    system[:, :, :2] = np.eye(2)
    system[:, :, 2] = -rays
    rhs = rays * rotated[:, 2:] - rotated[:, :2]
    shift = np.linalg.lstsq(system.reshape(-1, 3), rhs.ravel(), rcond=None)[0]
    initial = np.r_[rotation.as_rotvec(), shift]

    def project(parameters, xyz):
        return _projection(xyz, intrinsics, Rotation.from_rotvec(parameters[:3]).as_matrix(), parameters[3:])

    gradients = [cv2.Sobel(image, cv2.CV_32F, dx, dy, ksize=3, scale=1 / 8.)
                 for dx, dy in ((1, 0), (0, 1))]
    # Freeze validation membership before optimization. A bad pose must not improve
    # its score by discarding the withheld regions it moves off the object.
    all_initial = project(initial, points)
    initial_color_error = np.mean(abs(_sample(image, all_initial) - colors), axis=1)
    gradient = np.maximum(*[np.max(abs(_sample(g, all_initial)), axis=1) for g in gradients])
    validation = (_inside(mask, all_initial) & (rotated[:, 2] + shift[2] > .01)
                  & (initial_color_error < 28) & (gradient > 3) & ~training)
    if validation.sum() < 12:
        return None
    current = initial.copy()
    for iteration in range(5):
        uv = project(current, points)
        difference = np.mean(abs(_sample(image, uv) - colors), axis=1)
        gradient = np.maximum(*[np.max(abs(_sample(g, uv)), axis=1) for g in gradients])
        z = Rotation.from_rotvec(current[:3]).apply(points)[:, 2] + current[5]
        valid = _inside(mask, uv) & (z > .01) & (gradient > 3) & (difference < (28 if iteration == 0 else 20))
        selected = valid & training
        if selected.sum() < 32:
            return None
        samples, target_rgb = points[selected], colors[selected]

        def residual(parameters):
            return (_sample(image, project(parameters, samples)) - target_rgb).ravel()

        fit = least_squares(residual, current, loss='soft_l1', f_scale=5, max_nfev=35, diff_step=1e-4)
        if not np.isfinite(fit.x).all():
            return None
        current = fit.x

    initial_uv = all_initial[validation]
    test_points, test_colors = points[validation], colors[validation]
    final_uv = project(current, test_points)
    z = Rotation.from_rotvec(current[:3]).apply(test_points)[:, 2] + current[5]
    visible = (z > .01) & _inside(mask, final_uv)
    if (visible.mean() < .8 or np.median(np.linalg.norm(final_uv - initial_uv, axis=1)) > 15
            or np.linalg.eigvalsh(np.cov(final_uv[visible].T))[0] < 4):
        return None

    def test_errors(uv):
        return np.minimum(24, np.mean(abs(_sample(image, uv) - test_colors), axis=1))

    candidate_error = np.mean(np.where(visible, test_errors(final_uv), 24))
    seed_error = np.mean(test_errors(initial_uv))
    # A good initialization may already align the patch. In that case require
    # independent RGB evidence over the regular track-based pose before replacing it.
    prior_uv = _projection(test_points, intrinsics, prior[:3, :3], prior[:3, 3])
    prior_z = test_points @ prior[2, :3] + prior[2, 3]
    prior_visible = (prior_z > .01) & _inside(mask, prior_uv)
    prior_error = np.mean(np.where(prior_visible, test_errors(prior_uv), 24))
    improves_both = candidate_error < .98 * min(prior_error, seed_error)
    recovers_track = candidate_error < .85 * prior_error
    if candidate_error > 12 or candidate_error > 1.05 * seed_error or not (improves_both or recovers_track):
        return None
    return Rotation.from_rotvec(current[:3]).as_matrix(), current[3:]



def recover_surface_poses(frames, masks, intrinsics, views, warps, rotations, normal, depths, poses, times):
    """Use reference views from both sides of a gap in planar features, for observed frames only."""
    known = {0: np.eye(3), **{i: warps[i] for i in rotations}}
    # No extrapolation or matching in frames already deemed fully hidden.
    targets = [i for i in range(1, max(known)) if i not in known]
    if not targets:
        return {}
    images = {i: cv2.GaussianBlur(frames[views[i]], (3, 3), .65).astype(np.float32)
              for i in (*known, *targets)}
    template = _template(images, masks, views, known, normal, intrinsics)
    if template is None:
        return {}
    points, pixels, colors = template
    x, y = pixels.astype(int).T
    # The template plane is normal.X=1. Estimate its physical distance using
    # robust reference depths on those same consistently observed pixels.
    ratios = depths[views[0]][y, x] / points[:, 2]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) < 30:
        return {}
    distance = float(np.median(ratios))
    source_pose = poses[views[0]]
    # Exclude spatial blocks from every solve and score the result on those blocks.
    training = ((pixels[:, 0].astype(int) // 7 + pixels[:, 1].astype(int) // 7) % 3) != 1
    result = {}
    for i in targets:
        a = max(n for n in known if n < i)
        b = min(n for n in known if n > i)
        t, ta, tb = (times[views[j]] for j in (i, a, b))
        fraction = (t - ta) / (tb - ta)
        first = Rotation.from_matrix(rotations.get(a, np.eye(3)))
        relative = first.inv() * Rotation.from_matrix(rotations[b])
        seed = first * Rotation.from_rotvec(fraction * relative.as_rotvec())
        projected = (1 - fraction) * _warp(pixels, known[a]) + fraction * _warp(pixels, known[b])
        prior = poses[views[i]] @ np.linalg.inv(source_pose)
        prior[:3, 3] /= distance
        observed = _align(images[i], masks[views[i]], intrinsics[views[i]], points, colors, training,
                          seed, projected, prior)
        if observed is not None:
            relative = np.eye(4)
            relative[:3, :3] = observed[0]
            relative[:3, 3] = observed[1] * distance
            result[i] = relative @ source_pose
    return result
