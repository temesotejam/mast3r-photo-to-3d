"""Optional focal self-calibration from rigid, textured planar patches in a video.

A plane need not have a known shape. Its Euclidean tangent basis is shared
by all views. In calibrated rays, a plane homography equals R + t n^T/d
up to scale. Its restriction to that tangent plane must thus be a similarity.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

from video_parts import _box, _segment


def _plane_residuals(params, homographies):
    focal, nx, ny = params
    # Unitless image coordinates have principal point zero and width one.
    k = np.diag([focal, focal, 1.])
    rays = np.diag([1 / focal, 1 / focal, 1.]) @ homographies @ k
    tangent = np.linalg.qr(np.array([[1., 0.], [0., 1.], [-nx, -ny]]))[0]
    warped = rays @ tangent
    gram = warped.swapaxes(1, 2) @ warped
    trace = gram[:, 0, 0] + gram[:, 1, 1]
    return np.column_stack(((gram[:, 0, 0] - gram[:, 1, 1]) / trace,
                            2 * gram[:, 0, 1] / trace)).ravel()


def _fit_plane(hs):
    """Return a well-conditioned focal fit, or abstain on critical motions."""
    if len(hs) < 7:
        return None
    bounds = ([.25, -2., -2.], [4., 2., 2.])
    def solve(views, start):
        return least_squares(_plane_residuals, start, args=(views,), bounds=bounds,
                             loss="soft_l1", f_scale=.025, max_nfev=60)

    candidates = [solve(hs, (f, a, b)) for f in (.5, 1., 2.)
                  for a, b in ((0., 0.), (-.5, .5), (.5, -.5))]
    result = min(candidates, key=lambda s: s.cost)
    f, a, b = result.x
    error = np.sqrt(2 * result.cost / len(hs))
    if not .28 < f < 3.8 or max(abs(a), abs(b)) > 1.9 or error > .025:
        return None

    # Reject optical-axis spins, translations and near-repeats. Even at a
    # substantially different focal length a plane with such motion fits well.
    for tested in (f / 1.5, f * 1.5):
        fixed = least_squares(lambda n: _plane_residuals((tested, *n), hs),
                              [a, b], bounds=(-2., 2.), loss="soft_l1", f_scale=.025)
        alternative = np.sqrt(2 * fixed.cost / len(hs))
        if alternative < max(1.6 * error, error + .01):
            return None

    # Avoid estimates determined by a single perspective view or bad match.
    for fold in range(3):
        subset = hs[np.arange(len(hs)) % 3 != fold]
        partial = solve(subset, result.x).x[0]
        if abs(np.log(partial / f)) > np.log(1.18):
            return None
    return f, error


def _refine_homography(reference, target, corners, homography):
    # Bring the two planes into approximate alignment before subpixel flow.
    # The small residual search accommodates large rotations in the video.
    h, w = reference.shape
    aligned = cv2.warpPerspective(target, homography, (w, h),
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    refined, ok, error = cv2.calcOpticalFlowPyrLK(reference, aligned, corners, None,
                                                 winSize=(13, 13), maxLevel=1)
    if refined is None:
        return homography
    good = (ok.ravel() > 0) & (error.ravel() < 12) & (np.linalg.norm(refined - corners, axis=1) < 2.5)
    if good.sum() < 16:
        return homography
    destination = cv2.perspectiveTransform(refined[good, None], homography)
    refined_h, inliers = cv2.findHomography(corners[good], destination, cv2.USAC_MAGSAC, .65)
    if refined_h is None or inliers.sum() < max(16, .5 * len(destination)):
        return homography
    return refined_h


def _warps(indices, features, grays, mask, min_displacement):
    xy, desc = features[0]
    matcher = cv2.BFMatcher()
    groups = {}
    corners = cv2.goodFeaturesToTrack(grays[0], maxCorners=250, qualityLevel=.005, minDistance=4, mask=mask)
    for t, (target, target_desc) in enumerate(features[1:], start=1):
        if target_desc is None or len(target_desc) < 2:
            continue
        matches = [m for m, n in matcher.knnMatch(desc[indices], target_desc, k=2)
                   if m.distance < .82 * n.distance]
        if len(matches) < 15:
            continue
        src = xy[[indices[m.queryIdx] for m in matches]]
        dst = target[[m.trainIdx for m in matches]]
        _, inliers = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 1.0)
        if inliers is None or inliers.sum() < max(15, .35 * len(matches)):
            continue
        good = inliers.ravel().astype(bool)
        src, dst = src[good], dst[good]
        # Ignore static background leaks and homographies from collinear points.
        if np.median(np.linalg.norm(src - dst, axis=1)) < min_displacement:
            continue
        if np.linalg.eigvalsh(np.cov(src.T))[0] < 36:
            continue
        homography, _ = cv2.findHomography(src, dst, 0)
        if homography is None:
            continue
        if corners is not None:
            homography = _refine_homography(grays[0], grays[t], corners.reshape(-1, 2), homography)
        if np.isfinite(homography).all() and abs(homography[2, 2]) > 1.e-6:
            groups[t] = homography / homography[2, 2]
    return groups


def _features(grays):
    detector = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.015)
    result = []
    for gray in grays:
        keypoints, desc = detector.detectAndCompute(gray, None)
        result.append((np.float32([k.pt for k in keypoints]).reshape(-1, 2), desc))
    return result


def plane_warps(grays, mask, *, min_displacement=0., features=None):
    """Return supported template-to-frame homographies in input pixels."""
    if len(grays) < 2 or not np.any(mask):
        return {}
    features = _features(grays) if features is None else features
    xy, desc = features[0]
    if desc is None:
        return {}
    h, w = grays[0].shape
    uv = np.rint(xy).astype(int).clip([0, 0], [w - 1, h - 1])
    mask = cv2.erode(np.asarray(mask, dtype=np.uint8), np.ones((5, 5), np.uint8))
    indices = np.flatnonzero(mask[uv[:, 1], uv[:, 0]])
    if len(indices) < 15:
        return {}
    return _warps(indices, features, grays, mask, min_displacement)


def estimate_planar_focal(frames, boxes):
    """Estimate fx=fy in pixel units or return None; frames are uint8 RGB."""
    if len(frames) < 8 or not boxes:
        return None
    h, w = frames[0].shape[:2]
    center = np.array([[1 / w, 0., -(w - 1) / (2 * w)],
                       [0., 1 / w, -(h - 1) / (2 * w)], [0., 0., 1.]])
    sample = np.rint(np.linspace(0, len(frames) - 1, min(20, len(frames)))).astype(int)
    grays = [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in sample]
    features = _features(grays)
    if features[0][1] is None:
        return None
    estimates = []
    for box in boxes:
        rect = _box(box, w, h)
        x, y, r, b = rect
        seed = np.zeros((h, w), np.uint8)
        seed[y:b, x:r] = 1
        try:
            mask = _segment(frames[0], None, seed, rect, first=True)
        except cv2.error:  # GrabCut may have too few distinct pixels for its colour models.
            continue
        warps = plane_warps(grays, mask, min_displacement=5., features=features)
        if len(warps) < 7:
            continue
        hs = center @ np.asarray(list(warps.values())) @ np.linalg.inv(center)
        fit = _fit_plane(hs)
        if fit is not None:
            estimates.append(fit[0])
    if not estimates or max(estimates) / min(estimates) > 1.25:
        print("Focal calibration: no consistent, well-conditioned planar tracks; using model intrinsics.", flush=True)
        return None
    focal = float(np.median(estimates) * w)
    print(f"Focal calibration: {focal:.1f} px from {len(estimates)} independently tracked planar region(s).", flush=True)
    return focal
