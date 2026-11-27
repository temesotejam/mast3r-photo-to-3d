"""Track one rigid object's measured first-frame surface in a fixed-camera RGB video."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from reconstruct_video import SAMPLING, load_clip


def read_reference(args, original_size, frame_size):
    """Read axial depths along the optical axis and calibrated intrinsics in original pixels."""
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if args.depth.suffix.lower() == ".npy":
        depth = np.load(args.depth, allow_pickle=False)
        scale = 1. if args.depth_scale is None else args.depth_scale
    elif args.depth.suffix.lower() == ".png":
        if args.depth_scale is None:
            raise ValueError("For PNG depth, specify --depth-scale in metres per stored unit.")
        depth = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
        scale = args.depth_scale
    else:
        raise ValueError("Choose an aligned depth PNG or a numeric .npy depth array.")
    if (mask is None or depth is None or mask.shape != original_size[::-1]
            or depth.shape != mask.shape or depth.dtype.kind not in "uif"):
        raise ValueError("Mask and numeric depth must match the native video resolution.")
    data = json.loads(args.intrinsics.read_text())
    k = np.asarray(data.get("K") if isinstance(data, dict) else data, dtype=float)
    if k.size != 9:
        raise ValueError("Camera intrinsics must contain a 3×3 pinhole K matrix.")
    k = k.reshape(3, 3).copy()
    if (not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[[0, 1], [1, 0]], 0) or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError("Supply the calibrated pinhole matrix of the undistorted input.")
    zoom = np.asarray(frame_size, dtype=float) / original_size
    k[[0, 1], [0, 1]] *= zoom
    k[:2, 2] = (k[:2, 2] + .5) * zoom - .5
    depth = depth.astype(np.float64) * scale
    measured = (mask != 0) & np.isfinite(depth) & (depth > 0)
    # Leave out boundary pixels where a depth sensor may see the background instead.
    interior = cv2.erode(measured.astype(np.uint8), np.ones((5, 5), np.uint8))
    interior = cv2.resize(interior, frame_size, interpolation=cv2.INTER_NEAREST)
    x, y = [np.rint((np.arange(n) + .5) / z - .5).astype(int) for n, z in zip(frame_size, zoom)]
    depth = depth[np.ix_(y, x)]
    interior &= measured[np.ix_(y, x)].astype(np.uint8)
    return depth, interior, k


def unproject(pixels, depth, k):
    x, y = np.rint(pixels).astype(int).T
    rays = np.column_stack(((pixels - k[:2, 2]) / k.diagonal()[:2], np.ones(len(pixels))))
    return rays * depth[y, x, None]


def estimate_pose(landmarks, pixels, k):
    ok, rv, translation, inliers = cv2.solvePnPRansac(
        landmarks, pixels, k, None, iterationsCount=600, reprojectionError=2.25,
        confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < max(6, .35 * len(landmarks)):
        return None, 0
    ids = inliers.ravel()
    if cv2.contourArea(cv2.convexHull(pixels[ids])) < 12:
        return None, 0
    rv, translation = cv2.solvePnPRefineLM(landmarks[ids], pixels[ids], k, None, rv, translation)
    rotation = cv2.Rodrigues(rv)[0]
    if not np.isfinite(translation).all() or np.any(landmarks[ids] @ rotation[2] + translation[2] <= 0):
        return None, 0
    return np.column_stack((rotation, translation)), len(ids)


def reconstruct(frames, times, depth, mask, k, name, sampling):
    gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in frames]
    corners = cv2.goodFeaturesToTrack(gray[0], 700, .005, 2., mask=mask)
    if corners is None or len(corners) < 6:
        raise ValueError("Fewer than six textured, measured corners lie inside the initial mask.")
    pixels = corners[:, 0]
    landmarks = unproject(pixels, depth, k)
    y, x = np.nonzero(mask)
    if len(x) > 40000:
        selection = np.linspace(0, len(x) - 1, 40000).astype(int)
        x, y = x[selection], y[selection]
    points = unproject(np.column_stack((x, y)), depth, k)
    center = np.median(points, axis=0)
    rows = []
    pose, count = np.eye(3, 4), len(pixels)
    cv2.setRNGSeed(128)
    flow_args = dict(winSize=(21, 21), maxLevel=3)
    for i in range(len(frames)):
        if i:
            pose, count = None, 0
            if len(pixels) >= 6:
                forward, valid, error = cv2.calcOpticalFlowPyrLK(gray[i-1], gray[i], pixels, None, **flow_args)
                if forward is None:
                    pixels = pixels[:0]
                else:
                    back, returned, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[i-1], forward, None, **flow_args)
                    if back is None:
                        pixels = pixels[:0]
                    else:
                        keep = ((valid.ravel() != 0) & (returned.ravel() != 0) & (error.ravel() < 22)
                                & (np.linalg.norm(back - pixels, axis=1) < .8))
                        pixels, landmarks = forward[keep], landmarks[keep]
                    if len(pixels) >= 6:
                        pose, count = estimate_pose(landmarks, pixels, k)
        if pose is not None:
            # Canonical points are centered on the visible surface of the initial view.
            pose[:, 3] += pose[:, :3] @ center
        rows.append(dict(status="observed" if pose is not None else "unavailable", inliers=count,
                         pose=pose.round(8).ravel().tolist() if pose is not None else None))
    if all(row["pose"] is None for row in rows[1:]):
        raise ValueError("Could not establish a second pose from the measured reference surface.")
    h, w = depth.shape
    return dict(version=1, method="reference-rgbd-pnp", model="OpenCV LK + PnP",
                depth_method="reference-rgbd-pnp", units="metres",
                camera_mode="fixed", coordinate_system="opencv", width=w, height=h, frame_count=len(frames),
                intrinsics=k.ravel().tolist(), intrinsics_source="reference_calibration",
                horizontal_fov_deg=float(np.degrees(2 * np.arctan(w / (2 * k[0, 0])))), sampling=sampling,
                fps=float((len(frames)-1)/(times[-1]-times[0])), times=times.round(5).tolist(),
                camera_to_world=[np.eye(3, 4).ravel().tolist()] * len(frames),
                parts=[dict(name=name, points=(points-center).round(7).tolist(), colors=frames[0][y, x].tolist(),
                            frames=rows, new_points_per_frame=[len(points)] + [0] * (len(frames)-1))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("video", "mask", "depth", "intrinsics", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--name", default="object")
    parser.add_argument("--sampling", choices=SAMPLING, default="standard")
    parser.add_argument("--depth-scale", type=float, help="metres per depth unit, required for PNG; .npy defaults to metres")
    args = parser.parse_args()
    if args.depth_scale is not None and (not np.isfinite(args.depth_scale) or args.depth_scale <= 0):
        parser.error("Depth scale must be positive and finite.")
    if not 1 <= len(args.name.strip()) <= 80:
        parser.error("Object name must be 1–80 characters.")
    frames, times = load_clip(args.video, *SAMPLING[args.sampling])
    cap = cv2.VideoCapture(str(args.video))
    try:
        native = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    try:
        depth, mask, k = read_reference(args, native, frames[0].shape[1::-1])
        result = reconstruct(frames, times, depth, mask, k, args.name.strip(), args.sampling)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".part", delete=False) as draft:
        temp = Path(draft.name)
    try:
        temp.write_text(json.dumps(result, separators=(",", ":"), ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)
    observed = sum(row["status"] == "observed" for row in result["parts"][0]["frames"])
    print(f"{observed}/{len(frames)} poses observed from the initial measured surface.")


if __name__ == "__main__":
    main()
