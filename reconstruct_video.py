"""Reconstruct persistent rigid parts and their estimated poses from one short video."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from reconstruct import MODEL, MODEL_REVISION
from video_depth import estimate_depths


def load_clip(path, sample_fps=4, max_frames=24):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot decode this video. Try an H.264 MP4.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite([fps, total]).all() or not (0 < fps <= 480 and total >= 3):
            raise ValueError("The video needs a readable frame rate and at least three frames.")
        total = int(total)
        if total / fps > 30.05:
            raise ValueError("Choose a clip up to 30 seconds long.")
        count = min(total, max_frames, max(3, round((total - 1) / fps * sample_fps) + 1))
        indices = np.rint(np.linspace(0, total - 1, count)).astype(int)
        frames, times = [], []
        for index in range(total):
            if not cap.grab():
                break
            if index != indices[len(frames)]:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}. Try exporting the clip as H.264.")
            height, width = frame.shape[:2]
            if min(height, width) < 32 or max(height, width) > 8192:
                raise ValueError("Unsupported video frame dimensions.")
            scale = min(1, 560 / max(width, height))
            if scale < 1:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                   interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            if len(frames) == count:
                break
        if len(frames) != count or any(f.shape != frames[0].shape for f in frames):
            raise ValueError("Frame count or resolution changed during decoding. Re-encode the clip.")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            times = indices / fps
        times = np.asarray(times) - times[0]
        return frames, times
    finally:
        cap.release()


def load_parts(path, width, height):
    items = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(items, list) or not 1 <= len(items) <= 6:
        raise ValueError("Select 1–6 separate rigid parts in the first frame.")
    parts = []
    for n, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Each part requires a name and box.")
        name = item.get("name")
        box = np.asarray(item.get("box"), dtype=float)
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("Each part needs a short name.")
        if box.shape != (4,) or not np.isfinite(box).all() or box.min() < 0 or box.max() > 1:
            raise ValueError("Use normalized [left, top, right, bottom] boxes in [0, 1].")
        x0, y0, x1, y1 = np.rint(box * [width, height, width, height]).astype(int)
        if min(x1 - x0, y1 - y0) < 8:
            raise ValueError(f"{name}: the selected part is too small to track.")
        parts.append({"id": str(n + 1), "name": name.strip(), "box": [x0, y0, x1, y1]})
    return parts


def serialize_parts(parts):
    result = []
    for part in parts:
        points = np.asarray(part["points"], dtype=np.float64)
        frames = []
        for observation in part["frames"]:
            pose = observation["part_to_world"]
            frames.append({"status": observation["status"], "inliers": int(observation["inliers"]),
                           "pose": None if pose is None else np.round(pose[:3].ravel(), 7).tolist()})
        result.append({"name": part["name"], "points": points.round(6).tolist(),
                       "colors": np.asarray(part["colors"], dtype=np.uint8).tolist(), "frames": frames,
                       "new_points_per_frame": [int(v) for v in part["new_points_per_frame"]]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", choices=("fixed", "moving"), default="fixed")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    frames, times = load_clip(args.video)
    height, width = frames[0].shape[:2]
    parts = load_parts(args.parts, width, height)
    z, k = estimate_depths(frames, [p["box"] for p in parts], args.camera, args.device)
    if args.camera == "moving":
        from video_camera import estimate_camera_poses
        cameras = estimate_camera_poses(frames, z, k, parts)
    else:
        cameras = [np.eye(4) for _ in frames]
    from video_parts import reconstruct_parts
    reconstructed = reconstruct_parts(frames, z, k, parts, camera_poses=cameras, times=times)
    if all(sum(r["status"] == "observed" for r in p["frames"]) < 2 for p in reconstructed):
        raise ValueError("動画内で部品を継続して追跡できませんでした。最初のコマで模様のある部品を小さめに囲み直してください。")
    output = {"version": 1, "depth_method": "mast3r-pairwise", "model": MODEL,
              "model_revision": MODEL_REVISION, "camera_mode": args.camera,
              "coordinate_system": "opencv", "units": "model scale", "width": width,
              "height": height, "intrinsics": k.ravel().tolist(), "frame_count": len(frames),
              "fps": (len(frames) - 1) / float(times[-1]), "times": times.round(5).tolist(),
              "camera_to_world": [np.round(np.asarray(p)[:3].ravel(), 7).tolist() for p in cameras],
              "parts": serialize_parts(reconstructed)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".part", delete=False) as f:
        temp = Path(f.name)
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)
    for part in output["parts"]:
        print(f"{part['name']}: {len(part['points'])} persistent points; "
              f"{sum(f['status'] == 'observed' for f in part['frames'])}/{len(frames)} poses observed.")


if __name__ == "__main__":
    main()
