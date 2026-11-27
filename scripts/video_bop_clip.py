"""Prepare a short RGB clip from an extracted BOP scene or score an exported camera-relative reconstruction."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def annotations(folder, name):
    return json.loads((folder / name).read_text())


def target(gt, frame, object_id):
    instances = gt[str(frame)]
    ids = [i for i, entry in enumerate(instances) if entry["obj_id"] == object_id]
    if len(ids) != 1:
        raise ValueError(f"Expected one instance of object {object_id} at frame {frame}.")
    return ids[0], instances[ids[0]]


def image(path, mode):
    frame = cv2.imread(str(path), mode)
    if frame is None:
        raise ValueError(f"Missing image: {path}")
    return frame


def prepare(args, ids):
    folder = args.scene_dir
    first = ids[0]
    first_rgb = image(folder / "rgb" / f"{first:06d}.png", cv2.IMREAD_COLOR)
    h, w = first_rgb.shape[:2]
    idx, _ = target(annotations(folder, "scene_gt.json"), first, args.object_id)
    x, y, bw, bh = annotations(folder, "scene_gt_info.json")[str(first)][idx]["bbox_visib"]
    if bw <= 0 or bh <= 0:
        raise ValueError("The initial frame must contain the selected object.")
    k = annotations(folder, "scene_camera.json")[str(first)]["cam_K"]
    fov = np.degrees(2 * np.arctan(w / (2 * k[0])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clip = args.output_dir / "clip.mp4"
    boxfile = args.output_dir / "parts.json"
    command = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", "4", "-i", "pipe:0", "-c:v", "libx264", "-preset", "fast",
               "-crf", "12", "-pix_fmt", "yuv420p", str(clip)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for i in ids:
            rgb = first_rgb if i == first else image(folder / "rgb" / f"{i:06d}.png", cv2.IMREAD_COLOR)
            if rgb.shape != first_rgb.shape:
                raise ValueError("The image size changed during the clip.")
            encoder.stdin.write(rgb.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg could not encode the clip.")
    boxfile.write_text(json.dumps([{"name": f"obj_{args.object_id:06d}",
                                    "box": [x/w, y/h, (x+bw)/w, (y+bh)/h]}]) + "\n")
    print(shlex.join(["python", "reconstruct_video.py", "--video", str(clip), "--parts", str(boxfile),
                      "--camera", "fixed", "--horizontal-fov-deg", str(fov),
                      "--output", str(args.output_dir / "scene.json")]))


def score(args, ids):
    scene = json.loads(args.reconstruction.read_text())
    if (scene["camera_mode"] != "fixed" or scene["frame_count"] != len(ids)
            or not np.allclose(scene["times"], np.arange(len(ids)) / 4, atol=.005)):
        raise ValueError("Expected camera-relative poses at every prepared 4 Hz clip frame.")
    part, = scene["parts"]
    if part["name"] != f"obj_{args.object_id:06d}" or len(part["frames"]) != len(ids):
        raise ValueError("The reconstruction does not match the selected object and frames.")
    cloud = np.asarray(part["points"])
    k = np.asarray(scene["intrinsics"]).reshape(3, 3)
    gt = annotations(args.scene_dir, "scene_gt.json")
    _, initial = target(gt, ids[0], args.object_id)
    r0 = np.asarray(initial["cam_R_m2c"]).reshape(3, 3)
    est0 = np.asarray(part["frames"][0]["pose"]).reshape(3, 4)[:, :3]
    errors, ious = [], []
    splat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for i, obs in zip(ids, part["frames"]):
        index, gt_pose = target(gt, i, args.object_id)
        mask = image(args.scene_dir / "mask" / f"{i:06d}_{index:06d}.png", cv2.IMREAD_GRAYSCALE) > 0
        if obs["pose"] is None:
            ious.append(0.)
            continue
        pose = np.asarray(obs["pose"]).reshape(3, 4)
        gt_rot = np.asarray(gt_pose["cam_R_m2c"]).reshape(3, 3) @ r0.T
        errors.append(np.degrees(Rotation.from_matrix(pose[:, :3] @ est0.T @ gt_rot.T).magnitude()))
        xyz = cloud @ pose[:, :3].T + pose[:, 3]
        projection = xyz[xyz[:, 2] > 0] @ k.T
        uv = projection[:, :2] / projection[:, 2:]
        h, w = mask.shape
        uv = np.rint((uv + .5) * [w / scene["width"], h / scene["height"]] - .5).astype(int)
        u, v = uv[np.all((uv >= 0) & (uv < [w, h]), axis=1)].T
        rendered = np.zeros(mask.shape, np.uint8)
        rendered[v, u] = 1
        rendered = cv2.dilate(rendered, splat).astype(bool)
        ious.append(np.count_nonzero(rendered & mask) / max(1, np.count_nonzero(rendered | mask)))
    print(json.dumps({"frames": len(ids), "valid_poses": len(errors),
                      "observed_poses": sum(f["status"] == "observed" for f in part["frames"]),
                      "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(errors)))),
                      "mean_amodal_outline_iou": float(np.mean(ious))}, indent=2))


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scene-dir", type=Path, required=True, help="e.g. tudl/test/000001")
    common.add_argument("--object-id", type=int, default=1)
    common.add_argument("--first", type=int, required=True)
    common.add_argument("--last", type=int, required=True, help="Last source frame, inclusive")
    common.add_argument("--stride", type=int, default=8)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("prepare", parents=[common]).add_argument("--output-dir", type=Path, required=True)
    commands.add_parser("score", parents=[common]).add_argument("--reconstruction", type=Path, required=True)
    args = parser.parse_args()
    if args.first < 0 or args.last < args.first or args.stride <= 0:
        parser.error("Use nonnegative source frame IDs and a positive stride.")
    ids = list(range(args.first, args.last + 1, args.stride))
    if not 3 <= len(ids) <= 24:
        parser.error("Choose 3–24 frames for the standard 4 Hz sampling mode.")
    (prepare if args.mode == "prepare" else score)(args, ids)


if __name__ == "__main__":
    main()
