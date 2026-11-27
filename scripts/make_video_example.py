"""Render a repeatable articulated two-link scene for the video reconstruction example."""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT, FPS, FRAME_COUNT = 640, 416, 4, 16
K = np.array([[610., 0., WIDTH / 2], [0., 610., HEIGHT / 2], [0., 0., 1.]])
PARTS = (
    ("teal-link", (2.65, .58, .35), (137, 132, 51), 6.00),
    ("amber-link", (2.52, .48, .32), (53, 144, 212), 5.37),
)
FACE_NAMES = ("front", "back", "upper", "lower", "base", "tip")
# Vertices below are ordered so that the rendered face texture is continuous with motion.
FACES = ((0, 1, 2, 3), (5, 4, 7, 6), (4, 5, 1, 0),
         (3, 2, 6, 7), (4, 0, 3, 7), (1, 5, 6, 2))


def vertices(dimensions):
    length, width, thick = dimensions
    return np.array([[x, y, z] for z in (-thick / 2, thick / 2)
                     for x, y in ((0, -width / 2), (length, -width / 2),
                                  (length, width / 2), (0, width / 2))])


def rotation(axis, degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose(index, part):
    t = index / (FRAME_COUNT - 1)
    sweep = (1 - math.cos(math.pi * t)) / 2
    if part == 0:
        heading, yaw, roll = -30 + 40 * sweep, -6 - 10 * sweep, 18 - 52 * sweep
    else:
        heading, yaw, roll = 22 - 52 * sweep, 7 + 7 * sweep, -22 + 57 * sweep
    transform = np.eye(4)
    transform[:3, :3] = rotation("z", heading) @ rotation("y", yaw) @ rotation("x", roll)
    transform[:3, 3] = [-1.35, 0., PARTS[part][3]]
    return transform


def project(xyz):
    coordinates = xyz @ K.T
    return coordinates[:, :2] / coordinates[:, 2:3]


def texture(part, face):
    rng = np.random.default_rng(1703 + part * 41 + face * 151)
    width, height = (480, 112) if face < 4 else (112, 112)
    base = np.asarray(PARTS[part][2], dtype=float) * (1 if face == 0 else .84)
    pixels = base + rng.normal(0, 2, (height, width, 1))
    tex = np.clip(pixels, 0, 255).astype(np.uint8)
    dark, light = (39, 45, 43), (209, 223, 222)
    cv2.rectangle(tex, (4, 4), (width - 5, height - 5), dark, 3, cv2.LINE_AA)
    for _ in range(55 if face < 4 else 15):
        x, y = int(rng.integers(10, width - 12)), int(rng.integers(12, height - 15))
        color = light if rng.random() < .43 else dark
        if rng.random() < .5:
            cv2.rectangle(tex, (x, y), (x + int(rng.integers(4, 12)), y + 5), color, -1)
        else:
            cv2.circle(tex, (x, y), int(rng.integers(3, 7)), color, -1, cv2.LINE_AA)
    if face < 4:
        cv2.rectangle(tex, (145, 30), (342, 80), dark, -1)
        cv2.putText(tex, f"{'A' if part == 0 else 'B'}{face} / {27 + face * 5}",
                    (153, 66), cv2.FONT_HERSHEY_SIMPLEX, 1, light, 2, cv2.LINE_AA)
    for x in (14, width - 15):
        for y in (15, height - 15):
            cv2.circle(tex, (x, y), 7, dark, -1, cv2.LINE_AA)
            cv2.line(tex, (x - 2, y + 2), (x + 2, y - 2), light, 2)
    return tex


def background():
    """Static machine backplate with enough unique landmarks for camera tracking."""
    rng = np.random.default_rng(324)
    img = np.full((HEIGHT, WIDTH, 3), (60, 66, 67), np.float32)
    img += rng.normal(0, .8, (HEIGHT, WIDTH, 1)).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for y in (44, 354):
        cv2.line(img, (0, y), (WIDTH, y), (86, 92, 92), 1)
    for _ in range(60):
        x, y = (int(rng.integers(12, WIDTH - 12)), int(rng.integers(12, HEIGHT - 12)))
        cv2.circle(img, (x, y), int(rng.integers(2, 5)), (73, 80, 80), 1, cv2.LINE_AA)
    for x, y in ((43, 77), (590, 67), (78, 330), (567, 366)):
        cv2.circle(img, (x, y), 7, (35, 43, 44), -1, cv2.LINE_AA)
        cv2.line(img, (x - 3, y + 3), (x + 3, y - 3), (108, 112, 113), 1, cv2.LINE_AA)
    cv2.putText(img, "AXIS 02", (477, 39), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (127, 137, 137), 1, cv2.LINE_AA)
    return img


def draw_face(image, depth, labels, face_ids, mask, corners, tex, part_id, face_id):
    inward_normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    if inward_normal @ corners[0] <= 0:
        return
    xy = project(corners)
    x0, y0 = np.maximum(np.floor(xy.min(axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(xy.max(axis=0)).astype(int) + 1, (WIDTH, HEIGHT))
    if x1 <= x0 or y1 <= y0:
        return
    polygon = (xy - [x0, y0]).astype(np.float32)
    if abs(cv2.contourArea(polygon)) < 1:
        return
    shape = (y1 - y0, x1 - x0)
    footprint = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(footprint, np.int32(np.round(polygon * 16)), 1, shift=4)
    hit = footprint.astype(bool)
    mask[y0:y1, x0:x1] |= hit
    # Inverse depth varies affinely in screen space for one projected 3D plane.
    plane, *_ = np.linalg.lstsq(np.column_stack([polygon, np.ones(4)]), 1 / corners[:, 2], rcond=None)
    y, x = np.ogrid[:shape[0], :shape[1]]
    z = (1. / np.maximum(plane[0] * x + plane[1] * y + plane[2], 1.e-6)).astype(np.float32)
    region = np.s_[y0:y1, x0:x1]
    visible = hit & (z < depth[region])
    if not visible.any():
        return
    h, w = tex.shape[:2]
    uv = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    homography = cv2.getPerspectiveTransform(uv, polygon)
    colors = cv2.warpPerspective(tex, homography, (shape[1], shape[0]), borderMode=cv2.BORDER_REPLICATE)
    image[region][visible] = colors[visible]
    depth[region][visible] = z[visible]
    labels[region][visible] = part_id
    face_ids[region][visible] = face_id


def joint_fixture():
    """Fixed mounting plate and a through-shaft with a slotted retaining screw."""
    plate_center = np.array([-1.35, 0., 6.35])
    plate = np.full((96, 96, 3), (82, 89, 90), np.uint8)
    cv2.rectangle(plate, (3, 3), (92, 92), (130, 139, 140), 2)
    for u in (12, 83):
        for v in (12, 83):
            cv2.circle(plate, (u, v), 4, (39, 44, 45), -1, cv2.LINE_AA)
            cv2.line(plate, (u - 2, v), (u + 2, v), (150, 154, 155), 1)
    image = background()
    depth = np.full((HEIGHT, WIDTH), 8.25, dtype=np.float32)
    dummy = np.zeros((HEIGHT, WIDTH), np.uint8)
    local = vertices((.76, .62, .11))
    local[:, 0] -= .38
    for ids in FACES:
        draw_face(image, depth, dummy, dummy, np.zeros_like(dummy, dtype=bool),
                  local[list(ids)] + plate_center, plate, 0, 0)

    # Intersect camera rays with the circular cylinder and its near cap.
    py, px = np.indices(depth.shape, dtype=float)
    dx, dy = (px - K[0, 2]) / K[0, 0], (py - K[1, 2]) / K[1, 1]
    cx, cy = plate_center[:2]
    radius, near, far = .116, 5.12, plate_center[2]
    a, b = dx * dx + dy * dy, dx * cx + dy * cy
    disc = b * b - a * (cx * cx + cy * cy - radius * radius)
    side_z = (b - np.sqrt(np.maximum(disc, 0))) / np.maximum(a, 1.e-9)
    side = (disc >= 0) & (side_z >= near) & (side_z < far) & (side_z < depth)
    image[side] = (63, 70, 72)
    depth[side] = side_z[side]
    gx, gy = dx * near - cx, dy * near - cy
    r = np.hypot(gx, gy)
    cap = (r <= radius) & (near < depth)
    image[cap] = (113, 119, 123)
    image[cap & (r < radius * .78)] = (152, 157, 160)
    image[cap & (r < radius * .57) & (np.abs(gx + gy) < .014)] = (54, 58, 61)
    depth[cap] = near
    return image, depth


def render(index, bg, textures):
    image, depth = (array.copy() for array in bg)
    labels = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    face_ids = np.zeros_like(labels)
    alone, poses, states = [], [], []
    for p, (name, dims, _, _) in enumerate(PARTS):
        transform = pose(index, p)
        xyz = vertices(dims) @ transform[:3, :3].T + transform[:3, 3]
        mask = np.zeros(labels.shape, dtype=bool)
        for f, ids in enumerate(FACES):
            draw_face(image, depth, labels, face_ids, mask, xyz[list(ids)], textures[p][f], p + 1, f)
        length = dims[0]
        ends = np.array([[0., 0., 0.], [length, 0., 0.]]) @ transform[:3, :3].T + transform[:3, 3]
        center = ends.mean(axis=0, keepdims=True)
        states.append(dict(name=name, camera_from_part=transform.round(7).tolist(),
                           center_xyz=center[0].round(6).tolist(), center_px=project(center)[0].round(3).tolist(),
                           axis_px=project(ends).round(3).tolist(), landmarks_px=project(xyz).round(3).tolist()))
        alone.append(mask)
        poses.append(transform)
    for p, state in enumerate(states):
        visible = labels == p + 1
        state["visible_pixels"] = int(visible.sum())
        state["visible_fraction"] = round(float(visible.sum() / alone[p].sum()), 5)
        state["visible_faces"] = {name: int(np.count_nonzero(visible & (face_ids == f)))
                                  for f, name in enumerate(FACE_NAMES)}
    return image, depth, labels, poses, states


def write_video(path, frames):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        subprocess.run(command, input=b"".join(frame.tobytes() for frame in frames), check=True)
        return
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Install ffmpeg with its libx264 encoder to create the browser-playable example.")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-arrays", type=Path, help="Optional ground truth NPZ for development/evaluation only")
    args = parser.parse_args()
    dest = args.output_dir
    dest.mkdir(parents=True, exist_ok=True)
    textures = [[texture(p, f) for f in range(len(FACES))] for p in range(len(PARTS))]
    bg = joint_fixture()
    frames, frame_truth, depths, labels, transforms = [], [], [], [], []
    parts = []
    for i in range(FRAME_COUNT):
        frame, depth, ids, poses, states = render(i, bg, textures)
        frames.append(frame)
        frame_truth.append(dict(index=i, time=i / FPS, parts=states))
        if args.reference_arrays:
            depths.append(depth)
            labels.append(ids)
            transforms.append(poses)
        if i == 0:
            for p, (name, *_rest) in enumerate(PARTS):
                ys, xs = np.nonzero(ids == p + 1)
                box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) / [WIDTH, HEIGHT, WIDTH, HEIGHT]
                parts.append(dict(name=name, box=box.round(6).tolist()))
    write_video(dest / "video.mp4", frames)
    (dest / "parts.json").write_text(json.dumps(parts, indent=2) + "\n", encoding="utf-8")
    truth = dict(fps=FPS, image_size=[WIDTH, HEIGHT], camera_intrinsics=K.tolist(),
                 dimensions={name: dims for name, dims, _, _ in PARTS}, frames=frame_truth)
    (dest / "truth.json").write_text(json.dumps(truth) + "\n", encoding="utf-8")
    if args.reference_arrays:
        np.savez_compressed(args.reference_arrays, depth=np.array(depths), labels=np.array(labels),
                            K=K, T_camera_from_part=np.array(transforms))


if __name__ == "__main__":
    main()
