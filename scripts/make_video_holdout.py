"""Render the independent, partially occluded two-link clip used for the accuracy check."""
import cv2
import numpy as np

import make_video_example as renderer


WIDTH, HEIGHT, FPS, FRAME_COUNT = 576, 432, 4, 14
INTRINSICS = np.array([[562., 0., 288.], [0., 562., 216.], [0., 0., 1.]])
PARTS = (
    ("rear-bar", (2.37, .63, .31), (174, 104, 74), 6.40),
    ("front-arm", (1.97, .44, .40), (78, 133, 180), 5.88),
)
VIEW_ROTATION = renderer.rotation("x", -9) @ renderer.rotation("y", 14)


def pose(index, part):
    t = index / (FRAME_COUNT - 1)
    if part == 0:
        # The rear bar reverses direction before the front arm finishes its sweep.
        heading = 71 + 30 * np.sin(1.5 * np.pi * t - .25)
        yaw = 10 * np.cos(np.pi * t + .2)
        roll = -15 + 36 * t + 8 * np.sin(2 * np.pi * t)
        center = np.array([-.19 + .11 * t, -1.02 + .08 * np.sin(np.pi * t), .31])
    else:
        heading = 115 - 47 * t + 5 * np.sin(np.pi * t)
        yaw = -12 + 24 * np.sin(np.pi * t)
        roll = 19 - 38 * t + 5 * np.sin(2 * np.pi * t)
        center = np.array([.08 - .07 * t, -.97 - .06 * np.sin(np.pi * t), -.21])
    transform = np.eye(4)
    transform[:3, :3] = (VIEW_ROTATION @ renderer.rotation("z", heading)
                         @ renderer.rotation("y", yaw) @ renderer.rotation("x", roll))
    transform[:3, 3] = VIEW_ROTATION @ center + [0., -.10, 6.1]
    return transform


def texture(part, face):
    height, width = (144, 448) if face < 4 else (112, 112)
    rng = np.random.default_rng(304610 + 37 * face + 131 * part)
    color = np.asarray(PARTS[part][2])
    tone = color[None, None, :] * (1 if face == 0 else .8)
    pixels = np.rint(tone + rng.normal(0, 4.5, (height, width, 1))).clip(0, 255).astype(np.uint8)
    for i in range(65 if face < 4 else 22):
        x, y = rng.integers((3, 3), (width - 9, height - 9))
        mark = (color * rng.choice([.38, .68, 1.43])).clip(0, 255).astype(int).tolist()
        if i % 3:
            end = x + int(rng.integers(-7, 7)), y + int(rng.integers(-9, 9))
            cv2.line(pixels, (x, y), end, mark, 2, cv2.LINE_AA)
        else:
            cv2.circle(pixels, (x, y), int(rng.integers(2, 5)), mark, -1, cv2.LINE_AA)
    if face < 4:
        for j, label in enumerate([f"LX{part + 1}", f"{face + 3}9"]):
            cv2.putText(pixels, label, (20 + 255 * j, 78), cv2.FONT_HERSHEY_DUPLEX, 1.,
                        (37, 39, 38), 2, cv2.LINE_AA)
    return pixels


def backplate():
    rng = np.random.default_rng(59121)
    y, x = np.indices((HEIGHT, WIDTH))
    gray = 88 + 9 * x / WIDTH + 8 * y / HEIGHT + rng.normal(0, 2, y.shape)
    image = (gray[..., None] * [.88, .94, 1.05]).astype(np.uint8)
    for _ in range(90):
        x_pos, y_pos = rng.integers((0, 0), (WIDTH, HEIGHT))
        cv2.circle(image, (x_pos, y_pos), int(rng.integers(1, 5)), (82, 86, 95), 1)
    return image, np.full((HEIGHT, WIDTH), 9., np.float32)


if __name__ == "__main__":
    renderer.WIDTH, renderer.HEIGHT = WIDTH, HEIGHT
    renderer.FPS, renderer.FRAME_COUNT = FPS, FRAME_COUNT
    renderer.K, renderer.PARTS = INTRINSICS, PARTS
    renderer.pose, renderer.texture, renderer.joint_fixture = pose, texture, backplate
    renderer.main()
