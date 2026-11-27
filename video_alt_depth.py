"""Estimate RGB video depth with the indoor metric Depth Anything V2 Small checkpoint."""

import cv2
import numpy as np


MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
MODEL_REVISION = "8078d68a9c75a972131914f6afd0c1723be0da7f"


def estimate_depths(frames, boxes, camera_mode, device="auto"):
    """Return axial depth at each input pixel and a provisional 60-degree horizontal FoV."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL, revision=MODEL_REVISION, use_safetensors=True
    ).to(device).eval()
    height, width = frames[0].shape[:2]
    if min(height, width) < 14:
        raise ValueError("The video aspect ratio leaves too few pixels for a depth patch.")
    # Preserving the aspect ratio while resizing the short edge to the model's
    # default could otherwise produce thousands of tokens on a panoramic clip.
    sizing = {}
    if max(height, width) > 3 * min(height, width):
        factor = 518 / max(height, width)
        target = {name: max(14, round(value * factor / 14) * 14)
                  for name, value in (("height", height), ("width", width))}
        sizing = {"size": target, "keep_aspect_ratio": False}
    background = np.ones((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        background[y0:y1, x0:x1] = False
    depths = []
    with torch.inference_mode():
        for i, image in enumerate(frames):
            inputs = processor(images=image, return_tensors="pt", **sizing).to(device)
            prediction = model(**inputs)
            output = processor.post_process_depth_estimation(prediction, [(height, width)])[0]
            # This is metric axial depth, not a display-normalized image or relative disparity.
            z = output["predicted_depth"].float().cpu().numpy()
            valid = np.isfinite(z) & (z > 0)
            if valid.sum() < z.size // 3:
                raise ValueError(f"Frame {i + 1} has insufficient usable depth.")
            z[~valid] = np.nan
            if depths and camera_mode == "fixed":
                unchanged = np.max(cv2.absdiff(image, frames[0]), axis=2) < 24
                stable = unchanged & background & valid & np.isfinite(depths[0])
                if stable.sum() > z.size * .04:
                    z *= np.median(depths[0][stable] / z[stable])
            depths.append(z)
            print(f"Depth Anything V2: estimated depth {i + 1}/{len(frames)}.", flush=True)

    focal = width / (2 * np.tan(np.deg2rad(30.)))
    k = np.array([[focal, 0., (width - 1) / 2], [0., focal, (height - 1) / 2], [0., 0., 1.]])
    return depths, k
