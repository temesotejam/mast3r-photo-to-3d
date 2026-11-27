"""Per-frame axial depth priors from the pinned MASt3R model used by the photo app."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from reconstruct import MODEL, MODEL_REVISION


def estimate_depths(frames, boxes, camera_mode, device="auto"):
    import torch

    vendor = Path(os.environ.get("MAST3R_DIR") or Path(__file__).resolve().parent / "vendor/mast3r")
    sys.path.insert(0, str(vendor.resolve()))
    from mast3r.model import AsymmetricMASt3R

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    height, width = frames[0].shape[:2]
    edge = 320 if device == "cpu" else 512
    factor = min(1.0, edge / max(width, height))
    nw, nh = (max(16, round(v * factor / 16) * 16) for v in (width, height))
    xs = (np.arange(nw, dtype=np.float32) - (nw - 1) / 2)[None, :]
    ys = (np.arange(nh, dtype=np.float32) - (nh - 1) / 2)[:, None]
    rad2 = xs * xs + ys * ys
    model = AsymmetricMASt3R.from_pretrained(MODEL, revision=MODEL_REVISION).to(device).eval()

    def view(index):
        rgb = cv2.resize(frames[index], (nw, nh), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        return {"img": image.float().div_(127.5).sub_(1.0), "instance": [str(index)],
                "true_shape": torch.tensor([[nh, nw]], dtype=torch.int32, device=device)}

    depths, focals = [], []
    background = np.ones((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        background[y0:y1, x0:x1] = False
    with torch.inference_mode():
        for t, frame in enumerate(frames):
            # Head 1 returns coordinates in the current camera frame. The other
            # head reports points in that same frame and is not a depth map for
            # the neighbour's camera.
            other = t + 1 if t + 1 < len(frames) else t - 1
            pred, _ = model(view(t), view(other))
            xyz = pred["pts3d"][0].float().cpu().numpy()
            z = xyz[:, :, 2]
            finite = np.isfinite(xyz).all(axis=2) & (z > 1e-6)
            if np.count_nonzero(finite) < z.size // 3:
                raise ValueError(f"Frame {t + 1} has insufficient usable depth.")
            # Pinhole ray equation: (X/Z,Y/Z)=(u-cx,v-cy)/f.
            dot = xs * xyz[:, :, 0] + ys * xyz[:, :, 1]
            good = finite & (rad2 > (0.15 * max(nw, nh)) ** 2) & (dot > 1e-8)
            if good.any():
                focal = np.median(rad2[good] * z[good] / dot[good])
                if 0.25 * max(nw, nh) < focal < 5 * max(nw, nh):
                    focals.append(float(focal))
            z = cv2.resize(np.where(finite, z, np.nan).astype(np.float32), (width, height),
                           interpolation=cv2.INTER_LINEAR)
            if depths and camera_mode == "fixed":
                # Same background pixels should retain the same depth. Ignore
                # moving silhouettes using appearance change against frame zero.
                unchanged = np.max(cv2.absdiff(frame, frames[0]), axis=2) < 24
                keep = background & unchanged & np.isfinite(z) & np.isfinite(depths[0])
                if keep.sum() > z.size * 0.04:
                    scale = np.median(depths[0][keep] / z[keep])
                    if np.isfinite(scale) and scale > 0:
                        z *= scale
            depths.append(z)
            print(f"Estimated depth {t + 1}/{len(frames)}.", flush=True)
    focal = np.median(focals) if focals else max(nw, nh)
    k = np.array([[focal * width / nw, 0, (width - 1) / 2],
                  [0, focal * height / nh, (height - 1) / 2], [0, 0, 1]], dtype=np.float64)
    return depths, k
