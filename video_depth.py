"""Combine target-camera pointmaps from several support frames per video frame."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _estimate_focal(xyz, confidence, xs, ys):
    """Fit a centered pinhole camera by robustly minimizing pixel reprojection error."""
    # A coarse uniform grid keeps broad image coverage. Cap weights to stop a
    # single very textured patch from controlling the calibration.
    xyz = xyz[::3, ::3]
    conf = confidence[::3, ::3]
    z = xyz[..., 2]
    keep = np.isfinite(xyz).all(2) & np.isfinite(conf) & (z > 1e-6)
    px, py = np.broadcast_arrays(xs[:, ::3], ys[::3, :])
    keep &= np.hypot(px, py) > 0.1 * max(xs.size, ys.size)
    if keep.sum() < 16:
        return None
    rays = xyz[keep, :2] / z[keep, None]
    pixels = np.column_stack((px[keep], py[keep]))
    dot = (rays * pixels).sum(1)
    squared = (rays * rays).sum(1)
    weight = np.sqrt(np.clip(conf[keep] - 1, 0.01, 100))
    positive = (squared > 1e-9) & (dot > 0)
    if positive.sum() < 16:
        return None
    focal = np.median(dot[positive] / squared[positive])
    for _ in range(6):
        residual = np.linalg.norm(pixels - focal * rays, axis=1)
        weights = weight / np.maximum(0.5, residual)
        focal = (weights @ dot) / max(weights @ squared, 1e-9)
    return float(focal) if 0.25 * max(xs.size, ys.size) < focal < 5 * max(xs.size, ys.size) else None


def _fuse_predictions(predictions):
    """Align pair scales in one target view and suppress inconsistent depth predictions."""
    depths, confs = (np.stack(a) for a in zip(*predictions))
    valid = np.isfinite(depths) & (depths > 1e-6) & np.isfinite(confs)
    # A single bad pair must not determine the scale. Each pair observes the
    # same target pixels, even when the camera or the object moves in the clip.
    clean = np.where(valid, depths, np.nan)
    common = valid.all(axis=0)
    if common.sum() >= 32:
        pair_scales = np.median(depths[:, common], axis=1)
        clean *= (np.median(pair_scales) / pair_scales)[:, None, None]
    # Weighted Huber location on log depth. MAD derives the local agreement
    # threshold; the small floor preserves fine variations when views agree.
    log_depth = np.log(clean)
    observed = valid.any(axis=0)
    log_depth[:, ~observed] = 0
    center = np.nanmedian(log_depth, axis=0)
    deviations = np.abs(log_depth - center)
    width = np.maximum(0.015, 1.5 * np.nanmedian(deviations, axis=0))
    weights = np.where(valid, np.sqrt(np.clip(confs - 1, 0.01, 100)), 0)
    weights *= np.minimum(1, width / np.maximum(np.nan_to_num(deviations), 1e-6))
    consensus = np.sum(weights * np.nan_to_num(log_depth), axis=0) / np.maximum(weights.sum(axis=0), 1e-8)
    return np.where(observed, np.exp(consensus), np.nan).astype(np.float32)


def _guided_resize(z, coarse_rgb, frame):
    """Interpolate using spatial distance and RGB agreement at the original resolution."""
    h, w = frame.shape[:2]
    mh, mw = z.shape
    if (h, w) == (mh, mw):
        return z
    x = ((np.arange(w, dtype=np.float32) + 0.5) * mw / w - 0.5)[None, :]
    y = ((np.arange(h, dtype=np.float32) + 0.5) * mh / h - 0.5)[:, None]
    nearest_x, nearest_y = np.rint(x).astype(int), np.rint(y).astype(int)
    numerator = np.zeros((h, w), np.float32)
    denominator = np.zeros_like(numerator)
    frame = frame.astype(np.float32)
    for dy in (-1, 0, 1):
        iy = np.clip(nearest_y + dy, 0, mh - 1)
        for dx in (-1, 0, 1):
            ix = np.clip(nearest_x + dx, 0, mw - 1)
            spatial = np.square(ix - x) + np.square(iy - y)
            color = np.square(coarse_rgb[iy, ix] - frame).mean(axis=2)
            weight = np.exp(-spatial / (2 * 0.65**2)) * np.maximum(1e-5, np.exp(-color / (2 * 22**2)))
            sample = z[iy, ix]
            weight = np.where(np.isfinite(sample), weight, 0)
            numerator += np.nan_to_num(sample) * weight
            denominator += weight
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def estimate_depths(frames, boxes, camera_mode, device="auto"):
    """Return axial depth maps in input pixel order and their shared camera matrix."""
    vendor = Path(os.environ.get("MAST3R_DIR") or Path(__file__).resolve().parent / "vendor/mast3r")
    sys.path.insert(0, str(vendor.resolve()))
    from mast3r.model import AsymmetricMASt3R
    from reconstruct import MODEL, MODEL_REVISION

    model = AsymmetricMASt3R.from_pretrained(MODEL, revision=MODEL_REVISION)
    return estimate_pairwise_depths(frames, boxes, camera_mode, device, model)


def estimate_pairwise_depths(frames, boxes, camera_mode, device, model):
    """Fuse target-camera pointmaps from a DUSt3R-family model with a shared image encoder."""
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    original_h, original_w = frames[0].shape[:2]
    portrait = original_h > original_w
    if portrait:
        frames = [np.rot90(frame) for frame in frames]
        boxes = [(y0, original_w - x1, y1, original_w - x0) for x0, y0, x1, y1 in boxes]
    height, width = frames[0].shape[:2]
    factor = min(1.0, 512 / max(width, height))
    nw, nh = (max(16, round(v * factor / 16) * 16) for v in (width, height))
    xs = (np.arange(nw, dtype=np.float32) - (nw - 1) / 2)[None, :]
    ys = (np.arange(nh, dtype=np.float32) - (nh - 1) / 2)[:, None]
    model = model.to(device).eval()
    small = [cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA) for frame in frames]
    shape = torch.tensor([[nh, nw]], dtype=torch.int32, device=device)
    background = np.ones((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        background[y0:y1, x0:x1] = False
    depths, focals = [], []
    with torch.inference_mode():
        # The encoder is independent of its partner. Reuse its tokens across
        # all pairs and decode only the target pixels (head 2 describes the support image).
        encoded = []
        for rgb in small:
            img = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
            feature, position, _ = model._encode_image(img.float().div_(127.5).sub_(1.), shape)
            encoded.append((feature, position))
        for t, frame in enumerate(frames):
            # Close support helps retain correspondences during fast motion.
            # Wider support contributes another baseline to disambiguate depth.
            neighbors = [t + d for d in (-3, -1, 1, 3) if 0 <= t + d < len(frames)]
            if len(neighbors) < 2:
                neighbors = sorted((i for i in range(len(frames)) if i != t), key=lambda i: abs(i-t))[:2]
            predictions = []
            for other in neighbors:
                feature, position = encoded[t]
                support, support_position = encoded[other]
                target_tokens, _ = model._decoder(feature, position, support, support_position)
                pred = model._downstream_head(1, list(target_tokens), shape)
                xyz = pred["pts3d"][0].float().cpu().numpy()
                conf = pred["conf"][0].float().cpu().numpy()
                focal = _estimate_focal(xyz, conf, xs, ys)
                if focal is not None:
                    focals.append(focal)
                predictions.append((xyz[..., 2].copy(), conf))
            z = _fuse_predictions(predictions)
            if np.count_nonzero(np.isfinite(z)) < z.size // 3:
                raise ValueError(f"Frame {t + 1} has insufficient usable depth.")
            z = _guided_resize(z, small[t], frame)
            if depths and camera_mode == "fixed":
                unchanged = np.max(cv2.absdiff(frame, frames[0]), axis=2) < 24
                keep = background & unchanged & np.isfinite(z) & np.isfinite(depths[0])
                if keep.sum() > z.size * 0.04:
                    z *= np.median(depths[0][keep] / z[keep])
            depths.append(z)
            print(f"Estimated depth {t + 1}/{len(frames)} using {len(neighbors)} support frames.", flush=True)
    focal = np.median(focals) if focals else max(nw, nh)
    fx, fy = focal * width / nw, focal * height / nh
    if portrait:
        fx, fy = fy, fx
        depths = [np.ascontiguousarray(np.rot90(z, -1)) for z in depths]
    k = np.array([[fx, 0, (original_w - 1) / 2], [0, fy, (original_h - 1) / 2],
                  [0, 0, 1]], dtype=np.float64)
    return depths, k
