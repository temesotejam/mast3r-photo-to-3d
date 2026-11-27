"""Infer depth from the official MonST3R dynamic-scene point-pair model."""
import os
import sys
from pathlib import Path

from video_depth import estimate_pairwise_depths


MODEL = "Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"
MODEL_REVISION = "1ea2a9eade23a10898ec73be3fd44b5aa4ea0255"
DEPTH_METHOD = "monst3r-pairwise"


def estimate_depths(frames, boxes, camera_mode, device="auto"):
    vendor = Path(os.environ.get("MONST3R_DIR") or Path(__file__).resolve().parent / "vendor/monst3r")
    if not (vendor / "croco/models/croco.py").is_file():
        raise RuntimeError("Install the MonST3R source and dependencies with python scripts/setup_monst3r.py.")
    sys.path.insert(0, str(vendor.resolve()))
    from dust3r.model import AsymmetricCroCo3DStereo

    model = AsymmetricCroCo3DStereo.from_pretrained(MODEL, revision=MODEL_REVISION, strict=True)
    return estimate_pairwise_depths(frames, boxes, camera_mode, device, model)
