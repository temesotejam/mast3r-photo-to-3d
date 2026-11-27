"""Batch reconstruction with NAVER MASt3R. Output is an RGB binary PLY point cloud."""

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile


MODEL = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
MODEL_REVISION = "06e7259f34c3060f322df5cb0c7b9094f57e41fc"
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def list_photos(folder):
    if not folder.is_dir():
        raise ValueError(f"Input is not a directory: {folder}")
    photos = sorted((p for p in folder.iterdir() if p.suffix.lower() in PHOTO_EXTENSIONS),
                    key=lambda p: p.name.casefold())
    if not 2 <= len(photos) <= 10:
        raise ValueError(f"Select 2-10 JPG, PNG or WebP photos (found {len(photos)}).")
    if any(p.is_symlink() or not p.is_file() for p in photos):
        raise ValueError("Photo inputs must be regular files.")
    return photos


def prepare_photos(photos, temporary_dir, edge):
    from PIL import Image, ImageOps

    prepared = []
    for index, photo in enumerate(photos):
        # Stable numeric filenames preserve UI order and avoid using upload names as cache keys.
        destination = temporary_dir / f"view-{index:02d}.png"
        try:
            with Image.open(photo, formats=("JPEG", "PNG", "WEBP")) as source:
                width, height = source.size
                if width * height > 100_000_000:
                    raise ValueError("more than 100 megapixels")
                if round(edge * min(width, height) / max(width, height)) < 32:
                    raise ValueError("image is too narrow for the model")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("choose a still photo")
                source.draft("RGB", (edge, edge))
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
                image.save(destination, format="PNG")
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise ValueError(f"Cannot read {photo.name}: {error}") from error
        prepared.append(str(destination))
    return prepared


@contextmanager
def portable_cache(alignment):
    # The pinned upstream names a cache file with the repr of kwargs. On Windows,
    # the ':' in {'mode': 'avg-angle'} is illegal. canonical_view defaults to
    # exactly 'avg-angle', so omitting that redundant keyword preserves the math.
    if os.name != "nt":
        yield
        return
    original = alignment.prepare_canonical_data

    def prepare(*args, **kwargs):
        if kwargs.get("mode") == "avg-angle":
            kwargs.pop("mode")
        return original(*args, **kwargs)

    alignment.prepare_canonical_data = prepare
    try:
        yield
    finally:
        alignment.prepare_canonical_data = original


def export_ply(output, pointmaps, depthmaps, confidences, colors, first_pose, threshold=1.5):
    """Write tightly packed little-endian x/y/z float32 + red/green/blue uint8 (15 B)."""
    import numpy as np

    dtype = np.dtype([(k, "<f4") for k in ("x", "y", "z")] +
                     [(k, "u1") for k in ("red", "green", "blue")])
    chunks = []
    # World -> first camera, then to a viewer-friendly +Y up, -Z forward frame.
    world_to_first = np.linalg.inv(first_pose)
    for points, depth, confidence, image in zip(pointmaps, depthmaps, confidences, colors):
        points = np.asarray(points).reshape(-1, 3)
        depth = np.asarray(depth).ravel()
        confidence = np.asarray(confidence).ravel()
        rgb = np.asarray(image).reshape(-1, 3)
        valid = (np.isfinite(points).all(axis=1) & np.isfinite(depth) & (depth > 0)
                 & np.isfinite(confidence) & (confidence > threshold))
        if not valid.any():
            continue
        xyz = points[valid] @ world_to_first[:3, :3].T + world_to_first[:3, 3]
        xyz[:, 1:] *= -1
        rgb8 = np.rint(np.clip(rgb[valid], 0, 1) * 255).astype(np.uint8)
        records = np.empty(len(xyz), dtype=dtype)
        for axis, channel in enumerate(("x", "y", "z")):
            records[channel] = xyz[:, axis]
        for axis, channel in enumerate(("red", "green", "blue")):
            records[channel] = rgb8[:, axis]
        chunks.append(records)
    total = sum(len(chunk) for chunk in chunks)
    if not total:
        raise ValueError("No points passed the confidence filter. Try sharper views with more overlap.")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {total}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".part", delete=False) as file:
        pending = Path(file.name)
    try:
        with pending.open("wb") as file:
            file.write(header.encode("ascii"))
            for chunk in chunks:
                chunk.tofile(file)
        os.replace(pending, output)
    finally:
        pending.unlink(missing_ok=True)
    return total


def reconstruct(args):
    photos = list_photos(args.input)
    if args.output.suffix.lower() != ".ply":
        raise ValueError("Output filename must end in .ply")
    mast3r_dir = args.mast3r_dir.resolve()
    if not all((mast3r_dir / rel).is_file() for rel in
               ("mast3r/model.py", "dust3r/dust3r/model.py", "dust3r/croco/models/croco.py")):
        raise ValueError(f"MASt3R or its submodules are missing from {mast3r_dir}. Run the repository setup.")
    sys.path.insert(0, str(mast3r_dir))

    import torch
    from mast3r.cloud_opt import sparse_ga
    from mast3r.image_pairs import make_pairs
    from mast3r.model import AsymmetricMASt3R
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch could not find a CUDA device.")
    edge = args.image_size or (320 if device == "cpu" else 512)
    print(f"Preparing {len(photos)} photos for {device} ({edge} px).", flush=True)
    with tempfile.TemporaryDirectory(prefix="mast3r-") as scratch:
        paths = prepare_photos(photos, Path(scratch), edge)
        views = load_images(paths, size=edge, verbose=False)
        graph = "complete" if len(paths) <= 5 else "logwin-3-noncyclic"
        pairs = make_pairs(views, scene_graph=graph, symmetrize=True)
        print("Loading the MASt3R checkpoint (the first run downloads the weights).", flush=True)
        model = AsymmetricMASt3R.from_pretrained(MODEL, revision=MODEL_REVISION).to(device).eval()
        print(f"Matching {len(pairs) // 2} image pairs and aligning the views.", flush=True)
        with portable_cache(sparse_ga):
            scene = sparse_ga.sparse_global_alignment(
                paths, pairs, scratch, model, device=device, lr1=0.07, niter1=150,
                lr2=0.01, niter2=100, opt_depth=False, matching_conf_thr=0.0,
                verbose=False,
            )
        del model
        print("Building the color point cloud.", flush=True)
        with torch.no_grad():
            points, depths, confidence = to_numpy(scene.get_dense_pts3d(clean_depth=True))
            count = export_ply(args.output, points, depths, confidence, scene.imgs,
                               to_numpy(scene.get_im_poses()[0]))
    print(f"Saved {count:,} points to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Folder with 2-10 photos")
    parser.add_argument("--output", type=Path, required=True, help="Destination .ply file")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--image-size", type=int, choices=(320, 384, 512),
                        help="Longest model input edge (default: CPU 320, CUDA 512)")
    parser.add_argument("--mast3r-dir", type=Path,
                        default=os.getenv("MAST3R_DIR") or Path(__file__).resolve().parent / "vendor/mast3r")
    args = parser.parse_args()
    try:
        reconstruct(args)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Reconstruction failed: {error}\n")


if __name__ == "__main__":
    main()
