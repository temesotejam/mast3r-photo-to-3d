"""Compare fixed-camera point clouds and poses against the generated video's independent CAD truth."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def rotation_error(estimated, expected):
    """Shortest SO(3) angle between the two reference-to-current rotations."""
    relative = estimated.T @ expected
    skew = relative - relative.T
    sine = np.linalg.norm([skew[2, 1], skew[0, 2], skew[1, 0]]) / 2
    cosine = (np.trace(relative) - 1) / 2
    return float(np.degrees(np.arctan2(sine, cosine)))


def projected_outline_iou(xyz, intrinsics, truth_vertices, image_size):
    """Project the entire point cloud, including surfaces hidden by the other link.

    Both masks use original input pixels. A fixed 2 px point-splat radius tolerates
    discretization and small gaps without filling holes via a convex hull.
    The convex hull of the *known box* corners is its exact projected outline.
    """
    width, height = image_size
    truth_mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(np.rint(16 * np.asarray(truth_vertices)).astype(np.int32))
    cv2.fillConvexPoly(truth_mask, hull, 1, shift=4)

    cloud_mask = np.zeros_like(truth_mask)
    visible = xyz[xyz[:, 2] > 1.e-6]
    projected = visible @ intrinsics.T
    uv = np.rint(projected[:, :2] / projected[:, 2:3])
    inside = np.all((uv >= 0) & (uv < [width, height]), axis=1)
    u, v = uv[inside].astype(int).T
    cloud_mask[v, u] = 1
    cloud_mask = cv2.dilate(cloud_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    union = np.count_nonzero(cloud_mask | truth_mask)
    return float(np.count_nonzero(cloud_mask & truth_mask) / union)


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def evaluate(truth, scene):
    if scene['camera_mode'] != 'fixed':
        raise ValueError('This benchmark uses the fixed-camera reference clip.')
    width, height = truth['image_size']
    intrinsics = np.array(scene['intrinsics']).reshape(3, 3)
    # Map tracking-image pixel centers onto the original input pixel grid.
    sx, sy = width / scene['width'], height / scene['height']
    native_k = np.array([[sx, 0., (sx - 1) / 2], [0., sy, (sy - 1) / 2], [0., 0., 1.]]) @ intrinsics
    times = np.array([frame['time'] for frame in truth['frames']])
    indices = np.array([np.argmin(abs(times - value)) for value in scene['times']])
    if (len(indices) < 2 or len(set(indices)) != len(indices) or indices[0] != 0
            or max(abs(times[indices] - scene['times'])) > 0.5 / truth['fps'] + 1.e-6):
        raise ValueError('Scene timestamps must match distinct source frames starting at frame zero.')
    results = {}
    for part in scene['parts']:
        name = part['name']
        if len(part['frames']) != len(indices) or part['frames'][0]['pose'] is None:
            raise ValueError(f'{name}: poses must cover the sampled timeline and start with a valid pose.')
        cloud = np.asarray(part['points'])
        refs = [next(p for p in truth['frames'][i]['parts'] if p['name'] == name) for i in indices]
        initial_pose = np.array(part['frames'][0]['pose']).reshape(3, 4)
        true_initial = np.array(refs[0]['camera_from_part'])[:3, :3]
        # Recover size ratios along the independently known CAD axes. Translation
        # cancels in spans and model scale cancels when dividing by link length.
        cloud_in_cad = cloud @ initial_pose[:, :3].T @ true_initial
        spans = np.diff(np.quantile(cloud_in_cad, [.01, .99], axis=0), axis=0)[0]
        actual_ratio = float(spans[2] / spans[0])
        dimensions = truth['dimensions'][name]
        true_ratio = dimensions[2] / dimensions[0]
        by_frame = {}
        for row, ref, source_frame in zip(part['frames'], refs, indices):
            sample = {'status': row['status'], 'fully_hidden': ref['visible_fraction'] < .1,
                      'rotation_deg': None, 'iou': 0.}
            if row['pose'] is not None:
                pose = np.array(row['pose']).reshape(3, 4)
                expected = np.array(ref['camera_from_part'])[:3, :3] @ true_initial.T
                sample['rotation_deg'] = rotation_error(pose[:, :3] @ initial_pose[:, :3].T, expected)
                xyz = cloud @ pose[:, :3].T + pose[:, 3]
                sample['iou'] = projected_outline_iou(xyz, native_k, ref['landmarks_px'], (width, height))
            by_frame[int(source_frame)] = sample

        rows = list(by_frame.values())
        rotations = [r['rotation_deg'] for r in rows if r['rotation_deg'] is not None]
        counts = {s: sum(r['status'] == s for r in rows) for s in ('observed', 'interpolated', 'unavailable')}
        results[name] = {
            'frames': len(rows), 'status_counts': counts,
            'rotation_rmse_deg': rms(rotations),
            'outline_iou': float(np.mean([r['iou'] for r in rows])),
            'thickness_over_length': actual_ratio, 'true_thickness_over_length': true_ratio,
            'thickness_ratio_error': abs(actual_ratio - true_ratio), 'by_frame': by_frame,
        }
    return results


def compare(current, previous):
    if set(current) != set(previous):
        raise ValueError('Part names differ between scenes.')
    paired = {}
    for name, after in current.items():
        before = previous[name]
        a, b = after['by_frame'], before['by_frame']
        if set(a) != set(b):
            raise ValueError('Source frame coverage differs between scenes.')
        common = [i for i in a if all(v[i]['rotation_deg'] is not None for v in (a, b))]
        if not common:
            raise ValueError(f'No shared valid poses for {name}.')
        paired[name] = {
            'shared_poses': len(common),
            'rotation_rmse_deg_before': rms([b[i]['rotation_deg'] for i in common]),
            'rotation_rmse_deg_after': rms([a[i]['rotation_deg'] for i in common]),
            'outline_iou_before': before['outline_iou'], 'outline_iou_after': after['outline_iou'],
            'thickness_ratio_error_before': before['thickness_ratio_error'],
            'thickness_ratio_error_after': after['thickness_ratio_error'],
        }
    return paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('truth', type=Path, help='truth.json from make_video_example.py (evaluation only)')
    parser.add_argument('scene', type=Path)
    parser.add_argument('--baseline', type=Path, help='Optional older scene reconstructed from the same clip')
    parser.add_argument('--output', type=Path, help='Write measurements, including per-frame values, as JSON')
    parser.add_argument('--reference', type=Path, help='Renderer reference arrays for visible-surface precision and recall')
    parser.add_argument('--check-demo', action='store_true', help='Check accuracy for the public two-link fixture')
    args = parser.parse_args()
    truth = json.loads(args.truth.read_text())
    scene = json.loads(args.scene.read_text())
    current = evaluate(truth, scene)
    report = {'current': current}
    for name, value in current.items():
        counts = value['status_counts']
        print(f"{name}: 3D rotation RMSE {value['rotation_rmse_deg']:.2f} deg; "
              f"mean projected outline IoU {value['outline_iou']:.4f}; "
              f"thickness/length {value['thickness_over_length']:.3f} "
              f"(reference {value['true_thickness_over_length']:.3f}); poses "
              f"{counts['observed']} observed / {counts['interpolated']} interpolated / "
              f"{counts['unavailable']} unavailable.")
    if args.baseline:
        previous_scene = json.loads(args.baseline.read_text())
        previous = evaluate(truth, previous_scene)
        report['baseline'] = previous
        report['comparison'] = compare(current, previous)
        for name, p in report['comparison'].items():
            print(f"{name}: matched {p['shared_poses']} poses, relative rotation RMSE "
                  f"{p['rotation_rmse_deg_before']:.2f} -> {p['rotation_rmse_deg_after']:.2f} deg; "
                  f"all-frame outline IoU {p['outline_iou_before']:.4f} -> {p['outline_iou_after']:.4f}; "
                  f"absolute thickness/length error {p['thickness_ratio_error_before']:.3f} -> "
                  f"{p['thickness_ratio_error_after']:.3f}.")
    if args.reference:
        from video_surface_quality import evaluate_surface

        surfaces = {'current': evaluate_surface(truth, scene, args.reference)}
        if args.baseline:
            surfaces['baseline'] = evaluate_surface(truth, previous_scene, args.reference)
        report['visible_surfaces'] = surfaces
        for version, parts in surfaces.items():
            for name, scores in parts.items():
                print(f"{name} ({version}): visible surface precision {scores['precision']:.3f}, "
                      f"recall {scores['recall']:.3f}, F-score {scores['f_score']:.3f} "
                      f"at {100 * scores['tolerance_over_length']:.1f}% of part length.")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.check_demo:
        for name, value in current.items():
            # A flipped thin part can have a convincing 2D outline. Require
            # correct orientation, shape and time coverage independently.
            if (value['rotation_rmse_deg'] > 15 or value['outline_iou'] < .70
                    or value['thickness_ratio_error'] > .20 or value['status_counts']['unavailable']):
                raise SystemExit(f'{name}: generated fixture exceeds the reconstruction error limits')


if __name__ == '__main__':
    main()
