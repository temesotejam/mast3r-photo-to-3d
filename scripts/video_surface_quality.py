"""Measure reconstruction accuracy and coverage on the synthetic surfaces visible in the input.

The distance-based precision, recall and F-score follow the Tanks and Temples evaluation
(tanksandtemples.org/tutorial/). Here distance is relative to part length. Reference
depth and labels are used only for scoring. Each part has unknown scale and origin, so
a positive isotropic scale and translation are fitted per part, with no rotation fit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.spatial import cKDTree


def reduce_voxels(points, spacing):
    indices = np.floor(points / spacing).astype(np.int64)
    _, which = np.unique(indices, axis=0, return_inverse=True)
    count = np.bincount(which)
    return np.column_stack([np.bincount(which, weights=v) / count for v in points.T])


def visible_surface(depths, labels, camera_matrices, transforms, part_id, indices, length):
    """Unproject only rays visibly labelled as this object, into its CAD coordinate frame."""
    points = []
    for i in indices:
        z = depths[i]
        y, x = np.nonzero((labels[i] == part_id + 1) & np.isfinite(z) & (z > 0))
        k = camera_matrices[i]
        rays = np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(k).T
        camera_points = rays * (z[y, x] / rays[:, 2])[:, None]
        transform = transforms[i, part_id]
        points.append((camera_points - transform[:3, 3]) @ transform[:3, :3] / length)
    surface = np.vstack(points)
    if not len(surface):
        raise ValueError('Part has no visible surface in the sampled source frames.')
    return reduce_voxels(surface, .005)


def profile_shift(source, reference):
    """Initial translations from 1D density profiles resist missing patches and stray points."""
    shift = []
    for src, dst in zip(source.T, reference.T):
        step = max(.004, np.ptp(np.quantile(src, [.001, .999])) / 2048,
                   np.ptp(np.quantile(dst, [.001, .999])) / 2048)
        hists, origins = [], []
        for values in (dst, src):
            lo, hi = np.quantile(values, [.001, .999])
            ids = np.floor((values[(values >= lo) & (values <= hi)] - lo) / step).astype(int)
            hists.append(np.sqrt(np.bincount(ids, minlength=1)))
            origins.append(lo)
        correlation = correlate(*hists, mode='full')
        lags = correlation_lags(len(hists[0]), len(hists[1]), mode='full')
        shift.append(origins[0] - origins[1] + step * lags[np.argmax(correlation)])
    return np.asarray(shift)


def align_shape(reconstruction, reference):
    """Fit four nuisance parameters from equally weighted forward and reverse correspondences."""
    lo, hi = np.quantile(reference, [.01, .99], axis=0)
    axis = np.argmax(hi - lo)
    target_tree = cKDTree(reference)
    starts = []
    for q in (0., .01):
        ref_lo, ref_hi = np.quantile(reference, [q, 1-q], axis=0)
        lo, hi = np.quantile(reconstruction, [q, 1-q], axis=0)
        if hi[axis] - lo[axis] <= 1.e-10:
            continue
        scale = (ref_hi[axis] - ref_lo[axis]) / (hi[axis] - lo[axis])
        centers = [.5 * (ref_lo + ref_hi - scale * (lo + hi)),
                   np.median(reference, axis=0) - scale * np.median(reconstruction, axis=0),
                   profile_shift(reconstruction * scale, reference)]
        for shift in centers:
            source = reconstruction * scale + shift
            # Clipped distances limit the influence of surfaces missing in one cloud.
            forward = target_tree.query(source)[0]
            reverse = cKDTree(source).query(reference)[0]
            loss = np.mean(np.minimum(forward, .05)) + np.mean(np.minimum(reverse, .05))
            starts.append((loss, scale, shift))
    if not starts:
        raise ValueError('Reconstructed part has no measurable extent along its reference length.')
    _, initial_scale, initial_shift = min(starts, key=lambda s: s[0])
    source = reduce_voxels(reconstruction * initial_scale + initial_shift, .005)
    scale, shift = 1., np.zeros(3)
    def weights(distances):
        # A missing patch cannot pull both clouds towards its boundary.
        sigma = max(.003, np.median(distances))
        values = 1 / (1 + (distances / sigma)**2)
        return .5 * values / np.sum(values)

    for _ in range(25):
        aligned = scale * source + shift
        forward_dist, forward = target_tree.query(aligned)
        reverse_dist, reverse = cKDTree(aligned).query(reference)
        # Trim each direction separately. Points outside this fit still count in the final score.
        good_fwd = forward_dist <= np.quantile(forward_dist, .85)
        good_rev = reverse_dist <= np.quantile(reverse_dist, .85)
        x = np.vstack((source[good_fwd], source[reverse[good_rev]]))
        y = np.vstack((reference[forward[good_fwd]], reference[good_rev]))
        w = np.r_[weights(forward_dist[good_fwd]), weights(reverse_dist[good_rev])]
        mx, my = np.einsum('i,ij->j', w, x), np.einsum('i,ij->j', w, y)
        xx, yy = x - mx, y - my
        covariance = np.einsum('i,ij,ij->', w, xx, yy)
        variance = np.einsum('i,ij,ij->', w, xx, xx)
        slope = covariance / max(variance, 1.e-12)
        if not np.isfinite(slope) or slope <= 0:
            break
        new_shift = my - slope * mx
        change = abs(slope - scale) + np.linalg.norm(new_shift - shift)
        scale, shift = slope, new_shift
        if change < 1.e-6:
            break
    return initial_scale * scale, initial_shift * scale + shift


def evaluate_surface(truth, scene, reference_npz, threshold=.02):
    if scene['camera_mode'] != 'fixed':
        raise ValueError('The fixed-camera surface benchmark expects camera-relative part poses.')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Distance tolerance must be positive in units of part length.')
    times = np.array([f['time'] for f in truth['frames']])
    indices = np.array([np.argmin(abs(times - t)) for t in scene['times']])
    if (len(np.unique(indices)) != len(indices)
            or np.any(abs(times[indices] - scene['times']) > .5 / truth['fps'] + 1.e-6)):
        raise ValueError('Scene times do not match distinct source frames.')
    names = [part['name'] for part in truth['frames'][0]['parts']]
    if sorted(p['name'] for p in scene['parts']) != sorted(names):
        raise ValueError('Expected all reference parts in the reconstruction.')
    results = {}
    with np.load(reference_npz) as arrays:
        depths, labels = arrays['depth'], arrays['labels']
        ks = np.broadcast_to(arrays['K'], (len(depths), 3, 3))
        transforms = arrays['T_camera_from_part']
        width, height = truth['image_size']
        expected_poses = np.array([[p['camera_from_part'] for p in row['parts']] for row in truth['frames']])
        if (depths.shape != (len(times), height, width) or labels.shape != depths.shape
                or transforms.shape != expected_poses.shape or not np.allclose(transforms, expected_poses, atol=1.e-6)
                or not np.allclose(ks, np.asarray(truth['camera_intrinsics']))):
            raise ValueError('Reference arrays and trajectory annotations describe different renders.')
        for part in scene['parts']:
            name = part['name']
            part_id = names.index(name)
            length = max(truth['dimensions'][name])
            ref = visible_surface(depths, labels, ks, transforms, part_id, indices, length)
            points = np.asarray(part['points'], dtype=float).reshape(-1, 3)
            first = next((i for i,f in enumerate(part['frames']) if f['status'] == 'observed' and f['pose'] is not None), None)
            result = {'precision': 0., 'recall': 0., 'f_score': 0., 'reference_points': len(ref),
                      'reconstruction_points': 0, 'tolerance_over_length': threshold,
                      'alignment_scale': None, 'alignment_translation_over_length': None}
            if first is not None and len(points) > 2:
                # The observed pose defines the reconstructed reference axes; translation
                # and scale are free, while any orientation/shape errors remain measurable.
                rotation = np.array(part['frames'][first]['pose']).reshape(3, 4)[:, :3]
                cloud = points @ rotation.T @ transforms[indices[first], part_id, :3, :3]
                scale, shift = align_shape(cloud, ref)
                aligned = reduce_voxels(cloud * scale + shift, .005)
                p = float(np.mean(cKDTree(ref).query(aligned)[0] <= threshold))
                r = float(np.mean(cKDTree(aligned).query(ref)[0] <= threshold))
                result.update(precision=p, recall=r, f_score=2*p*r/(p+r) if p+r else 0.,
                              reconstruction_points=len(aligned), alignment_scale=float(scale),
                              alignment_translation_over_length=shift.tolist())
            results[name] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('truth', type=Path)
    parser.add_argument('reference', type=Path, help='NPZ emitted by make_video_example.py --reference-arrays')
    parser.add_argument('scene', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    truth = json.loads(args.truth.read_text())
    scenes = {'current': args.scene}
    if args.baseline:
        scenes['baseline'] = args.baseline
    report = {}
    for version, path in scenes.items():
        report[version] = evaluate_surface(truth, json.loads(path.read_text()), args.reference)
        for name, result in report[version].items():
            print(f"{version} {name}: visible surface at 2% of part length: "
                  f"F={result['f_score']:.3f}, precision={result['precision']:.3f}, recall={result['recall']:.3f}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
